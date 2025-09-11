import torch
import torch.distributed as dist
import os
import socket
from dataclasses import dataclass
from typing import Optional, Dict, Any, Sequence
import multiprocessing as mp

def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()

def is_master() -> bool:
    return (not is_dist()) or dist.get_rank() == 0

def _sum_reduce_scalar(x: float, device: torch.device) -> float:
    if not is_dist():
        return float(x)
    t = torch.tensor([float(x)], device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t.item())

def globalize_epoch_totals(*, 
    local_bce_sum: float,
    local_mmd_sum: float,
    local_correct: int,
    local_count: int,
    device: torch.device,
):
    g_bce  = _sum_reduce_scalar(local_bce_sum, device)
    g_mmd  = _sum_reduce_scalar(local_mmd_sum, device)
    g_corr = _sum_reduce_scalar(float(local_correct), device)
    g_cnt  = _sum_reduce_scalar(float(local_count), device)
    return g_bce, g_mmd, int(round(g_corr)), int(round(g_cnt))

def epoch_metrics_from_globals(*, g_bce_sum: float, g_mmd_sum: float, g_correct: int, g_count: int):
    if g_count == 0:
        return dict(bce=0.0, mmd=0.0, acc=0.0)
    return dict(
        bce = g_bce_sum / g_count,
        mmd = g_mmd_sum / g_count,
        acc = g_correct / g_count,
    )

@dataclass
class DistInfo:
    backend: str
    initialized: bool
    rank: int
    local_rank: int
    world_size: int
    node_rank: int
    master_addr: str
    master_port: int
    device_type: str
    device_name: str
    has_cuda: bool
    is_primary: bool
    hostname: str
    # NEW:
    num_workers: int
    cpus_per_task: Optional[int] = None
    cpu_affinity_count: Optional[int] = None
    
    def shared_dict(
        self,
        keys: Optional[Sequence[str]] = None,
        include: Optional[Sequence[str]] = None,
        exclude: Optional[Sequence[str]] = None,
        drop_none: bool = True,
    ) -> Dict[str, Any]:
        """
        Return a dict of attributes we *expect* to be identical across ranks.
        No distributed checks—pure selection.

        Args:
          keys: base set of fields to include. If None, uses sensible defaults.
          include: additional field names to add.
          exclude: field names to remove.
          drop_none: omit keys whose value is None.
        """
        default_keys = (
            "backend",
            "world_size",
            "master_addr",
            "master_port",
            "device_type",
            "has_cuda",
            "num_workers",          # include if you expect same per rank
            "cpus_per_task",        # often same under SLURM
        )

        selected = list(keys or default_keys)
        if include:
            selected.extend(include)
        if exclude:
            selected = [k for k in selected if k not in set(exclude)]

        out: Dict[str, Any] = {}
        for k in selected:
            if hasattr(self, k):
                v = getattr(self, k)
                if not (drop_none and v is None):
                    out[k] = v
        return out

def _infer_rank_world_local():
    """
    Returns (rank, world_size, local_rank) from a variety of environments.
    Priority order:
      torchrun/launch > SLURM > OMPI/PMI > fallback single-process
    """
    # torchrun / legacy launch
    rank       = _get_env_int("RANK")
    world_size = _get_env_int("WORLD_SIZE")
    local_rank = _get_env_int("LOCAL_RANK")

    if rank is not None and world_size is not None:
        # LOCAL_RANK might be missing on multi-node w/some setups; try other hints
        if local_rank is None:
            # SLURM_LOCALID is often set even under torchrun inside srun
            local_rank = _get_env_int("SLURM_LOCALID", "MPI_LOCALRANKID", "OMPI_COMM_WORLD_LOCAL_RANK", default=0)
        return rank, world_size, local_rank or 0

    # SLURM
    slurm_rank       = _get_env_int("SLURM_PROCID")
    slurm_world      = _get_env_int("SLURM_NTASKS")
    slurm_local_rank = _get_env_int("SLURM_LOCALID")
    if slurm_rank is not None and slurm_world is not None:
        return slurm_rank, slurm_world, slurm_local_rank or 0

    # OMPI / PMI (e.g., OpenMPI, MPICH)
    ompi_rank  = _get_env_int("OMPI_COMM_WORLD_RANK", "PMI_RANK")
    ompi_world = _get_env_int("OMPI_COMM_WORLD_SIZE", "PMI_SIZE")
    ompi_local = _get_env_int("OMPI_COMM_WORLD_LOCAL_RANK", "MPI_LOCALRANKID")
    if ompi_rank is not None and ompi_world is not None:
        return ompi_rank, ompi_world, ompi_local or 0

    # Fallback single process
    return 0, 1, 0

def _infer_node_rank():
    # Explicit override first (PyTorch elastic/DeepSpeed sometimes set NODE_RANK)
    node_rank = _get_env_int("NODE_RANK")
    if node_rank is not None:
        return node_rank
    # SLURM
    node_rank = _get_env_int("SLURM_NODEID")
    if node_rank is not None:
        return node_rank
    # Kubernetes / others sometimes pass GROUP_RANK
    node_rank = _get_env_int("GROUP_RANK")
    if node_rank is not None:
        return node_rank
    # Fallback
    return 0

def _choose_device(local_rank: int):
    # CUDA takes precedence
    if torch.cuda.is_available():
        num = torch.cuda.device_count()
        # Respect CUDA_VISIBLE_DEVICES mapping: local_rank indexes into visible set
        if num > 0:
            try:
                torch.cuda.set_device(local_rank % num)
            except Exception:
                # As a backup, set device 0 to avoid crashing
                torch.cuda.set_device(0)
            name = torch.cuda.get_device_name(torch.cuda.current_device())
            return "cuda", name, True
    # Apple MPS (Apple Silicon)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps", "MPS", False
    # CPU fallback
    return "cpu", "CPU", False

def _choose_backend(device_type: str) -> str:
    # NCCL only works with CUDA; Gloo is safe everywhere
    if device_type == "cuda":
        return "nccl"
    return "gloo"

def _resolve_master():
    # torchrun sets these; otherwise default to rank0 hostname and a common port
    addr = _get_env_str("MASTER_ADDR", default=None)
    port = _get_env_int("MASTER_PORT", default=None)

    if addr and port:
        return addr, port

    # SLURM usually has node list; but simplest safe fallback:
    # If not provided, try envs sometimes set by launchers
    addr = addr or _get_env_str("SLURM_LAUNCH_NODE_IPADDR", "HOSTNAME", default="127.0.0.1")
    port = port or 29500
    return addr, int(port)

def _maybe_init_process_group(backend: str, rank: int, world_size: int, master_addr: str, master_port: int):
    if dist.is_available() and not dist.is_initialized():
        # Ensure essential envs exist for init_method="env://"
        os.environ.setdefault("MASTER_ADDR", str(master_addr))
        os.environ.setdefault("MASTER_PORT", str(master_port))
        os.environ.setdefault("RANK", str(rank))
        os.environ.setdefault("WORLD_SIZE", str(world_size))
        dist.init_process_group(backend=backend, init_method="env://", rank=rank, world_size=world_size)


def _get_env_int(*keys: str, default: Optional[int] = None) -> Optional[int]:
    for k in keys:
        v = os.environ.get(k)
        if v:
            try: return int(v)
            except ValueError: pass
    return default

def _get_env_str(*keys: str, default: Optional[str] = None) -> Optional[str]:
    for k in keys:
        v = os.environ.get(k)
        if v: return v
    return default

# --- existing _infer_rank_world_local / _infer_node_rank / _choose_device / _choose_backend / _resolve_master / _maybe_init_process_group unchanged ---

# NEW: detect available CPUs inside containers / affinity-limited jobs
def _cpu_count_available() -> int:
    # 1) Linux CPU affinity
    try:
        return len(os.sched_getaffinity(0))  # type: ignore[attr-defined]
    except Exception:
        pass
    # 2) cgroups v2 quota
    try:
        with open("/sys/fs/cgroup/cpu.max", "r") as f:
            quota_str, period_str = f.read().strip().split()
            if quota_str != "max":
                quota = int(quota_str); period = int(period_str)
                if period > 0:
                    return max(1, quota // period)
    except Exception:
        pass
    # 3) cpuset mask (v1/v2)
    for p in ("/sys/fs/cgroup/cpuset.cpus.effective", "/sys/fs/cgroup/cpuset.cpus"):
        try:
            with open(p, "r") as f:
                spec = f.read().strip()
                if spec:
                    total = 0
                    for part in spec.split(","):
                        if "-" in part:
                            a, b = part.split("-")
                            total += int(b) - int(a) + 1
                        else:
                            total += int(part)
                    if total > 0:
                        return total
        except Exception:
            continue
    # 4) fallback
    try:
        return mp.cpu_count()
    except Exception:
        return 1

# NEW: infer num_workers (per process)
def _infer_num_workers(arg_num_workers: Optional[int] = None) -> tuple[int, Optional[int], Optional[int]]:
    # 0) CLI wins
    if arg_num_workers is not None:
        return max(0, int(arg_num_workers)), None, None

    # 1) explicit env overrides
    for k in ("NUM_WORKERS", "PYTORCH_NUM_WORKERS", "DATALOADER_NUM_WORKERS"):
        v = _get_env_int(k)
        if v is not None:
            return max(0, v), None, None

    # 2) SLURM: per-task CPUs is the right “budget” for this rank
    cpt = _get_env_int("SLURM_CPUS_PER_TASK")
    if cpt is not None and cpt > 0:
        reserve = _get_env_int("NUM_WORKER_RESERVE_THREADS", default=2)  # leave a little headroom
        return max(0, cpt - reserve), cpt, None

    # 3) Generic availability (affinity / cgroups)
    avail = _cpu_count_available()
    reserve = _get_env_int("NUM_WORKER_RESERVE_THREADS", default=2)
    return max(0, avail - reserve), None, avail

def setup_dist(arg_num_workers: Optional[int] = None) -> DistInfo:  # NEW: optional arg
    hostname = socket.gethostname()

    rank, world_size, local_rank = _infer_rank_world_local()
    node_rank = _infer_node_rank()
    master_addr, master_port = _resolve_master()

    device_type, device_name, has_cuda = _choose_device(local_rank)
    backend = _choose_backend(device_type)

    try:
        _maybe_init_process_group(backend, rank, world_size, master_addr, master_port)
    except Exception as e:
        if world_size > 1:
            raise e

    # NEW:
    num_workers, cpus_per_task, cpu_affinity = _infer_num_workers(arg_num_workers)

    return DistInfo(
        backend=backend,
        initialized=dist.is_available() and dist.is_initialized(),
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        node_rank=node_rank,
        master_addr=str(master_addr),
        master_port=int(master_port),
        device_type=device_type,
        device_name=device_name,
        has_cuda=has_cuda,
        is_primary=(rank == 0),
        hostname=hostname,
        num_workers=num_workers,
        cpus_per_task=cpus_per_task,
        cpu_affinity_count=cpu_affinity,
    )

