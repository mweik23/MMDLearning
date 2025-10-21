# tests/smoke_cpu_ddp_var.py
import os, torch, torch.distributed as dist
from torch.distributed.nn.functional import all_gather as dist_all_gather
from pathlib import Path
import sys
import time 
SRC_path = Path(__file__).parents[1].resolve() / 'src' / 'MMDLearning'
sys.path.append(str(SRC_path))
from utils.distributed import (
    dist_global_variance_autograd,
    dist_global_variance_nograd,
)

def init():
    # Pick backend based on availability; for GPUs use NCCL.
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)

    # Map this rank to a single CUDA device
    if backend == "nccl":
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)

def teardown():
    dist.destroy_process_group()

def device():
    return torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")

def gather_full_batch(x):
    # autograd-aware all_gather; all parts must have same shape
    parts = dist_all_gather(x)
    return torch.cat(parts, dim=0)

def main():
    init()
    rank, world = dist.get_rank(), dist.get_world_size()
    dev = device()

    # Make sure each rank gets different data
    torch.manual_seed(1234 + rank)
    if dev.type == "cuda":
        torch.cuda.manual_seed_all(1234 + rank)

    # Keep same N per rank so we can gather easily without padding
    N, D = 8, 7
    x = torch.randn(N, D, dtype=torch.float32, device=dev, requires_grad=True)

    # ---- AUTOGRAD VERSION (GPU) ----
    v_auto = dist_global_variance_autograd(x, unbiased=True)  # scalar
    loss = v_auto
    loss.backward()  # should populate x.grad and include cross-rank deps

    # Reference (gather to check value only)
    with torch.no_grad():
        x_all = gather_full_batch(x.detach())
        ref = x_all.var(dim=0, unbiased=True).sum()

    # Print on rank 0
    if rank == 0:
        print(f"[autograd] var={float(v_auto):.6f}  ref={float(ref):.6f}")

    # Close value match on all ranks
    assert torch.allclose(v_auto.detach(), ref, atol=1e-6, rtol=1e-6)

    # Gradients should be finite and non-null
    assert x.grad is not None and torch.isfinite(x.grad).all()

    # ---- NOGRAD VERSION (GPU) ----
    with torch.no_grad():
        v_eval = dist_global_variance_nograd(x.detach(), unbiased=True)
        if rank == 0:
            print(f"[nograd]   var={float(v_eval):.6f}  ref={float(ref):.6f}")
        assert torch.allclose(v_eval, ref, atol=1e-6, rtol=1e-6)
    if rank == 0:
        #create a slight delay so that all ranks' prints don't jumble
        time.sleep(0.1)
    print(f"Rank {rank} passed.")
    
    # Optional: sync before exit for cleaner teardown
    if dev.type == "cuda":
        torch.cuda.synchronize()

    teardown()

if __name__ == "__main__":
    main()
