import torch
import torch.distributed as dist

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
