from pathlib import Path
SRC_DIR = (Path(__file__).parent).resolve()
import sys
sys.path.append(str(SRC_DIR))
from MMDLearning.utils.distributed import is_master

def display_epoch_summary(*, partition: str, epoch: int, tot_epochs: int, bce: float, mmd: float, acc: float, time_s: float, logger=None):
    if not is_master(): 
        return
    msg = (f"[{partition}] Epoch {epoch}/{tot_epochs} — "
           f"BCE {bce:.4f}, MMD {mmd:.4f}, Acc {acc:.4f}, Time {time_s:.1f}s")
    (logger.info if logger else print)(msg)
    return msg

def display_status(*, partition: str, domain: str, epoch: int, tot_epochs: int,
                   batch_idx: int, num_batches: int,
                   running_bce: float, running_mmd: float,
                   running_acc: float, total_acc: float, avg_batch_time: float,
                   logger=None):
    if not is_master():
        return
    msg = (f">> {partition} ({domain}):\tEpoch {epoch}/{tot_epochs}\t"
           f"Batch {batch_idx}/{num_batches}\t"
           f"BCE {running_bce:.4f}\tMMD {running_mmd:.4f}\t"
           f"RunAcc {running_acc:.3f}\tTotAcc {total_acc:.3f}\t"
           f"AvgBatchTime {avg_batch_time:.4f}s")
    (logger.info if logger else print)(msg)
    return msg
