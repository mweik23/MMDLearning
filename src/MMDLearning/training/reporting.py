from pathlib import Path
SRC_DIR = (Path(__file__).parent).resolve()
import sys
sys.path.append(str(SRC_DIR))
from MMDLearning.utils.distributed import is_master

def display_epoch_summary(*, 
                          partition: str, 
                          epoch: int, 
                          tot_epochs: int, 
                          bce: float, 
                          mmd: float, 
                          acc: float, 
                          time_s: float, 
                          domain: str = 'Source',
                          best_epoch: int = None,
                          best_val: float = None,
                          logger=None):
    if not is_master(): 
        return
    msg = (f"Domain: {domain}[{partition}] Epoch {epoch}/{tot_epochs} — "
           f"BCE {bce:.4f}, MMD {mmd:.4f}, Acc {acc:.4f}, Time {time_s:.1f}s")
    (logger.info if logger else print)(msg)
    if partition == 'validation' and best_epoch is not None and best_val is not None:
        msg = f"  (best val epoch {best_epoch} with loss {best_val:.4f})"
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

def finish_roc_plot(path, ax, is_primary=True):
    if is_primary:
        ax.set_xlabel('tpr')
        ax.set_ylabel('1/fpr')
        ax.set_xlim([0, 1])
        ax.set_yscale('log')
        ax.legend(frameon=False)
        fig = ax.figure
        fig.tight_layout()
        fig.savefig(f"{path}/ROC_curve.pdf", dpi=300, bbox_inches="tight")
        return ax
    return None
