from pathlib import Path
import matplotlib.pyplot as plt
import torch
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

def display_status(*, phase: str, domain: str, epoch: int, tot_epochs: int,
                   batch_idx: int, num_batches: int,
                   running_bce: float, running_mmd: float,
                   running_acc: float, total_acc: float, avg_batch_time: float,
                   logger=None):
    if not is_master():
        return
    msg = (f">> {phase} ({domain}):\tEpoch {epoch}/{tot_epochs}\t"
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



def make_train_plt(train_metrics, path, pretrained=False, do_MMD=False, rename_map={}):
    keys_needed = ['epochs', 'train_BCE_loss', 'val_BCE_loss']
    if do_MMD:
        keys_needed += ['train_MMD_loss', 'val_MMD_loss', 'train_loss', 'val_loss'] 
    
    for k, v in rename_map.items():
        if k in train_metrics:
            if v not in keys_needed: 
                print(f"Warning: Attempting to rename {k} to {v} but {v} not in keys_needed")
            train_metrics[v] = train_metrics.pop(k)
    assert all(k in train_metrics for k in keys_needed), f"Missing keys in train_metrics. Found {list(train_metrics.keys())}, need {keys_needed}"
    if pretrained:
        train_start = 1
    else:
        train_start = 0
    fig, ax = plt.subplots()
    ax.plot(train_metrics['epochs'][train_start:], train_metrics['train_BCE_loss'], color='b', linestyle='dotted' if do_MMD else 'solid', label='train BCE')
    ax.plot(train_metrics['epochs'], train_metrics['val_BCE_loss'], color='r', linestyle='dotted' if do_MMD else 'solid', label='val BCE')

    if do_MMD:
        ax.plot(train_metrics['epochs'][train_start:], train_metrics['train_MMD_loss'], color='b', linestyle='dashed', label='train MMD')
        ax.plot(train_metrics['epochs'][train_start:], train_metrics['train_loss'], color='b', linestyle='solid', label='train total')
        ax.plot(train_metrics['epochs'], train_metrics['val_MMD_loss'], color='r', linestyle='dashed', label='val MMD')
        ax.plot(train_metrics['epochs'], train_metrics['val_loss'], color='r', linestyle='solid', label='val total')
    ax.legend(frameon=False)
    ax.set_ylim([-0.1, .6])
    fig.savefig(f"{path}/loss_vs_epochs.pdf")
    plt.close(fig)
    return None

def make_logits_plt(logit_diffs, path, name='final'):
    plt.figure()
    for d, l in logit_diffs.items():
        plt.hist(l, bins=100, histtype='step', label=d, density=True)

    plt.legend(frameon=False)
    plt.savefig(f"{path}/logit_diff_{name}.pdf")
    plt.close()
    return None