from typing import TypedDict
import torch

class LossDict(TypedDict):
    bce: torch.nn.Module
    mmd: torch.nn.Module
    label: str

def split_batch(batch_vals, n_s=None, domains=['Source', 'Target']):
    out = {}
    for d in domains:
        if d=='Source':
            out[d] = {
                k: v[:n_s] for k, v in batch_vals.items()
            }
        elif d=='Target':
            out[d] = {
                k: v[n_s:] for k, v in batch_vals.items()
            }
        else:
            raise ValueError(f"Unknown domain: {d}")
    return out  

class MetricHistory:
    def __init__(self):
        self.storage = {}  # e.g., {'train_loss': [..], 'val_acc': [..]}

    def append(self, **kwargs):
        # usage: history.append(train_loss=0.23, val_loss=0.31, train_acc=0.88)
        for k, v in kwargs.items():
            self.storage.setdefault(k, []).append(v)
            
    def update(self, **kwargs):
        for k, v in kwargs.items():
            self.storage[k] = v

    def get(self, key, default=None):
        return self.storage.get(key, default)
    
    @property
    def mmd_norm(self):
        if 'init_loss' in self.storage:
            init_loss = self.storage['init_loss']
            if isinstance(init_loss, dict):
                bce = init_loss.get('BCE', None)
                mmd = init_loss.get('MMD', None)
                if bce is not None and mmd is not None and mmd != 0:
                    return bce / mmd
        return None

    def to_dict(self):
        return dict(self.storage)

    def reset(self):
        self.storage.clear()