from typing import TypedDict
import torch

class LossDict(TypedDict):
    bce: torch.nn.Module
    mmd: torch.nn.Module
    label: str

def split_batch(batch_vals, n_s, domains=['Source', 'Target']):
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
           