import torch
import torch.nn as nn
from typing import Dict
from dataclasses import dataclass
from typing import Any

import sys
from pathlib import Path

from metrics import get_batch_metrics

SRC_PATH = Path(__file__).parents[1].resolve() 
sys.path.append(str(SRC_PATH))
from utils.utils import split_batch
from utils.buffers import EpochLogitBuffer

class TrainingPolicy:
    def compute_loss(self, data, model) -> Dict[str, torch.Tensor]: ...
    
@dataclass
class SupervisedPolicy(TrainingPolicy):
    loss_fns: Dict[str, nn.Module]
    bufs: Dict[str, EpochLogitBuffer]
    device: torch.device
    dtype: torch.dtype

    def compute_batch_metrics(self, *, data, model, state=None):
        #prepare the batch
        prepared = model.prepare_batch(data, self.device, self.dtype)

        pred, = model(prepared)

        #get labels and masks
        label = prepared['is_signal'].to(self.device, self.dtype).long()

        batch_output = {d: {'pred': pred, 'label': label} for d in state['track domains']}

        
        for d in state['get_buffers']:
            self.bufs[d].add(
                logit_diffs=batch_output[d]['pred'][:, 1] - batch_output[d]['pred'][:, 0],
                labels=batch_output[d]['label']
            )

        #calculate losses and metrics
        batch_metrics = get_batch_metrics(batch_output, self.loss_fns, domains=state['track domains'])

        tot_loss = batch_metrics['Source']['BCE_loss'] if 'Source' in batch_metrics else None

        return batch_metrics, tot_loss
    
@dataclass
class MMDPolicy(TrainingPolicy):
    loss_fns: Dict[str, nn.Module]
    bufs: Dict[str, EpochLogitBuffer]
    device: torch.device
    dtype: torch.dtype
    mmd_sched: Any  # MMDScheduler
    MMD_frac: float

    def compute_batch_metrics(self, *, data, model, state=None):
        #prepare the batch
        prepared = model.prepare_batch(data, self.device, self.dtype)

        pred, encoded = model(prepared, intermediates=['encoder'])

        #get labels and masks
        label = prepared['is_signal'].to(self.device, self.dtype).long()

        #slice batch tensors by domain
        batch_output = split_batch({'pred': pred, 'label': label, 'encoded': encoded}, n_s=prepared['n_s'])

        for d in state['get_buffers']:
            self.bufs[d].add(
                logit_diffs=batch_output[d]['pred'][:, 1] - batch_output[d]['pred'][:, 0],
                labels=batch_output[d]['label']
            )

        #calculate losses and metrics
        batch_metrics = get_batch_metrics(batch_output, self.loss_fns, mmd_coef=self.MMD_frac*state['mmd_norm'], domains=state['track domains'])

        mmd_scale = self.mmd_sched(state['epochs_completed']) if state['phase']=='train' else 1

        tot_loss = mmd_scale*batch_metrics['Source']['MMD_loss'] + batch_metrics['Source']['BCE_loss']

        return batch_metrics, tot_loss

def make_policy(policy_type: str = 'MMD', **kwargs) -> TrainingPolicy:
    policy_type = policy_type.lower()
    if policy_type == 'mmd':
        return MMDPolicy(**kwargs)
    elif policy_type == 'supervised':
        return SupervisedPolicy(**kwargs)
    else:
        raise ValueError(f"Unknown policy type: {policy_type}")