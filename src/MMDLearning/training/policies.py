import torch
import torch.nn as nn
from typing import Dict
from dataclasses import dataclass
from typing import Any

import sys
from pathlib import Path

from .metrics import get_batch_metrics

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
        prepared = model.module.prepare_batch(next(d for d in data if d is not None), self.device, self.dtype) #gets the non-None element of data

        pred, = model(prepared)

        #get labels and masks
        label = prepared['is_signal'].to(self.device, self.dtype).long()

        batch_output = {d: {'pred': pred, 'label': label} for d in state['track_domains']}

        for d in state['get_buffers']:
            self.bufs[d].add(
                logit_diffs=batch_output[d]['pred'][:, 1] - batch_output[d]['pred'][:, 0],
                labels=batch_output[d]['label']
            )

        #calculate losses and metrics
        batch_metrics = get_batch_metrics(batch_output, self.loss_fns, domains=state['track_domains'])

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
        prepared = model.module.prepare_batch(data[0], self.device, self.dtype)

        pred, encoded = model(prepared, intermediates=('encoder',))

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
        batch_metrics = get_batch_metrics(batch_output, self.loss_fns, mmd_coef=self.MMD_frac*state['mmd_norm'], domains=state['track_domains'])

        mmd_scale = self.mmd_sched(state['epochs_completed']) if state['phase']=='train' else 1

        tot_loss = mmd_scale*batch_metrics['Source']['MMD_loss'] + batch_metrics['Source']['BCE_loss']

        return batch_metrics, tot_loss

@dataclass
class TwinMMDPolicy(TrainingPolicy):
    loss_fns: Dict[str, nn.Module]
    bufs: Dict[str, EpochLogitBuffer]
    device: torch.device
    dtype: torch.dtype
    mmd_sched: Any  # MMDScheduler
    MMD_frac: float

    def compute_batch_metrics(self, *, data, model, state=None):
        #prepare the batch
        prepared = [model.module.prepare_batch(d, self.device, self.dtype) for d in data]

        pred, encoded_s = model(prepared[0], intermediates=('encoder',))
        encoded_t = model.module.encode_target(prepared[1])
        batch_output = {'Source': {'pred': pred,
                                   'label': prepared[0]['is_signal'].to(self.device, self.dtype).long(),
                                   'encoded': encoded_s},
                        'Target': {'encoded': encoded_t}}
        if state['use_tar_labels']:
            batch_output['Target']['label'] = prepared[1]['is_signal'].to(self.device, self.dtype).long()
        if state['phase'] == 'test' or 'Target' in state['get_buffers']:
            batch_output['Target']['pred'] = model.module.stages['classifier'](encoded_t)
            batch_output['Target']['label'] = prepared[1]['is_signal'].to(self.device, self.dtype).long()

        for d in state['get_buffers']:
            self.bufs[d].add(
                logit_diffs=batch_output[d]['pred'][:, 1] - batch_output[d]['pred'][:, 0],
                labels=batch_output[d]['label']
            )

        #calculate losses and metrics
        batch_metrics = get_batch_metrics(batch_output, 
                                          self.loss_fns, 
                                          mmd_coef=self.MMD_frac*state['mmd_norm'], 
                                          domains=state['track_domains'],
                                          use_tar_labels=state['use_tar_labels'])

        mmd_scale = self.mmd_sched(state['epochs_completed']) if state['phase']=='train' else 1

        tot_loss = mmd_scale*batch_metrics['Source']['MMD_loss'] + batch_metrics['Source']['BCE_loss']

        return batch_metrics, tot_loss

@dataclass
class SourceTargetClassifier(TrainingPolicy):
    loss_fns: Dict[str, nn.Module]
    bufs: Dict[str, EpochLogitBuffer]
    device: torch.device
    dtype: torch.dtype
    target_encoder_groups: Tuple[str] = ()

    def compute_batch_metrics(self, *, data, model, state=None):
        #prepare the batch
        prepared = model.module.prepare_batch(data[0], self.device, self.dtype)
        pred, _, _ = model(prepared, domains=('Source', 'Target'), target_preds=True)

        #TODO: continue from here
        #get labels and masks
        label = prepared['is_target'].to(self.device, self.dtype).long()
        batch_output = {d: {'pred': pred, 'label': label} for d in ['mixed']}

        for d in state['get_buffers']:
            self.bufs[d].add(
                logit_diffs=batch_output[d]['pred'][:, 1] - batch_output[d]['pred'][:, 0],
                labels=batch_output[d]['label']
            )

        #calculate losses and metrics
        batch_metrics = get_batch_metrics(batch_output, self.loss_fns, domains=state['track_domains'])

        tot_loss = batch_metrics[state['track_domains'][0]]['BCE_loss']

        return batch_metrics, tot_loss

def make_policy(do_MMD=False,
                mode='qt_classifier',
                **kwargs) -> TrainingPolicy:
    if mode=='st_classifier':
        return SourceTargetClassifier(**kwargs)
    else:
        
    if policy_type == 'mmd':
        return MMDPolicy(**kwargs)
        return TwinMMDPolicy(**kwargs)
    elif policy_type == 'supervised':
        return SupervisedPolicy(**kwargs)
    elif policy_type == 'stc':
        
    else:
        raise ValueError(f"Unknown policy type: {policy_type}")