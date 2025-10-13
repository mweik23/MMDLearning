import torch
import torch.nn as nn
from typing import Dict
from dataclasses import dataclass
from typing import Any, Tuple

import sys
from pathlib import Path

from .metrics import get_batch_metrics

SRC_PATH = Path(__file__).parents[1].resolve() 
sys.path.append(str(SRC_PATH))
from utils.utils import split_output
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
        prepared = [model.module.prepare_batch(d, self.device, self.dtype) for d in data if d is not None]

        preds, _ = model(*prepared, domains=state['track_domains'], target_preds=state['phase']=='test')

        #get labels and masks
        labels = [p['is_signal'].to(self.device, self.dtype).long() for p in prepared]

        batch_output = {d: {'pred': pred, 'label': label} for d, pred, label in zip(state['track_domains'], preds, labels)}

        for d in state['get_buffers']:
            self.bufs[d].add(
                logit_diffs=batch_output[d]['pred'][:, 1] - batch_output[d]['pred'][:, 0],
                labels=batch_output[d]['label']
            )

        #calculate losses and metrics
        batch_metrics = get_batch_metrics(batch_output, self.loss_fns, domains=state['track_domains'])

        tot_loss = batch_metrics[state['track_domains'][0]]['BCE_loss']

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
        prepared = [model.module.prepare_batch(d, self.device, self.dtype) for d in data if d is not None]
        target_preds = 'Target' in state['track_domains'] or 'Target' in state['get_buffers']
        preds, taps = model(*prepared, domains=('Source', 'Target'), target_preds=target_preds)

        #get labels and masks
        labels = [p['is_signal'].to(self.device, self.dtype).long() for p in prepared]
        if target_preds:
            if len(prepared)==1:
                assert preds[0] is not None, "If one batch is provided, preds[0] must not be None"
                #slice batch tensors by domain
                batch_output = split_output({'pred': preds[0], 'label': labels[0]}, n_s=prepared[0]['n_s'])
            else:
                assert preds[0] is not None and preds[1] is not None, "If two batches are provided, neither preds[0] nor preds[1] can be None"
                batch_output = {d: {'pred': pred, 'label': label} for d, pred, label in zip(('Source', 'Target'), preds, labels)}
        
        else:
            assert preds[0] is not None, "In training phase, preds[0] must not be None"
            if len(prepared)==1:
                batch_output = split_output({'label': labels[0]}, n_s=prepared[0]['n_s'])
                batch_output['Source']['pred'] = preds[0]
            else:
                batch_output = {'Source': {'label': labels[0]}}
                batch_output['Source']['pred'] = preds[0]
                   
        for k, v in taps['encoder'].items():
            batch_output[k]['encoder'] = v

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

    def compute_batch_metrics(self, *, data, model, state=None):
        #prepare the batch
        prepared = model.module.prepare_batch(data[0], self.device, self.dtype)
        (pred, _), _ = model(prepared, domains=('Source', 'Target'), target_preds=True)

        #TODO: continue from here
        #get labels and masks
        label = prepared['is_target'].to(self.device, self.dtype).long()
        batch_output = {d: {'pred': pred, 'label': label} for d in state['track_domains']}
        
        if len(state['get_buffers']) > 0:
            for_bufs = split_output(batch_output, n_s=prepared['n_s'])
        for d in state['get_buffers']:
            self.bufs[d].add(
                logit_diffs=for_bufs[d]['pred'][:, 1] - for_bufs[d]['pred'][:, 0],
                labels=for_bufs[d]['label']
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
    elif mode=='qt_classifier':
        if do_MMD:
            return MMDPolicy(**kwargs)
        else:
            return SupervisedPolicy(**kwargs)
    else:
        raise ValueError(f"Unknown mode: {mode}")