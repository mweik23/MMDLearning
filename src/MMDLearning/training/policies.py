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
    #TODO: refactor based on new buffer logic
    def compute_batch_metrics(self, *, data, model, state=None):
        #prepare the batch
        prepared = [model.module.prepare_batch(d, self.device, self.dtype) for d in data if d is not None]
        if len(prepared)==1:
            assert len(state['track_domains'])==1, "If one batch is provided, only one track_domain should be specified"
        elif len(prepared)==2:
            assert list(state['track_domains']) == ['Source', 'Target'], f"all_domains {state['track_domains']} do not match expected domains ['Source', 'Target']"

        preds, _ = model(*prepared, domains=tuple(state['track_domains']), target_preds=state['phase']=='test')

        #get labels and masks
        labels = [p['is_signal'].to(self.device, self.dtype).long() for p in prepared]

        batch_output = {d: {'pred': preds[i], 'label': labels[i]} for i, d in enumerate(state['track_domains'])}

        if state['get_buffers']:
            for d in state['all_domains']:
                assert d in self.bufs.keys(), f"buffer for domain {d} not found in bufs"
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

    #TODO: refactor based on new buffer logic
    def compute_batch_metrics(self, *, data, model, state=None):
        #prepare the batch
        prepared = [model.module.prepare_batch(d, self.device, self.dtype) for d in data if d is not None]
        target_preds = 'Target' in state['track_domains'] or state['get_buffers']
        preds, taps = model(*prepared, domains=('Source', 'Target'), target_preds=target_preds)

        #get labels and masks
        labels = [p['is_signal'].to(self.device, self.dtype).long() for p in prepared]
        if len(prepared)==1:
            assert preds[0] is not None, "If one batch is provided, preds[0] must not be None"
            if state['get_buffers']:
                assert target_preds, "If buffers are requested target_preds must be True"
                #TODO: gotta decide whether all domains represents buffer domains or gathered buffer domains
                for d in state['all_domains']:
                    self.bufs[d].add(
                        logit_diffs=preds[0][:, 1] - preds[0][:, 0],
                        labels=labels[0],
                        domains=prepared[0]['is_target']
                    )
            batch_output = split_output({'pred': preds[0], 'label': labels[0]}, n_s=prepared[0]['n_s'])
        else:
            assert preds[0] is not None and preds[1] is not None, "If two batches are provided, neither preds[0] nor preds[1] can be None"
            assert list(state['all_domains']) == ['Source', 'Target'], f"all_domains {state['all_domains']} do not match expected domains ['Source', 'Target']"
            batch_output = {d: {'pred': pred, 'label': label} for d, pred, label in zip(state['all_domains'], preds, labels)}
            if state['get_buffers']:
                assert target_preds, "If buffers are requested target_preds must be True"
                for d in state['all_domains']:
                    self.bufs[d].add(
                        logit_diffs=batch_output[d]['pred'][:, 1] - batch_output[d]['pred'][:, 0],
                        labels=batch_output[d]['label']
                    )
        if not target_preds:
            batch_output['Target'] = {}
        
        for k, v in taps['encoder'].items():
            batch_output[k]['encoder'] = v

        #calculate losses and metrics
        batch_metrics = get_batch_metrics(
            batch_output, 
            self.loss_fns, 
            mmd_coef=self.MMD_frac*state['mmd_norm'], 
            domains=state['track_domains'], 
            grad=state['phase']=='train'
        )

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

        batch_output = {state['track_domains'][0]: {'pred': pred, 'label': label}}
        assert list(batch_output.keys()) == state['all_domains'], f"all_domains {state['all_domains']} do not match batch output domains {list(batch_output.keys())}"

        if state['get_buffers']:
            for d, out in batch_output.items():
                self.bufs[d].add(
                    logit_diffs=out['pred'][:, 1] - out['pred'][:, 0],
                    labels=out['label'],
                    domains=out['label']
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