# src/MMDLearning/training/trainer.py
from dataclasses import dataclass
from pathlib import Path
import torch, json, time
from torch import distributed as dist
import os

#---------local imports-----------------------
from .metrics import get_test_metrics, RunningStats
from .reporting import display_epoch_summary, make_logits_plt, make_train_plt, display_status, finish_roc_plot
from .losses import LambdaAdjust
from .schedulers import SchedConfig, make_scheduler
from .policies import make_policy
from .training_utils import pair_by_source_len, Initialization, TrainEpochStart, ValidEpochStart, TestEpochStart, EndFirstVal
#---------------------------------------------

# put SRC on path
SRC_DIR = (Path(__file__).parents[1]).resolve()
import sys
sys.path.append(str(SRC_DIR))

#-----------non-local imports from src--------------
from utils.distributed import globalize_epoch_totals, epoch_metrics_from_globals
from utils.buffers import EpochLogitBuffer  
from utils.utils import LossDict, MetricHistory
from utils.distributed import DistInfo
#---------------------------------------------------


@dataclass
class Trainer:
    # core state you currently use as globals
    cfg: any
    ddp_model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    dist_info: DistInfo
    device: torch.device
    sched_config: SchedConfig
    # “context” shared across methods (use your existing objects)
    loss_fns: LossDict # {'bce': torch.nn.Module 'mmd': torch.nn.Module}
    dataloaders: dict   # {'train': ..., 'valid': ..., 'test': ...}
    metrics: MetricHistory
    start_epoch: int = 0
    mmd_sched: any = None
    mode: str = 'qt_classifier'  # 'qt_classifier', 'st_classifier'
    lambda_adjust: LambdaAdjust = LambdaAdjust(1.0, 0.0)  # default no-op
    dtype: torch.dtype = torch.float32
    mixed_batch: bool = False  # whether batches contain both source and target data
    def __post_init__(self):
        if self.mmd_sched is None:
            self.mmd_sched = lambda epoch: 1.0
        self.final_epoch = self.cfg.epochs + self.start_epoch
        self.is_target = {'Source': 0, 'Target': 1}
        self.state = {}
        policy_kwargs = {}
        self._handlers = {
            Initialization: self._initialize,
            TrainEpochStart: self._start_train_epoch,
            ValidEpochStart: self._start_valid_epoch,
            TestEpochStart: self._start_test_epoch,
            EndFirstVal: self._end_first_val,
        }
        self.update_state(Initialization(bce_estimate=0.27, mmd_estimate=0.01)) #guess values
        if self.cfg.do_MMD:
            policy_kwargs = {'mmd_sched': self.mmd_sched, 'MMD_frac': self.cfg.MMD_frac}
        self.scheduler = make_scheduler(self.optimizer, self.sched_config)
        self.buffers = {d: EpochLogitBuffer(keep_indices=False, 
                                            keep_domains=self.mixed_batch,
                                            assume_equal_lengths=True) for d in self.state['all_domains']
                        }
        self.policy = make_policy(loss_fns=self.loss_fns, 
                                  bufs=self.buffers, 
                                  device=self.device, 
                                  dtype=self.dtype,
                                  do_MMD=self.cfg.do_MMD,
                                  mode=self.mode,
                                  **policy_kwargs)
        self.loader_names = ['primary_loader', 'secondary_loader']

    def _save_ckp(self, state, is_best, epoch, save_all=False):
        p = Path(self.cfg.logdir) / self.cfg.exp_name
        p.mkdir(parents=True, exist_ok=True)
        if save_all and self.dist_info.is_primary:
            torch.save(state, p / f"checkpoint-epoch-{epoch}.pt")
        if is_best and self.dist_info.is_primary:
            for _ in range(3):
                try:
                    torch.save(state, p / "best-val-model.pt"); break
                except OSError:
                    time.sleep(5)
    
    def _update_mmd_norm(self, bce, mmd):
        self.metrics.update(init_loss={'BCE': bce, 'MMD': mmd}) #guess values
        self.state['mmd_norm'] = self.metrics.mmd_norm
        
    def _initialize(self, event):
        self.state['get_buffers'] = False
        if self.cfg.do_MMD:
            self._update_mmd_norm(event.bce_estimate, event.mmd_estimate)
            self.state['use_tar_labels'] = self.cfg.use_tar_labels
        if self.mode == 'st_classifier':
            self.state['track_domains'] = ['Mixed']
        elif self.mode == 'qt_classifier':
            self.state['track_domains'] = ['Source']
        self.state['epochs_completed'] = -1
        self.state['all_domains'] = ['Mixed'] if self.mixed_batch else ['Source', 'Target']

    def _start_train_epoch(self, event):
        self.state['phase'] = 'train'
        self.state['get_buffers'] = False
        self.state['epochs_completed'] += 1
        
    def _start_valid_epoch(self, event):
        self.state['phase'] = 'valid'
        if self.state['epochs_completed'] == -1:
            self.state['get_buffers'] = True if (self.cfg.do_MMD or self.mode=='st_classifier') else False
            
    def _start_test_epoch(self, event):
        self.state['phase'] = 'test'
        self.state['get_buffers'] = True
        if self.mode == 'st_classifier':
            self.state['track_domains'] = ['Mixed']
        elif self.mode == 'qt_classifier':
            self.state['track_domains'] = ['Source', 'Target']
        
    def _end_first_val(self, event):
        if self.cfg.do_MMD:
            self._update_mmd_norm(event.bce, event.mmd/self.metrics.mmd_norm/self.cfg.MMD_frac)
            self.state['mmd_norm'] = self.metrics.mmd_norm
            
    def update_state(self, event):
        handler = self._handlers.get(type(event), None)
        if handler is not None:
            handler(event)

    #TODO: configure for multi gpu training. Currently not supported
    def get_mmd_floor(self, loader, quantiles=[0.5, 0.9]):
        print('>> getting MMD stats...')
        self.ddp_model.eval()
        mmd_floor_dist = []
        for i, data in enumerate(loader):
            prepared = self.predictor.prepare_batch(data, self.device, dtype=self.dtype)
            _, encoded = self.predictor.forward(self.ddp_model, prepared)
            z1, z2 = torch.chunk(encoded, 2, dim=0)
            mmd_floor_dist.append(self.loss_fns['mmd'](z1, z2))
        mmd_floor_dist = torch.stack(mmd_floor_dist, dim=0)
        assert mmd_floor_dist.device == torch.device(f"cuda:{self.dist_info.local_rank}")
        mmd_med, mmd_upper = torch.quantile(mmd_floor_dist, torch.tensor(quantiles, device=mmd_floor_dist.device, dtype=torch.float32))
        return mmd_med, mmd_upper

    def _run_epoch(self, epoch: int, primary_loader=None, secondary_loader=None):
        if self.state['phase'] == 'train':
            self.ddp_model.train()
            for ld in [primary_loader, secondary_loader]:
                if ld is not None:
                    ld.sampler.set_epoch(epoch)
        else:
            self.ddp_model.eval()

        trackers = {d: RunningStats(window=self.cfg.log_interval, domain=d) for d in self.state['track_domains']}

        loader_length = len(primary_loader) if primary_loader is not None else len(secondary_loader)
        #need to make prediction for source and target data
        for batch_idx, data in pair_by_source_len(primary_loader, secondary_loader):
            if self.state['phase'] == 'train':
                self.optimizer.zero_grad()
            #TODO: make sure batch_metrics has the right structure
            batch_metrics, tot_loss = self.policy.compute_batch_metrics(data=data, 
                                                                  model=self.ddp_model, 
                                                                  state=self.state)
            if self.state['phase'] == 'train':
                tot_loss.backward()
                self.optimizer.step()

            for name, metrics in batch_metrics.items():
                trackers[name].update(**metrics)

            if (batch_idx+1) % self.cfg.log_interval == 0:
                for d, tr in trackers.items():
                    display_status(phase=self.state['phase'], domain=d, epoch=epoch, 
                                   tot_epochs=0 if self.state['phase']=='test' else self.final_epoch-1, #TODO: check if this gives correct tot_epochs
                                   batch_idx=batch_idx+1, num_batches=loader_length,
                                   running_acc=tr.running_acc, avg_batch_time=tr.avg_batch_time(),
                                   running_bce=tr.running_bce, running_mmd=tr.running_mmd,
                                   logger=None)
    
        torch.cuda.empty_cache() #can put this in the batch loop to free memory at the end of each batch but it slows things down
        # ---------- reduce -----------
        #globalize epoch metrics
        metrics = {}
        device = next(self.ddp_model.module.parameters()).device
        for d, tr in trackers.items():
            g_bce, g_mmd, g_corr, g_cnt = globalize_epoch_totals(
                local_bce_sum=tr.epoch_bce_sum,
                local_mmd_sum=tr.epoch_mmd_sum,
                local_correct=tr.epoch_correct,
                local_count=tr.epoch_count,
                device=device
                
            )
            metrics[d] = epoch_metrics_from_globals(g_correct=g_corr, g_count=g_cnt, g_bce_sum=g_bce, g_mmd_sum=g_mmd)
            metrics[d]['time'] = tr.epoch_time()
        #gather logits and labels if buffers are requested
        gathered_buffers = {d: self.buffers[d].gather_to_rank0(cast_fp16=False) if self.state['get_buffers'] else None for d in self.state['all_domains']}
        # split mixed buffers into source and target if needed
        if self.state['get_buffers'] and self.state['all_domains'][0]=='Mixed' \
        and gathered_buffers[self.state['all_domains'][0]] is not None and self.mode=='qt_classifier':
            gathered_buffers = {d: {k: arr[gathered_buffers[self.state['all_domains'][0]]['domains']==v] 
                                    for k, arr in gathered_buffers[self.state['all_domains'][0]].items()}
                                for d, v in self.is_target.items()}
        if self.buffers is not None:
            for buf in self.buffers.values():
                buf.clear()

        return metrics, gathered_buffers if self.state['get_buffers'] else None

    def train(self):
        if self.cfg.pretrained !='':
            self.update_state(ValidEpochStart())
            with torch.no_grad():
                # first validation run to get initial MMD and BCE
                #TODO: change the way secondary loader is passed and conditions are checked
                val_metrics, val_buffers = self._run_epoch(self.start_epoch-1, 
                                                           primary_loader=self.dataloaders[0]['valid'], 
                                                           secondary_loader=self.dataloaders[1]['valid'] if (not self.mixed_batch and self.cfg.do_MMD) else None)
            #save logits and labels for validation
            #TODO: make saving more robust
            #TODO: decide about splitting mixed buffers for saving when using qt_classifier and mmd
            if val_buffers is not None:
                if all(buf is not None for buf in val_buffers.values()):
                    torch.save(val_buffers, f'{self.cfg.logdir}/{self.cfg.exp_name}/init_val_buffers.pt')

            self.metrics.append(
                epochs = self.start_epoch-1,
                val_BCE = val_metrics[self.state['track_domains'][0]]['BCE_loss'],
                val_MMD = self.cfg.MMD_frac*val_metrics[self.state['track_domains'][0]]['BCE_loss'], # by construction
                val_loss = (1+self.cfg.MMD_frac)*val_metrics[self.state['track_domains'][0]]['BCE_loss'], 
                val_acc = val_metrics[self.state['track_domains'][0]]['acc'],
                val_time = val_metrics[self.state['track_domains'][0]]['time'],
            )

            self.update_state(EndFirstVal(bce=val_metrics[self.state['track_domains'][0]]['BCE_loss'], mmd=val_metrics[self.state['track_domains'][0]]['MMD_loss']))

            self.metrics.update(best_val=self.metrics.get('val_loss')[-1],
                                best_epoch=self.start_epoch-1
                                )
            
            
            ## save best model (minimum BCE + MMD with the MMD_coef only - no epoch dependent coefs) 
            if self.dist_info.is_primary:
                display_epoch_summary(partition="validation", epoch=self.start_epoch-1, tot_epochs=self.final_epoch-1,
                                acc=self.metrics.get("val_acc")[-1], time_s=self.metrics.get("val_time")[-1], 
                                bce=self.metrics.get("val_BCE")[-1], mmd=self.metrics.get("val_MMD")[-1],
                                logger=getattr(self, "logger", None))
        else:
            self.metrics.update(best_val=float('inf'), best_epoch=-1)
        ### training and validation
        for ld in self.dataloaders:
            if 'train' in ld:
                ld['train'].sampler.set_epoch(self.start_epoch-1)
        for epoch in range(self.start_epoch, self.final_epoch):
            self.update_state(TrainEpochStart())
            is_best=False
            # lambda adjust setting
            #TODO: either update this for target_model or remove it
            assert self.cfg.mmd_interval == -1, "Currently does not support lambda_adust so please set mmd_interval=-1"
            if self.cfg.mmd_interval != -1 and self.cfg.do_MMD:
                if (epoch-self.start_epoch) % self.cfg.mmd_interval == 0:
                    with torch.no_grad():
                        quantiles =[0.5, 0.975]
                        mmd_med, mmd_upper = self.get_mmd_floor(self.dataloaders['train'], quantiles=quantiles)
                        print('MMD quantile ', quantiles[0], ': ',  mmd_med)
                        print('MMD quantile ', quantiles[1], ': ',  mmd_upper)
                    self.lambda_adjust = LambdaAdjust(2*mmd_upper, 2*(mmd_upper-mmd_med)**(-1))
            lr_message =  'Learning rates\n'   
            for g in self.optimizer.param_groups:      
                lr_message += g['name'] + f": {g['lr']:.3e}  "
            lr_message += '\n' + 124*'-'
            if self.dist_info.is_primary:
                print(lr_message)
            #----------training------------
            train_metrics, _ = self._run_epoch(epoch, 
                                               primary_loader=self.dataloaders[0]['train'], 
                                               secondary_loader=self.dataloaders[1]['train'] if (not self.mixed_batch and self.cfg.do_MMD) else None)

            self.metrics.append(
                    epochs = epoch,
                    train_BCE = train_metrics[self.state['track_domains'][0]]['BCE_loss'],
                    train_MMD = train_metrics[self.state['track_domains'][0]]['MMD_loss'], # by construction
                    train_loss = train_metrics[self.state['track_domains'][0]]['BCE_loss'] + train_metrics[self.state['track_domains'][0]]['MMD_loss'], 
                    train_acc = train_metrics[self.state['track_domains'][0]]['acc'],
                    train_time = train_metrics[self.state['track_domains'][0]]['time'],
                    classifier_lr = [g['lr'] for g in self.optimizer.param_groups if g['name']=='classifier'][0]
                )
            if self.dist_info.is_primary:
                display_epoch_summary(partition="train", epoch=epoch, tot_epochs=self.final_epoch-1,
                            acc=self.metrics.get("train_acc")[-1], time_s=self.metrics.get("train_time")[-1],
                            bce=self.metrics.get("train_BCE")[-1], mmd=self.metrics.get("train_MMD")[-1],
                            logger=getattr(self, "logger", None))
            #----------validation------------
            if epoch % self.cfg.val_interval == 0:
                self.update_state(ValidEpochStart())
                with torch.no_grad():
                    val_metrics, _ = self._run_epoch(epoch,
                                                     primary_loader=self.dataloaders[0]['valid'], 
                                                     secondary_loader=self.dataloaders[1]['valid'] if (not self.mixed_batch and self.cfg.do_MMD) else None)
                if self.cfg.pretrained =='' and epoch==self.start_epoch:
                    self.update_state(EndFirstVal(bce=val_metrics[self.state['track_domains'][0]]['BCE_loss'], mmd=val_metrics[self.state['track_domains'][0]]['MMD_loss']))
                    if self.cfg.do_MMD:
                        val_metrics[self.state['track_domains'][0]]['MMD_loss'] = self.cfg.MMD_frac*val_metrics[self.state['track_domains'][0]]['BCE_loss']
                val_loss = val_metrics[self.state['track_domains'][0]]['BCE_loss'] + val_metrics[self.state['track_domains'][0]]['MMD_loss']
                self.metrics.append(
                        val_BCE = val_metrics[self.state['track_domains'][0]]['BCE_loss'],
                        val_MMD = val_metrics[self.state['track_domains'][0]]['MMD_loss'],
                        val_loss = val_loss,
                        val_acc = val_metrics[self.state['track_domains'][0]]['acc'],
                        val_time = val_metrics[self.state['track_domains'][0]]['time'],
                    )
                
                if val_loss < self.metrics.get('best_val'):
                    is_best=True
                    self.metrics.update(best_val=val_loss,
                                        best_epoch=epoch
                                        )
                    
                checkpoint = {'epoch': epoch + 1, 'state_dict': self.ddp_model.state_dict(), 'optimizer': self.optimizer.state_dict()}
                self._save_ckp(checkpoint, is_best, epoch)

                json_object = json.dumps(self.metrics.to_dict(), indent=4)
                with open(f"{self.cfg.logdir}/{self.cfg.exp_name}/train-result.json", "w") as outfile:
                    outfile.write(json_object)
                
                if self.dist_info.is_primary:      
                    display_epoch_summary(partition="validation", epoch=epoch, tot_epochs=self.final_epoch-1,
                                bce=self.metrics.get("val_BCE")[-1], mmd=self.metrics.get("val_MMD")[-1], acc=self.metrics.get("val_acc")[-1],
                                time_s=self.metrics.get("val_time")[-1], best_epoch=self.metrics.get('best_epoch'),
                                best_val=self.metrics.get('best_val'), logger=getattr(self, "logger", None))
            self.scheduler.step_epoch(val_metric=val_loss)
            dist.barrier() # syncronize
        # keys needed: ['epochs', 'train_BCE', 'val_BCE'] if do_MMD: += ['train_MMD', 'val_MMD', 'train_loss', 'val_loss']
        # otherwise use argument rename_map = {'old_key': 'new_key', ...}
        if self.dist_info.is_primary:
            make_train_plt(self.metrics.to_dict(), 
                            f"{self.cfg.logdir}/{self.cfg.exp_name}", 
                            pretrained=(self.cfg.pretrained !=''), 
                            do_MMD=(self.cfg.MMD_frac>0))
    
    def test(self):
        ### test on best model
        best_model = torch.load(f"{self.cfg.logdir}/{self.cfg.exp_name}/best-val-model.pt", map_location=self.device, weights_only=True)
        self.ddp_model.load_state_dict(best_model['state_dict'])
        self.update_state(TestEpochStart())
        
        with torch.no_grad():
            test_metrics, test_buffers = self._run_epoch(0, 
                                                         primary_loader=self.dataloaders[0]['valid'], 
                                                         secondary_loader=self.dataloaders[1]['valid'] if not self.mixed_batch else None)
    
        #print(f'rank: {self.dist_info.rank}, test_buffers: {test_buffers}')
        if all(buf is not None for buf in test_buffers.values()):
            torch.save(test_buffers, f'{self.cfg.logdir}/{self.cfg.exp_name}/best_val_buffers.pt')
            # plot final logits
            make_logits_plt({k: v['logit_diffs'] for k, v in test_buffers.items()},
                            f"{self.cfg.logdir}/{self.cfg.exp_name}", 
                            domains={k: v['domains'] for k, v in test_buffers.items()} if self.mode=='st_classifier' else None)
            ax = None
            for d, buf in test_buffers.items():
                metrics, ax = get_test_metrics(buf['labels'].numpy(), buf['logit_diffs'].numpy(), domain=d, ax=ax)
                test_metrics[d].update(metrics)
            finish_roc_plot(f"{self.cfg.logdir}/{self.cfg.exp_name}", ax, is_primary=self.dist_info.is_primary)

        if os.path.exists(f'{self.cfg.logdir}/{self.cfg.exp_name}/init_val_buffers.pt'):
            # plot initial logits
            if self.dist_info.is_primary:
                #load initial validation buffers
                init_val_buffers = torch.load(f'{self.cfg.logdir}/{self.cfg.exp_name}/init_val_buffers.pt', weights_only=True)
                make_logits_plt({k: v['logit_diffs'] for k, v in init_val_buffers.items()},
                                f"{self.cfg.logdir}/{self.cfg.exp_name}", name='initial',
                                domains={k: v['domains'] for k, v in init_val_buffers.items()} if self.mode=='st_classifier' else None)

        for domain, met in test_metrics.items():
            if self.dist_info.is_primary:
                display_epoch_summary(partition="test", epoch=1, tot_epochs=0,
                                bce=met.get('BCE_loss', None), mmd=met.get('MMD_loss', None), acc=met.get('acc', None), time_s=met.get('time', None),
                                logger=getattr(self, "logger", None), domain=domain, auc=met.get('auc', None), r30=met.get('1/eB ~ 0.3', None))

            if self.dist_info.is_primary:
                json_object = json.dumps(test_metrics, indent=4)
                with open(f"{self.cfg.logdir}/{self.cfg.exp_name}/test-result.json", "w") as outfile:
                    outfile.write(json_object)