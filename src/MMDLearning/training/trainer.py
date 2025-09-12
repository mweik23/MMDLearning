# src/MMDLearning/training/trainer.py
from dataclasses import dataclass
from pathlib import Path
import torch, json, time
from copy import deepcopy

#---------local imports-----------------------
from metrics import get_batch_metrics, get_correct, display_status, RunningStats
from reporting import display_epoch_summary
from losses import LambdaAdjust
from schedulers import SchedConfig, make_scheduler
#---------------------------------------------

# put SRC on path
SRC_DIR = (Path(__file__).parents[1]).resolve()
import sys
sys.path.append(str(SRC_DIR))

#-----------non-local imports from src--------------
from MMDLearning.models.predictors import make_predictor
from MMDLearning.utils.distributed import globalize_epoch_totals, epoch_metrics_from_globals, gather_scores, gather_preds
from MMDLearning.utils.buffers import EpochLogitBuffer
from MMDLearning.utils.utils import split_batch, LossDict, MetricHistory
#---------------------------------------------------


@dataclass
class Trainer:
    # core state you currently use as globals
    args: any
    ddp_model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    rank: int
    local_rank: int
    world_size: int
    sched_config: SchedConfig
    # “context” shared across methods (use your existing objects)
    loss_fns: LossDict # {'bce': torch.nn.Module 'mmd': torch.nn.Module}
    mmd_sched: any
    loss_stats: dict
    train_sampler: any
    dataloaders: dict   # {'train': ..., 'valid': ..., 'test': ...}
    lambda_adjust: LambdaAdjust = LambdaAdjust(1.0, 0.0)  # default no-op

    def __post_init__(self):
        # create the predictor once, based on model choice
        self.predictor = make_predictor(self.args.model)
        self.dtype = torch.float32  # or whatever you consistently use
        if self.val_logits is None:
            self.val_logits = {'init': [], 'best': [], 'last': [[], []]}
        #TODO: load pretrained model and define start_epoch and final_epoch
        self.start_epoch = 0
        self.final_epoch = self.args.epochs + self.start_epoch
        self.is_target = {'Source': 0, 'Target': 1}
        self.metrics = MetricHistory()
        self.metrics.update(init_loss={'BCE': 0.27, 'MMD': 0.006}) #guess values
        self.scheduler = make_scheduler(self.optimizer, self.sched_config)


    def _set_train(self, epoch):
        self.train_sampler.set_epoch(epoch)
        self.ddp_model.train()
        #TODO: remove once I implement batch norm eval option
        if self.args.bn_eval:
            for m in self.ddp_model.modules():
                if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                    m.eval()

    def _save_ckp(self, state, is_best, epoch):
        p = Path(self.args.logdir) / self.args.exp_name
        p.mkdir(parents=True, exist_ok=True)
        torch.save(state, p / f"checkpoint-epoch-{epoch}.pt")
        if is_best and self.rank == 0:
            for _ in range(3):
                try:
                    torch.save(state, p / "best-val-model.pt"); break
                except OSError:
                    time.sleep(5)
                    
    def _calc_BCE(self, res, pred, label):
        batch_size = pred.size(dim=0) #keep in mind this is the batch_size of only the dataset that is passed into this function
        correct = get_correct(pred, label)
        loss_BCE = self.bce(pred, label)
        res['counter'] += batch_size
        res['correct'] += correct
        res['BCE loss'] += loss_BCE.item() * batch_size
        res['BCE loss_arr'].append(loss_BCE.item())
        res['correct_arr'].append(correct)
        return loss_BCE
    
    @torch.no_grad()
    def _gather_scores(self, t):
        pred = [torch.zeros_like(t) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(pred, t)
        return torch.cat(pred).cpu()
    
        #TODO: configure for multi gpu training
    def get_mmd_floor(self, loader, quantiles=[0.5, 0.9]):
        print('>> getting MMD stats...')
        self.ddp_model.eval()
        mmd_floor_dist = []
        for i, data in enumerate(loader):
            prepared = self.predictor.prepare_batch(data, self.local_rank, dtype=self.dtype)
            _, encoded = self.predictor.forward(self.ddp_model, prepared)
            z1, z2 = torch.chunk(encoded, 2, dim=0)
            mmd_floor_dist.append(self.loss_fns['mmd'](z1, z2))
        mmd_floor_dist = torch.stack(mmd_floor_dist, dim=0)
        assert mmd_floor_dist.device == torch.device(f"cuda:{self.local_rank}")
        mmd_med, mmd_upper = torch.quantile(mmd_floor_dist, torch.tensor(quantiles, device=mmd_floor_dist.device, dtype=torch.float32))
        return mmd_med, mmd_upper
    
    def _run_epoch(self, epoch: int, loader, partition: str, get_buffers: bool = False, domains=['Source', 'Target']):
        #TODO: make this be an optional argument
        lambda_adjust = self.loss_stats['lambda_adjust']
        if partition == 'train':
            #TODO: change this when I change the batchnorm eval option
            self._set_train(epoch)
        else:
            self.ddp_model.eval()
        if get_buffers:
            bufs = {d: EpochLogitBuffer(keep_indices=False, keep_domains=False, assume_equal_lengths=True) for d in domains}
            
        if partition=='test':
            track_domains = domains
        else:
            track_domains = ['Source']

        trackers = {d: RunningStats(window=self.args.log_interval, domain=d) for d in track_domains}
        
        loader_length = len(loader)
        #need to make prediction for source and target data
        for batch_idx, data in enumerate(loader):
            if partition == 'train':
                self.optimizer.zero_grad()
            
            #prepare the batch
            prepared = self.predictor.prepare_batch(data, self.local_rank, dtype=self.dtype)
            
            #forward pass
            pred, encoded = self.predictor.forward(self.ddp_model, prepared, intermediates=['encoder'])
            
            #get labels and masks
            label = prepared['is_signal'].to(self.local_rank, self.dtype).long()
            
            #get source and target batch sizes
            batch_sizes = [prepared['n_s'], label.size(0)-prepared['n_s']]
            batch_size_avg = (sum(batch_sizes)/2).to(self.local_rank)
            
            #slice batch tensors by domain
            sb = split_batch({'pred': pred, 'label': label, 'encoded': encoded}, prepared['n_s'])
        
            if get_buffers:
                for d, buf in bufs.items():
                    buf.add(
                        logits=sb[d]['pred'],
                        labels=sb[d]['label']
                    )

            #calculate losses and metrics
            batch_metrics, mmd_val = get_batch_metrics(sb, self.loss_fns, domains=track_domains)

            mmd_scale = 1 if partition=='test' else self.mmd_sched(epoch-self.start_epoch)

            #TODO: setup control of lambda adjust, calculate the multiplicative factor before epoch/training
            loss_mmd = mmd_scale*self.args.MMD_frac*self.metrics.mmd_norm*mmd_val

            if partition == 'train':
                (batch_metrics['BCE loss'][0] + loss_mmd).backward()
                self.optimizer.step()

            for name, tracker in trackers.items():
                tracker.update(
                    bce=batch_metrics[name]['BCE loss'].detach().cpu().item(),
                    mmd=((loss_mmd/mmd_scale).detach().cpu().item() if loss_mmd is not None else 0.0), # TODO: need update batch metrics or change how we source MMD here
                    correct=int(batch_metrics[name]['correct'].detach().cpu().item()),
                    batch_size=int(batch_metrics[name]['batch_size'].detach().cpu().item()),
                )

            if (batch_idx+1) % self.args.log_interval == 0:
                for domain, tr in trackers.items():
                    display_status(partition=partition, domain=domain, epoch=epoch, tot_epochs=self.tot_epochs, #TODO: check if this gives correct tot_epochs
                                   batch_idx=batch_idx, num_batches=loader_length,
                                   running_bce=tr.running_bce, running_mmd=tr.running_mmd,
                                   running_acc=tr.running_acc, total_acc=tr.total_acc, avg_batch_time=tr.avg_batch_time(),
                                   logger=None)
    
        torch.cuda.empty_cache() #can put this in the batch loop to free memory at the end of each batch but it slows things down
        # ---------- reduce -----------
        #globalize epoch metrics
        metrics = {}
        device = next(self.model.parameters()).device
        for d, tr in trackers.items():
            g_bce, g_mmd, g_corr, g_cnt = globalize_epoch_totals(
                local_bce_sum=tr.epoch_bce_sum,
                local_mmd_sum=tr.epoch_mmd_sum,
                local_correct=tr.epoch_correct,
                local_count=tr.epoch_count,
                device=device,
            )
            metrics[d] = epoch_metrics_from_globals(g_bce_sum=g_bce, g_mmd_sum=g_mmd, g_correct=g_corr, g_count=g_cnt)
            metrics[d]['time'] = tr.epoch_time()
        #gather logits and labels if buffers are requested
        if get_buffers:
            buffers = {d: buf.gather_to_rank0() for d, buf in bufs.items()}
            for buf in bufs.values():
                buf.clear()
        #TODO: save metrics and buffers - probably will do it outside _run_epoch
        
        return metrics, buffers if get_buffers else None

    @torch.no_grad()
    def test(self, res: dict):
        # ——— a near copy of your current test() ———
        # load best-val-model.pt, call self._run_epoch(..., 'test'), gather preds,
        # write metrics/plots/json (rank==0)
        pass
    

    def train(self, metrics: dict):
        #start with a validation run if starting with a pretrained model
        print('Learning rates (before val): ', [g['lr'] for g in self.optimizer.param_groups])
        if self.args.pretrained !='':
            with torch.no_grad():
                # first validation run to get initial MMD and BCE
                val_metrics, val_buffers = self._run_epoch(self.start_epoch-1, self.dataloaders['valid'], 'valid', get_buffers=True)
            #save logits and labels for validation
            #TODO: make saving more robust
            if val_buffers['Source'] is not None:
                torch.save(val_buffers, f'{self.args.logdir}/{self.args.exp_name}/init_val_buffers.pt')
            
            print('loss stats: ', self.metrics.get('init_loss'))
            self.metrics.append(
                epochs = self.start_epoch-1,
                val_BCE = val_metrics['BCE loss'],
                val_MMD = self.args.MMD_frac*val_metrics['BCE loss'], # by construction
                val_loss = val_metrics['BCE loss'] + self.args.MMD_frac*val_metrics['BCE loss'], 
                val_acc = val_metrics['acc'],
                val_time = val_metrics['time'],
            )
            if self.args.MMD_frac==0:
                self.metrics['init_loss']['MMD'] = 1
            self.metrics.update(init_loss={'BCE': val_metrics['BCE loss'], 'MMD': val_metrics['MMD loss']/self.metrics.mmd_norm/self.args.MMD_frac},
                                best_val=self.metrics.get('val_loss')[-1],
                                best_epoch=self.start_epoch-1
                                )

            ## save best model (minimum BCE + MMD with the MMD_coef only - no epoch dependent coefs) 
            
            display_epoch_summary(partition="validation", epoch=self.start_epoch-1, tot_epochs=self.final_epoch,
                            bce=self.metrics.get("val_BCE"), mmd=self.metrics.get("val_MMD"), acc=self.metrics.get("val_acc"), time_s=self.metrics.get("val_time"),
                            logger=getattr(self, "logger", None))

        ### training and validation
        self.train_sampler.set_epoch(self.start_epoch-1)
        self.scheduler.step_epoch()
        for epoch in range(self.start_epoch, self.final_epoch):
            #percentiles = get_mmd_floor(ddp_model, data1, data2)
            is_best=False
            
            # lambda adjust setting
            #TODO: rename mmd_interval to mmd_adjust_interval
            if self.args.mmd_interval != -1:
                if (epoch-self.start_epoch) % self.args.mmd_interval == 0:
                    with torch.no_grad():
                        quantiles =[0.5, 0.975]
                        mmd_med, mmd_upper = self.get_mmd_floor(self.dataloaders['train'], quantiles=quantiles)
                        print('MMD quantile ', quantiles[0], ': ',  mmd_med)
                        print('MMD quantile ', quantiles[1], ': ',  mmd_upper)
                    self.lambda_adjust = LambdaAdjust(2*mmd_upper, 2*(mmd_upper-mmd_med)**(-1))
                            
            print('Learning rate: ', [g['lr'] for g in self.optimizer.param_groups])
            #----------training------------
            train_metrics, _ = self._run_epoch(epoch, self.dataloaders['train'], 'train')
            
            display_epoch_summary(partition="train", epoch=epoch, tot_epochs=self.final_epoch,
                            bce=train_metrics['BCE loss'], mmd=train_metrics['MMD loss'], acc=train_metrics['acc'], time_s=train_metrics['time'],
                            logger=getattr(self, "logger", None))
            self.metrics.append(
                    epochs = self.start_epoch-1,
                    train_BCE = train_metrics['BCE loss'],
                    train_MMD = train_metrics['MMD loss'], # by construction
                    train_loss = train_metrics['BCE loss'] + train_metrics['MMD loss'], 
                    train_acc = train_metrics['acc'],
                    train_time = train_metrics['time'],
                    lr = [g['lr'] for g in self.optimizer.param_groups]
                )
            #----------validation------------
            if epoch % self.args.val_interval == 0:
                with torch.no_grad():
                    val_metrics, _ = self._run_epoch(self.start_epoch-1, self.dataloaders['valid'], 'valid')
                
                display_epoch_summary(partition="validation", epoch=self.start_epoch-1, tot_epochs=self.final_epoch,
                            bce=val_metrics['BCE'], mmd=val_metrics['MMD'], acc=val_metrics['acc'], time_s=val_metrics['time'],
                            logger=getattr(self, "logger", None))
                
                val_loss = val_metrics['BCE loss'] + val_metrics['MMD loss']
                self.metrics.append(
                        val_BCE = val_metrics['BCE loss'],
                        val_MMD = val_metrics['MMD loss'],
                        val_loss = val_loss,
                        val_acc = val_metrics['acc'],
                        val_time = val_metrics['time'],
                    )
                if val_loss < self.metrics.get('best_val'):
                    is_best=True
                    self.metrics.update(best_val=val_loss,
                                        best_epoch=epoch
                                        )
                    
                    checkpoint = {'epoch': epoch + 1, 'state_dict': self.ddp_model.state_dict(), 'optimizer': self.optimizer.state_dict()}
                    self._save_ckp(checkpoint, is_best, epoch)

                    json_object = json.dumps(self.metrics.to_dict(), indent=4)
                    with open(f"{self.args.logdir}/{self.args.exp_name}/train-result.json", "w") as outfile:
                        outfile.write(json_object)
            self.scheduler.step_epoch()
            dist.barrier() # syncronize

def test(res):
    ### test on best model
    best_model = torch.load(f"{args.logdir}/{args.exp_name}/best-val-model.pt", map_location=torch.cuda.set_device(local_rank))
    ddp_model.load_state_dict(best_model['state_dict'])
    with torch.no_grad():
        test_res = run(0, dataloaders['test'], partition='test')
    preds = [gather_preds(r) for r in test_res]
    if (rank == 0):
        np.save(f"{args.logdir}/{args.exp_name}/score_source.npy", preds[0])
        np.save(f"{args.logdir}/{args.exp_name}/score_target.npy", preds[1])
        metrics = [get_metric(pred, r) for pred, r in zip(preds, test_res)]
        #The form of each metric is: {'domain': r['domain'],'test_loss': r['BCE loss'], 'test_acc': r['acc'],
        #          'test_auc': auc, 'test_1/eB_0.3':1./eB[0],'test_1/eB_0.5':1./eB[1]}
        fig=plt.figure()
        first_tpr = []
        for m in metrics:
            idx = int(max(np.max(np.where(m['fpr']==0)[0]), np.max(np.where(m['tpr']==0)[0]))) + 1
            first_tpr.append(m['tpr'][idx])
            plt.plot(m['tpr'][idx:], 1/m['fpr'][idx:], label=m['domain'])
            del m['fpr']
            del m['tpr']
        dummy_x = np.linspace(min(first_tpr), 1, 1000)
        plt.plot(dummy_x, 1/dummy_x, label='random')
        plt.xlabel('tpr')
        plt.ylabel('1/fpr')
        plt.xlim([0, 1])
        plt.yscale('log')
        plt.legend(frameon=False)
        plt.savefig(f"{args.logdir}/{args.exp_name}/ROC_curve.pdf")
        res = [res, {}]
        for r, m in zip(res, metrics):
            r.update(m)
            print("Test domain: " + r['domain'] +  "\t BCE Loss: %.4f \t MMD Loss: %.4f \t Acc: %.4f \t AUC: %.4f \t 1/eB 0.3: %.4f \t 1/eB 0.5: %.4f"
               % (r['test_BCE_loss'], r['test_MMD_loss'], r['test_acc'], r['test_auc'], r['test_1/eB_0.3'], r['test_1/eB_0.5']))
        json_objects = [json.dumps(r, indent=4) for r in res]
        with open(f"{args.logdir}/{args.exp_name}/test-result.json", "w") as outfile:
            for obj in json_objects:
                outfile.write(obj)