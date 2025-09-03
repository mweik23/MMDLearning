# src/MMDLearning/training/trainer.py
from dataclasses import dataclass
from pathlib import Path
import torch, json, time
from copy import deepcopy

#---------local imports-----------------------
from metrics import get_batch_metrics, get_correct, display_status, RunningStats
from reporting import display_epoch_summary
#---------------------------------------------

# put SRC on path
SRC_DIR = (Path(__file__).parent).resolve()
import sys
sys.path.append(str(SRC_DIR))

#-----------non-local imports from src--------------
from MMDLearning.models.predictors import make_predictor
from MMDLearning.utils.distributed import globalize_epoch_totals, epoch_metrics_from_globals, gather_scores, gather_preds
from MMDLearning.utils.buffers import EpochLogitBuffer
from MMDLearning.utils.utils import split_batch, LossDict
#---------------------------------------------------


@dataclass
class Trainer:
    # core state you currently use as globals
    args: any
    ddp_model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    lr_scheduler: any | None
    rank: int
    local_rank: int
    world_size: int

    # “context” shared across methods (use your existing objects)
    loss_fns: LossDict # {'bce': torch.nn.Module 'mmd': torch.nn.Module}
    mmd_sched: any
    loss_stats: dict
    train_sampler: any
    dataloaders: dict   # {'train': ..., 'valid': ..., 'test': ...}
    val_logits: dict | None = None  

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
        self.loss_stats = {'init_MMD': 0.006, 'init_BCE': 0.27, 'lambda_adjust': (lambda x: 1)} #guessing these values, will be updated after the first validation run

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
    
    def _run_epoch(self, epoch: int, loader, partition: str, get_buffers: bool = False, domains=['Source', 'Target']):
        #TODO: make this be an optional argument
        lambda_adjust = self.loss_stats['lambda_adjust']
        if partition == 'train':
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
            pred, encoded = self.predictor.forward(self.ddp_model, prepared)
            
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
            loss_mmd = lambda_adjust(mmd_val)*mmd_scale*self.args.MMD_frac*mmd_val*self.loss_stats['init_BCE']/self.loss_stats['init_MMD']

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

        #gather logits and labels if buffers are requested
        if get_buffers:
            payload = {d: buf.gather_to_rank0() for d, buf in bufs.items()}
        
        #globalize epoch metrics
        res = {}
        device = next(self.model.parameters()).device
        for d, tr in trackers.items():
            g_bce, g_mmd, g_corr, g_cnt = globalize_epoch_totals(
                local_bce_sum=tr.epoch_bce_sum,
                local_mmd_sum=tr.epoch_mmd_sum,
                local_correct=tr.epoch_correct,
                local_count=tr.epoch_count,
                device=device,
            )
            res[d] = epoch_metrics_from_globals(g_bce_sum=g_bce, g_mmd_sum=g_mmd, g_correct=g_corr, g_count=g_cnt)
            display_epoch_summary(partition="train", epoch=epoch, tot_epochs=self.args.epochs,
                            bce=res[d]["bce"], mmd=res[d]["mmd"], acc=res[d]["acc"], time_s=tr.epoch_time(),
                            logger=getattr(self, "logger", None))
        return res

    @torch.no_grad()
    def test(self, res: dict):
        # ——— a near copy of your current test() ———
        # load best-val-model.pt, call self._run_epoch(..., 'test'), gather preds,
        # write metrics/plots/json (rank==0)
        pass
    

    def train(self, res: dict):
        #start with a validation run if starting with a pretrained model
        print('Learning rates (before val): ', [g['lr'] for g in self.optimizer.param_groups])
        if self.args.pretrained !='':
            with torch.no_grad():
                val_res = self._run_epoch(self.start_epoch-1, self.dataloaders['valid'], partition='valid')[0]
            val_logits['init'] = [gather_scores(l) for l in val_logits['last']]
            val_logits['best'] = deepcopy(val_logits['init'])
            if (self.rank == 0): # only master process save
                is_best=False
                if self.args.MMD_frac==0:
                    loss_stats['init_MMD'] = 1
                loss_stats['init_MMD'] = val_res['MMD loss']*loss_stats['init_MMD']/loss_stats['init_BCE']/args.MMD_frac
                loss_stats['init_BCE'] = val_res['BCE loss']
                print('loss stats: ', loss_stats)
                res['val_time'].append(val_res['time'])
                res['val_BCE_loss'].append(val_res['BCE loss'])
                res['val_MMD_loss'].append(args.MMD_frac*val_res['BCE loss'])
                res['val_acc'].append(val_res['acc'])
                res['epochs'].append(start_epoch-1)
                ## save best model (minimum BCE + MMD with the MMD_coef only - no epoch dependent coefs) 
                val_loss = val_res['BCE loss'] + res['val_MMD_loss'][-1]
                if val_loss < res['best_val']:
                    is_best=True
                    res['best_val'] = val_loss
                    res['best_epoch'] = start_epoch-1
                print("Val BCE loss: %.4f \t Val MMD loss: %.4f \t  Val acc: %.4f" % (val_res['BCE loss'], val_res['MMD loss'], val_res['acc']))
                print("Best val loss: %.4f at epoch %d." % (res['best_val'],  res['best_epoch']))
                res['initial val MMD'] = loss_stats['init_MMD']
                res['initial val BCE'] = loss_stats['init_BCE']


        ### training and validation
        train_sampler.set_epoch(start_epoch-1)
        for epoch in range(start_epoch, start_epoch+args.epochs):
            #percentiles = get_mmd_floor(ddp_model, data1, data2)
            is_best=False
            if args.mmd_interval != -1:
                if (epoch-start_epoch) % args.mmd_interval == 0:
                    with torch.no_grad():
                        quantiles =[0.5, 0.975]
                        mmd_med, mmd_upper = get_mmd_floor(dataloaders['train'], quantiles=quantiles)
                        print('MMD quantile ', quantiles[0], ': ',  mmd_med)
                        print('MMD quantile ', quantiles[1], ': ',  mmd_upper)
                    lambda_adjust = src.LambdaAdjust(2*mmd_upper, 2*(mmd_upper-mmd_med)**(-1))
                    loss_stats['lambda_adjust'] = lambda_adjust
            print('Learning rate: ', [g['lr'] for g in optimizer.param_groups])
            train_res = run(epoch, dataloaders['train'], partition='train')[0]
            print("Time: train: %.2f \t Train BCE loss %.4f \t Train MMD loss %.4f \t Train acc: %.4f" % (train_res['time'], train_res['BCE loss'], train_res['MMD loss'], train_res['acc']))
            if epoch % args.val_interval == 0:
                val_logits['last'] = [[], []]
                with torch.no_grad():
                    val_res = run(epoch, dataloaders['valid'], partition='valid')[0]
                val_loss = val_res['BCE loss'] + val_res['MMD loss']
                if (rank == 0): # only master process save
                    res['lr'].append([g['lr'] for g in optimizer.param_groups])
                    res['train_time'].append(train_res['time'])
                    res['val_time'].append(val_res['time'])
                    res['train_BCE_loss'].append(train_res['BCE loss'])
                    res['train_MMD_loss'].append(train_res['MMD loss'])
                    res['train_acc'].append(train_res['acc'])
                    res['val_BCE_loss'].append(val_res['BCE loss'])
                    res['val_MMD_loss'].append(val_res['MMD loss'])
                    res['val_acc'].append(val_res['acc'])
                    res['epochs'].append(epoch)
                    res['val_tot_loss'].append(val_loss)
                    if val_loss < res['best_val']:
                        is_best=True
                        val_logits['best'] = [gather_scores(l) for l in val_logits['last']]
                        res['best_val'] = val_loss
                        res['best_epoch'] = epoch
                    checkpoint = {'epoch': epoch + 1, 'state_dict': ddp_model.state_dict(), 'optimizer': optimizer.state_dict()}
                    save_ckp(checkpoint, is_best, args.logdir, args.exp_name, epoch)
                    print("Epoch %d/%d finished." % (epoch, start_epoch+args.epochs-1))
                    print("Train time: %.2f \t Val time %.2f" % (train_res['time'], val_res['time']))
                    print("Train BCE loss %.4f \t Train MMD loss %.4f \t Train acc: %.4f" % (train_res['BCE loss'], train_res['MMD loss'], train_res['acc']))
                    print("Val BCE loss: %.4f \t Val MMD loss: %.4f \t  Val acc: %.4f" % (val_res['BCE loss'], val_res['MMD loss'], val_res['acc']))
                    print("Best val loss: %.4f at epoch %d." % (res['best_val'],  res['best_epoch']))

                    json_object = json.dumps(res, indent=4)
                    with open(f"{args.logdir}/{args.exp_name}/train-result.json", "w") as outfile:
                        outfile.write(json_object)
            #print('DEBUG: ', args.lr_scheduler)
            ## adjust learning rate
            if args.lr_scheduler=='Reduce':
                lr_scheduler.step(val_loss)
            elif args.lr_scheduler=='CosineAnealing':
                ## adjust learning rate
                if (epoch < 31*int(round(1/ratio**(1/2)))):
                    lr_scheduler.step(metrics=val_res['BCE loss'] + val_res['MMD loss'])
                else:
                    for g in optimizer.param_groups:
                        g['lr'] = g['lr']*0.5**(ratio**(1/2))
            elif args.lr_scheduler=='ParticleNet' or args.lr_scheduler=='ParticleNet-Lite':
                #sched_message = f"Epoch {epoch}/{args.epochs+start_epoch}, LR: {lr_scheduler.get_last_lr()[0]:.9f}"
                #print(sched_message)
                if epoch < start_epoch + args.warmup_epochs:
                    lambda_scheduler.step()
                else:
                    reduce_scheduler.step(val_loss)
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