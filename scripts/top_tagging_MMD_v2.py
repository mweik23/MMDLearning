
from src.MMDLearning.training.schedulers import SchedConfig
import torch
from torch import nn, optim
import argparse, json, time
import src
import numpy as np
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts, LambdaLR, LinearLR, ConstantLR, SequentialLR
import domain_adapt as da
from copy import deepcopy
import os
import matplotlib.pyplot as plt
from pathlib import Path
SRC_path = Path(__file__).parents[1].resolve() / 'src' / 'MMDLearning'
import sys
sys.path.append(str(SRC_path))
from utils.distributed import setup_dist, DistInfo
from utils.io import config_init
from models.predictors import build_predictor
from data.dataset_v2 import initialize_datasets, retrieve_dataloaders
from utils.model_utils import print_stage_param_summary
from training.schedulers import SchedConfig
from training.losses import MMDLoss, RBF, MMDScheduler, LambdaAdjust
from training.trainer import Trainer

def build_parser():
    parser = argparse.ArgumentParser(description='Top tagging')
    parser.add_argument('--exp_name', type=str, default='', metavar='N',
                        help='experiment_name')
    parser.add_argument('--test_mode', action='store_true', default=False,
                        help = 'test best model')
    parser.add_argument('--batch_size', type=int, default=32, metavar='N',   
                        help='input batch size for training')
    parser.add_argument('--num_train', type=int, default=-1, metavar='N',
                        help='number of training samples')
    parser.add_argument('--epochs', type=int, default=35, metavar='N',
                        help='number of training epochs')
    parser.add_argument('--model_config', type=str, default='', metavar='N',
                        help='model config file')
    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='number of warm-up epochs')             
    parser.add_argument('--seed', type=int, default=99, metavar='N',
                        help='random seed')
    parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--mmd_interval', type=int, default=-1, metavar='N',
                        help='how many batches to wait before calculating the null MMD')
    parser.add_argument('--val_interval', type=int, default=1, metavar='N',
                        help='how many epochs to wait before validation')
    parser.add_argument('--datadir', nargs='+', default='./data/top', metavar='N',
                        help='data directories')
    parser.add_argument('--logdir', type=str, default='./logs/top', metavar='N',
                        help='folder to output logs')
    parser.add_argument('--dropout', type=float, default=0.2, metavar='N',
                        help='dropout probability')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='N',
                        help='learning rate')
    parser.add_argument('--final_scale', type=float, default=50, metavar='N',
                        help='max learning rate scale')
    parser.add_argument('--n_hidden', type=int, default=72, metavar='N',
                        help='dim of latent space')
    parser.add_argument('--n_layers', type=int, default=6, metavar='N',
                        help='number of LGEBs')
    parser.add_argument('--num_workers', type=int, default=None, metavar='N',
                        help='number of workers for the dataloader')
    parser.add_argument('--weight_decay', type=float, default=1e-2, metavar='N',
                        help='weight decay')
    parser.add_argument('--no_batchnorm', action='store_true', default=False,
                        help = 'test best model')
    parser.add_argument('--no_layernorm', action='store_true', default=False,
                        help = 'remove batchnorm layers')
    parser.add_argument('--auto_scale', action='store_true', default=False,
                        help = 'scale network and epochs with amount of data')
    parser.add_argument('--lr_scheduler', type=str, default='CosineAnealing', metavar='N',
                        help='patience for ReduceLROnPlateau scheduler')
    parser.add_argument('--patience', type=int, default=10, metavar='N',
                        help='learning rate scheduler')
    parser.add_argument('--reduce_factor', type=float, default=0.1, metavar='N',
                        help='factor for LR scheduler if reduce')
    parser.add_argument('--starting_lr', type=float, default=0.1, metavar='N',
                        help='starting learning rate factor for warmup')
    parser.add_argument('--MMDturnon_epoch', type=int, default=5, metavar='N',
                        help='epoch when MMD turns on')
    parser.add_argument('--MMD_coef', type=float, default=0, metavar='N',
                        help='prefactor for the MMD loss term')
    parser.add_argument('--MMD_frac', type=float, default=0, metavar='N',
                        help='prefactor for the MMD loss term')
    parser.add_argument('--MMDturnon_width', type=int, default=5, metavar='N',
                        help='the number of epochs it takes for MMD to smoothly turnon')
    parser.add_argument('--intermed_mmd', action='store_true', default=False,
                        help = 'MMD is calculated on an intermediate layer')
    parser.add_argument('--pretrained', type=str, default='', metavar='N',
                        help='directory with model to start the run with')
    parser.add_argument('--ld_optim_state', action='store_true', default=False,
                        help='want to load the optimizer state from pretrained run?')
    parser.add_argument('--n_kernels', type=int, default=5, metavar='N', 
                        help='number of kernels summed for MMD kernel')
    parser.add_argument('--use_tar_labels', action='store_true', default=False,
                        help = 'Use target labels for MMD')
    parser.add_argument('--devices', type=int,  nargs='+', default=[0], metavar='N',
                        help='device numbers')
    parser.add_argument('--bn_eval', action='store_true', default=False,
                        help='use batchnorm in eval mode')
    #added this input for compatibility with ParticleNet
    ############################################################
    parser.add_argument('--model_name', type=str, default='LorentzNet', metavar='N',
                        help='model name')
    ############################################################                    
    parser.add_argument('--local_rank', type=int, default=0)
    
    return parser

def make_train_plt(output, pretrained):
    if pretrained:
        train_start = 1
    else:
        train_start = 0
    plt.figure()
    plt.plot(output['epochs'][train_start:], output['train_BCE_loss'], color='b', linestyle='dotted', label='train BCE')
    plt.plot(output['epochs'][train_start:], output['train_MMD_loss'], color='b', linestyle='dashed', label='train MMD')
    total_loss_train = [m+b for m, b in zip(output['train_MMD_loss'], output['train_BCE_loss'])]
    plt.plot(output['epochs'][train_start:], total_loss_train, color='b', linestyle='solid', label='train total')

    total_loss_val = [m+b for m, b in zip(output['val_MMD_loss'], output['val_BCE_loss'])]
    plt.plot(output['epochs'], output['val_BCE_loss'], color='r', linestyle='dotted', label='val BCE')
    plt.plot(output['epochs'], output['val_MMD_loss'], color='r', linestyle='dashed', label='val MMD')
    plt.plot(output['epochs'], total_loss_val, color='r', linestyle='solid', label='val total')
    plt.legend(frameon=False)
    plt.ylim([-0.1, .6])
    plt.savefig(f"{args.logdir}/{args.exp_name}/loss_vs_epochs.pdf")
    plt.close()
    return None
def make_logits_plt(logits, domains=['Source', 'Target']):
    plot_data = {k:v for k,v in logits.items() if k != 'last'}
    for name, log in plot_data.items():
        for d, l in zip(domains, log):
            arr = l.numpy()
            plt.figure()
            plt.hist2d(arr[:, 0], arr[:, 1], bins=100)
            plt.savefig(f"{args.logdir}/{args.exp_name}/logits2D_{name}_{d}.pdf")
            plt.close()
        plt.figure()
        for d, l in zip(domains, log):
            arr = l.numpy()
            plt.hist(arr[:, 1]-arr[:, 0], bins=100, histtype='step', label=d, density=True)

        plt.legend(frameon=False)
        plt.savefig(f"{args.logdir}/{args.exp_name}/logit_diff_{name}.pdf")
        plt.close()
    return None

def bn_eval(model):
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.eval()            # use running stats, don't update them

def save_ckp(state, is_best, logdir, exp_name, epoch, retries=5, delay=10):
    torch.save(state, f"{logdir}/{exp_name}/checkpoint-epoch-{epoch}.pt")
    if is_best:
        print("New best validation model, saving...")
        for attempt in range(retries):
            try:
                torch.save(state, f"{logdir}/{exp_name}/best-val-model.pt")
                return
            except OSError as e:
                print(f"Attempt {attempt+1}: Failed to save checkpoint due to {e}")
                time.sleep(delay)
        print(f"Checkpoint failed after {retries} attempts. Try again later.")

# def get_pred(data):
#     #print('start get_pred()')
#     batch_size, n_nodes, _ = data['Pmu'].size()
#     atom_positions = data['Pmu'].view(batch_size * n_nodes, -1).to(local_rank, dtype)
#     #os.system('nvidia-smi')
#     atom_mask = data['atom_mask'].view(batch_size * n_nodes, -1).to(local_rank)
#     edge_mask = data['edge_mask'].reshape(batch_size * n_nodes * n_nodes, -1).to(local_rank)
#     nodes = data['nodes'].view(batch_size * n_nodes, -1).to(local_rank,dtype)
#     #os.system('nvidia-smi')
#     nodes = psi(nodes)
#     edges = [a.to(local_rank) for a in data['edges']]
#     pred, h = ddp_model(scalars=nodes, x=atom_positions, edges=edges, node_mask=atom_mask,
#                           edge_mask=edge_mask, n_nodes=n_nodes)
#     #os.system('nvidia-smi')
#     return pred, h

def load_ckp(checkpoint_fpath, model, optimizer=None):
    checkpoint = torch.load(checkpoint_fpath, map_location=torch.cuda.set_device(local_rank))
    model.load_state_dict(checkpoint['state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint['epoch']


def gather_scores(scores):
    pred = [torch.zeros_like(scores) for _ in range(dist.get_world_size())]
    dist.all_gather(pred, scores )
    return torch.cat(pred).cpu()

def gather_preds(res):
    return gather_scores(res['score'])

def get_metric(pred, res):
    fpr, tpr, thres, eB, eS  = src.buildROC(pred[...,0], pred[...,2])
    auc = src.roc_auc_score(pred[...,0], pred[...,2])
    metric = {'domain': res['domain'],'test_BCE_loss': res['BCE loss'], 'test_MMD_loss': res['MMD loss'], 'test_acc': res['acc'],
                  'test_auc': auc, 'test_1/eB_0.3':1./eB[0],'test_1/eB_0.5':1./eB[1],  'fpr': fpr, 'tpr': tpr}
    return metric

#TODO: configure for multi gpu training
def get_mmd_floor(loader, quantiles=[0.5, 0.9]):
    print('>> getting MMD stats...')
    ddp_model.eval()
    mmd_floor_dist = []
    for i, data in enumerate(loader):
        if args.model=='ParticleNet' or args.model=='ParticleNet-Lite':
            mmd_in, _ = ddp_model(data['points'].to(local_rank, dtype), data['features'].to(local_rank, dtype), mask=data['label'].to(local_rank, dtype))
        else:
            print('get MMD floor not configured for ', args.model)
        z1, z2 = torch.chunk(mmd_in, 2, dim=0)
        mmd_floor_dist.append(mmd(z1, z2))
    mmd_floor_dist = torch.stack(mmd_floor_dist, dim=0)
    assert mmd_floor_dist.device == torch.device(f"cuda:{local_rank}")
    mmd_med, mmd_upper = torch.quantile(mmd_floor_dist, torch.tensor(quantiles, device=mmd_floor_dist.device, dtype=torch.float32))
    return mmd_med, mmd_upper

def main(argv=None):
    #get arguments
    parser = build_parser()
    args = parser.parse_args(argv)
    
    #set up distributed training
    dist_info: DistInfo = setup_dist(arg_num_workers=args.num_workers)
    if dist_info.is_primary:
        print(dist_info)
    device = torch.device(dist_info.device_name)
    dtype = torch.float32
    
    #set up training configuration
    cfg = config_init(args, dist_info)
    
    ### set random seed
    torch.manual_seed(cfg.seed)#+ rank)
    np.random.seed(cfg.seed)#+ rank)
    
    
    datasets = initialize_datasets(
        datadir=cfg.datadir, 
        splits=['train', 'valid'], 
        num_data=cfg.num_train, 
        rank=dist_info.rank, 
        model=cfg.model_name)
    
    train_sampler, dataloaders = retrieve_dataloaders(
        datasets,
        2*args.batch_size,
        num_workers=dist_info.num_workers,
        rank=dist_info.rank,
        num_replicas=dist_info.world_size,
        model_arch=cfg.model_name
    )
    
    #load model config
    if cfg.model_config != '':     
        with open(cfg.model_config, 'r') as f:
            model_config = json.load(f)
    
    # get model with predictor wrapper    
    model = build_predictor(cfg.model_name, 
                            input_dims=7, #TODO: get this from the dataset shape
                            num_classes=2,
                            cfg=model_config)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.to(device)
    stage_param_groups = [
        {
            "params": list(model.stages[name].parameters()), 
            "lr": specs['optim'].get('lr', model_config['defaults']['lr']),
            "weight_decay": specs['optim'].get('weight_decay', model_config['defaults']['weight_decay']), 
            "name": name}
        for name, specs in cfg['group_specs'].items()
    ]
    print_stage_param_summary(model, print_output = dist_info.is_primary)
    optimizer = optim.AdamW(stage_param_groups)
    ddp_model = DistributedDataParallel(model, broadcast_buffers=True, device_ids=[dist_info.local_rank])
    
    if cfg.pretrained != '':
        start_epoch = load_ckp(f"{cfg.logdir}/{cfg.pretrained}/best-val-model.pt", ddp_model, optimizer = optimizer if cfg.ld_optim_state else None)
    else:
        start_epoch = 0
        
    for g in optimizer.param_groups:
        g.setdefault('initial_lr', g['lr'])
        
    sched_config = SchedConfig(
        kind = "warmup_plateau",
        lr_min=cfg.start_lr,
        warmup_epochs = cfg.warmup_epochs,
        mode = "min",
        factor = cfg.reduce_factor,
        patience = cfg.patience,
    )
    
    mmd_sched = MMDScheduler(cfg.MMDturnon_epoch, cfg.MMDturnon_width)
    
    loss_fns = {
        'bce': nn.CrossEntropyLoss(),
        'mmd': MMDLoss()
    }
    
    trainer = Trainer(
        cfg=cfg,
        ddp_model=ddp_model,
        start_epoch=start_epoch,
        optimizer=optimizer,
        dist_info=dist_info,
        device=device,
        sched_config=sched_config,
        loss_fns=loss_fns,
        mmd_sched=mmd_sched,
        train_sampler=train_sampler,
        dataloaders=dataloaders
    )
    if not args.test_mode:
        trainer.train()

    trainer.test()
    torch.cuda.empty_cache()
    
if __name__ == "__main__":
    main()

   
    
    
    