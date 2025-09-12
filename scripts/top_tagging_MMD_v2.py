
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
    parser.add_argument('--factor', type=float, default=0.1, metavar='N',
                        help='factor for LR scheduler if reduce')
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
    ddp_model = DistributedDataParallel(model, broadcast_buffers=True, device_ids=[dist_info.local_rank])
    
    stage_param_groups = [
        {
            "params": list(model.stages[name].parameters()), 
            "lr": specs['optim'].get('lr', model_config['defaults']['lr']),
            "weight_decay": specs['optim'].get('weight_decay', model_config['defaults']['weight_decay']), 
            "name": name}
        for name, specs in cfg['group_specs'].items()
    ]

if __name__ == "__main__":
    main()



    if not args.no_batchnorm:
        if rank==0:
            print('converting batchnorm to sync_batchnorm') #can turn into comment once I confirm this works
        
    model = model.to(local_rank)
    #need broadcast_buffers=True for multi GPU training
    ddp_model = DistributedDataParallel(model, broadcast_buffers=True, device_ids=[local_rank])

    ### print model and data information
    if (rank == 0):
        pytorch_total_params = sum(p.numel() for p in ddp_model.parameters())
        print("Network Size:", pytorch_total_params)

   
    #optimizer.param_groups[0]['initial_lr'] = args.lr
        #load pretrained model
    if args.pretrained != '' and not args.test_mode:
        if args.ld_optim_state:
            optimizer = optim.AdamW(ddp_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            optim_init=optimizer
        else:
             ### optimizer
            backbone = []
            mmd_btlnck = []
            classifier = []
            for name, param in ddp_model.named_parameters():
                if name.startswith('module.bn_fts') or name.startswith('module.edge_convs'):
                    backbone.append(param)
                    print('group: backbone')
                elif name.startswith('module.fc_block.fc.0'):
                    mmd_btlnck.append(param)
                    print('group: mmd bottleneck')
                elif name.startswith('module.fc_block.fc.1'):
                    classifier.append(param)
                    print('group: classifier')
                else:
                    print('param names did not match')
                print(name, ' number of parameters: ', param.numel())
            optimizer = optim.AdamW([
            { "params": backbone, "lr": 1e-6 },
            { "params": mmd_btlnck,     "lr": 1e-6 },
            { "params": classifier, "lr": 1e-6}], weight_decay=args.weight_decay)
            optim_init=None
        print('loading pretrained model')
        start_epoch = load_ckp(f"{args.logdir}/{pt_epoch}.pt", ddp_model, optimizer=optim_init)
        #optimizer.param_groups[0]['initial_lr'] = optimizer.param_groups[0]['lr']
        for group in optimizer.param_groups:
            group['initial_lr'] = group['lr']
        #best_model = torch.load(f"{args.logdir}/{args.pretrained}.pt", map_location=torch.cuda.set_device(local_rank))
        #ddp_model.load_state_dict(best_model)
        best_val_BCE = train_res_init['val_loss'][start_epoch-1]
        print('best val BCE: ', best_val_BCE)
    else:
        start_epoch=0
    print('start epoch: ', start_epoch)
    ### lr scheduler
    if args.pretrained!='':
        #print('Learning rate (before scheduler definition): ', optimizer.param_groups[0]['lr'])
        constant_steps = args.MMDturnon_epoch+args.MMDturnon_width
        linear_steps = args.warmup_epochs-constant_steps
        start_scale = 1
        factors = [[start_scale, start_scale], [start_scale, args.final_scale]]
        milestones = [[start_epoch, start_epoch+constant_steps], [start_epoch+constant_steps, start_epoch+constant_steps+linear_steps]]
        post_train_lambda = src.ParticleNetLambda(factors, milestones, ['linear', 'linear'])
        lambda_scheduler = LambdaLR(optimizer, post_train_lambda, last_epoch=start_epoch - 1)
        reduce_scheduler = plateau_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=3, min_lr=1e-6, verbose=True)
        #lr_scheduler = SequentialLR(optimizer, schedulers=[lambda_scheduler, reduce_scheduler], milestones=[start_epoch+args.warmup_epochs+1])
        #print('Learning rate (after scheduler definition): ', optimizer.param_groups[0]['lr'])
    else:
        if args.lr_scheduler=='CosineAnealing':
            base_scheduler = CosineAnnealingWarmRestarts(optimizer, restart_epoch, 2, verbose = False)
            lr_scheduler = src.GradualWarmupScheduler(optimizer, multiplier=1,
                                                    warmup_epoch=args.warmup_epochs,
                                                    after_scheduler=base_scheduler) ## warmup
        
        elif args.lr_scheduler=='Reduce':
            base_scheduler = ReduceLROnPlateau(optimizer, patience=args.patience, factor=args.factor) 
            lr_scheduler = base_scheduler
        elif args.lr_scheduler=='ParticleNet' or args.lr_scheduler=='ParticleNet-Lite':
            if args.lr_scheduler=='ParticleNet':
                LRs = [3e-4-args.warmup_epochs*5e-5, 3e-4, 3e-3, 3e-4, 5e-7]
            else:
                LRs = [5e-4-args.warmup_epochs*8e-5, 5e-4, 5e-3, 5e-4, 1e-6]
            factors = [[LRs[i], LRs[i+1]] for i in range(len(LRs)-1)]
            one_fifth = int((args.epochs-1-args.warmup_epochs)//5)
            epochs_main = [[0, 2*one_fifth], [2*one_fifth, 4*one_fifth], [4*one_fifth, 5*one_fifth]]
            epochs = [[0, args.warmup_epochs]]
            for ep in epochs_main:
                epochs.append([e+args.warmup_epochs for e in ep])
            schedule_types = ['linear', 'linear', 'linear', 'exp']
            pn_lambda = src.ParticleNetLambda(factors, epochs, schedule_types)
            lr_scheduler = LambdaLR(optimizer, pn_lambda)
        else:
            print(args.lr_scheduler + " is not a valid learning rate scheduler")
    loss_stats = {'init_MMD': 0.006, 'init_BCE': 0.27, 'lambda_adjust': (lambda x: 1)}
    mmd_sched = src.MMDScheduler(args.MMDturnon_epoch, args.MMDturnon_width)
    ### loss function
    bce = nn.CrossEntropyLoss()
    mmd = da.MMDLoss(kernel=da.RBF(n_kernels=args.n_kernels, device=local_rank))

    ### initialize logs
    res = {'epochs': [], 'lr' : [],
           'train_time': [], 'val_time': [],  'train_BCE_loss': [], 
           'train_MMD_loss': [], 'val_BCE_loss': [], 'val_MMD_loss': [], 'val_tot_loss': [],
           'train_acc': [], 'val_acc': [], 'best_val': 1e30, 'best_epoch': 0}
    val_logits = {'init': [], 'best': [], 'last': [[], []]}
    if not args.test_mode:
        ### training and testing
        train(res)
        if rank==0:
            make_train_plt(res, args.pretrained != '')
            print('shape val_logits components:')
            print('init:', [l.size() for l in val_logits['init']])
            print('best:', [l.size() for l in val_logits['best']])
            make_logits_plt(val_logits)

        test(res)
    else:
        ### only test on best model
        test(res)
    torch.cuda.empty_cache()