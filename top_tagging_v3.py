from src import dataset as dset
import torch
from torch import nn, optim
import argparse, json, time
import src
import numpy as np
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts, LambdaLR
import os
import matplotlib.pyplot as plt

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
parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                    help='number of warm-up epochs')
parser.add_argument('--c_weight', type=float, default=5e-3, metavar='N',
                    help='weight of x model')                 
parser.add_argument('--seed', type=int, default=99, metavar='N',
                    help='random seed')
parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
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
parser.add_argument('--n_hidden', type=int, default=72, metavar='N',
                    help='dim of latent space')
parser.add_argument('--n_layers', type=int, default=6, metavar='N',
                    help='number of LGEBs')
parser.add_argument('--n_latent', type=int, default=0, metavar='N',
                    help='dim of latent space')
parser.add_argument('--num_workers', type=int, default=4, metavar='N',
                    help='number of workers for the dataloader')
parser.add_argument('--weight_decay', type=float, default=1e-2, metavar='N',
                    help='weight decay')
parser.add_argument('--no_batchnorm', action='store_true', default=False,
                    help = 'remove batchnorm layers')
parser.add_argument('--no_layernorm', action='store_true', default=False,
                    help = 'removed layernorm layers')
parser.add_argument('--auto_scale', action='store_true', default=False,
                    help = 'test best model')
parser.add_argument('--lr_scheduler', type=str, default='CosineAnealing', metavar='N',
                    help='learning rate scheduler')
parser.add_argument('--patience', type=int, default=10, metavar='N',
                    help='learning rate scheduler')
parser.add_argument('--factor', type=float, default=0.1, metavar='N',
                    help='factor for LR scheduler if reduce')
parser.add_argument('--manual', action='store_true', default=False,
                    help = 'run manually')
parser.add_argument('--devices', type=int,  nargs='+', default=[0], metavar='N',
                    help='device numbers')
#added this input for compatibility with ParticleNet
############################################################
parser.add_argument('--model', type=str, default='LorentzNet', metavar='N',
                    help='model_name')
############################################################
parser.add_argument('--local_rank', type=int, default=0)

def make_train_plt(output, pretrained=False):
    if pretrained:
        train_start = 1
    else:
        train_start = 0
    plt.figure()
    plt.plot(output['epochs'], output['train_loss'], color='b', label='train')
    plt.plot(output['epochs'], output['val_loss'], color='r', label='valid')
    plt.legend(frameon=False)
    plt.ylim([0, .4])
    plt.savefig(f"{args.logdir}/{args.exp_name}/loss_vs_epochs.pdf")
    return None


def save_ckp(state, is_best, logdir, exp_name, epoch):
    torch.save(state, f"{logdir}/{exp_name}/checkpoint-epoch-{epoch}.pt")
    if is_best:
        print("New best validation model, saving...")
        torch.save(state, f"{logdir}/{exp_name}/best-val-model.pt")

def run(epoch, loader, partition):
    if partition == 'train':
        train_samplers[0].set_epoch(epoch)
        ddp_model.train()
    else:
        ddp_model.eval()

    res = {'time':0, 'correct':0, 'loss': 0, 'counter': 0, 'acc': 0,
           'loss_arr':[], 'correct_arr':[],'label':[],'score':[]}

    tik = time.time()
    loader_length = len(loader)
    for i, data in enumerate(loader):
        if partition == 'train':
            optimizer.zero_grad()
        #print('start of batch ', i)
        #os.system('nvidia-smi')
        if args.model=='LorentzNet':
            batch_size, n_nodes, _ = data['Pmu'].size()
            atom_positions = data['Pmu'].view(batch_size * n_nodes, -1).to(local_rank, dtype)
            atom_mask = data['atom_mask'].view(batch_size * n_nodes, -1).to(local_rank)
            edge_mask = data['edge_mask'].reshape(batch_size * n_nodes * n_nodes, -1).to(local_rank)
            nodes = data['nodes'].view(batch_size * n_nodes, -1).to(local_rank,dtype)
            nodes = psi(nodes)
            edges = [a.to(local_rank) for a in data['edges']]
            pred, _ = ddp_model(scalars=nodes, x=atom_positions, edges=edges, node_mask=atom_mask,
                         edge_mask=edge_mask, n_nodes=n_nodes)
        elif args.model=='ParticleNet' or args.model=='ParticleNet-Lite':
            batch_size, _, _ = data['label'].shape
            pred = ddp_model(data['points'].to(local_rank, dtype), data['features'].to(local_rank, dtype), mask=data['label'].to(local_rank, dtype))[-1]
        
        label = data['is_signal'].to(local_rank, dtype).long()
        #print('before model eval batch ', i)
        #os.system('nvidia-smi')
        #print('before pred batch ', i)

        #print(torch.cuda.memory_summary())
        
        #print('after pred batch ', i)
        #print(torch.cuda.memory_summary())
        #print('after model eval batch ', i)
        #os.system('nvidia-smi')
        predict = pred.max(1).indices
        correct = torch.sum(predict == label).item()
        loss = loss_fn(pred, label)
        
        if partition == 'train':
            loss.backward()
            optimizer.step()
        elif partition == 'test': 
            # save labels and probilities for ROC / AUC
            score = torch.nn.functional.softmax(pred, dim = -1)
            res['label'].append(label)
            res['score'].append(score)

        res['time'] = time.time() - tik
        res['correct'] += correct
        res['loss'] += loss.item() * batch_size
        res['counter'] += batch_size
        res['loss_arr'].append(loss.item())
        res['correct_arr'].append(correct)

        if i != 0 and i % args.log_interval == 0:
            running_loss = sum(res['loss_arr'][-args.log_interval:])/len(res['loss_arr'][-args.log_interval:])
            running_acc = sum(res['correct_arr'][-args.log_interval:])/(len(res['correct_arr'][-args.log_interval:])*batch_size)
            avg_time = res['time']/res['counter'] * batch_size
            tmp_counter = src.sum_reduce(res['counter'], device = local_rank)
            tmp_loss = src.sum_reduce(res['loss'], device = local_rank) / tmp_counter
            tmp_acc = src.sum_reduce(res['correct'], device = local_rank) / tmp_counter
            if (rank == 0):
                print(">> %s \t Epoch %d/%d \t Batch %d/%d \t Loss %.4f \t Running Acc %.3f \t Total Acc %.3f \t Avg Batch Time %.4f" %
                     (partition, epoch + 1, args.epochs, i, loader_length, running_loss, running_acc, tmp_acc, avg_time))
        #print('end of batch')
        #os.system('nvidia-smi')
    #print(torch.cuda.memory_summary())
    torch.cuda.empty_cache()
    # ---------- reduce -----------
    if partition == 'test':
        res['label'] = torch.cat(res['label']).unsqueeze(-1)
        res['score'] = torch.cat(res['score'])
        res['score'] = torch.cat((res['label'],res['score']),dim=-1)
    res['counter'] = src.sum_reduce(res['counter'], device = local_rank).item()
    res['loss'] = src.sum_reduce(res['loss'], device = local_rank).item() / res['counter']
    res['acc'] = src.sum_reduce(res['correct'], device = local_rank).item() / res['counter']
    return res

def train(res):
    ### training and validation
    for epoch in range(0, args.epochs):
        train_res = run(epoch, dataloaders_set[0]['train'], partition='train')
        if rank==0:
            print("Time: train: %.2f \t Train loss %.4f \t Train acc: %.4f" % (train_res['time'],train_res['loss'],train_res['acc']))
        dist.barrier()
        if epoch % args.val_interval == 0:
            with torch.no_grad():
                val_res = run(epoch, dataloaders_set[0]['valid'], partition='valid')
            if rank == 0: # only master process save
                is_best = False
                res['lr'].append(optimizer.param_groups[0]['lr'])
                res['train_time'].append(train_res['time'])
                res['val_time'].append(val_res['time'])
                res['train_loss'].append(train_res['loss'])
                res['train_acc'].append(train_res['acc'])
                res['val_loss'].append(val_res['loss'])
                res['val_acc'].append(val_res['acc'])
                res['epochs'].append(epoch)

                ## save best model
                if val_res['loss'] < res['best_val']:
                    is_best = True
                    res['best_val'] = val_res['loss']
                    res['best_epoch'] = epoch
                checkpoint = {'epoch': epoch + 1, 'state_dict': ddp_model.state_dict(), 'optimizer': optimizer.state_dict()}
                save_ckp(checkpoint, is_best, args.logdir, args.exp_name, epoch)
                print("Epoch %d/%d finished." % (epoch, args.epochs))
                print("Train time: %.2f \t Val time %.2f" % (train_res['time'], val_res['time']))
                print("Train loss %.4f \t Train acc: %.4f" % (train_res['loss'], train_res['acc']))
                print("Val loss: %.4f \t Val acc: %.4f" % (val_res['loss'], val_res['acc']))
                print("Best val loss: %.4f at epoch %d." % (res['best_val'],  res['best_epoch']))

                json_object = json.dumps(res, indent=4)
                with open(f"{args.logdir}/{args.exp_name}/train-result.json", "w") as outfile:
                    outfile.write(json_object)

        ## adjust learning rate
        if args.lr_scheduler=='Reduce':
            lr_scheduler.step(val_res['loss'])
        elif args.lr_scheduler=='CosineAnealing':
            if (epoch < (31/35)*args.epochs):
                lr_scheduler.step(metrics=val_res['loss'])
                
            else:
                for g in optimizer.param_groups:
                    g['lr'] = g['lr']*0.5**(35/args.epochs)
            
        elif args.lr_scheduler=='ParticleNet' or args.lr_scheduler=='ParticleNet-Lite':
            sched_message = f"Epoch {epoch}/{args.epochs}, LR: {lr_scheduler.get_last_lr()[0]:.9f}"
            lr_scheduler.step()

        dist.barrier() # syncronize

def test(res, domain):
    ### test on best model
    best_model = torch.load(f"{args.logdir}/{args.exp_name}/best-val-model.pt", map_location=torch.cuda.set_device(local_rank))
    ddp_model.load_state_dict(best_model['state_dict'])
    with torch.no_grad():
        test_res = run(0, dataloaders_set[domain]['test'], partition='test')

    pred = [torch.zeros_like(test_res['score']) for _ in range(dist.get_world_size())]
    dist.all_gather(pred, test_res['score'] )
    pred = torch.cat(pred).cpu()

    if (rank == 0):
        np.save(f"{args.logdir}/{args.exp_name}/score_{domain_names[domain]}.npy",pred)
        fpr, tpr, thres, eB, eS  = src.buildROC(pred[...,0], pred[...,2])
        auc = src.roc_auc_score(pred[...,0], pred[...,2])

        metric = {'test_loss': test_res['loss'], 'test_acc': test_res['acc'],
                  'test_auc': auc, 'test_1/eB_0.3':1./eB[0],'test_1/eB_0.5':1./eB[1]}
        res.update(metric)
        print("Test: Loss %.4f \t Acc %.4f \t AUC: %.4f \t 1/eB 0.3: %.4f \t 1/eB 0.5: %.4f"
               % (test_res['loss'], test_res['acc'], auc, 1./eB[0], 1./eB[1]))
        json_object = json.dumps(res, indent=4)
        json_name = f"{args.logdir}/{args.exp_name}/test-result_{domain_names[domain]}.json"
        with open(json_name,  "w") as outfile:
            outfile.write(json_object)
        return fpr, tpr
if __name__ == "__main__":
    ### FOR SLURM
    ##########################
    #### set rank and local rank
    args = parser.parse_args()
    world_size = int(os.environ["WORLD_SIZE"])
    
    if args.manual:
        if len(args.devices) != world_size:
            devices = range(args.devices[0], args.devices[0]+world_size)
        else:
            devices = args.devices
        rank = args.local_rank
        local_rank = devices[args.local_rank]
        num_workers = args.num_workers
    else:
        rank  = int(os.environ["SLURM_PROCID"])
        gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])
        device_name = torch.cuda.get_device_name()
        local_rank = rank - gpus_per_node * (rank // gpus_per_node)
        num_workers = int(os.environ["SLURM_CPUS_PER_TASK"])
    #can come back to this. Might be better to remove some contents from this function
    src.args_init(args, rank, world_size)
    ##########################

    #added for compatibilty with ParticleNet
    #######################################
    if args.model=='LorentzNet':
        from src.MMDLearning.models.model_LNet import psi, LorentzNet
    elif args.model=='ParticleNet' or args.model=='ParticleNet-Lite':
        from src.MMDLearning.models.model_PNet import ParticleNet
    ### set random seed
    torch.manual_seed(args.seed)#+rank)
    np.random.seed(args.seed)#+rank)

    num_train = args.num_train
    if num_train==-1:
        num_test = -1
        num_val = -1
    else:
        num_test = .2*num_train
        if num_test<1e5:
            num_test=1e5
        num_val = num_test
    num_pts=[{'train':num_train,'test':num_test,'valid':num_val}]
    num_pts.append({'test': num_pts[0]['test']})

    #added for compatiblility with ParticleNet
    ###########################################
    if args.model=='LorentzNet':
        reg_params=None
    elif args.model=='ParticleNet' or args.model=='ParticleNet-Lite':
        #order of reg_params: logpt, logE, logpt_ptjet, logE_Ejet, DeltaR
        reg_params=None
        #reg_params=[(1.7, 0.7), (2.0, 0.7), (-4.7, 0.7), (-4.7, 0.7), (0.2, 4.0)]
    ###########################################
    #load data
    print('data directories: ', args.datadir)
    datasets_domains = [dset.initialize_datasets(dd, num_pts=n, rank=rank, reg_params=reg_params, logdir=args.logdir) for dd, n in zip(args.datadir, num_pts)]

    ### FOR SLURM
    ##########################
    # initialize cuda
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    if rank == 0: print(f"Group initialized? {dist.is_initialized()}", flush=True)

    torch.cuda.set_device(local_rank)
    ##########################
    ### initialize cuda
    #dist.init_process_group(backend='nccl')
    #device = torch.device("cuda:{}".format(args.local_rank))
    dtype = torch.float32

    domains = range(len(args.datadir))
    domain_names =['Source', 'Target']
    ### create data loader (modified for SLURM) #TODO come back to this after I hear from Ranit
    samplers_dataloaders = [dset.retrieve_dataloaders(
                                    datasets,
                                    args.batch_size,
                                    num_workers=num_workers,
                                    rank=rank,
                                    num_replicas=world_size,
                                    collate_config=args.model) for datasets in datasets_domains]
    train_samplers = [sd[0] for sd in samplers_dataloaders]
    dataloaders_set = [sd[1] for sd in samplers_dataloaders]
    if (rank==0):
        for (split, dataloader) in dataloaders_set[0].items():
                s = len(dataloader.dataset)
                num_pts[0][split] = s
                print(f" {split} samples: {s}")
    ### create parallel model
    if args.model=='LorentzNet':
        restart_epoch = 4
        ratio =1
        if args.auto_scale:
            ratio = num_pts[0]['train']/1.2e6
            args.n_hidden = int(round(args.n_hidden*ratio**(1/3)))
            args.n_layers = int(round(args.n_layers*ratio**(1/3)))
            args.epochs = int(round(args.epochs/ratio**(1/2)))
            args.warmup_epochs = int(round(args.warmup_epochs/ratio**(1/2)))
            restart_epoch = int(round(restart_epoch/ratio**(1/2)))
        if rank==0:
            print('hidden: ', args.n_hidden)
            print('layers: ', args.n_layers)
            print('epochs: ', args.epochs)
        model = LorentzNet(n_scalar = 2, n_hidden = args.n_hidden, n_class = 2,
                       dropout = args.dropout, n_layers = args.n_layers,
                       c_weight = args.c_weight, no_batchnorm=args.no_batchnorm, no_layernorm=args.no_layernorm)
    else: 
        if args.model=='ParticleNet':
            kwargs={'fc_params': [(256, 0), (32, 0), (256, args.dropout)], 'conv_params': [(16, (64, 64, 64)),(16, (128, 128, 128)),(16, (256, 256, 256))]}
        elif args.model=='ParticleNet-Lite': 
            kwargs={'fc_params': [(128, 0), (32, 0), (128, args.dropout)], 'conv_params': [(7, (32, 32, 32)),(7, (64, 64, 64))]}

        model = ParticleNet(7,
        num_classes=2,
        conv_params=kwargs.get('conv_params', None),
        fc_params=kwargs.get('fc_params', None),
        use_fusion=kwargs.get('use_fusion', False),
        use_fts_bn=kwargs.get('use_fts_bn', True),
        use_counts=kwargs.get('use_counts', True),
        for_inference=kwargs.get('for_inference', False),
        out_lyrs=[1, -1])
    if not args.no_batchnorm:
        if rank==0:
            print('converting batchnorm to sync_batchnorm') #can turn into comment once I confirm this works
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model = model.to(local_rank)
    print('model details')
    print(model)
    ddp_model = DistributedDataParallel(model, device_ids=[local_rank])

   ### print model and data information
    if (rank == 0):
        pytorch_total_params = sum(p.numel() for p in ddp_model.parameters())
        print("Network Size:", pytorch_total_params)

    ### optimizer
    optimizer = optim.AdamW(ddp_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    ### lr scheduler
    if args.lr_scheduler=='CosineAnealing':
        base_scheduler = CosineAnnealingWarmRestarts(optimizer, restart_epoch, 2, verbose = False)
        lr_scheduler = src.GradualWarmupScheduler(optimizer, multiplier=1,
                                                warmup_epoch=args.warmup_epochs,
                                                after_scheduler=base_scheduler) 
    elif args.lr_scheduler=='Reduce':
        base_scheduler = ReduceLROnPlateau(optimizer, patience=args.patience, factor=args.factor)
        lr_scheduler = base_scheduler
    elif args.lr_scheduler=='ParticleNet' or args.lr_scheduler=='ParticleNet-Lite':
        if args.lr_scheduler=='ParticleNet':
            LRs = [3e-4, 3e-3, 3e-4, 5e-7]
        else:
            LRs = [5e-4, 5e-3, 5e-4, 1e-6]
        factors = [[LRs[i], LRs[i+1]] for i in range(len(LRs)-1)]
        one_fifth = int((args.epochs-1)//5)
        epochs = [[0, 2*one_fifth], [2*one_fifth, 4*one_fifth], [4*one_fifth, args.epochs-1]]
        schedule_types = ['linear', 'linear', 'exp']
        pn_lambda = src.ParticleNetLambda(factors, epochs, schedule_types)
        lr_scheduler = LambdaLR(optimizer, pn_lambda)
    else:
        print(args.lr_scheduler + " is not a valid learning rate scheduler")
    ## warmup

    ### loss function
    loss_fn = nn.CrossEntropyLoss()

    ### initialize logs
    res = {'epochs': [], 'lr' : [],
           'train_time': [], 'val_time': [],  'train_loss': [], 'val_loss': [],
           'train_acc': [], 'val_acc': [], 'best_val': 1e30, 'best_epoch': 0}

    if not args.test_mode:
        ### training and testing
        train(res)
        if rank==0:
            make_train_plt(res)

    if rank==0:
        fig = plt.figure()
        #make random roc curve
        start_tpr = 1e-3
        dummy_x = np.linspace(start_tpr, 1, 1000)
        plt.plot(dummy_x, 1/dummy_x, label='random')
        for d in domains:
            fpr, tpr = test(res, d)
            idx = int(max(np.max(np.where(fpr==0)[0]), np.max(np.where(tpr==0)[0]))) + 1
            plt.plot(tpr[idx:], 1/fpr[idx:], label=domain_names[d])
        #finish plot of roc curves and save
        plt.xlabel('tpr')
        plt.ylabel('1/fpr')
        plt.xlim([0, 1])
        plt.yscale('log')
        plt.legend(frameon=False)
        plt.savefig(f"{args.logdir}/{args.exp_name}/ROC_curve.pdf")
    else:
        for d in domains:
            test(res, d)
    
    print('rank ', rank, ' is done!')
    torch.cuda.empty_cache()
