from src import dataset_v2 as dset
from src.MMDLearning.models.model_LNet import psi, LorentzNet
import torch
from torch import nn, optim
import argparse, json, time
import src
import numpy as np
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
import domain_adapt as da
from copy import deepcopy
import os
import matplotlib.pyplot as plt
import gc

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
parser.add_argument('--num_workers', type=int, default=4, metavar='N',
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
parser.add_argument('--MMD_coef', type=float, default=0, metavar='N', help='prefactor for the MMD loss term')
parser.add_argument('--MMDturnon_width', type=int, default=5, metavar='N',
                    help='the number of epochs it takes for MMD to smoothly turnon')
parser.add_argument('--intermed_mmd', action='store_true', default=False,
                    help = 'MMD is calculated on an intermediate layer')
parser.add_argument('--pascal', action='store_true', default=False,
                    help = 'run on pascal')
parser.add_argument('--devices', type=int,  nargs='+', default=[0], metavar='N',
                    help='device numbers')
parser.add_argument('--local_rank', type=int, default=0)

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
    return None

def save_ckp(state, is_best, logdir, exp_name, epoch):
    torch.save(state, f"{logdir}/{exp_name}/checkpoint-epoch-{epoch}.pt")
    if is_best:
        print("New best validation model, saving...")
        torch.save(state, f"{logdir}/{exp_name}/best-val-model.pt")

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

def get_correct(pred, label):
    predict = pred.max(1).indices
    correct = torch.sum(predict == label).item()
    return correct
def calc_BCE(res, pred, label):
    batch_size = pred.size(dim=0) #keep in mind this is the batch_size of only the dataset that is passed into this function
    correct = get_correct(pred, label)
    loss_BCE = bce(pred, label)
    res['counter'] += batch_size
    res['correct'] += correct
    res['BCE loss'] += loss_BCE.detach().item() * batch_size
    res['BCE loss_arr'].append(loss_BCE.item())
    res['correct_arr'].append(correct)
    return loss_BCE
def display_status(res, num_batches, partition, epoch, batch_num, batch_size):
    running_BCE_loss = sum(res['BCE loss_arr'][-args.log_interval:])/len(res['BCE loss_arr'][-args.log_interval:])
    running_MMD_loss = sum(res['MMD loss_arr'][-args.log_interval:])/len(res['MMD loss_arr'][-args.log_interval:])
    running_acc = sum(res['correct_arr'][-args.log_interval:])/(len(res['correct_arr'][-args.log_interval:])*batch_size)
    avg_time = res['time']/res['counter'] * batch_size
    tmp_counter = src.sum_reduce(res['counter'], device = local_rank)
    #tmp_BCE_loss = utils.sum_reduce(res['BCE loss'], device = local_rank) / tmp_counter
    #tmp_MMD_loss = utils.sum_reduce(res['MMD loss'], device = local_rank) / tmp_counter
    tmp_acc = src.sum_reduce(res['correct'], device = local_rank) / tmp_counter
    if (rank == 0):
        print('domain: ', res['domain'])
        print(">> %s \t Epoch %d/%d \t Batch %d/%d \t BCE Loss %.4f \t MMD Loss %.4f \t Running Acc %.3f \t Total Acc %.3f \t Avg Batch Time %.4f" %
                (partition, epoch, args.epochs, batch_num, num_batches, running_BCE_loss, running_MMD_loss,  running_acc, tmp_acc, avg_time))

def gather_preds(res):
    pred = [torch.zeros_like(res['score']) for _ in range(dist.get_world_size())]
    dist.all_gather(pred, res['score'] )
    return torch.cat(pred).cpu()

def get_metric(pred, res):
    fpr, tpr, thres, eB, eS  = src.buildROC(pred[...,0], pred[...,2])
    auc = src.roc_auc_score(pred[...,0], pred[...,2])
    metric = {'domain': res['domain'],'test_BCE_loss': res['BCE loss'], 'test_MMD_loss': res['MMD loss'], 'test_acc': res['acc'],
                  'test_auc': auc, 'test_1/eB_0.3':1./eB[0],'test_1/eB_0.5':1./eB[1],  'fpr': fpr, 'tpr': tpr}
    return metric

def run(epoch, loader, partition):
    if partition == 'train':
        train_sampler.set_epoch(epoch)
        ddp_model.train()
    else:
        ddp_model.eval()

    res = [{'domain': 'Source', 'time':0, 'correct':0, 'BCE loss': 0, 'MMD loss': 0, 'counter': 0, 'acc': 0,
        'BCE loss_arr':[], 'MMD loss_arr':[], 'correct_arr':[],'label':[],'score':[]}]
    if partition=='test':
        res = [deepcopy(res[0]), deepcopy(res[0])]
        res[1]['domain']= 'Target'
    tik = time.time()
    loader_length = len(loader)
    #print(partition, loader_lengths)
    #need to make prediction for source and target data
    for i, data in enumerate(loader):
        if partition == 'train':
            optimizer.zero_grad()
        #print('start batch')
        #os.system('nvidia-smi')
        #calculate prediction for target and source but only keep labels for the source data
        source_tar_mask = [data['is_source']==s for s in is_source]
        batch_sizes = [torch.sum(s) for s in source_tar_mask]
        batch_size_avg = (sum(batch_sizes)/2).to(local_rank)
        #os.system('nvidia-smi')
        batch_size, n_nodes, _ = data['Pmu'].size()
        atom_positions = data['Pmu'].view(batch_size * n_nodes, -1).to(local_rank, dtype)
        atom_mask = data['atom_mask'].view(batch_size * n_nodes, -1).to(local_rank)
        edge_mask = data['edge_mask'].reshape(batch_size * n_nodes * n_nodes, -1).to(local_rank)
        nodes = data['nodes'].view(batch_size * n_nodes, -1).to(local_rank, dtype)
        nodes = psi(nodes)
        edges = [a.to(local_rank) for a in data['edges']]
        #print('before pred batch ', i)
        #print(torch.cuda.memory_summary())
        pred, _ = ddp_model(scalars=nodes, x=atom_positions, edges=edges, node_mask=atom_mask,
                            edge_mask=edge_mask, n_nodes=n_nodes)
        
        #print('pred size', pred.element_size()*pred.nelement())
        label = data['is_signal'].to(local_rank, dtype).long()
        #print('after pred batch ', i)
        #print(torch.cuda.memory_summary())
        #os.system('nvidia-smi')
        if partition =='test':
            score = torch.nn.functional.softmax(pred, dim = -1)
            for r, s in zip(res, source_tar_mask):
                r['label'].append(label[s].detach())
                r['score'].append(score[s].detach())
        
        #note the method calc_BCE() also updates the res variable
        #loss_BCE = [calc_BCE(res[i], preds[i], l) for i, l in enumerate(labels)]
        loss_BCE = [calc_BCE(r, pred[source_tar_mask[j]] , label[source_tar_mask[j]]) for j, r in enumerate(res)]
        #print('memory of bce loss: ', loss_BCE[0].element_size() * loss_BCE[0].nelement())
        #os.system('nvidia-smi')
        #MMD_value = args.MMD_coef*mmd(pred[source_tar_mask[0]], pred[source_tar_mask[1]]) #still contains MMD_coef
        #loss_MMD = mmd_sched(epoch)*MMD_value
        MMD_scale = mmd_sched(epoch)
        loss_MMD = MMD_scale*args.MMD_coef*mmd(pred[source_tar_mask[0]], pred[source_tar_mask[1]])
        #print('memory of mmd loss: ', loss_MMD.element_size() * loss_MMD.nelement())
        #print(f'loss MMD: {loss_MMD}')

        #losses = [l + loss_MMD for l in loss_BCE]

        if partition == 'train':
            #print('loss size: ', (loss_BCE[0]+loss_MMD).element_size())
            (loss_BCE[0] + loss_MMD).backward()
            #loss_BCE[0].backward()
            #print('loss.backward() complete')
            optimizer.step()
            #print('optimizer.step() complete')

        elapsed = time.time() - tik
        for r in res:
            r['time'] = elapsed
            r['MMD loss'] += (loss_MMD/MMD_scale).detach().item() * args.batch_size*(batch_size_avg/(args.batch_size))**2
            r['MMD loss_arr'].append((loss_MMD/MMD_scale).item())
        if i != 0 and i % args.log_interval == 0:
            #TODO check if this should be batch_sizes[0] or args.batch_size
            display_status(res[0], loader_length, partition, epoch, i, batch_size_avg)
            if partition=='test':
                display_status(res[1], loader_length, partition, epoch, i, batch_size_avg)
        #print('end of batch')
        #print(os.system('nvidia-smi'))

    torch.cuda.empty_cache() #can put this in the batch loop to free memory at the end of each batch
    # ---------- reduce -----------
    if partition == 'test':
        for r in res:
            r['label'] = torch.cat(r['label']).unsqueeze(-1)
            r['score'] = torch.cat(r['score'])
            r['score'] = torch.cat((r['label'], r['score']),dim=-1)
    for r in res:
        r['counter'] = src.sum_reduce(r['counter'], device = local_rank).item()
        r['BCE loss'] = src.sum_reduce(r['BCE loss'], device = local_rank).item() / r['counter']
        r['MMD loss'] = src.sum_reduce(r['MMD loss'], device = local_rank).item() / r['counter']
        r['acc'] = src.sum_reduce(r['correct'], device = local_rank).item() / r['counter']
    print(r['counter'])
    return res

def train(res):
    ### training and validation
    for epoch in range(0, args.epochs):
        train_res = run(epoch, dataloaders['train'], partition='train')[0]
        print("Time: train: %.2f \t Train BCE loss %.4f \t Train MMD loss %.4f \t Train acc: %.4f" % (train_res['time'], train_res['BCE loss'], train_res['MMD loss'], train_res['acc']))
        if epoch % args.val_interval == 0:
            with torch.no_grad():
                val_res = run(epoch, dataloaders['valid'], partition='valid')[0]
            val_loss = val_res['BCE loss'] + val_res['MMD loss']
            if (rank == 0): # only master process save
                res['lr'].append(optimizer.param_groups[0]['lr'])
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
                    res['best_val'] = val_loss
                    res['best_epoch'] = epoch
                checkpoint = {'epoch': epoch + 1, 'state_dict': ddp_model.state_dict(), 'optimizer': optimizer.state_dict()}
                save_ckp(checkpoint, is_best, args.logdir, args.exp_name, epoch)
                print("Epoch %d/%d finished." % (epoch, args.epochs))
                print("Train time: %.2f \t Val time %.2f" % (train_res['time'], val_res['time']))
                print("Train BCE loss %.4f \t Train MMD loss %.4f \t Train acc: %.4f" % (train_res['BCE loss'], train_res['MMD loss'], train_res['acc']))
                print("Val BCE loss: %.4f \t Val MMD loss: %.4f \t  Val acc: %.4f" % (val_res['BCE loss'], val_res['MMD loss'], val_res['acc']))
                print("Best val acc: %.4f at epoch %d." % (res['best_val'],  res['best_epoch']))

                json_object = json.dumps(res, indent=4)
                with open(f"{args.logdir}/{args.exp_name}/train-result.json", "w") as outfile:
                    outfile.write(json_object)

        ## adjust learning rate
        if args.lr_scheduler=='Reduce':
            lr_scheduler.step(val_loss)
        else:
            ## adjust learning rate
            if (epoch < 31*int(round(1/ratio**(1/2)))):
                lr_scheduler.step(metrics=val_res['BCE loss'] + val_res['MMD loss'])
            else:
                for g in optimizer.param_groups:
                    g['lr'] = g['lr']*0.5**(ratio**(1/2))

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

if __name__ == "__main__":
    print('starting script')
    ### initialize args
    args = parser.parse_args()
    world_size = int(os.environ["WORLD_SIZE"])
    if args.pascal:
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
    
    src.args_init(args, rank, world_size)

    ### set random seed
    torch.manual_seed(args.seed)#+ rank)
    np.random.seed(args.seed)#+ rank)

    num_train = args.num_train
    if num_train==-1:
        num_test = -1
        num_val = -1
    else:
        num_test = .2*num_train
        num_val = num_test
    num_pts={'train':num_train,'test':num_test,'valid':num_val}
    is_source = [1, 0]
    domains = ['Source', 'Target']
    #load data
    datasets = dset.initialize_datasets(datadir=args.datadir, num_pts=num_pts, is_source=is_source, rank=rank)
    num_pts = [num_pts]
    num_pts.append(deepcopy(num_pts[0]))
    # if (rank==0):
    #     for n, s, d in zip(num_pts, is_source, domains):
    #         print(f"Domain: {d}")
    #         for (split, dataset) in datasets.items():
    #             print('type of dataset', type(dataset))
    #             n[split] = (dataset[0]['is_source']==s).size() + (dataset[1]['is_source']==s).size()[0]
    #             print(f"{split} samples: {n[split]}")

    ### FOR SLURM
    ##########################
    ### initialize cuda
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    if rank == 0: print(f"Group initialized? {dist.is_initialized()}", flush=True)

    torch.cuda.set_device(local_rank)
    ##########################

    #device = torch.device("cuda:{}".format(args.local_rank + first_device))
    dtype = torch.float32

    ### create data loader
    train_sampler, dataloaders = dset.retrieve_dataloaders(
                                    datasets,
                                    2*args.batch_size,
                                    num_workers=num_workers,
                                    rank=rank,
                                    num_replicas=world_size)

    restart_epoch = 4
    ratio = 1
    if args.auto_scale:
        ratio = num_pts['train']/1.2e6
        args.n_hidden = int(round(args.n_hidden*ratio**(1/3)))
        args.n_layers = int(round(args.n_layers*ratio**(1/3)))
        args.epochs = int(round(args.epochs/ratio**(1/2)))
        args.warmup_epochs = int(round(args.warmup_epochs/ratio**(1/2)))
        restart_epoch = int(round(restart_epoch/ratio**(1/2)))
        args.MMD_turnon = int(round(args.MMD_turnon/ratio**(1/2)))
    print('hidden: ', args.n_hidden)
    print('layers: ', args.n_layers)
    print('epochs: ', args.epochs)
    ### create parallel model
    model = LorentzNet(n_scalar = 2, n_hidden = args.n_hidden, n_class = 2,
                       dropout = args.dropout, n_layers = args.n_layers,
                       c_weight = args.c_weight, no_batchnorm=args.no_batchnorm, no_layernorm=args.no_layernorm)
    if not args.no_batchnorm:
        if rank==0:
            print('converting batchnorm to sync_batchnorm') #can turn into comment once I confirm this works
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.to(local_rank)
    #need broadcast_buffers=True for multi GPU training
    ddp_model = DistributedDataParallel(model, broadcast_buffers=True, device_ids=[local_rank])

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
                                                after_scheduler=base_scheduler) ## warmup
    
    elif args.lr_scheduler=='Reduce':
        base_scheduler = ReduceLROnPlateau(optimizer, patience=args.patience, factor=args.factor) 
        lr_scheduler = base_scheduler
    else:
        print(args.lr_scheduler + " is not a valid learning rate scheduler")
    
    mmd_sched = src.MMDScheduler(args.MMDturnon_epoch, args.MMDturnon_width)
    ### loss function
    bce = nn.CrossEntropyLoss()
    mmd = da.MMDLoss(kernel=da.RBF(device=local_rank))

    ### initialize logs
    res = {'epochs': [], 'lr' : [],
           'train_time': [], 'val_time': [],  'train_BCE_loss': [], 
           'train_MMD_loss': [], 'val_BCE_loss': [], 'val_MMD_loss': [], 'val_tot_loss': [],
           'train_acc': [], 'val_acc': [], 'best_val': 1e30, 'best_epoch': 0}

    if not args.test_mode:
        ### training and testing
        train(res)
        if rank==0:
            make_train_plt(res, False)
        test(res)
    else:
        ### only test on best model
        test(res)
