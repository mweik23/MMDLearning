from src import dataset as dset
from models import psi, LorentzNet
import torch
from torch import nn, optim
import argparse, json, time
import src
import numpy as np
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import domain_adapt as da
from copy import deepcopy
import os

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
#new parameter
parser.add_argument('--MMD_coef', type=float, default=0, metavar='N', help='prefactor for the MMD loss term')
parser.add_argument('--auto_scale', action='store_true', default=False,
                    help = 'scale network and epochs with amount of data')

parser.add_argument('--local_rank', type=int, default=0)

def get_pred(data):
    batch_size, n_nodes, _ = data['Pmu'].size()
    atom_positions = data['Pmu'].view(batch_size * n_nodes, -1).to(device, dtype)
    atom_mask = data['atom_mask'].view(batch_size * n_nodes, -1).to(device)
    edge_mask = data['edge_mask'].reshape(batch_size * n_nodes * n_nodes, -1).to(device)
    nodes = data['nodes'].view(batch_size * n_nodes, -1).to(device,dtype)
    nodes = psi(nodes)
    edges = [a.to(device) for a in data['edges']]
    pred = ddp_model(scalars=nodes, x=atom_positions, edges=edges, node_mask=atom_mask,
                          edge_mask=edge_mask, n_nodes=n_nodes)
    return pred

def get_correct(pred, label):
    predict = pred.max(1).indices
    correct = torch.sum(predict == label).item()
    return correct
def calc_BCE(res, pred, label):
    correct = get_correct(pred, label)
    batch_size = pred.size(dim=0)
    loss_BCE = bce(pred, label)
    res['counter'] += batch_size
    res['correct'] += correct
    res['BCE loss'] += loss_BCE.item() * batch_size
    res['BCE loss_arr'].append(loss_BCE.item())
    res['correct_arr'].append(correct)
    return loss_BCE
def display_status(res, num_batches, partition, epoch, batch_num, batch_size):
    running_BCE_loss = sum(res['BCE loss_arr'][-args.log_interval:])/len(res['BCE loss_arr'][-args.log_interval:])
    running_MMD_loss = sum(res['MMD loss_arr'][-args.log_interval:])/len(res['MMD loss_arr'][-args.log_interval:])
    running_acc = sum(res['correct_arr'][-args.log_interval:])/(len(res['correct_arr'][-args.log_interval:])*batch_size)
    avg_time = res['time']/res['counter'] * batch_size
    tmp_counter = src.sum_reduce(res['counter'], device = device)
    tmp_BCE_loss = src.sum_reduce(res['BCE loss'], device = device) / tmp_counter
    tmp_MMD_loss = src.sum_reduce(res['MMD loss'], device = device) / tmp_counter
    tmp_acc = src.sum_reduce(res['correct'], device = device) / tmp_counter
    if (args.local_rank == 0):
        print('domain: ', res['domain'])
        print(">> %s \t Epoch %d/%d \t Batch %d/%d \t BCE Loss %.4f \t MMD Loss %.4f \t Running Acc %.3f \t Total Acc %.3f \t Avg Batch Time %.4f" %
                (partition, epoch + 1, args.epochs, batch_num, num_batches, running_BCE_loss, running_MMD_loss,  running_acc, tmp_acc, avg_time))

def gather_preds(res):
    pred = [torch.zeros_like(res['score']) for _ in range(dist.get_world_size())]
    dist.all_gather(pred, res['score'] )
    return torch.cat(pred).cpu()

def get_metric(pred, res):
    fpr, tpr, thres, eB, eS  = src.buildROC(pred[...,0], pred[...,2])
    auc = src.roc_auc_score(pred[...,0], pred[...,2])
    metric = {'domain': res['domain'],'test_BCE_loss': res['BCE loss'], 'test_MMD_loss': res['MMD loss'], 'test_acc': res['acc'],
                  'test_auc': auc, 'test_1/eB_0.3':1./eB[0],'test_1/eB_0.5':1./eB[1]}
    return metric

def run(epoch, loaders, partition):
    if partition == 'train':
        for ts in train_samplers:
            ts.set_epoch(epoch)
        ddp_model.train()
    else:
        ddp_model.eval()

    res = [{'domain': 'Source', 'time':0, 'correct':0, 'BCE loss': 0, 'MMD loss': 0, 'counter': 0, 'acc': 0,
           'BCE loss_arr':[], 'MMD loss_arr':[], 'correct_arr':[],'label':[],'score':[]}]
    if partition=='test':
        res = [deepcopy(res[0]), deepcopy(res[0])]
        res[1]['domain']= 'Target'
    tik = time.time()
    loader_lengths = [len(loader) for loader in loaders]
    #print(partition, loader_lengths)
    #need to make prediction for source and target data
    for i, data in enumerate(zip(*loaders)):

        #calculate prediction for target and source but only keep labels for the source data
        batch_size, _, _ = data[0]['Pmu'].size()
        preds = [get_pred(d) for d in data]
        #labels contains source and target labels if partition=='test'. Otherwise it only contains source labels
        if partition =='test':
            labels = [d['is_signal'].to(device, dtype).long() for d in data]
            # save labels and probilities for ROC / AUC
            scores = [torch.nn.functional.softmax(p, dim = -1) for p in preds]
            #print('res set: ', res)
            for r, l, s in zip(res, labels, scores):
                #print('res:', r)
                #print('type of res: ', type(r))
                r['label'].append(l)
                r['score'].append(s)
        else:
            labels = [data[0]['is_signal'].to(device, dtype).long()]
        
        #note the method calc_BCE() also updates the res variable
        #loss_BCE = [calc_BCE(res[i], preds[i], l) for i, l in enumerate(labels)]
        loss_BCE = [calc_BCE(res[i], preds[i] , l) for i, l in enumerate(labels)]
        loss_MMD = mmd(preds[0], preds[1])
        #print(f'loss MMD: {loss_MMD}')

        losses = [l + loss_MMD for l in loss_BCE]
        if partition == 'train':
            optimizer.zero_grad()
            losses[0].backward()
            optimizer.step()
        
        elapsed = time.time() - tik
        for r in res:
            r['time'] = elapsed
            r['MMD loss'] += loss_MMD.item() * batch_size
            r['MMD loss_arr'].append(loss_MMD.item())
        #print('MMD loss: ', tuple(r['MMD loss'] for r in res))
        #print('MMD loss array: ', tuple(r['MMD loss_arr'] for r in res))
        if i != 0 and i % args.log_interval == 0:
            #print('data size:', data[0]['Pmu'].size())
            display_status(res[0], loader_lengths[0], partition, epoch, i, batch_size)
            if partition=='test':
                display_status(res[1], loader_lengths[1], partition, epoch, i)
    torch.cuda.empty_cache()
    # ---------- reduce -----------
    if partition == 'test':
        for r in res:
            r['label'] = torch.cat(r['label']).unsqueeze(-1)
            r['score'] = torch.cat(r['score'])
            r['score'] = torch.cat((r['label'], r['score']),dim=-1)
    for r in res:
        r['counter'] = src.sum_reduce(r['counter'], device = device).item()
        r['BCE loss'] = src.sum_reduce(r['BCE loss'], device = device).item() / r['counter']
        r['MMD loss'] = src.sum_reduce(r['MMD loss'], device = device).item() / r['counter']
        r['acc'] = src.sum_reduce(r['correct'], device = device).item() / r['counter']
    return res

def train(res):
    ### training and validation
    for epoch in range(0, args.epochs):
        dataloaders_train = [d['train'] for d in dataloaders_set]
        train_res = run(epoch, dataloaders_train, partition='train')[0]
        del dataloaders_train
        print("Time: train: %.2f \t Train BCE loss %.4f \t Train MMD loss %.4f \t Train acc: %.4f" % (train_res['time'], train_res['BCE loss'], train_res['MMD loss'], train_res['acc']))
        if epoch % args.val_interval == 0:
            if (args.local_rank == 0):
                torch.save(ddp_model.state_dict(), f"{args.logdir}/{args.exp_name}/checkpoint-epoch-{epoch}.pt")
            dist.barrier() # wait master to save model
            with torch.no_grad():
                dataloaders_val = [d['valid'] for d in dataloaders_set]
                val_res = run(epoch, dataloaders_val, partition='valid')[0]
                del dataloaders_val
            if (args.local_rank == 0): # only master process save
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

                ## save best model (minimum BCE + MMD)
                val_loss = val_res['BCE loss'] + val_res['MMD loss']
                if val_loss < res['best_val']:
                    print("New best validation model, saving...")
                    torch.save(ddp_model.state_dict(), f"{args.logdir}/{args.exp_name}/best-val-model.pt")
                    res['best_val'] = val_loss
                    res['best_epoch'] = epoch

                print("Epoch %d/%d finished." % (epoch, args.epochs))
                print("Train time: %.2f \t Val time %.2f" % (train_res['time'], val_res['time']))
                print("Train BCE loss %.4f \t Train MMD loss %.4f \t Train acc: %.4f" % (train_res['BCE loss'], train_res['MMD loss'], train_res['acc']))
                print("Val BCE loss: %.4f \t Val MMD loss: %.4f \t  Val acc: %.4f" % (val_res['BCE loss'], val_res['MMD loss'], val_res['acc']))
                print("Best val acc: %.4f at epoch %d." % (res['best_val'],  res['best_epoch']))

                json_object = json.dumps(res, indent=4)
                with open(f"{args.logdir}/{args.exp_name}/train-result.json", "w") as outfile:
                    outfile.write(json_object)

        ## adjust learning rate
        if (epoch < 31*int(round(1/ratio**(1/2)))):
            lr_scheduler.step(metrics=val_res['BCE loss'] + val_res['MMD loss'])
        else:
            for g in optimizer.param_groups:
                g['lr'] = g['lr']*0.5**(ratio**(1/2))

        dist.barrier() # syncronize

def test(res):
    ### test on best model
    best_model = torch.load(f"{args.logdir}/{args.exp_name}/best-val-model.pt", map_location=device)
    ddp_model.load_state_dict(best_model)
    dataloaders_test = [d['test'] for d in dataloaders_set]
    with torch.no_grad():
        test_res = run(0, dataloaders_test, partition='test')
        del dataloaders_test
    preds = [gather_preds(r) for r in test_res]
    if (args.local_rank == 0):
        np.save(f"{args.logdir}/{args.exp_name}/score_source.npy", preds[0])
        np.save(f"{args.logdir}/{args.exp_name}/score_target.npy", preds[1])
        metrics = [get_metric(pred, r) for pred, r in zip(preds, test_res)]
        #The form of each metric is: {'domain': r['domain'],'test_loss': r['BCE loss'], 'test_acc': r['acc'],
        #          'test_auc': auc, 'test_1/eB_0.3':1./eB[0],'test_1/eB_0.5':1./eB[1]}
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
    ### initialize args
    args = parser.parse_args()
    src.args_init(args)

    ### set random seed
    torch.manual_seed(args.seed + args.local_rank)
    np.random.seed(args.seed + args.local_rank)

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
    num_pts.append(deepcopy(num_pts[0]))

    #load data
    datasets_domains = [dset.initialize_datasets(dd, num_pts=n) for dd,n in zip(args.datadir, num_pts)]
    ### initialize cuda
    first_device = 2
    dist.init_process_group(backend='nccl')
    device = torch.device("cuda:{}".format(args.local_rank + first_device))
    dtype = torch.float32
    domains = ['Source', 'Target']
    ### load data
    #TODO modify dataloaders to make batches of samples with source and target in equal numbers. The following might be usefull
    #https://discuss.pytorch.org/t/how-to-use-my-own-sampler-when-i-already-use-distributedsampler/62143/8
    samplers_dataloaders = [dset.retrieve_dataloaders(
                                                datasets,
                                                args.batch_size, 
                                                args.num_workers) for datasets in datasets_domains]
    train_samplers = [sd[0] for sd in samplers_dataloaders]
    dataloaders_set = [sd[1] for sd in samplers_dataloaders]
    if (args.local_rank==0):
        for n, dataloaders in zip(num_pts, dataloaders_set):
            for (split, dataloader) in dataloaders.items():
                    s = len(dataloader.dataset)
                    n[split] = s
                    print(f" {split} samples: {s}")
    restart_epoch = 4
    ratio = 1
    if args.auto_scale:
        ratio = args.num_train/1.2e6
        args.n_hidden = int(round(args.n_hidden*ratio**(1/3)))
        args.n_layers = int(round(args.n_layers*ratio**(1/3)))
        args.epochs = int(round(args.epochs/ratio**(1/2)))
        args.warmup_epochs = int(round(args.warmup_epochs/ratio**(1/2)))
        restart_epoch = int(round(restart_epoch/ratio**(1/2)))
        #args.MMD_turnon = int(round(args.MMD_turnon/ratio**(1/2)))
    print('hidden: ', args.n_hidden)
    print('layers: ', args.n_layers)
    print('epochs: ', args.epochs)
    ### create parallel model
    model = LorentzNet(n_scalar = 2, n_hidden = args.n_hidden, n_class = 2,
                       dropout = args.dropout, n_layers = args.n_layers,
                       c_weight = args.c_weight)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.to(device)
    #need broadcast_buffers=True for multi GPU training
    ddp_model = DistributedDataParallel(model, broadcast_buffers=False, device_ids=[args.local_rank+first_device])
    ### print model and data information
    if (args.local_rank == 0):
        pytorch_total_params = sum(p.numel() for p in ddp_model.parameters())
        print("Network Size:", pytorch_total_params)


    ### optimizer
    optimizer = optim.AdamW(ddp_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    ### lr scheduler
    base_scheduler = CosineAnnealingWarmRestarts(optimizer, restart_epoch, 2, verbose = False)
    lr_scheduler = src.GradualWarmupScheduler(optimizer, multiplier=1,
                                                warmup_epoch=args.warmup_epochs,
                                                after_scheduler=base_scheduler) ## warmup
    ### loss function
    bce = nn.CrossEntropyLoss()
    mmd = da.MMDLoss(kernel=da.RBF(device=device), coef=args.MMD_coef)

    ### initialize logs
    res = {'epochs': [], 'lr' : [],
           'train_time': [], 'val_time': [],  'train_BCE_loss': [], 
           'train_MMD_loss': [], 'val_BCE_loss': [], 'val_MMD_loss': [],
           'train_acc': [], 'val_acc': [], 'best_val': 1e30, 'best_epoch': 0}

    if not args.test_mode:
        ### training and testing
        train(res)
        test(res)
    else:
        ### only test on best model
        test(res)
