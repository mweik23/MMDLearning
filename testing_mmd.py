import torch
import domain_adapt as da
import matplotlib.pyplot as plt
import os
import argparse
import numpy as np
from top import dataset as dset
import torch.distributed as dist
import json
from torch.nn.parallel import DistributedDataParallel
import time
import torch.distributed as dist

def load_ckp(checkpoint_fpath, model, optimizer=None, device=None):
    if device is None:
        device = local_rank
    checkpoint = torch.load(checkpoint_fpath, map_location=torch.cuda.set_device(device))
    model.load_state_dict(checkpoint['state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint['epoch']
def get_pred(data, model_name='LorentzNet'):
    if model_name=='LorentzNet':
        batch_size, n_nodes, _ = data['Pmu'].size()
        atom_positions = data['Pmu'].view(batch_size * n_nodes, -1).to(local_rank, dtype)
        atom_mask = data['atom_mask'].view(batch_size * n_nodes, -1).to(local_rank)
        edge_mask = data['edge_mask'].reshape(batch_size * n_nodes * n_nodes, -1).to(local_rank)
        nodes = data['nodes'].view(batch_size * n_nodes, -1).to(local_rank,dtype)
        nodes = psi(nodes)
        edges = [a.to(local_rank) for a in data['edges']]
        pred, h = ddp_model(scalars=nodes, x=atom_positions, edges=edges, node_mask=atom_mask,
                            edge_mask=edge_mask, n_nodes=n_nodes)
    elif model_name=='ParticleNet' or model_name=='ParticleNet-Lite':
        batch_size, _, _ = data['label'].shape
        h, pred = ddp_model(data['points'].to(local_rank, dtype), data['features'].to(local_rank, dtype), mask=data['label'].to(local_rank, dtype))
    return pred, h
def gather_preds(domain):
    res[domain] = torch.cat(res[domain])
    pred = [torch.zeros_like(res[domain]) for _ in range(dist.get_world_size())]
    dist.all_gather(pred, res[domain])
    return torch.cat(pred).cpu()
def get_layer(layer=1, model_name='LorentzNet'):
    with torch.no_grad():
        ddp_model.eval()
        #tik = time.time()
        loader_lengths = [len(loader) for loader in dataloaders]
        for i, data in enumerate(zip(*dataloaders)):

            outputs = [get_pred(d, model_name=model_name)[-1-layer] for d in data]
            for d, o in zip(domains, outputs):
                res[d].append(o)
            if i%args.log_interval == 0:
                print(f"batch {i}/{loader_lengths[0]}")
        preds = [gather_preds(d) for d in domains]
        if layer==1:
            output = 'logits'
        else:
            output = 'intermed'
        for pred, domain in zip(preds, domains):
            torch.save(pred, f"{args.logdir}/{args.pretrained}/{domain}_{output}.pt", _use_new_zipfile_serialization=False)

if __name__=='__main__':

    parser = argparse.ArgumentParser(description='testing MMD')

    parser.add_argument('--pascal', action='store_true', default=False,
                        help = 'run on pascal')
    parser.add_argument('--devices', type=int,  nargs='+', default=[0], metavar='N',
                        help='device numbers')
    parser.add_argument('--seed', type=int, default=99, metavar='N',
                    help='random seed')
    parser.add_argument('--num_workers', type=int, default=4, metavar='N',
                    help='number of workers for the dataloader')
    parser.add_argument('--num_val', type=int, default=-1, metavar='N',
                        help='number of validation events')
    #parser.add_argument('--datadir', nargs='+', default='./data/top', metavar='N',
    #                help='data directories')
    parser.add_argument('--pretrained', type=str, default='', metavar='N',
                    help='experiment name to load')
    parser.add_argument('--logdir', type=str, default='./logs/top', metavar='N',
                        help='directory to load experiments from')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--log_interval', type=int, metavar='N',
                        help='number of batches between each printout')
    parser.add_argument('--datadir', nargs='+', default='pretrained', metavar='N',
                    help='data directories')
    parser.add_argument('--intermed_MMD', action='store_true', default=False,
                        help = 'run on pascal')
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

    with open(f"{args.logdir}/{args.pretrained}/args.json") as json_data:
        pt_args = json.load(json_data)
        json_data.close()
    
     #added for compatibilty with ParticleNet
    #######################################
    if pt_args['model']=='LorentzNet':
        from models import psi, LorentzNet
    elif pt_args['model']=='ParticleNet' or pt_args['model']=='ParticleNet-Lite':
        from model_PNet import ParticleNet
    ### set random seed
    torch.manual_seed(args.seed) #+ rank)
    np.random.seed(args.seed) # + rank)
    torch.cuda.set_device(local_rank)
    dtype = torch.float32

    num_pts = [{'valid': args.num_val}, {'valid': args.num_val}]
    if args.datadir=='pretrained':
        datadir = pt_args['datadir']
    else:
        datadir = args.datadir
    #load data
    datasets_domains = [dset.initialize_datasets(dd, num_pts=n, rank=rank, model=pt_args['model']) for dd,n in zip(datadir, num_pts)]
    ### initialize cuda
   
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    if rank == 0: print(f"Group initialized? {dist.is_initialized()}", flush=True)

    samplers_dataloaders = [dset.retrieve_dataloaders(
                                    datasets,
                                    pt_args['batch_size'],
                                    num_workers=num_workers,
                                    rank=rank,
                                    num_replicas=world_size,
                                    collate_config=pt_args['model']) for datasets in datasets_domains]
    
    dataloaders = [sd[1]['valid'] for sd in samplers_dataloaders]
    
    if rank==0:
        print('hidden: ', pt_args['n_hidden'])
        print('layers: ', pt_args['n_layers'])

    ### create parallel model
    if pt_args['model']=='LorentzNet':
        model = LorentzNet(n_scalar = 2, n_hidden = pt_args['n_hidden'], n_class = 2,
                       dropout = pt_args['dropout'], n_layers = pt_args['n_layers'],
                       c_weight = pt_args['c_weight'], no_batchnorm=pt_args['no_batchnorm'], 
                       no_layernorm=pt_args['no_layernorm'])
    else: 
        if pt_args['model']=='ParticleNet':
            kwargs={'fc_params': [(256, pt_args['dropout'])], 'conv_params': [(16, (64, 64, 64)),(16, (128, 128, 128)),(16, (256, 256, 256))]}
        elif pt_args['model']=='ParticleNet-Lite': 
            kwargs={'fc_params': [(128, pt_args['dropout'])], 'conv_params': [(7, (32, 32, 32)),(17, (64, 64, 64))]}

        model = ParticleNet(7,
        num_classes=2,
        conv_params=kwargs.get('conv_params', None),
        fc_params=kwargs.get('fc_params', None),
        use_fusion=kwargs.get('use_fusion', False),
        use_fts_bn=kwargs.get('use_fts_bn', True),
        use_counts=kwargs.get('use_counts', True),
        for_inference=kwargs.get('for_inference', False),
        intermed_access=args.intermed_MMD)
    
    if not pt_args['no_batchnorm']:
        if rank==0:
            print('converting batchnorm to sync_batchnorm') #can turn into comment once I confirm this works
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.to(local_rank)
    ddp_model = DistributedDataParallel(model, device_ids=[local_rank])
    epoch = load_ckp(f"{args.logdir}/{args.pretrained}/best-val-model.pt", ddp_model) - 1
    ### initialize logs
    res = {'Source': [], 'Target': []}
    domains = list(res.keys())
    if args.intermed_MMD:
        layer = 0
    else:
        layer= 1
    get_layer(layer=layer, model_name=pt_args['model'])