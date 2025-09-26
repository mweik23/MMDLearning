import torch
from torch.utils.data import DataLoader, ConcatDataset, Subset
from torch.utils.data.distributed import DistributedSampler
from . import collate_fn
import numpy as np
import h5py, glob
from . import JetDataset
import os
import matplotlib.pyplot as plt

def retrieve_dataloaders(datasets, batch_size, num_workers = 4, rank=None, num_replicas=None, collate_config='LorentzNet'):
    # Initialize dataloader
     
    # if num_train==-1:
    #     num_test=-1
    #     num_val=-1
    # else:
    #     num_test= .2*num_train
    #     num_val = num_test
    # datasets = initialize_datasets(datadir, num_pts={'train':num_train,'test':num_test,'valid':num_val})

    # distributed training
    if 'train' in datasets:
        train_sampler = DistributedSampler(datasets['train'], shuffle=True, num_replicas=num_replicas, rank=rank)
    else:
        train_sampler = None
    # Construct PyTorch dataloaders from datasets
    if collate_config=='LorentzNet':
        collate = lambda data: collate_fn(data, scale=1, add_beams=True, beam_mass=1)
    elif collate_config=='ParticleNet' or collate_config=='ParticleNet-Lite':
        collate=None
    #print('collate is None: ', collate is None)
    dataloaders = {split: DataLoader(dataset,
                                     batch_size=batch_size if (split == 'train') else batch_size, # prevent CUDA memory exceeded
                                     sampler=train_sampler if (split == 'train') else DistributedSampler(dataset, shuffle=False, num_replicas=num_replicas, rank=rank),
                                     pin_memory=True,
                                     persistent_workers=True,
                                     drop_last= True if (split == 'train') else False,
                                     num_workers=num_workers,
                                     collate_fn=collate)
                        for split, dataset in datasets.items()}

    return train_sampler, dataloaders

def regularize(data, reg_params=(0, 1.0), masks=None):
    if masks is None:
        res = reg_params[1]*(data-reg_params[0])
    else:
        res = reg_params[1]*(data-reg_params[0])*masks
    return res
    
#order of reg_params: logpt, logE, logpt_ptjet, logE_Ejet, DeltaR

def initialize_datasets(datadir='./data', num_pts=None, rank=0, reg_params=None, logdir=None, model='ParticleNet'):
    """
    Initialize datasets.
    """
    if rank==0:
        print('initialize datasets for datadir: ', datadir)
    ### ------ 1: Get the file names ------ ###
    # datadir should be the directory in which the HDF5 files (e.g. out_test.h5, out_train.h5, out_valid.h5) reside.
    # There may be many data files, in some cases the test/train/validate sets may themselves be split across files.
    # We will look for the keywords defined in splits to be be in the filenames, and will thus determine what
    # set each file belongs to.
    if num_pts is None:
        num_pts={'train':-1,'test':-1,'valid':-1}
        
    splits = list(num_pts.keys()) # Our data categories -- training set, testing set and validation set
    patterns_all = {'train':'train', 'test':'test', 'valid':'val'} # Patterns to look for in data files, to identify which data category each belongs in
    patterns = {split: patterns_all[split] for split in splits}

    files = glob.glob(datadir + '/*.h5')
    #print('files:', files)
    datafiles = {split:[] for split in splits}
    for file in files:
        for split,pattern in patterns.items():
            if pattern in file: datafiles[split].append(file)
    nfiles = {split:len(datafiles[split]) for split in splits}
    
    ### ------ 2: Set the number of data points ------ ###
    # There will be a JetDataset for each file, so we divide number of data points by number of files,
    # to get data points per file. (Integer division -> must be careful!)
    #TODO: nfiles > npoints might cause issues down the line, but it's an absurd use case
    num_pts_per_file = {}
    for split in splits:
        num_pts_per_file[split] = []
        
        if num_pts[split] == -1:
            for n in range(nfiles[split]): num_pts_per_file[split].append(-1)
        else:
            for n in range(nfiles[split]): num_pts_per_file[split].append(int(np.ceil(num_pts[split]/nfiles[split])))
            num_pts_per_file[split][-1] = int(np.maximum(num_pts[split] - np.sum(np.array(num_pts_per_file[split])[0:-1]),0))
    
    ### ------ 3: Load the data ------ ###
    datasets = {}
    for split in splits:
        datasets[split] = []
        for file in datafiles[split]:
            with h5py.File(file,'r') as f:
                datasets[split].append({key: torch.from_numpy(val[:]) for key, val in f.items()})
            #regularize
            if reg_params is not None:
                for i, p in enumerate(reg_params):    
                    datasets[split][-1]['features'][:, :, i] = regularize(datasets[split][-1]['features'][:, :, i], reg_params=p, masks=datasets[split][-1]['label'])
                #include the coordinates in the feature vector
            #make plots
            # if split=='train':
            #     if logdir is not None:
            #         plotdir = logdir + '/plots_' + datadir.split('/')[-1] + '/'
            #         if os.path.isdir(plotdir):
            #             os.system('rm -r ' + plotdir)
            #         os.system('mkdir ' + plotdir)
            #         num_plots=5
            #         idxs = np.random.randint(len(datasets[split][-1]['Nobj']), size=num_plots)
            #         print('idxs: ', idxs)
            #         for idx in idxs:
            #             print('Nobj: ', datasets[split][-1]['Nobj'][idx])
            #             fig = plt.figure()
            #             ax = fig.add_subplot()
            #             ax.scatter(datasets[split][-1]['points'][idx, :, 0], datasets[split][-1]['points'][idx, :, 1])
            #             plt.savefig(plotdir + 'eta_phi_' + str(idx) + '.pdf')
            if model=='ParticleNet' or model=='ParticleNet-Lite':
                datasets[split][-1]['features'] = torch.cat((datasets[split][-1]['features'], datasets[split][-1]['points']), dim=-1)
                #reshape the tensors
                datasets[split][-1]['features'] = datasets[split][-1]['features'].transpose(1, 2)
                datasets[split][-1]['points'] = datasets[split][-1]['points'].transpose(1, 2)
                datasets[split][-1]['label'] = datasets[split][-1]['label'][:, None, :]
                #print(split)
        #print('number of datasets in ' + split + ': ', len(datasets[split]))
 
    ### ------ 4: Error checking ------ ###
    # Basic error checking: Check the training/test/validation splits have the same set of keys.
    keys = []
    for split in splits:
        for dataset in datasets[split]:
            #print(split, len(dataset['Nobj']))
            keys.append(dataset.keys())
    assert all([key == keys[0] for key in keys]), 'Datasets must have same set of keys!'
    ### ------ 5: Initialize datasets ------ ###
    # Now initialize datasets based upon loaded data
    torch_datasets = {split:ConcatDataset([JetDataset(data, num_pts=num_pts_per_file[split][idx], shuffle=True) for idx, data in enumerate(datasets[split])]) for split in splits}
    torch_datasets = {split:Subset(dataset, torch.randperm(len(dataset))) for split, dataset in torch_datasets.items()}
    #for split in splits:
    #    print(split + ' dataset is None: ', torch_datasets[split] is None)
    return torch_datasets
