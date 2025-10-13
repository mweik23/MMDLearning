from pyexpat import model
import torch
from torch.utils.data import DataLoader, ConcatDataset, Subset
from torch.utils.data.distributed import DistributedSampler
from .collate import collate_fn, collate_sorted
import numpy as np 
import h5py, glob
from .jetdatasets import JetDataset
from copy import deepcopy

def retrieve_dataloaders(datasets, batch_size, num_workers = 4, rank=None, num_replicas=None, model_arch='LorentzNet', collate=None):

    # distributed training
    if 'train' in datasets:
        train_sampler = DistributedSampler(datasets['train'], shuffle=True, num_replicas=num_replicas, rank=rank)
    else:
        train_sampler = None
    # Construct PyTorch dataloaders from datasets
    if model_arch=='LorentzNet':
        collate = lambda data: collate_fn(data, scale=1, add_beams=True, beam_mass=1) #TODO: update to give a sorted batch
    elif model_arch=='ParticleNet' or model_arch=='ParticleNet-Lite':
        if collate == 'sorted':
            collate=collate_sorted
    dataloaders = {split: DataLoader(dataset,
                                     batch_size=batch_size if (split == 'train') else batch_size, # prevent CUDA memory exceeded
                                     sampler=train_sampler if (split == 'train') else DistributedSampler(dataset, shuffle=False, num_replicas=num_replicas, rank=rank),
                                     pin_memory=True,
                                     persistent_workers=True,
                                     drop_last= True,
                                     num_workers=num_workers,
                                     collate_fn=collate)
                        for split, dataset in datasets.items()}

    return dataloaders

def create_dataset_args(project_root, 
                        datadir, 
                        do_MMD=False, 
                        mixed_batch=False,
                        **dataset_args):
    
    if mixed_batch:
        dataset_args.setdefault("is_target", [0, 1])
        dataset_args.setdefault("splits", ['train', 'valid'])
        dataset_args.setdefault("datadir", [str(project_root / d) for d in datadir])
        dataset_args = [dataset_args]
    else:
        dataset_args = [deepcopy(dataset_args) for _ in range(2)]
        for i, da in enumerate(dataset_args):
            da.setdefault("is_target", i) 
            da.setdefault("splits", ['train', 'valid'] if i==0 or do_MMD else ['valid'])
            da.setdefault("datadir", str(project_root / datadir[i]))
            
    return dataset_args

def initialize_datasets(datadir='data', splits=['train', 'valid'], num_data=None, is_target=[0, 1], tv_fracs={'train': 0.6, 'valid': 0.2}, model='ParticleNet'):
    """
    Initialize datasets.
    """
    if type(datadir)==str:
        datadir = [datadir]
    if np.isscalar(is_target) or is_target is None:
        is_target=[is_target]
    if num_data == -1:
        num_pts = {split:-1 for split in splits}
    else:
        num_pts = {split:int(num_data*tv_fracs[split]) for split in splits}
    if type(num_pts)==dict:
        num_pts = 2*[deepcopy(num_pts)]
        
    ### ------ 1: Get the file names ------ ###
    # datadir should be the directory in which the HDF5 files (e.g. out_test.h5, out_train.h5, out_valid.h5) reside.
    # There may be many data files, in some cases the test/train/validate sets may themselves be split across files.
    # We will look for the keywords defined in splits to be be in the filenames, and will thus determine what
    # set each file belongs to.
        
    patterns_all = {'train':'train', 'test':'test', 'valid':'val'} # Patterns to look for in data files, to identify which data category each belongs in
    patterns = {split: patterns_all[split] for split in splits}

    files_set = [glob.glob(d + '/*.h5') for d in datadir]
    datafiles_set = [{split:[] for split in splits} for _ in datadir]
    nfiles_set = []
    for datafiles, files in zip(datafiles_set, files_set):
        for file in files:
            for split,pattern in patterns.items():
                if pattern in file: datafiles[split].append(file)
        nfiles_set.append({split:len(datafiles[split]) for split in splits})
    
    ### ------ 2: Set the number of data points ------ ###
    # There will be a JetDataset for each file, so we divide number of data points by number of files,
    # to get data points per file. (Integer division -> must be careful!)
    #TODO: nfiles > npoints might cause issues down the line, but it's an absurd use case
    num_pts_per_file_set = [{} for _ in datadir]
    #TODO: continue with edits making everything compatible with having source and target
    for num_pts_per_file, num, nfiles in zip(num_pts_per_file_set, num_pts, nfiles_set):
        for split in splits:
            num_pts_per_file[split] = []
            if num[split] == -1:
                for n in range(nfiles[split]): num_pts_per_file[split].append(-1)
            else:
                for n in range(nfiles[split]): num_pts_per_file[split].append(int(np.ceil(num[split]/nfiles[split])))
                num_pts_per_file[split][-1] = int(np.maximum(num[split] - np.sum(np.array(num_pts_per_file[split])[0:-1]),0))
    
    ### ------ 3: Load the data ------ ###
    num_pts_per_file = {}
    datasets = {}
    for split in splits:
        num_pts_per_file[split] = []
        datasets[split] = []
        for t, datafiles, nums in zip(is_target, datafiles_set, num_pts_per_file_set):
            for file, n in zip(datafiles[split], nums[split]):
                num_pts_per_file[split].append(n)
                with h5py.File(file,'r') as f:
                    datasets[split].append({key: torch.from_numpy(val[:]) for key, val in f.items()})
                if t is not None:
                    datasets[split][-1]['is_target'] = t*torch.ones_like(datasets[split][-1]['Nobj'])
                if model=='ParticleNet' or model=='ParticleNet-Lite':
                    datasets[split][-1]['features'] = torch.cat((datasets[split][-1]['features'], datasets[split][-1]['points']), dim=-1)
                    #reshape the tensors
                    datasets[split][-1]['features'] = datasets[split][-1]['features'].transpose(1, 2)
                    datasets[split][-1]['points'] = datasets[split][-1]['points'].transpose(1, 2)
                    datasets[split][-1]['label'] = datasets[split][-1]['label'][:, None, :]
                    #print(split)
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

    torch_datasets = {split: ConcatDataset([JetDataset(data, num_pts=num_pts_per_file[split][idx]) for idx, data in enumerate(datasets[split])]) for split in splits}
    torch_datasets = {split:Subset(dataset, torch.randperm(len(dataset))) for split, dataset in torch_datasets.items()}

    return torch_datasets

if __name__=='__main__':
    #test initialize
    num_train = -1
    num_test = num_train
    num_val = num_train
    source = 'py83'
    target = 'hw72'
    datadir_base = '/scratch/mjw283/LorentzNet/datasets_Lnet/'
    datadir = [datadir_base + source, datadir_base + target]
    datasets = initialize_datasets(datadir, num_pts={'train':num_train,'test':num_test,'valid':num_val}, is_source = [1, 0])