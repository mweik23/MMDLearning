import torch
import numpy as np

import h5py, glob

num_pts=None
datadir = './data/top'

### ------ 1: Get the file names ------ ###                                                                                                                                                  
# datadir should be the directory in which the HDF5 files (e.g. out_test.h5, out_train.h5, out_valid.h5) reside.                                                                             
# There may be many data files, in some cases the test/train/validate sets may themselves be split across files.                                                                             
# We will look for the keywords defined in splits to be be in the filenames, and will thus determine what                                                                                    
# set each file belongs to.                                                                                                                                                                  
splits = ['train', 'test', 'valid'] # Our data categories -- training set, testing set and validation set                                                                                    
patterns = {'train':'train', 'test':'test', 'valid':'val'} # Patterns to look for in data files, to identify which data category each belongs in                                             

files = glob.glob(datadir + '/*.h5')
print('files: ', files)
datafiles = {split:[] for split in splits}
for file in files:
    for split,pattern in patterns.items():
        if pattern in file: datafiles[split].append(file)
nfiles = {split:len(datafiles[split]) for split in splits}

### ------ 2: Set the number of data points ------ ###                                                                                                                                       
# There will be a JetDataset for each file, so we divide number of data points by number of files,                                                                                           
# to get data points per file. (Integer division -> must be careful!)                                                                                                                        
#TODO: nfiles > npoints might cause issues down the line, but it's an absurd use case                                                                                                        
if num_pts is None:
    num_pts={'train':-1,'test':-1,'valid':-1}

num_pts_per_file = {}
for split in splits:
    num_pts_per_file[split] = []

    if num_pts[split] == -1:
        for n in range(nfiles[split]): num_pts_per_file[split].append(-1)
    else:
        for n in range(nfiles[split]): num_pts_per_file[split].append(int(np.ceil(num_pts[split]/nfiles[split])))
        num_pts_per_file[split][-1] = int(np.maximum(num_pts[split] - np.sum(np.array(num_pts_per_file[split])[0:-1]),0))

print('num_pts: ', num_pts)
print('num_pts_per_file: ', num_pts_per_file)
print('datafiles: ', datafiles)

datasets = {}
for split in splits:
    datasets[split] = []
    for file in datafiles[split]:
        with h5py.File(file,'r') as f:
            datasets[split].append({key: torch.from_numpy(val[:]) for key, val in f.items()})

print('datasets: ', datasets)
