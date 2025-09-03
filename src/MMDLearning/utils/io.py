import os
import json
import random
import string
import numpy as np
import torch

def makedir(path):
    try:
        os.makedirs(path)
    except OSError:
        pass

def args_init(args, rank, world_size):
    r''' Initialize seed and exp_name.
    '''
    if args.seed is None: # use random seed if not specified
        args.seed = np.random.randint(100)
    if args.exp_name == '': # use random strings if not specified
        args.exp_name = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
    if (rank == 0): # master
        print(args)
        makedir(f"{args.logdir}/{args.exp_name}")
        d = args.__dict__
        d['world_size'] = world_size
        with open(f"{args.logdir}/{args.exp_name}/args.json", 'w') as f:
            json.dump(d, f, indent=4)