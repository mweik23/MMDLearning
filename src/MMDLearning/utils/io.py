import os
import json
import random
import string
import numpy as np
import torch

from dataclasses import dataclass, field, asdict

def check_path(name, root='', suffix='', overwrite=False):
    parent_dir = root.split('/' + root.split('/')[-1])[0]
    makedir(parent_dir)
    if os.path.exists(root+name+suffix):
        if not overwrite:
            int_name = int(name)
            int_name += 1
            name = str(int_name).zfill(3)
            name = check_path(name, root, suffix, overwrite)
    return name

def makedir(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError:
            pass

@dataclass
class TrainingConfig:
    # store everything here dynamically
    _config: dict = field(default_factory=dict)

    @classmethod
    def from_args_and_dist(cls, args, dist_info, extra: dict = None):
        args_dict = vars(args).copy()
        dist_dict = dist_info.shared_dict()
        merged = {**args_dict, **dist_dict, **(extra or {})}
        return cls(merged)

    def __getattr__(self, name):
        if name in self._config:
            return self._config[name]
        raise AttributeError(name)

    def as_dict(self):
        return dict(self._config)


def config_init(args, dist_info, pt_overwrite_keys=[]):
    r''' Initialize seed and exp_name.
    '''
    if args.seed is None: # use random seed if not specified
        args.seed = np.random.randint(100)
    if args.exp_name == '': # use random strings if not specified
        args.exp_name = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
    if args.pretrained != '':
        if '/' in args.pretrained:
            pt_exp = args.pretrained.split('/')[0]
            pt_model = args.pretrained
        else:
            pt_exp = args.pretrained
            pt_model = args.pretrained + '/best-val-model'
        with open(f"{args.logdir}/{pt_exp}/args.json", 'r') as file:
            pt_args = json.load(file)
        #TODO: combine with args and dist_info to make config
        with open(f"{args.logdir}/{pt_exp}/train-result.json") as json_data:
            train_res_init = json.load(json_data)
        json_data.close()
        if len(pt_overwrite_keys) == 0:
            #use default keys if not specified
            pt_overwrite_keys = ['datadir', 'model', 'no_batchnorm']
    pt_args_overwrite = {k: pt_args[k] for k in pt_overwrite_keys if k in pt_args}
    cfg = TrainingConfig.from_args_and_dist(args, dist_info, pt_args_overwrite)
    if (dist_info.rank == 0): # master
        print(cfg)
        makedir(f"{args.logdir}/{args.exp_name}")
        d = cfg.__dict__
        with open(f"{args.logdir}/{args.exp_name}/args.json", 'w') as f:
            json.dump(d, f, indent=4)
            f.close()
    return cfg
