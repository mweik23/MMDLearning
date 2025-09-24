import os
import json
import random
import string
import numpy as np
import torch

from dataclasses import dataclass, field

def check_path(name, root='', suffix='', overwrite=False):
    parent_dir = root.split('/' + root.split('/')[-1])[0]
    makedir(parent_dir)
    if os.path.exists(f'{root}_{name}{suffix}'):
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

def load_ckp(checkpoint_fpath, model, optimizer=None, device=torch.device('cpu')):
    checkpoint = torch.load(checkpoint_fpath, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint['epoch']

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


def config_init(args, dist_info, pt_overwrite_keys=['datadir', 'model']):
    r''' Initialize seed and exp_name.
    '''
    if args.seed is None: # use random seed if not specified
        args.seed = np.random.randint(100)
    if args.exp_name == '': # use random strings if not specified
        args.exp_name = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
    pt_args_overwrite = {}
    if args.pretrained != '':
        if '/' in args.pretrained:
            pt_exp = args.pretrained.split('/')[0]
        else:
            pt_exp = args.pretrained
        with open(f"{args.logdir}/{pt_exp}/config.json", 'r') as file:
            pt_args = json.load(file)
        pt_args_overwrite = {k: pt_args[k] for k in pt_overwrite_keys if k in pt_args}
    pt_args_overwrite['do_MMD'] = args.MMD_frac > 0
    cfg = TrainingConfig.from_args_and_dist(args, dist_info, pt_args_overwrite)
    if (dist_info.rank == 0): # master
        print(cfg)
        makedir(f"{args.logdir}/{args.exp_name}")
        d = cfg.as_dict()
        with open(f"{args.logdir}/{args.exp_name}/config.json", 'w') as f:
            json.dump(d, f, indent=4)
            f.close()
    return cfg
