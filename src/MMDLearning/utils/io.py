import os
import json
import random
import string
import numpy as np
import torch
import shutil

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

def make_clean_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)   # remove the directory and all its contents
    os.makedirs(path) 

def load_ckp(checkpoint_fpath, model, optimizer=None, device=torch.device('cpu'), use_target_model=False):
    checkpoint = torch.load(checkpoint_fpath, map_location=device, weights_only=True)
    print('initial load of model state dict...')
    incompat = model.load_state_dict(checkpoint['state_dict'], strict=False)
    assert all('target_model' in k.split('.') for k in incompat.missing_keys), 'some non-target_model keys from the current model are not present in the loaded state: ' + str(incompat.missing_keys)
    print("all keys in the current model present in loaded state except for target_model keys")     # expected: keys for target_* (new modules)
    assert len(incompat.unexpected_keys)==0, 'some unexpected keys in loaded state: ' + str(incompat.unexpected_keys)
    print("no unexpected keys in loaded state")  # expected: no unexpected
    if len(incompat.missing_keys) > 0 and use_target_model:
        print('target_model detected but some parameters are not matched, copying state dict for target_model from model...')
        incompat = model.module.target_model.load_state_dict(
            model.module.model.state_dict(), strict=False
        )
        assert len(incompat.missing_keys)==0, 'some keys are missing in loaded state: ' + str(incompat.missing_keys)
        print("no missing keys in loaded state for target_model")  # expected: no missing
        if all('classifier' in k.split('.') for k in incompat.unexpected_keys):
            print("the only unexpected keys are for the classifier as expected.") #classifier weights will be unexpected
        else: 
            print('WARNING: some keys from main model other than classifier head are not matched to the target model: ' + str(incompat.unexpected_keys))
        
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


def config_init(args, dist_info, project_root, pt_overwrite_keys=['datadir', 'model']):
    r''' Initialize seed and exp_name.
    '''
    if args.seed is None: # use random seed if not specified
        args.seed = np.random.randint(100)
    if args.exp_name == '': # use random strings if not specified
        args.exp_name = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
    pt_args_overwrite = {}
    args.logdir = str(project_root / args.logdir)
    args.target_model_groups = tuple(args.target_model_groups or [])
    for k in args.frozen_groups.keys():
        args.frozen_groups[k] = tuple(args.frozen_groups[k])
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
        make_clean_dir(f"{args.logdir}/{args.exp_name}")
        d = cfg.as_dict()
        with open(f"{args.logdir}/{args.exp_name}/config.json", 'w') as f:
            json.dump(d, f, indent=4)
            f.close()
    return cfg
