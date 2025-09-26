import torch
import json
import argparse
from pathlib import Path
PROJECT_ROOT = Path(__file__).parents[1].resolve()
SRC_ROOT = PROJECT_ROOT / 'src' / 'MMDLearning'
print('src root: ', str(SRC_ROOT))
import sys
sys.path.append(str(SRC_ROOT))
from utils.io import check_path

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

if __name__=='__main__':
    default_lr = 1e-3
    default_wd = 1e-4
    parser = argparse.ArgumentParser(description='Generate a model configuration for GroupedMLP.')
    parser.add_argument('--name', type=str, default='000', help='Name of the model, e.g., "000"')
    parser.add_argument('--group_names', type=str, nargs='+', default='all', help='Names of the parameter groups of the model, e.g., "000"')
    parser.add_argument('--conv_params', type=json.loads, nargs='+', default=None, help='convolutional parameters')
    parser.add_argument('--fc_params', type=json.loads, nargs='+', default=None, help='fully connected parameters')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing config file if it exists.')
    parser.add_argument('--lr', type=float, nargs='+', default=-1, help='learning rate for each group.')
    parser.add_argument('--weight_decay', type=float, nargs='+', default=-1, help='weight decay for each group.')
    parser.add_argument('--activation', type=str, nargs= '+', default='ReLU', help='Activation function to use in all groups (default ReLU).')
    parser.add_argument("--freeze_bn", type=str2bool, nargs="+", default=[False], help="List of booleans (e.g., --freeze_bn true false true)")
    args = parser.parse_args()
    
    
    if args.conv_params is None:
        args.conv_params = [[]*len(args.group_names)]
    if args.fc_params is None:
        args.fc_params = [[]*len(args.group_names)]
    if type(args.weight_decay) is float:
        args.weight_decay = [args.weight_decay]*len(args.group_names)
    if type(args.lr) is float:
        args.lr = [args.lr]*len(args.group_names)
    if len(args.freeze_bn)==1:
        args.freeze_bn = args.freeze_bn*len(args.group_names)
        
    arch_config = {
        "group_order": args.group_names,
        "group_specs": {
            gn: {
                "conv_params":    conv if conv is not None else [],
                "fc_params":    fc if fc is not None else [],
                "freeze_bn":    freeze if freeze is not None else [],
                "optim_params": {},
            } for gn, conv, fc, freeze in zip(args.group_names, args.conv_params, args.fc_params, args.freeze_bn)
        }
    }
    
    for gn, lr, wd in zip(args.group_names, args.lr, args.weight_decay):
        if lr > 0:
            arch_config["group_specs"][gn]["optim_params"]["lr"] = lr
        if wd > 0:
            arch_config["group_specs"][gn]["optim_params"]["weight_decay"] = wd
    
    arch_config['defaults'] = {
        "lr": default_lr,
        "weight_decay": default_wd,
        "activation": args.activation
    }
    
    root = str(PROJECT_ROOT / 'model_configs' / 'config')
    suffix = '.json'
    name = check_path(args.name, root=root, suffix=suffix, overwrite=args.overwrite)
    config_path = f'{root}_{name}{suffix}'
    with open(config_path, "w") as f:
        json.dump(arch_config, f, indent=4)

    #usage example
    '''
    python scripts/generate_model_config.py --group_names backbone encoder classifier 
    --conv_params '[[7, [32, 32, 32]]]' '[[7, [64, 64, 64]]]' '[]' 
    --fc_params '[]' '[[64, 0], [32, 0]]' '[[32, 0.1], [32, 0.1]]' 
    --lr 1e-2 1e-1 1 --weight_decay 1e-4 1e-4 1e-4
    '''