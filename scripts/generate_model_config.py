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

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Generate a model configuration for GroupedMLP.')
    parser.add_argument('--name', type=str, default='000', help='Name of the model, e.g., "000"')
    parser.add_argument('--group_names', type=str, nargs='+', default='all', help='Names of the parameter groups of the model, e.g., "000"')
    parser.add_argument('--conv_params', type=json.loads, nargs='+', default=None, help='convolutional parameters')
    parser.add_argument('--fc_params', type=json.loads, nargs='+', default=None, help='fully connected parameters')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing config file if it exists.')
    parser.add_argument('--dropout', type=float, nargs='+', default=0.0, help='Dropout rate for each group.')
    parser.add_argument('--activation', type=str, nargs= '+', default='ReLU', help='Activation function to use in all groups (default ReLU).')
    args = parser.parse_args()
    
    if args.conv_params is None:
        conv_params = [[]*len(args.group_names)]
    else:
        conv_params = args.conv_params
    if args.fc_params is None:
        fc_params = [[]*len(args.group_names)]
    
    arch_config = {
        "group_order": args.group_names,
        "group_specs": {
            gn: {
                "conv_params":    conv if conv is not None else [],
                "fc_params":    fc if fc is not None else [],
            } for gn, conv, fc in zip(args.group_names, args.conv_params, args.fc_params)
        }
    }
    
    root = str(PROJECT_ROOT / 'model_configs' / 'config_')
    suffix = '.json'
    name = check_path(args.name, root=root, suffix=suffix, overwrite=args.overwrite)
    config_path = f'{root}_{name}{suffix}'
    with open(config_path, "w") as f:
        json.dump(arch_config, f, indent=4)

    #usage example
    '''python
    python generate_model_config.py 
    '''