from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).parents[1].resolve()
sys.path.append(str(PROJECT_ROOT))

SRC_PATH = Path(__file__).parents[1].resolve() / 'src' / 'MMDLearning'
sys.path.append(str(SRC_PATH))
from utils.distributed import setup_dist
from utils.io import config_init
from utils.cli import build_parser

def test_config_init(tmp_path):
    parser = build_parser()
    args = parser.parse_args(['--exp_name', 'test_exp', '--logdir', str(tmp_path)])
    dist_info = setup_dist(arg_num_workers=args.num_workers)
    assert dist_info.device_type in ['mps', 'cpu', 'cuda'], f"Unexpected device name: {dist_info.device_type}"
    #set up training configuration
    cfg = config_init(args, dist_info, PROJECT_ROOT)
    print ('saved to: ', tmp_path)
    print(cfg.as_dict())