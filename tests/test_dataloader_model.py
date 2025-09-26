import torch
import numpy as np
import json
from pathlib import Path
import sys
import pytest
SRC_PATH = Path(__file__).parents[1].resolve() / 'src' / 'MMDLearning'
sys.path.append(str(SRC_PATH))
from utils.distributed import setup_dist, DistInfo
from utils.cli import build_parser
from utils.io import config_init
from data.dataset_v2 import initialize_datasets, retrieve_dataloaders, create_dataset_args
from models.predictors import make_predictor
PROJECT_ROOT = Path(__file__).parents[1].resolve()

@pytest.mark.parametrize("mmd_frac", [0, 1.0])
def test(tmp_path, mmd_frac):
    print('temporary path: ', str(tmp_path))
    num_points = 100
    num_dims = 7
    batch_size = 32
    num_classes = 2
    target_thresh = 4
    #get arguments
    cli = f'--exp_name {str(tmp_path)} --model_name ParticleNet-Lite --model_config config_002.json '\
        + f'--batch_size {batch_size} --MMD_frac {mmd_frac} --seed 42 --num_data 500 --datadir data/datasets_Pnet100_Njets1000/py83 '\
        + f'data/datasets_Pnet100_Njets1000/hw72'
    cli_split = cli.split(' ')
    print(cli_split)

    parser = build_parser()
    args = parser.parse_args(cli_split)
    
    #set up distributed training
    dist_info: DistInfo = setup_dist(arg_num_workers=args.num_workers)
    if dist_info.is_primary:
        print(dist_info)
    device = torch.device(dist_info.device_type)
    dtype = torch.float32
    
    #set up training configuration
    cfg = config_init(args, dist_info, PROJECT_ROOT)
    assert cfg.do_MMD == (mmd_frac>0), f"cfg.do_MMD {cfg.do_MMD} inconsistent with mmd_frac {mmd_frac}"
    
    ### set random seed
    torch.manual_seed(cfg.seed)#+ rank)
    np.random.seed(cfg.seed)#+ rank)
    
    #load data sets
    dataset_args = create_dataset_args(cfg, PROJECT_ROOT)
    datasets = [initialize_datasets(**d_args) for d_args in dataset_args]
    _, dataloaders = retrieve_dataloaders(cfg.do_MMD, datasets, cfg.batch_size,
                                                      num_workers=dist_info.num_workers,
                                                      rank=dist_info.rank,
                                                      num_replicas=dist_info.world_size,
                                                      model_arch=cfg.model_name)

    batch = next(iter(dataloaders[0]['train']))
    assert torch.all(batch['is_target'][:-1] <= batch['is_target'][1:]), f"Batch is_target not sorted: {batch['is_target']}"

    if cfg.do_MMD:
        print('testing MMD mode')
        assert len(dataloaders)== 1, f"Expected 1 dataloader, got {len(dataloaders)}"
        assert list(dataloaders[0].keys())== ['train', 'valid'], f"Expected ['train', 'valid'], got {list(dataloaders[0].keys())}"
        bs_effective = 2*batch_size
        n_target = batch['is_target'].sum().item()
        assert n_target>batch_size-4*np.sqrt(bs_effective)/2 and n_target<batch_size+4*np.sqrt(bs_effective)/2, f"Expected ~{batch_size} target points, got {n_target}"
    else:
        print('testing non-MMD mode')
        assert len(dataloaders)== 2, f"Expected 2 dataloaders, got {len(dataloaders)}"
        assert list(dataloaders[0].keys())== ['train', 'valid'], f"Expected ['train', 'valid'], got {list(dataloaders[0].keys())}"
        assert list(dataloaders[1].keys())== ['valid'], f"Expected ['valid'], got {list(dataloaders[1].keys())}"
        bs_effective = batch_size

    num_train = len(datasets[0]['train'])
    num_batch_expected = int(num_train//bs_effective)
    assert len(dataloaders[0]['train'])==num_batch_expected, f"Expected {num_batch_expected} batches, got {len(dataloaders[0]['train'])}"
    assert batch['features'].size() == (bs_effective, num_dims, num_points), f"Expected features size {(bs_effective, num_dims, num_points)}, got {batch['features'].size()}"
    #load model config
    if cfg.model_config != '':     
        with open(PROJECT_ROOT / 'model_configs' / cfg.model_config, 'r') as f:
            model_config = json.load(f)
    
    # get model with predictor wrapper    
    model = make_predictor(cfg.model_name, 
                            input_dims=num_dims, #TODO: get this from the dataset shape
                            num_classes=num_classes,
                            cfg=model_config)
    
    model = model.to(device)
    
    if cfg.do_MMD:
        prepared = model.prepare_batch(batch, device, dtype)
        pred, encoded = model(prepared, intermediates=['encoder'])
        assert pred.size() == (bs_effective, num_classes), f"Expected pred size {(bs_effective, num_classes)}, got {pred.size()}"
        latent_dim = model_config['group_specs']['encoder']['fc_params'][-1][0]
        assert encoded.size() == (bs_effective, latent_dim), f"Expected encoded size {(bs_effective, latent_dim)}, got {encoded.size()}"
    else:
        prepared = model.prepare_batch(batch, device, dtype=dtype)
        pred, = model(prepared)
        assert pred.size() == (bs_effective, num_classes), f"Expected pred size {(bs_effective, num_classes)}, got {pred.size()}"
