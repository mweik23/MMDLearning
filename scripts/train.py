import torch
from torch import nn, optim
import argparse, json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
SRC_path = Path(__file__).parents[1].resolve() / 'src' / 'MMDLearning'
import sys
sys.path.append(str(SRC_path))
from utils.distributed import setup_dist, DistInfo, wrap_like_ddp, maybe_convert_syncbn
from utils.io import config_init, load_ckp
from utils.cli import build_parser
from models.predictors import make_predictor
from data.dataset_v2 import initialize_datasets, retrieve_dataloaders, create_dataset_args
from utils.model_utils import print_stage_param_summary
from training.schedulers import SchedConfig
from training.losses import MMDLoss, MMDScheduler
from training.trainer import Trainer
PROJECT_ROOT = Path(__file__).parents[1].resolve()

'''
usage: python train.py --exp_name <name> --logdir <logdir> --model_name <model> --model_config <json file> 
--batch_size <int> --num_workers <int> --start_lr <float> --warmup_epochs <int> --patience <int> 
--reduce_factor <float> --final_epoch <int> --val_interval <int> --MMDturnon_epoch <int> 
--MMDturnon_width <int> --MMD_frac <float> --do_MMD (flag) --pretrained <ckp path> --ld_optim_state (flag) --seed <int>
'''

def main(argv=None):
    #get arguments
    parser = build_parser()
    args = parser.parse_args(argv)
    
    #set up distributed training
    dist_info: DistInfo = setup_dist(arg_num_workers=args.num_workers)
    if dist_info.is_primary:
        print(dist_info)
    device = torch.device(dist_info.device_type)
    
    #set up training configuration
    cfg = config_init(args, dist_info, PROJECT_ROOT)

    ### set random seed
    torch.manual_seed(cfg.seed)#+ rank)
    np.random.seed(cfg.seed)#+ rank)
    
    #load data sets
    dataset_args = create_dataset_args(cfg, PROJECT_ROOT)
    datasets = [initialize_datasets(**d_args) for d_args in dataset_args]
    train_sampler, dataloaders = retrieve_dataloaders(cfg.do_MMD, datasets, cfg.batch_size,
                                                      num_workers=dist_info.num_workers,
                                                      rank=dist_info.rank,
                                                      num_replicas=dist_info.world_size,
                                                      model_arch=cfg.model_name)
    
    #load model config
    if cfg.model_config != '':
        with open(PROJECT_ROOT / 'model_configs' / cfg.model_config, 'r') as f:
            model_config = json.load(f)
    
    # get model with predictor wrapper    
    model = make_predictor(cfg.model_name,
                            input_dims=7, #TODO: get this from the dataset shape
                            num_classes=2,
                            cfg=model_config)
    
    model = maybe_convert_syncbn(model, dist_info.device_type, dist_info.world_size)
    model = model.to(device)
    stage_param_groups = [{
            "params": list(model.stages[name].parameters()), 
            "lr": specs['optim_params'].get('lr', model_config['defaults']['lr'])*cfg.peak_lr,
            "weight_decay": specs['optim_params'].get('weight_decay', model_config['defaults']['weight_decay']), 
            "name": name
        }
        for name, specs in model_config['group_specs'].items()
    ]
    if dist_info.is_primary:
        print_stage_param_summary(model)
    optimizer = optim.AdamW(stage_param_groups)
    ddp_model = wrap_like_ddp(model, device, dist_info.local_rank, use_ddp=(dist_info.world_size>1))

    if cfg.pretrained != '':
        start_epoch = load_ckp(f"{cfg.logdir}/{cfg.pretrained}/best-val-model.pt",
                               ddp_model,
                               optimizer=optimizer if cfg.ld_optim_state else None,
                               device=device)
    else:
        start_epoch = 0
        
    for g in optimizer.param_groups:
        g.setdefault('initial_lr', g['lr'])
        
    sched_config = SchedConfig(
        kind = "warmup_plateau",
        lr_min=cfg.start_lr/cfg.peak_lr,
        warmup_epochs = cfg.warmup_epochs,
        mode = "min",
        factor = cfg.reduce_factor,
        patience = cfg.patience,
    )
    
    mmd_sched = MMDScheduler(cfg.MMDturnon_epoch, cfg.MMDturnon_width) if cfg.do_MMD else None
    
    loss_fns = {'bce': nn.CrossEntropyLoss()}
    if cfg.do_MMD:
        loss_fns['mmd'] = MMDLoss()

    trainer = Trainer(
        cfg=cfg,
        ddp_model=ddp_model,
        start_epoch=start_epoch,
        optimizer=optimizer,
        dist_info=dist_info,
        device=device,
        sched_config=sched_config,
        loss_fns=loss_fns,
        mmd_sched=mmd_sched,
        train_sampler=train_sampler,
        dataloaders=dataloaders
    )
    
    #for n,m in ddp_model.named_modules():
    #    pass
    if not args.test_mode:
        trainer.train()

    trainer.test()
    torch.cuda.empty_cache()
    
if __name__ == "__main__":
    main()

   
    
    
    