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
from utils.utils import MetricHistory
from models.predictors import make_predictor
from data.dataset_v2 import initialize_datasets, retrieve_dataloaders, create_dataset_args
from utils.model_utils import print_stage_param_summary, get_param_groups, freeze_param_groups
from training.schedulers import SchedConfig
from training.losses import MMDLoss, MMDScheduler
from training.trainer import Trainer
import torch.distributed as dist

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
    dataset_args = create_dataset_args(PROJECT_ROOT,
                                       datadir=cfg.datadir,
                                       do_MMD=cfg.do_MMD,
                                       mixed_batch=(cfg.do_MMD and 'backbone' not in cfg.target_encoder_groups) or cfg.mode == 'st_classifier',
                                       model=cfg.model_name,
                                       num_data=cfg.num_data,
                                       tv_fracs={'train': 0.6, 'valid': 0.2})
    
    datasets = [initialize_datasets(**d_args) for d_args in dataset_args]
    dataloaders = [
        retrieve_dataloaders(
            dsets,
            2*cfg.batch_size if len(datasets)==1 else cfg.batch_size,
            num_workers=dist_info.num_workers,
            rank=dist_info.rank,
            num_replicas=dist_info.world_size,
            model_arch=cfg.model_name,
            collate='sorted' if len(datasets)==1 else None
        ) 
        for dsets in datasets
    ]
    
    #load model config
    if cfg.model_config != '':
        with open(PROJECT_ROOT / 'model_configs' / cfg.model_config, 'r') as f:
            model_config = json.load(f)
    
    # get model with predictor wrapper  
    model = make_predictor(cfg.model_name,
                           target_encoder_groups=cfg.target_encoder_groups,
                           input_dims=7, #TODO: get this from the dataset shape
                           num_classes=2,
                           tap_keys=('encoder',) if cfg.do_MMD else (),
                           encoder_layer='encoder',
                           cfg=model_config)
    
    model = maybe_convert_syncbn(model, dist_info.device_type, dist_info.world_size)
    model = model.to(device)
    
    param_groups = get_param_groups(model, model_config, peak_lr=cfg.peak_lr, target_encoder_groups=cfg.target_encoder_groups)
    param_groups, train_groups = freeze_param_groups(param_groups, frozen_groups={'main': ()}) # no frozen groups for now TODO: add input arg
    
    metrics = MetricHistory()
    metrics.update(peak_lr_by_group={k: [g['lr'] for g in groups] for k, groups in param_groups.items()})
    #TODO: update usage of param_groups ahead

    if dist_info.is_primary:
        model_total, model_trainable = print_stage_param_summary(model.model)
        if len(cfg.target_encoder_groups)>0:
            t_encoder_total, t_encoder_trainable = print_stage_param_summary(model.target_encoder, name='Target Model')
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}, Trainable parameters: {trainable_params:,}")
        assert total_params == model_total + (t_encoder_total if len(cfg.target_encoder_groups)>0 else 0), "Total params do not match sum of model and target encoder!"
        assert trainable_params == model_trainable + (t_encoder_trainable if len(cfg.target_encoder_groups)>0 else 0), "Trainable params do not match sum of model and target encoder!"
    optimizer = optim.AdamW([g for group in train_groups.values() for g in group])
    ddp_model = wrap_like_ddp(model, device, dist_info.local_rank, use_ddp=(dist_info.world_size>1))

    if cfg.pretrained != '':
        start_epoch = load_ckp(f"{cfg.logdir}/{cfg.pretrained}/best-val-model.pt",
                               ddp_model,
                               optimizer=optimizer if cfg.ld_optim_state else None,
                               device=device,
                               twin_encoder=len(cfg.target_encoder_groups)>0)
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
        threshold = cfg.threshold
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
        dataloaders=dataloaders,
        metrics=metrics,
        mode=cfg.mode
    )
    
    #for n,m in ddp_model.named_modules():
    #    pass
    if not args.test_mode:
        trainer.train()

    trainer.test()
    
    if dist_info.initialized:
        dist.destroy_process_group()
        
    torch.cuda.empty_cache()
    
    
if __name__ == "__main__":
    main()

   
    
    
    