#!/bin/bash
job_type=MMD_Lite
exp_name="test_run_mmd2"
model_name="ParticleNet-Lite"
model_config="config_007.yaml"
pretrained="test_run"
batch_size=100
mmd_frac=0.175
mmd_turnon_epoch=3
mmd_turnon_width=3
num_data=-1
epochs=5
warmup_epochs=7
log_interval=2
patience=3
threshold=5e-3
reduce_factor=0.3
start_lr=1e-5
peak_lr=1e-4
target_model_groups='encoder'
mode='qt_classifier'
use_tar_labels= #--use_tar_labels

##########################
source=py83
target=hw72
##########################

source_dir=data/datasets_Pnet100_Njets1000/${source}
target_dir=data/datasets_Pnet100_Njets1000/${target}

#!/usr/bin/env bash

exp_name="${job_type}_${source}_${target}"
export FORCE_CPU=1
python scripts/train.py --exp_name ${exp_name} --model_name ${model_name} --model_config ${model_config} --batch_size ${batch_size} \
    --MMD_frac ${mmd_frac} --num_data ${num_data} --datadir ${source_dir} ${target_dir} \
    --epochs ${epochs} --warmup_epochs ${warmup_epochs} --log_interval ${log_interval} \
    --start_lr ${start_lr} --peak_lr ${peak_lr} --patience ${patience} --reduce_factor ${reduce_factor} \
    --MMDturnon_epoch ${mmd_turnon_epoch} --MMDturnon_width ${mmd_turnon_width} --threshold ${threshold} --pretrained "${pretrained}" \
    --target_model_groups ${target_model_groups} ${use_tar_labels} --mode ${mode}