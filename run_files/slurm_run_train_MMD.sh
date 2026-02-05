#!/bin/bash
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "WORLD_SIZE="$WORLD_SIZE
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR
#########

project=domain_adaptation
group=MMDLearning
job_type=MMD_Lite

model_name="ParticleNet-Lite"
model_config="config_007.yaml"
pretrained=supervised_Lite_py83_hw72_43163255
batch_size=1024
mmd_frac=0.175
mmd_turnon_epoch=3
mmd_turnon_width=3
num_data=-1
epochs=1
warmup_epochs=7
log_interval=50
patience=3
threshold=5e-3
reduce_factor=0.3
start_lr=1e-6
peak_lr=1e-5
target_model_groups='encoder'
mode='qt_classifier'
use_tar_labels= #--use_tar_labels
n_fourier_features=1024

##########################
source=py83
target=hw72
##########################

source_dir=data/datasets_Pnet100_Njets-1/${source}
target_dir=data/datasets_Pnet100_Njets-1/${target}

source ~/.bashrc
cd $PSCRATCH/${group}

nvidia-smi
echo "ASSIGNED DEVICES: $CUDA_VISIBLE_DEVICES"
#############
# FOR SLURM
##### without singularity, activate conda environment and then : srun python train_lorentznet.py .......

conda activate ml

#!/usr/bin/env bash

exp_name="${job_type}_${source}_${target}_$SLURM_JOB_ID"

srun python -u scripts/train.py --exp_name ${exp_name} --model_name ${model_name} --model_config ${model_config} --batch_size ${batch_size} \
    --MMD_frac ${mmd_frac} --num_data ${num_data} --datadir ${source_dir} ${target_dir} \
    --epochs ${epochs} --warmup_epochs ${warmup_epochs} --log_interval ${log_interval} \
    --start_lr ${start_lr} --peak_lr ${peak_lr} --patience ${patience} --reduce_factor ${reduce_factor} \
    --MMDturnon_epoch ${mmd_turnon_epoch} --MMDturnon_width ${mmd_turnon_width} --threshold ${threshold} --pretrained ${pretrained} \
    --target_model_groups ${target_model_groups} ${use_tar_labels} --mode ${mode} --n_fourier_features ${n_fourier_features}