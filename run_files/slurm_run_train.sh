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
job_type=supervised_Lite

model_name="ParticleNet-Lite"
model_config="config_000.json"
batch_size=1024
mmd_frac=0.0
num_data=-1
epochs=30
warmup_epochs=10

log_interval=50
patience=3
threshold=5e-3
reduce_factor=0.3
start_lr=5e-4
peak_lr=5e-3

##########################
source=py83
target=hw72
##########################
#source=hw72
#target=py83
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
    --start_lr ${start_lr} --peak_lr ${peak_lr} --patience ${patience} --reduce_factor ${reduce_factor} --threshold ${threshold}