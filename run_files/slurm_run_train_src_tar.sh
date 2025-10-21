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
job_type=Source-Target_Lite

model_name="ParticleNet-Lite"
model_config="config_001.json"
pretrained=""
batch_size=256
mmd_frac=0
mmd_turnon_epoch=3
mmd_turnon_width=3
num_data=-1
epochs=1
warmup_epochs=8
log_interval=50
patience=3
threshold=5e-3
reduce_factor=0.3
start_lr=3e-4
peak_lr=3e-3
target_model_groups="" #'backbone encoder'
mode='st_classifier'
frozen_groups='{}' #'{"main": ["backbone", "encoder"], "target_model": ["backbone", "encoder"]}'
use_tar_labels= #--use_tar_labels

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

#debug
#------------------------------------
#export NCCL_DEBUG=INFO
#export NCCL_DEBUG_SUBSYS=INIT,COLL
#export TORCH_DISTRIBUTED_DEBUG=DETAIL
#export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
#export TORCH_NCCL_BLOCKING_WAIT=1
#------------------------------------

export SLURM_CPU_BIND=cores
export SLURM_HINT=nomultithread

srun python -u scripts/train.py --exp_name ${exp_name} --model_name ${model_name} --model_config ${model_config} --batch_size ${batch_size} \
    --MMD_frac ${mmd_frac} --num_data ${num_data} --datadir ${source_dir} ${target_dir} \
    --epochs ${epochs} --warmup_epochs ${warmup_epochs} --log_interval ${log_interval} \
    --start_lr ${start_lr} --peak_lr ${peak_lr} --patience ${patience} --reduce_factor ${reduce_factor} \
    --MMDturnon_epoch ${mmd_turnon_epoch} --MMDturnon_width ${mmd_turnon_width} --threshold ${threshold} \
    ${use_tar_labels} --mode ${mode} --frozen_groups "${frozen_groups}"