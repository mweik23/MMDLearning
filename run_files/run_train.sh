#!/bin/zsh

exp_name="test_run"
model_name="ParticleNet-Lite"
model_config="config_003.json"
batch_size=64
mmd_frac=0.0
num_data=-1
epochs=20
warmup_epochs=5
log_interval=20
patience=3
reduce_factor=0.5
mmd_turnon_epoch=5
mmd_turnon_width=5
source_dir=data/datasets_Pnet100_Njets-1/py83
target_dir=data/datasets_Pnet100_Njets-1/hw72

export FORCE_CPU=1
python scripts/train.py --exp_name ${exp_name} --model_name ${model_name} --model_config ${model_config} --batch_size ${batch_size} \
        --MMD_frac ${mmd_frac} --num_data ${num_data} --datadir ${source_dir} ${target_dir} \
        --epochs ${epochs} --warmup_epochs ${warmup_epochs} --log_interval ${log_interval} \
        --start_lr 1e-4 --peak_lr 1e-3 --patience ${patience} --reduce_factor ${reduce_factor} --MMDturnon_epoch ${mmd_turnon_epoch} \
        --MMDturnon_width ${mmd_turnon_width}