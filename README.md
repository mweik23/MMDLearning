# MMD-Net Algorithm
A machine learning training algorithm for classifying jets from collider physics experiments.

## Overview
This project is a proof of concept of an unsupervised domain adaptation technique to optimize performance on a target domain where labels are not known a priori. The technique requires a source dataset where we do have labels that is expected to be similar to the target domain. The model is then trained with a loss that is made up of the binary cross entropy loss evaluated on the source domain and a maximum mean discrepency regularization term to constrain the model to use a latent representation in which both datasets follow the same distribution. With this setup, the source dataset can be more informative for classification of the target dataset.


## Installation
### Prerequisites
- [python 3.9](https://www.python.org/downloads/release/python-390/)
- [pip](https://pip.pypa.io/en/stable/installation/)

### Local Development
#### CPU
```bash
conda env create -f environment_cpu.yml
conda activate MMDLearning
pip install -e ".[dev]"

```
#### CUDA
```bash
conda env create -f environment_cuda.yml
conda activate MMDLearning
pip install -e ".[dev]"
```

## Command-Line Usage

### Preprocessing
Details about preprocessing still to come.

### Training
#### Example Usage
```bash
job_type=MMD_Lite
model_name="ParticleNet-Lite"
model_config="config_006.yaml"
pretrained=supervised_Lite_py83_hw72
batch_size=1024
mmd_frac=0.175
mmd_turnon_epoch=3
mmd_turnon_width=3
num_data=-1
epochs=30
warmup_epochs=7
log_interval=50
patience=3
threshold=5e-3
reduce_factor=0.3
start_lr=1e-6
peak_lr=1e-5
target_model_groups='encoder'
mode='qt_classifier'

##########################
source=py83
target=hw72
##########################

source_dir=data/datasets_Pnet100_Njets-1/${source}
target_dir=data/datasets_Pnet100_Njets-1/${target}

exp_name="${job_type}_${source}_${target}"

python scripts/train.py --exp_name ${exp_name} \
    --model_name ${model_name} --model_config ${model_config} \
    --batch_size ${batch_size} --MMD_frac ${mmd_frac} \
    --num_data ${num_data} --datadir ${source_dir} ${target_dir} \
    --epochs ${epochs} --warmup_epochs ${warmup_epochs} \
    --log_interval ${log_interval} --start_lr ${start_lr} \
    --peak_lr ${peak_lr} --patience ${patience} \
    --reduce_factor ${reduce_factor} --MMDturnon_epoch ${mmd_turnon_epoch} \
    --MMDturnon_width ${mmd_turnon_width} --threshold ${threshold} \
    --pretrained ${pretrained} --target_model_groups ${target_model_groups} \
    --mode ${mode}
```

#### Training Script Arguments

| Argument | Type | Default | Description |
|---------|------|---------|-------------|
| `--exp_name` | str | `""` | Experiment name |
| `--test_mode` | flag | `False` | Test best model |
| `--batch_size` | int | `32` | Input batch size for training |
| `--num_data` | int | `-1` | Number of samples |
| `--epochs` | int | `35` | Number of training epochs |
| `--model_config` | str | `""` | Model config file |
| `--warmup_epochs` | int | `5` | Number of warm-up epochs |
| `--seed` | int | `99` | Random seed |
| `--log_interval` | int | `100` | Batches between logging |
| `--mmd_interval` | int | `-1` | Batches between null MMD calculation |
| `--val_interval` | int | `1` | Epochs between validation |
| `--datadir` | list[str] | `data/top` | Data directories |
| `--logdir` | str | `logs/top` | Output log directory |
| `--peak_lr` | float | `1e-3` | Peak learning rate |
| `--num_workers` | int | `None` | DataLoader workers |
| `--patience` | int | `10` | LR scheduler patience |
| `--threshold` | float | `1e-4` | LR scheduler threshold |
| `--reduce_factor` | float | `0.1` | LR reduction factor |
| `--start_lr` | float | `1e-4` | Starting LR for warmup |
| `--MMDturnon_epoch` | int | `5` | Epoch when MMD turns on |
| `--MMD_frac` | float | `0` | MMD loss prefactor |
| `--MMDturnon_width` | int | `5` | Smooth MMD turn-on width (epochs) |
| `--pretrained` | str | `""` | Pretrained model directory |
| `--ld_optim_state` | flag | `False` | Load optimizer state |
| `--n_kernels` | int | `5` | Number of MMD kernels |
| `--use_tar_labels` | flag | `False` | Use target labels for MMD |
| `--model_name` | str | `LorentzNet` | Model name |
| `--target_model_groups` | list[str] | `None` | Separate target models for groups |
| `--mode` | str | `qt_classifier` | Mode of operation |
| `--frozen_groups` | json | `{}` | Model groups to freeze |
| `--local_rank` | int | `0` | Distributed training rank |

Anytime a new training experiment is run, a new directory will be created in `--logdir` with the name `{exp_name}`. This directory will contain the model checkpoints and logs for that experiment.

