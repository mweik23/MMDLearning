# Domain Adaptation With Maximum Mean Discrepency for Top Tagging

In particle phyiscs experiments, collisions between high energy particles lead to the production of thousands of particles which can be detected downstream. By studying the distribution of particles produced by these events, we can probe our most advanced theory of particle physics, the Standard Model. Of particular interest for studying extensions of the Standard Model are top quarks, which decay quickly after they are produced. These decays lead to *jets*, which are beams of (in this case $\mathcal{O}(100)$) stable particles that come from these decays and reach our detectors. These are quantum mechanical processes so our best theory can predict only the statistical distribution of the particles in a jet. 

The statistical distributions of jets are intractible, with many complex correlations between constituents. However, we have fairly reliable simulations that sample these distrubtions and can be used for training *top taggers*. Top taggers are classiers that typically are trained to distinguish *top jets* (jets from top quarks) from *QCD jets* (jets from light quarks), based large datasets of simulated jets of each type. The primary technique that has been explored in the literature is supervised training with labels coming from simulations. Two of the most advanced models for this task are [LorentzNet](https://arxiv.org/abs/2201.08187) and [ParticleNet](https://arxiv.org/abs/1902.08570). As models become more advanced, they become limited by the limitations of our simulations. If the simulations are not perfectly faithful to the real data, the models will become optimized on top tagging based on the simulation distribution and may not perform as well on real data. 

The goal of this project was to provide a proof of concept for a domain adaptation technique that could be used to optimize a classiifcation model for use on the real dataset (target) from training on simulated data (source). For this proof of concept, we use simulated data from two different simulation softwares that have small but significant differences and treat one dataset as the source and the other as the target. When the real data is the target, labels will not be availible, so the problem is to improve evaluation outcomes on the target dataset while using input and labels from the source and only the input from the target. We attempt this problem using the Maximum Mean Discrepency (MMD), a statistical metric that measures the difference between two datasets. We include the target dataset inputs in the training process and augment the binary cross entropy loss by adding the the batch-evaluated MMD between the model outputs for target and source.

---

## Table of Contents


- [Installation](#installation)
- [Usage](#usage)
- [Acknowledgements](#acknowledgements)

---

## Installation

Clone the repository:

```bash
git clone https://github.com/mweik23/MMDLearning
cd MMDLearning
```
Environment Setup:

```bash
conda create -n EnvName python=3.7
conda activate EnvName
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
```
---
## Usage

Compatible with SLURM or manual job execution

Basic usage example for ParticleNet using SLURM:

```bash
python top_tagging_MMD_v2.py --batch_size=512 --epochs=46 --num_train=-1 --warmup_epochs=5\
        --lr=1 --num_workers=4 --dropout=.1 --log_interval=100 --weight_decay=0.0001\
        --exp_name=ExpName --datadir SourceDir TargetDir --logdir ./logs/top/ --no_layernorm\
        --MMD_coef=4 --MMDturnon_epoch=5 --intermed_mmd --MMDturnon_width=2\
        --lr_scheduler=ParticleNet --model=ParticleNet --n_kernels=5 --n_latent=64
```
For ParticleNet-Lite, replace `ParticleNet` with `ParticleNet-Lite` everywhere in the above command

Basic usage example for LorentzNet using SLURM:

```bash
python top_tagging_MMD_v2.py --batch_size=512 --model=Lorentznet\
        --epochs=35 --num_train=-1 --warmup_epochs=5 --lr=.001 --num_workers=4 --c_weight=0.005\
        --dropout=0.2 --log_interval=100 --weight_decay=0.01\
        --exp_name=ExpName --datadir SourceDir TargetDir --no_layernorm\
        --MMD_coef=4 --MMDturnon_epoch=5 --intermed_mmd --MMDturnon_width=2\
        --lr_scheduler=CosineAnealing --n_hidden=72 --n_layers=6 
```

For training without MMD, either use '--MMD_coef=0' or for a more efficient run use

```bash
python top_tagging_v3.py --batch_size=384 --model=ParticleNet\
        --epochs=21 --num_train=${num_train} --lr=1 --num_workers=4\
        --dropout=.1 --log_interval=100 --weight_decay=0.0001\
        --exp_name=ExpName --datadir SourceDir TargetDir --no_layernorm\
        --lr_scheduler=ParticleNet
```
or similar for other variants

For manual_job execution use:
```bash
python -m torch.distributed.launch --nproc_per_node=1 top_tagging_MMD_v2.py --manual ...
```
which works for `top_tagging_v3.py` as well.

For more detailed options, see:

```bash
python top_tagging_MMD_v2.py --help
```
or
```bash
python top_tagging_v3.py --help
```

---

## Acknowledgements

This repo is based off of the [LorentzNet](https://github.com/sdogsq/LorentzNet-release) repo <br>
The LorentzNet model is described in detail in [An Efficient Lorentz Equivariant Graph Neural Network for Jet Tagging](https://arxiv.org/abs/2201.08187)<br>
Thank you to the creaters of LorentzNet

This repo also includes the [ParticleNet architecture](https://github.com/hqucms/weaver-core/blob/main/weaver/nn/model/ParticleNet.py)
described in detail in [ParticleNet: Jet Tagging via Particle Clouds](https://arxiv.org/abs/1902.08570)<br>
Thank you to the creaters of ParticleNet
