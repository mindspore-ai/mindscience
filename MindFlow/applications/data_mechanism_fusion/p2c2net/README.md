ENGLISH | [简体中文](README_CN.md)

# Solving 2d Burgers Equation by Using P2C2Net

## Overview

**P2C2Net (PDE-Preserved Coarse Correction Network)** is a novel neural network architecture designed to efficiently solve spatiotemporal partial differential equations (PDEs) on coarse mesh grids with limited training data. Original paper is *"P2C2Net: PDE-Preserved Coarse Correction Network for Efficient Prediction of Spatiotemporal Dynamics"*.
![model architecture](images/model_architecture.png)
The model consists of two synergistic modules: (1) a trainable PDE block that learns to update the coarse solution (i.e., the system state), based on a high-order numerical scheme with boundary condition encoding, and (2) a neural network block that consistently corrects the solution on the fly. In particular, the model adopts a learnable symmetric Conv filter, with weights shared over the entire model, to accurately estimate the spatial derivatives of PDE based on the neural-corrected system state.

The Burgers’ equation is a nonlinear PDE that models the propagation and reflection of shock waves. It is widely used in fluid mechanics, nonlinear acoustics, gas dynamics, and other fields. In this project, we focus on solving the **2D Burgers’ equation** efficiently using P2C2Net.

## Quick Start

### 1. Data Generation

First, generate training and testing data by running:

```shell
cd src
python dataGen.py
```

### 2. Training

Run the following command to train P2C2Net on the generated data:

```shell
python p2c2net/train_burgers.py --experiment p2c2net
```

where

`--experiment` is the the experiment directory. It should include experiment specifications under 'config/';

`--mode` is the running mode. 'GRAPH' indicates static graph mode. 'PYNATIVE' indicates dynamic graph mode. Default 'GRAPH';

`--device_target` represents the type of computing platform used, which can be selected as 'Ascend' or 'GPU', default 'Ascend';

`--device_id` represents the calculation card number used, which can be filled in according to the actual situation, default 0;

`--continue` represents whether to resume training from a saved checkpoint, default False;

`--config_filename` is the name of the configuration file (under the `configs/` directory) that defines experiment settings such as model parameters, training schedule, default 'burgers.json';

`--train_stage` specifies whether to enable the training mode, default True;

`--test_stage` specifies whether to enable the testing mode, default True;

### 3. Result

After training, experiment outputs (checkpoints and evaluation results) are saved in result directory under the --experiment directory you provided. Use the saved checkpoints to reproduce evaluations or continue training.

#### inference result

![inference result](images/inference.png)

## Rquirements

1. Python>=3.9
2. MindSpore>=2.5

## Contributor

gitee id：[liuguangyuu](https://gitee.com/liuguangyuu)

email: liuguangyuu@outlook.com