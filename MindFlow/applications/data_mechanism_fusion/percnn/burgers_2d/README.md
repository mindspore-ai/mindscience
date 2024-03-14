ENGLISH | [简体中文](README_CN.md)

# Solving 2d Burgers Equation by Using PeRCNN

## Overview

Burgers' equation is a nonlinear partial differential equation that simulates the propagation and reflection of shock waves. It is widely used in the fields of fluid mechanics, nonlinear acoustics, gas dynamics et al. It is named after Johannes Martins Hamburg (1895-1981). In this case, the 2D Burgers' equation with viscosity is solved based on PeRCNN method.

## Quick Start

Download the data from [link](https://download.mindspore.cn/mindscience/mindflow/dataset/applications/data_mechanism_fusion/PeRCNN) and save them in `./dataset`. Need to install `prettytable` and `imageio` packages.

```shell
pip install prettytable
pip install imageio
```

### Stage 1: call `train.py` to carry out data-driven PeRCNN simulation

```shell
python train.py --config_file_path ./percnn_burgers.yaml --mode GRAPH --device_target Ascend --device_id 0
```

where

`--mode` is the running mode. 'GRAPH' indicates static graph mode. 'PYNATIVE' indicates dynamic graph mode. Default 'GRAPH'.

`--save_graphs` represents whether to save the computing graph. Default 'False'；

`--save_graphs_path` represents save path of the computing graph. Default './graphs'

`--device_target` represents the type of computing platform used, which can be selected as 'Ascend' or 'GPU', default 'Ascend';

`--device_id` represents the calculation card number used, which can be filled in according to the actual situation, default 0;

`--config_file_path` represents the parameter and path control file, default './configs/percnn_burgers.yaml';

`--pattern` represents research pattern. Choose 'data_driven' in this stage;

### Stage 2: call `uncover_coef.py` uncover the underlying physical equations by sparse regression

```shell
python uncover_coef.py --mode GRAPH --save_graphs_path ./graphs --device_target Ascend --device_id 0 --config_file_path ./percnn_burgers.yaml --pattern data_driven
```

where,

`--data_path` represents the path of the data file. You can select the data obtained from stage1, default './data/Burgers_2001x2x100x100_[dt=00025].mat';

### Stage 3: call `train.py` to fine-tune the coefficients with physics-driven PeRCNN

```shell
python train.py --mode GRAPH --save_graphs_path ./graphs --device_target Ascend --device_id 0 --config_file_path ./percnn_burgers.yaml --pattern data_driven
```

where,

`--config_file_path` represents the parameter and path control file, default './configs/percnn_burgers.yaml'；

`--pattern` represents research pattern. Choose 'physics_driven' in this stage;

## Results

![Burgers PINNs](images/results.gif)

## Performance

| Parameter               | Ascend               | GPU                |
|:----------------------:|:--------------------------:|:---------------:|
| Hardware                | Ascend 32G           | NVIDIA V100 32G    |
| MindSpore version       | >=2.1.0                 | >=2.1.0                   |
| dataset                 | [PeRCNN Dataset](https://download-mindspore.osinfra.cn/mindscience/mindflow/dataset/applications/data_mechanism_fusion/PeRCNN/)      | [PeRCNN Dataset](https://download-mindspore.osinfra.cn/mindscience/mindflow/dataset/applications/data_mechanism_fusion/PeRCNN/)                   |
| Parameters              | 4e3                  | 4e3                   |
| Train Config            | batch_size=1, steps_per_epoch=1, epochs=15000 | batch_size=8, steps_per_epoch=1, epochs=15000 |
| Evaluation Config       | batch_size=1      | batch_size=1               |
| Optimizer               | Adam                 | Adam                   |
| Train Loss(MSE)         | 0.001                | 0.001             |
| Evaluation Error(RMSE)  | 0.06                | 0.10              |
| Speed(ms/step)          | 40                   | 150                 |

## Contributor

gitee id：[yi-zhang95](https://gitee.com/yi-zhang95), [chengzrz](https://gitee.com/chengzrz)

email: zhang_yi_1995@163.com