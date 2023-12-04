ENGLISH | [简体中文](README_CN.md)

# PDE-Net for Convection-Diffusion Equation

## Overview

PDE-Net is a feedforward deep network proposed by Zichao Long et al. to learn partial differential equations from data, predict the dynamic characteristics of complex systems accurately and uncover potential PDE models. The basic idea of PDE-Net is to approximate differential operators by learning convolution kernels (filters). Neural networks or other machine learning methods are applied to fit unknown nonlinear responses. Numerical experiments show that the model can identify the observed dynamical equations and predict the dynamical behavior over a relatively long period of time, even in noisy environments. More information can be found in [PDE-Net: Learning PDEs from Data](https://arxiv.org/abs/1710.09668).

![coe label benchmark](images/coe_label_benchmark.png)

![coe trained step-1](images/coe_trained_step-1.png)

![result](images/result.jpg)

![extrapolation](images/extrapolation.jpg)

[See More](./pde_net.ipynb)

## Quick Start

### Training Method 1: Call the `train.py` script on the command line

```shell
python train.py --config_file_path ./configs/pde_net.yaml --device_target Ascend --device_id 0 --mode GRAPH
```

Among them,

`--config_file_path` represents the parameter and path control file, default './configs/pde_net.yaml'

`--device_target` represents the type of computing platform used, which can be selected as 'Ascend' or 'GPU', default 'Ascend';

`--device_id` represents the calculation card number used, which can be filled in according to the actual situation, default 0;

`--mode` represents the running mode, 'GRAPH' indicates the static Graphical model, 'PYNATIVE' indicates the dynamic Graphical model, default 'GRAPH';

### Training Method 2: Running Jupyter Notebook

You can run training and validation code line by line using both the [Chinese version](pde_net_CN.ipynb) and the [English version](pde_net.ipynb) of Jupyter Notebook.

## Performance

| Parameter               | Ascend               | GPU                |
|:----------------------:|:--------------------------:|:---------------:|
| Hardware                | Ascend 32G           | NVIDIA V100 32G    |
| MindSpore version       | >=2.1.0                | >=2.1.0                   |
| Parameters              | 3.5e4       |         3.5e4                   |
| Train Config            | batch_size=16, steps_per_epoch=70, epochs=500 | batch_size=16, steps_per_epoch=70, epochs=500 |
| Evaluation Config       | batch_size=16   | batch_size=16               |
| Optimizer               | Adam                 | Adam                   |
| Train Loss(MSE)         | 0.9        |     0.6             |
| Evaluation Error(RMSE)  | 0.06       |       0.04              |
| Speed(ms/step)          | 45       |    105                 |

## Contributor

gitee id：[liulei277](https://gitee.com/liulei277)

email: liulei2770919@163.com