ENGLISH | [简体中文](README_CN.md)

# PDE-Net for Convection-Diffusion Equation

## Overview

PDE-Net is a feedforward deep network proposed by Zichao Long et al. to learn partial differential equations from data, predict the dynamic characteristics of complex systems accurately and uncover potential PDE models. The basic idea of PDE-Net is to approximate differential operators by learning convolution kernels (filters). Neural networks or other machine learning methods are applied to fit unknown nonlinear responses. Numerical experiments show that the model can identify the observed dynamical equations and predict the dynamical behavior over a relatively long period of time, even in noisy environments. More information can be found in [PDE-Net: Learning PDEs from Data](https://arxiv.org/abs/1710.09668).

![coe label benchmark](images/coe_label_benchmark.png)

![coe trained step-1](images/coe_trained_step-1.png)

![result](images/result.jpg)

![extrapolation](images/extrapolation.jpg)

[See More](https://gitee.com/mindspore/mindscience/blob/master/MindFlow/applications/data_mechanism_fusion/variant_linear_coe_pde_net/pde_net.ipynb)

## Quick Start

### Training Method 1: Call the `train.py` script on the command line

python train.py --config_file_path ./pde_net.yaml --device_target Ascend --device_id 0 --mode GRAPH --save_graphs False --save_graphs_path ./summary

Among them,

`--config_file_path` represents the parameter and path control file, default './pde_net.yaml'

`--device_target` represents the type of computing platform used, which can be selected as 'Ascend' or 'GPU', default 'Ascend';

`--device_id` represents the calculation card number used, which can be filled in according to the actual situation, default 0;

`--mode` represents the running mode, 'GRAPH' indicates the static Graphical model, 'PYNATIVE' indicates the dynamic Graphical model, default 'GRAPH';

`--save_graphs` represents whether to save the calculation graph, default 'False';

`--save_graphs_path` represents the path where the calculation graph is saved, default './summary';

### Training Method 2: Running Jupyter Notebook

You can run training and validation code line by line using both the [Chinese version](pde_net_CN.ipynb) and the [English version](pde_net.ipynb) of Jupyter Notebook.

## Contributor

liulei277