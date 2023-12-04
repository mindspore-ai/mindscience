ENGLISH | [简体中文](README_CN.md)

# 1D Burgers

## Overview

Burgers' equation is a nonlinear partial differential equation that simulates the propagation and reflection of shock waves. It is widely used in the fields of fluid mechanics, nonlinear acoustics, gas dynamics et al. It is named after Johannes Martins Hamburg (1895-1981). In this case, MindFlow fluid simulation suite is used to solve the Burgers' equation in one-dimensional viscous state based on the physical-driven PINNs (Physics Informed Neural Networks) method.

## QuickStart

You can download dataset from [physics_driven/burgers_pinns/dataset](https://download.mindspore.cn/mindscience/mindflow/dataset/applications/physics_driven/burgers_pinns/dataset/) for model evaluation. Save these dataset at `./dataset`.

### Run Option 1: Call `train.py` from command line

```shell
python train.py --config_file_path ./configs/burgers.yaml --mode GRAPH --device_target Ascend --device_id 0
```

where:

`--config_file_path` indicates the path of the parameter file. Default './configs/burgers.yaml'；

`--mode` is the running mode. 'GRAPH' indicates static graph mode. 'PYNATIVE' indicates dynamic graph mode. Default 'GRAPH'.

`--device_target` indicates the computing platform. You can choose 'Ascend' or 'GPU'. Default 'Ascend'.

`--device_id` indicates the index of NPU or GPU. Default 0.

### Run Option 2: Run Jupyter Notebook

You can use [Chinese](./burgers1D_CN.ipynb) or [English](./burgers1D.ipynb)Jupyter Notebook to run the training and evaluation code line-by-line.

## Results

![Burgers PINNs](images/result.jpg)

## Performance

| Parameter               | Ascend               | GPU                |
|:----------------------:|:--------------------------:|:---------------:|
| Hardware                | Ascend 32G           | NVIDIA V100 32G    |
| MindSpore version       | >=2.0.0                | >=2.0.0                   |
| dataset                 | [Burgers Dataset](https://download.mindspore.cn/mindscience/mindflow/dataset/applications/physics_driven/burgers_pinns/)      | [Burgers Dataset](https://download.mindspore.cn/mindscience/mindflow/dataset/applications/physics_driven/burgers_pinns/)                   |
| Parameters              | 6e4                  | 6e4                   |
| Train Config            | batch_size=8192, steps_per_epoch=1, epochs=15000 | batch_size=8192, steps_per_epoch=1, epochs=15000 |
| Evaluation Config       | batch_size=8192, steps=4      | batch_size=8192, steps=4               |
| Optimizer               | Adam                 | Adam                   |
| Train Loss(MSE)         | 0.001                | 0.0001             |
| Evaluation Error(RMSE)  | 0.010                | 0.008              |
| Speed(ms/step)          | 10                   | 40                |

## Contributor

gitee id：[liulei277](https://gitee.com/liulei277)

email: liulei2770919@163.com
