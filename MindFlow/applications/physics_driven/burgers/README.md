ENGLISH | [简体中文](README_CN.md)

# 1D Burgers

## Overview

Burgers' equation is a nonlinear partial differential equation that simulates the propagation and reflection of shock waves. It is widely used in the fields of fluid mechanics, nonlinear acoustics, gas dynamics et al. It is named after Johannes Martins Hamburg (1895-1981). In this case, MindFlow fluid simulation suite is used to solve the Burgers' equation in one-dimensional viscous state based on the physical-driven PINNs (Physics Informed Neural Networks) method.

## QuickStart

You can download dataset from [physics_driven/burgers_pinns/dataset](https://download.mindspore.cn/mindscience/mindflow/dataset/applications/physics_driven/burgers_pinns/dataset/) for model evaluation. Save these dataset at `./dataset`.

### Run Option 1: Call `train.py` from command line

```shell
python --mode GRAPH --save_graphs_path ./graphs --device_target Ascend --device_id 0 --config_file_path ./burgers_cfg.yaml
```

where:

`--mode` is the running mode. 'GRAPH' indicates static graph mode. 'PYNATIVE' indicates dynamic graph mode. You can refer to [MindSpore official website](https://www.mindspore.cn/docs/en/r2.0.0-alpha/design/dynamic_graph_and_static_graph.html) for details.Default 'GRAPH'.

`--save_graphs` indicates whether to save the computational graph. Default 'False'.

`--save_graphs_path` indicates the path to save the computational graph. Default './graphs'.

`--device_target` indicates the computing platform. You can choose 'Ascend' or 'GPU'. Default 'Ascend'.

`--device_id` indicates the index of NPU or GPU. Default 0.

`--config_file_path` indicates the path of the parameter file. Default './burgers_cfg.yaml'；

### Run Option 2: Run Jupyter Notebook

You can use [Chinese](https://gitee.com/mindspore/mindscience/blob/master/MindFlow/applications/physics_driven/burgers/burgers1D_CN.ipynb) or [English](https://gitee.com/mindspore/mindscience/blob/master/MindFlow/applications/physics_driven/burgers/burgers1D.ipynb)Jupyter Notebook to run the training and evaluation code line-by-line.

## Results

![Burgers PINNs](images/result.jpg)

## Contributor

liulei277
