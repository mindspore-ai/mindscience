ENGLISH | [简体中文](README_CN.md)

# 2D Cylinder Flow

## Overview

### Problem Description

Flow past cylinder problem is a two-dimensional low velocity steady flow around a cylinder which is only related to the `Re` number. When `Re` is less than or equal to 1, the inertial force in the flow field is secondary to the viscous force, the streamlines in the upstream and downstream directions of the cylinder are symmetrical, and the drag coefficient is approximately inversely proportional to `Re` . The flow around this `Re` number range is called the Stokes zone; With the increase of `Re` , the streamlines in the upstream and downstream of the cylinder gradually lose symmetry. This special phenomenon reflects the peculiar nature of the interaction between the fluid and the surface of the body. Solving flow past a cylinder is a classical problem in hydromechanics. This case uses PINNs to solve the wake flow field around a cylinder.

### Technical Path

## Problem Description

The Navier-Stokes equation, referred to as `N-S` equation, is a classical partial differential equation in the field of fluid mechanics. In the case of viscous incompressibility, the dimensionless `N-S` equation has the following form:

$$
\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} = 0
$$

$$
\frac{\partial u} {\partial t} + u \frac{\partial u}{\partial x} + v \frac{\partial u}{\partial y} = - \frac{\partial p}{\partial x} + \frac{1} {Re} (\frac{\partial^2u}{\partial x^2} + \frac{\partial^2u}{\partial y^2})
$$

$$
\frac{\partial v} {\partial t} + u \frac{\partial v}{\partial x} + v \frac{\partial v}{\partial y} = - \frac{\partial p}{\partial y} + \frac{1} {Re} (\frac{\partial^2v}{\partial x^2} + \frac{\partial^2v}{\partial y^2})
$$

where `Re` stands for Reynolds number.

In this case, the PINNs method is used to learn the mapping from the location and time to flow field quantities to solve the `N-S` equation.

$$
(x, y, t) \mapsto (u, v, p)
$$

## QuickStart

You can download dataset from [physics_driven/flow_past_cylinder/dataset/](https://download.mindspore.cn/mindscience/mindflow/dataset/applications/physics_driven/flow_past_cylinder/dataset/). Save these dataset at `./dataset`.

### Run Method 1: Call `train.py` from command line

```shell
python train.py --config_file_path ./configs/cylinder_flow.yaml --mode GRAPH --device_target Ascend --device_id 0
```

where:

`--config_file_path` indicates the path of the parameter file. Default './configs/cylinder_flow.yaml'；

`--device_target` indicates the computing platform. You can choose 'Ascend' or 'GPU'. Default 'Ascend'.

`--device_id` indicates the index of NPU or GPU. Default 0.

`--mode` is the running mode. 'GRAPH' indicates static graph mode. 'PYNATIVE' indicates dynamic graph mode.

### Run Method 2: Run Jupyter Notebook

You can run the training and validation code line by line using the Chinese or English version of the Jupyter Notebook [Chinese Version](./navier_stokes2D_CN.ipynb) and [English Version](./navier_stokes2D_CN.ipynb).

## Results Display

![cylinder flow](images/cylinder_flow.gif)

![flow](images/image-flow.png)

![Time Error](images/TimeError_epoch5000.png)

## Performance

| Parameter               | Ascend               | GPU                |
|:----------------------:|:--------------------------:|:---------------:|
| Hardware                | Ascend 32G           | NVIDIA V100 32G    |
| MindSpore version       | >=2.0.0                | >=2.0.0                   |
| dataset                 | [2D Cylinder Fow Dataset](https://download.mindspore.cn/mindscience/mindflow/dataset/applications/physics_driven/burgers_pinns/)      | [2D Cylinder Fow Dataset](https://download.mindspore.cn/mindscience/mindflow/dataset/applications/physics_driven/burgers_pinns/)                    |
| Parameters              | 1.3e5                  | 1.3e5                   |
| Train Config            | batch_size=8192, steps_per_epoch=2, epochs=12000 | batch_size=8192, steps_per_epoch=2, epochs=12000 |
| Evaluation Config       | batch_size=8192      | batch_size=8192               |
| Optimizer               | Adam                 | Adam                   |
| Train Loss(MSE)         | 9e-5                 | 4e-5             |
| Evaluation Error(RMSE)  | 2e-2                 | 2e-2              |
| Speed(ms/step)          | 100                  | 450                |

## Contributor

gitee id：[liulei277](https://gitee.com/liulei277)

email: liulei2770919@163.com
