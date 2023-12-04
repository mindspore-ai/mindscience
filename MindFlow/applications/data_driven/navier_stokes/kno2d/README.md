# 2D Navier-Stokes Equation

## Overview

## Navier-Stokes equation

Navier-Stokes equation is a classical equation in computational fluid dynamics. It is a set of
partial differential equations describing the conservation of fluid momentum, called N-S equation
for short. Its vorticity form in two-dimensional incompressible flows is as follows:

$$
\partial_t w(x, t)+u(x, t) \cdot \nabla w(x, t)=\nu \Delta w(x, t)+f(x), \quad x \in(0,1)^2, t \in(0, T]
$$

$$
\nabla \cdot u(x, t)=0, \quad x \in(0,1)^2, t \in[0, T]
$$

$$
w(x, 0)=w_0(x), \quad x \in(0,1)^2
$$

where $u$ is the velocity field, $w=\nabla \times u$ is the vorticity, $w_0(x)$ is the initial
vorticity, $\nu$ is the viscosity coefficient, $f(x)$ is the forcing function.

We aim to solve two-dimensional incompressible N-S equation by learning the operator mapping from
each time step to the next time step:

$$
w_t \mapsto w(\cdot, t+1)
$$

![](images/kno.jpg)

[See More](./KNO2D.ipynb)

## Train

### Run Option 1: Call `train.py` from command line

```shell
python --mode GRAPH --save_graphs_path ./graphs --device_target Ascend --device_id 0 --config_file_path ./configs/kno2d.yaml
```

where:

`--mode` is the running mode. 'GRAPH' indicates static graph mode. 'PYNATIVE' indicates dynamic graph mode. Default 'GRAPH'.

`--device_target` indicates the computing platform. You can choose 'Ascend' or 'GPU'. Default 'Ascend'.

`--device_id` indicates the index of NPU or GPU. Default 0.

`--config_file_path` indicates the path of the parameter file. Default './configs/kno2d.yaml'；

### Run Option 2: Run Jupyter Notebook

You can use [Chinese](./KNO2D_CN.ipynb) or [English](./KNO2D.ipynb) Jupyter Notebook to run the training and evaluation code line-by-line.

## Results

Take 1 samples, and do 10 consecutive steps of prediction. Visualize the prediction as follows.

![Inference Error](images/result.gif)

## Performance

| Parameter               | Ascend               | GPU                |
|:----------------------:|:--------------------------:|:---------------:|
| Hardware                | Ascend 32G           | NVIDIA V100 32G    |
| MindSpore version       | >=2.1.0              | >=2.1.0                   |
| dataset                 | [2D Navier-Stokes Equation Dataset](https://download-mindspore.osinfra.cn/mindscience/mindflow/dataset/applications/data_driven/navier_stokes/)      | [2D Navier-Stokes Equation Dataset](https://download-mindspore.osinfra.cn/mindscience/mindflow/dataset/applications/data_driven/navier_stokes/)                   |
| Parameters              | 4e5                  | 4e5                   |
| Train Config            | batch_size=19, steps_per_epoch=1000, epochs=50 | batch_size=19, steps_per_epoch=1000, epochs=50 |
| Evaluation Config       | batch_size=19      | batch_size=19                |
| Optimizer               | Adam                 | Adam                   |
| Train Loss(MSE)         | 0.2                 | 0.2             |
| Evaluation Error(RMSE)  | 0.03                | 0.03              |
| Speed(ms/step)          | 30                  | 40                 |

## Contributor

gitee id：[dyonghan](https://gitee.com/dyonghan)

email: dyonghan@qq.com