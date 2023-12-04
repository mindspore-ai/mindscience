# Solve Navier-Stokes Equation by FNO-3D

## Overview

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

We aim to solve two-dimensional incompressible N-S equation by learning the Fourier Operator mapping from
each time step to the next time step:

$$
w_t \mapsto w(\cdot, t+1)
$$

![Fourier Neural Operator model structure](images/FNO.png)

## QuickStart

You can download dataset from  [data_driven/navier_stokes3d/](https://download.mindspore.cn/mindscience/mindflow/dataset/applications/data_driven/navier_stokes_3d/). Save these dataset at `./dataset`.

### Run Option 1: Call `train.py` from command line

```shell
python train.py --config_file_path ./configs/fno3d.yaml --mode GRAPH --device_target Ascend --device_id 0
```

where:

`--config_file_path` indicates the path of the parameter file. Default './configs/fno3d.yaml'；

`--mode` is the running mode. 'GRAPH' indicates static graph mode. 'PYNATIVE' indicates dynamic graph mode. Default 'GRAPH'.

`--device_target` indicates the computing platform. You can choose 'Ascend' or 'GPU'. Default 'Ascend'.

`--device_id` indicates the index of NPU or GPU. Default 0.

### Run Option 2: Run Jupyter Notebook

You can use [Chinese](./FNO3D_CN.ipynb) or [English](./FNO3D.ipynb)Jupyter Notebook to run the training and evaluation code line-by-line.

## Results

get the prediction as follows.

epoch: 141 train loss: 0.019202268 epoch time: 19.43s<br>
epoch: 142 train loss: 0.01606355 epoch time: 19.24s<br>
epoch: 143 train loss: 0.027023558 epoch time: 19.28s<br>
epoch: 144 train loss: 0.017647993 epoch time: 19.36s<br>
epoch: 145 train loss: 0.017342577 epoch time: 19.29s<br>
epoch: 146 train loss: 0.01618614 epoch time: 19.29s<br>
epoch: 147 train loss: 0.018703096 epoch time: 19.31s<br>
epoch: 148 train loss: 0.014269419 epoch time: 19.29s<br>
epoch: 149 train loss: 0.02165638 epoch time: 19.32s<br>
epoch: 150 train loss: 0.016264874 epoch time: 19.32s<br>
loss: 0.016265<br>
step: 150, time elapsed: 19317.4147605896ms<br>
================================Start Evaluation================================<br>
mean rms_error: 0.01986466<br>
predict total time: 8.004547119140625 s<br>
=================================End Evaluation=================================

## Performance

| Parameter               | Ascend               | GPU                |
|:----------------------:|:--------------------------:|:---------------:|
| Hardware                | Ascend 32G           | NVIDIA V100 32G    |
| MindSpore version       | >=2.1.0                | >=2.1.0                   |
| dataset                 | [2D Navier-Stokes Equation Dataset](https://download-mindspore.osinfra.cn/mindscience/mindflow/dataset/applications/data_driven/navier_stokes/)      | [2D Navier-Stokes Equation Dataset](https://download-mindspore.osinfra.cn/mindscience/mindflow/dataset/applications/data_driven/navier_stokes/)                   |
| Parameters              | 6.5e6                  | 6.5e6                    |
| Train Config            | batch_size=10, steps_per_epoch=1, epochs=150 | batch_size=10, steps_per_epoch=1, epochs=150 |
| Evaluation Config       | batch_size=1      | batch_size=1               |
| Optimizer               | Adam                 | Adam                   |
| Train Loss(MSE)         | 0.02                | 0.02             |
| Evaluation Error(RMSE)  | 0.02                | 0.02              |
| Speed(ms/step)          | 20000                   | 50000                 |

## Contributor

gitee id：[chengzrz](https://gitee.com/chengzrz)

email: czrzrichard@gmail.com
