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

You can download dataset from  [data_driven/navier_stokes_3d/](https://download.mindspore.cn/mindscience/mindflow/dataset/applications/data_driven/navier_stokes_3d/). Save these dataset at `./dataset`.

### Run Option 1: Call `train.py` from command line

```shell
export PYTHONPATH=$(cd ../../../../../ && pwd):$PYTHONPATH
python train.py --config_file_path ./configs/fno3d.yaml --mode GRAPH --device_target Ascend --device_id 0
```

where:

`--config_file_path` indicates the path of the parameter file. Default './configs/fno3d.yaml'；

`--mode` is the running mode. 'GRAPH' indicates static graph mode. 'PYNATIVE' indicates dynamic graph mode. Default 'GRAPH'.

`--device_target` indicates the computing platform. You can choose 'Ascend' or 'GPU'. Default 'Ascend'.

`--device_id` indicates the index of NPU or GPU. Default 0.

### Run Option 2: Run Jupyter Notebook

You can use [Chinese](./FNO3D_CN.ipynb) or [English](./FNO3D.ipynb) Jupyter Notebook to run the training and evaluation code line-by-line.

## Results

get the prediction as follows:

```text
epoch: 141 train loss: 0.014623 epoch time: 12.42s step time: 0.12s
epoch: 142 train loss: 0.012681 epoch time: 12.46s step time: 0.12s
epoch: 143 train loss: 0.022258 epoch time: 12.48s step time: 0.12s
epoch: 144 train loss: 0.014924 epoch time: 12.48s step time: 0.12s
epoch: 145 train loss: 0.015092 epoch time: 12.54s step time: 0.13s
epoch: 146 train loss: 0.013113 epoch time: 12.45s step time: 0.12s
epoch: 147 train loss: 0.013670 epoch time: 12.42s step time: 0.12s
epoch: 148 train loss: 0.011905 epoch time: 12.53s step time: 0.13s
epoch: 149 train loss: 0.018703 epoch time: 12.48s step time: 0.12s
epoch: 150 train loss: 0.013277 epoch time: 12.45s step time: 0.12s
loss: 0.013277
step: 150, time elapsed: 12457.710981369019ms
================================Start Evaluation================================
mean rms_error: 0.016683187
predict total time: 5.692582845687866 s
=================================End Evaluation=================================
```

## Performance

| Parameter               | Ascend               |
|:----------------------:|:--------------------------:|
| Hardware                | Ascend 32G           |
| MindSpore version       | 2.7.0               |
| dataset                 | [3D Navier-Stokes Equation Dataset](https://download.mindspore.cn/mindscience/mindflow/dataset/applications/data_driven/navier_stokes_3d/)      |
| Parameters              | 6.5e6                  |
| Train Config            | batch_size=10, steps_per_epoch=100, epochs=150 |
| Evaluation Config       | batch_size=1      |
| Optimizer               | Adam                 |
| Train Loss(MSE)         | 0.01                |
| Evaluation Error(RMSE)  | 0.02                |
| Speed(ms/epoch)          | 12179                   |

## Contributor

gitee id：[chengzrz](https://gitee.com/chengzrz),[huangwangwen2025](https://gitee.com/huangwangwen2025)

email: czrzrichard@gmail.com,wangwen@isrc.iscas.ac.cn
