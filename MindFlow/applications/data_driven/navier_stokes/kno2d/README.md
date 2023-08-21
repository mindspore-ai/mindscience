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

[See More](https://gitee.com/mindspore/mindscience/blob/master/MindFlow/applications/data_driven/navier_stokes/kno2d/KNO2D.ipynb)

## Results Display

Take 1 samples, and do 10 consecutive steps of prediction. Visualize the prediction as follows.

[Inference Error](images/result.gif)

## Performance

|        Parameter         |        Ascend               |    GPU       |
|:----------------------:|:--------------------------:|:---------------:|
|     Hardware         |     Ascend 910A, 32G；CPU: 2.6GHz, 192 cores      |      NVIDIA V100 32G       |
|     MindSpore版本   |        2.1            |      2.1       |
|        train loss      |       0.17                |       0.16       |
|        valid loss      |        3e-2               |       3e-2    |
|        speed          |     25s/epoch        |    160s/epoch  |

## Contributor

gitee id：[dyonghan](https://gitee.com/dyonghan)

email: dyonghan@qq.com