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

## Train

The training log, training loss, and validation loss is shown as follows.

```text
epoch: 1, time cost: 55.562426, recons loss: 0.467314, pred loss: 0.237300
epoch: 2, time cost: 32.804436, recons loss: 0.175188, pred loss: 0.050888
epoch: 3, time cost: 32.946971, recons loss: 0.167865, pred loss: 0.041778
epoch: 4, time cost: 33.064430, recons loss: 0.170181, pred loss: 0.038075
epoch: 5, time cost: 32.907211, recons loss: 0.171853, pred loss: 0.035849
epoch: 6, time cost: 33.799230, recons loss: 0.173322, pred loss: 0.034017
epoch: 7, time cost: 32.612255, recons loss: 0.174376, pred loss: 0.032719
epoch: 8, time cost: 32.896673, recons loss: 0.175445, pred loss: 0.031596
epoch: 9, time cost: 33.907305, recons loss: 0.176131, pred loss: 0.030644
epoch: 10, time cost: 33.175130, recons loss: 0.176701, pred loss: 0.029969
Eval epoch: 10, recons loss: 0.23137304687500002, relative pred loss: 0.03798459614068269

...

epoch: 41, time cost: 32.962233, recons loss: 0.185430, pred loss: 0.017872
epoch: 42, time cost: 33.296847, recons loss: 0.185595, pred loss: 0.017749
epoch: 43, time cost: 33.803700, recons loss: 0.185646, pred loss: 0.017651
epoch: 44, time cost: 32.776349, recons loss: 0.185723, pred loss: 0.017564
epoch: 45, time cost: 33.377666, recons loss: 0.185724, pred loss: 0.017497
epoch: 46, time cost: 33.228983, recons loss: 0.185827, pred loss: 0.017434
epoch: 47, time cost: 33.244342, recons loss: 0.185854, pred loss: 0.017393
epoch: 48, time cost: 33.211263, recons loss: 0.185912, pred loss: 0.017361
epoch: 49, time cost: 35.656644, recons loss: 0.185897, pred loss: 0.017349
epoch: 50, time cost: 33.527458, recons loss: 0.185899, pred loss: 0.017344
Eval epoch: 50, recons loss: 0.2389616699218751, relative pred loss: 0.03355878115445375
```

## Test

Take 1 samples, and do 10 consecutive steps of prediction. Visualize the prediction as follows.

![](images/result.gif)

## Contributor

dyonghan
