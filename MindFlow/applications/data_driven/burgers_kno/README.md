# 1D Burgers Equation

## Overview

Burgers' equation is a nonlinear partial differential equation that simulates the propagation and
reflection of shock waves. It is widely used in the fields of fluid mechanics, nonlinear acoustics,
gas dynamics et al. It is named after Johannes Martins Hamburg (1895-1981).

The 1-d Burgers’ equation applications include modeling the one dimensional flow of a viscous fluid.
It takes the form

$$
\partial_t u(x, t)+\partial_x (u^2(x, t)/2)=\nu \partial_{xx} u(x, t), \quad x \in(0,1), t \in(0, 1]
$$

$$
u(x, 0)=u_0(x), \quad x \in(0,1)
$$

where $u$ is the velocity field, $u_0$ is the initial condition and $\nu$ is the viscosity coefficient.

We aim to learn the operator mapping the initial condition to the solution at time one:

$$
u_0 \mapsto u(\cdot, 1)
$$

## Train

The training log, training loss, and validation loss is shown as follows.

```text
Data preparation finished
input_path:  (1000, 1024, 1)
label_path:  (1000, 1024)
Data preparation finished
input_path:  (200, 1024, 1)
label_path:  (200, 1024)
name:KNO1D_channels:32_modes:64_depths:4_resolution:1024
./summary_dir/name:KNO1D_channels:32_modes:64_depths:4_resolution:1024
epoch: 1, time cost: 21.623394, recons loss: 0.295491, pred loss: 0.085095
epoch: 2, time cost: 2.527564, recons loss: 0.161590, pred loss: 0.002210
epoch: 3, time cost: 2.598942, recons loss: 0.027091, pred loss: 0.000967
epoch: 4, time cost: 2.517585, recons loss: 0.000775, pred loss: 0.000502
epoch: 5, time cost: 2.573697, recons loss: 0.000057, pred loss: 0.000282
epoch: 6, time cost: 2.562175, recons loss: 0.000048, pred loss: 0.000244
epoch: 7, time cost: 2.491402, recons loss: 0.000048, pred loss: 0.000214
epoch: 8, time cost: 2.530793, recons loss: 0.000048, pred loss: 0.000237
epoch: 9, time cost: 2.504641, recons loss: 0.000048, pred loss: 0.000231
epoch: 10, time cost: 2.544668, recons loss: 0.000049, pred loss: 0.000227
---------------------------start evaluation-------------------------
Eval epoch: 10, recons loss: 4.7650219457864295e-05, relative pred loss: 0.01156728882342577
---------------------------end evaluation---------------------------

...

epoch: 91, time cost: 2.539794, recons loss: 0.000042, pred loss: 0.000006
epoch: 92, time cost: 2.521379, recons loss: 0.000042, pred loss: 0.000007
epoch: 93, time cost: 3.142074, recons loss: 0.000042, pred loss: 0.000006
epoch: 94, time cost: 2.569737, recons loss: 0.000042, pred loss: 0.000006
epoch: 95, time cost: 2.545627, recons loss: 0.000042, pred loss: 0.000006
epoch: 96, time cost: 2.568123, recons loss: 0.000042, pred loss: 0.000006
epoch: 97, time cost: 2.547843, recons loss: 0.000042, pred loss: 0.000006
epoch: 98, time cost: 2.709663, recons loss: 0.000042, pred loss: 0.000006
epoch: 99, time cost: 2.529918, recons loss: 0.000042, pred loss: 0.000006
epoch: 100, time cost: 2.502929, recons loss: 0.000042, pred loss: 0.000006
---------------------------start evaluation-------------------------
Eval epoch: 100, recons loss: 4.1765865171328186e-05, relative pred loss: 0.004054672718048095
---------------------------end evaluation---------------------------
```

## Test

Take 6 samples, and do 10 consecutive steps of prediction. Visualize the prediction as follows.

![](images/result.jpg)

## Contributor

dyonghan
