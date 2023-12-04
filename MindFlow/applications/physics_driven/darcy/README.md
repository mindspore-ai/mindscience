ENGLISH | [简体中文](README_CN.md)

# 2D Stabilized Darcy Problem

## Overview

Darcy equation is a second-order, elliptic PDE (partial differential equation), which describes the flow through a porous medium at low speed. It is widely used in hydraulic engineering and petroleum engineering. The Darcy equation was originally formulated by Henry Darcy on the basis of experimental results of permeability  experiments in sandy soil, and later derived from the Navier-Stokes equation by Stephen Whitaker via the homogenization method. This case uses PINNs to solve the two-dimensional stabilized Darcy equation based on PINNs method.

## Results

![Darcy PINNs](images/result.png)

## Performance

| Parameter               | Ascend               | GPU                |
|:----------------------:|:--------------------------:|:---------------:|
| Hardware                | Ascend 32G           | NVIDIA V100 32G    |
| MindSpore version       | >=2.0.0                | >=2.0.0                   |
| Parameters              | 6e4                  | 6e4                   |
| Train Config            | batch_size=8192, steps_per_epoch=8, epochs=4000 | batch_size=8192, steps_per_epoch=8, epochs=4000 |
| Evaluation Config       | batch_size=8192      | batch_size=8192               |
| Optimizer               | Adam                 | Adam                   |
| Train Loss(MSE)         | 0.001                | 0.025              |
| Evaluation Error(RMSE)  | 0.01                 | 0.01               |
| Speed(ms/step)          | 150                  | 400                |

## Contributor

haojiwei