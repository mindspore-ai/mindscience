ENGLISH | [简体中文](README_CN.md)

# PDE-Net for Convection-Diffusion Equation

## Overview

PDE-Net is a feedforward deep network proposed by Zichao Long et al. to learn partial differential equations from data, predict the dynamic characteristics of complex systems accurately and uncover potential PDE models. The basic idea of PDE-Net is to approximate differential operators by learning convolution kernels (filters). Neural networks or other machine learning methods are applied to fit unknown nonlinear responses. Numerical experiments show that the model can identify the observed dynamical equations and predict the dynamical behavior over a relatively long period of time, even in noisy environments. More information can be found in [PDE-Net: Learning PDEs from Data](https://arxiv.org/abs/1710.09668).

![coe label benchmark](images/coe_label_benchmark.png)

![coe trained step-1](images/coe_trained_step-1.png)

![result](images/result.jpg)

![extrapolation](images/extrapolation.jpg)

[See More](https://gitee.com/mindspore/mindscience/blob/master/MindFlow/applications/data_mechanism_fusion/variant_linear_coe_pde_net/pde_net.ipynb)

## Contributor

liulei277