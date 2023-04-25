# MindFlow Release Notes

MindFlow is a flow simulation suite developed based on MindSpore. It supports AI flow simulation in industries such as aerospace, ship manufacturing, and energy and power. It aims to provide efficient and easy-to-use AI computing flow simulation software for industrial research engineers, university professors, and students.

[查看中文](./RELEASE_CN.md)

## MindFlow 0.1.0.rc1 Release Notes

### Major Features and Improvements

#### Data Driven

- [STABLE] Provide the Kupmann KNO neural operator to improve the simulation accuracy of NS equations. Provide a large model of Dongfang Yufeng, supporting end-to-end rapid simulation of airfoils.

## MindFlow 0.1.0-alpha Release Notes

### Major Features and Improvements

#### Physics Driven

- [STABLE] Solve partial differential equations (PDEs) based on physics informed neural network. PDEs and basic equations can be defined by sympy. Users can calculate the Hessian and Jacobian matrix of network output to input. Basic geometrics, time domains and their operations are supported, which can be used for sampling within the geometric region and on the boundary.

#### Data Driven

- [STABLE] Various neural networks are supported, including fully connected networks, residual networks, Fourier neural operators and Vision Transformer. Dataset merging and multiple data formats are supported. High level API is provided for training and evaluation. Multiple learning rates and losses are supported.

#### Physics Plus Data Driven

- [STABLE] A physics plus data driven deep learning method, PDE-Net, is provided for unsteady flow field prediction and regression of PDEs.

#### Differentiable CFD Solver

- [STABLE] An end-to-end differentiable compressible CFD solver, MindFlow-CFD, is introduced. WENO5 reconstruction, Rusanov flux, Runge-Kutta integrator are supported. Symmetry, periodic, solid wall and Neumann boundary conditions are supported.

### Contributors

Thanks goes to these wonderful people:

yufan, wangzidong, liuhongsheng, zhouhongye, zhangyi, dengzhiwen, liulei, libokai, yangge, longzichao, yqiuu, haojiwei, leiyixiang
