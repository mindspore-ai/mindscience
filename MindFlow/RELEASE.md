# MindSpore Flow Release Notes

[查看中文](./RELEASE_CN.md)

MindSpore Flow is a flow simulation suite developed based on MindSpore. It supports AI flow simulation in industries such as aerospace, ship manufacturing, and energy and power. It aims to provide efficient and easy-to-use AI computing flow simulation software for industrial research engineers, university professors, and students.

## MindFlow 0.1.0 Release Notes

### Major Feature and Improvements

#### Data Driven

- [STABLE] [`CAE-LSTM`](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/research/cae_lstm) : Support data-driven implementation of convolutional autoencoder-long short memory neural network for processing unsteady compressible flow.
- [STABLE] [`Move Boundary Hdnn`](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/research/move_boundary_hdnn) : Support data-driven implementation of HDNN network for solving unsteady flow field problems with moving boundaries.

#### Data-Mechanism Fusion

- [STABLE] [`PeRCNN`](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/data_mechanism_fusion/percnn) : Support physical encoded recursive Convolutional neural network (PeRCNN).

#### Physics Driven

- [STABLE] [`Boltzmann`](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/physics_driven/boltzmann) : Support PINNs method for solving Boltzmann equations.
- [STABLE] [`Poisson with Point Source`](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/physics_driven/poisson/point_source) : Support PINNs method to solve Poisson's equation with point source.

## MindSpore Flow 0.1.0.rc1 Release Notes

### Major Features and Improvements

#### Data Driven

- [STABLE] [`KNO`](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/data_driven/navier_stokes/kno2d): Provide the Kupmann KNO neural operator to improve the simulation accuracy of NS equations.
- [STABLE] [`DongFang·YuFeng`](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/data_driven/airfoil/2D_steady): Provide a large model of Dongfang Yufeng, supporting end-to-end rapid simulation of airfoils.

## MindFlow 0.1.0-alpha Release Notes

### Major Features and Improvements

#### Data Driven

- [STABLE] Various neural networks are supported, including fully connected networks, residual networks, Fourier neural operators and Vision Transformer. Dataset merging and multiple data formats are supported. High level API is provided for training and evaluation. Multiple learning rates and losses are supported.

#### Data-Mechanism Fusion

- [STABLE] [`PDE-Net`](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/data_mechanism_fusion/pde_net): A physics plus data driven deep learning method, PDE-Net, is provided for unsteady flow field prediction and regression of PDEs.

#### Physics Driven

- [STABLE] Solve partial differential equations (PDEs) based on physics informed neural network. PDEs and basic equations can be defined by sympy. Users can calculate the Hessian and Jacobian matrix of network output to input. Basic geometrics, time domains and their operations are supported, which can be used for sampling within the geometric region and on the boundary.

#### Differentiable CFD Solver

- [STABLE] An end-to-end differentiable compressible CFD solver, MindFlow-CFD, is introduced. WENO5 reconstruction, Rusanov flux, Runge-Kutta integrator are supported. Symmetry, periodic, solid wall and Neumann boundary conditions are supported.

### Contributors

Thanks goes to these wonderful people:

hsliu_ustc, Yi_zhang95, zwdeng, liulei277, chengzrz, liangjiaming2023, yanglin2023