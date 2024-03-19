# MindSpore Flow Release Notes

[查看中文](./RELEASE_CN.md)

MindSpore Flow is a flow simulation suite developed based on MindSpore. It supports AI flow simulation in industries such as aerospace, ship manufacturing, and energy and power. It aims to provide efficient and easy-to-use AI computing flow simulation software for industrial research engineers, university professors, and students.

## MindSpore Flow 0.2.0 Release Notes

### Major Feature and Improvements

#### Data Driven

- [STABLE] [Airfoil2D_Unsteady](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/data_driven/airfoil/2D_unsteady): Transonic airfoil flow is simulated by data-driven methods (using FNO2D and Unet2D).
- [STABLE] [API-FNO1D/2D/3D](https://gitee.com/mindspore/mindscience/blob/master/MindFlow/mindflow/cell/neural_operators/fno.py): FNO1D, FNO2D and FNO3D APIs are refactored to improve the commonality. "Channels_last" and "channels_first" input formats are supported. Activation functions can be set respectively. Users can set compute data type of SpectralConvDft and FNO-skip.  Hyper parameters of projection and lifting layers, residual connection and positional embedding are supported.
- [STABLE] [API-UNet2D](https://gitee.com/mindspore/mindscience/blob/master/MindFlow/mindflow/cell/unet2d.py): UNet2D API are refactored. Users can define the improving and reducing of UpConv and DownCov by 'base_channels'. Data formats of 'NCHW' and 'NHWC' are supported.

#### Data-Mechanism Fusion

- [STABLE] [API-Percnn](https://gitee.com/mindspore/mindscience/blob/master/MindFlow/mindflow/cell/neural_operators/percnn.py): The percnn API is added to learn the spatiotemporal evolution rules of physical fields on coarse grids through the recursive convolutional neural network. By default, the input of two physical components is supported. The number of conv layers and kernel size can be customized to implement applications on different physical phenomena.
- [STABLE] [PeRCNN-gsrd3d](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/data_mechanism_fusion/percnn/gsrd_3d): Add case of solving 3d GS reaction-diffusion equation by PeRCNN.

#### Physics Driven

- [STABLE] [Boltzmann](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/physics_driven/boltzmann): Boltzmann equation with D1V3-BGK and secondary collision term is solved. The relevant papers are published in [SIAM Journal on Scientific Computing](https://www.siam.org/publications/journals/siam-journal-on-scientific-computing-sisc).
- [STABLE] [Periodic Hill](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/physics_driven/navier_stokes/periodic_hill): Periodic hill flow are solved by PINNs.
- [STABLE] [Possion](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/physics_driven/poisson/continuous): Poisson equations with periodic and robin boundary conditions are solved by PINNs.
- [RESEARCH] [Cma_Es_Mgda](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/research/cma_es_mgda): Add CMA-ES and Multi-objective Gradient Optimization Algorithm(mgda) to solve PINNs.
- [RESEARCH] [Moe_Pinns](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/research/moe_pinns): Support MOE-PINNs.
- [RESEARCH] [Allen-Cahn](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/research/allen_cahn): Allen-Cahn equation is solved by PINNs.

### Contributors

Thanks to the following developers for their contributions:

hsliu_ustc, Yi_zhang95, zwdeng, liulei277, chengzrz, mengqinghe0909, xingzhongfan, jiangchenglin3, positive-one, yezhenghao2023, lunhao2023, lin109, xiaoruoye, b_rookie, Marc-Antoine-6258, yf-Li21, lixin07, ddd000g, huxin2023, leiyixiang1, dyonghan, huangxiang360729, liangjiaming2023, yanglin2023

Contributions to the project in any form are welcome!

## MindSpore Flow 0.1.0 Release Notes

### Major Feature and Improvements

#### Data Driven

- [STABLE] [CAE-LSTM](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/research/cae_lstm) : Support data-driven implementation of convolutional autoencoder-long short memory neural network for processing unsteady compressible flow.
- [STABLE] [Move Boundary Hdnn](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/research/move_boundary_hdnn) : Support data-driven implementation of HDNN network for solving unsteady flow field problems with moving boundaries.

#### Data-Mechanism Fusion

- [STABLE] [PeRCNN](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/data_mechanism_fusion/percnn) : Support physical encoded recursive Convolutional neural network (PeRCNN).

#### Physics Driven

- [STABLE] [Boltzmann](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/physics_driven/boltzmann) : Support PINNs method for solving Boltzmann equations.
- [STABLE] [Poisson with Point Source](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/physics_driven/poisson/point_source) : Support PINNs method to solve Poisson's equation with point source.

## MindSpore Flow 0.1.0.rc1 Release Notes

### Major Features and Improvements

#### Data Driven

- [STABLE] [KNO](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/data_driven/navier_stokes/kno2d): Provide the Kupmann KNO neural operator to improve the simulation accuracy of NS equations.
- [STABLE] [DongFang·YuFeng](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/data_driven/airfoil/2D_steady): Provide a large model of Dongfang Yufeng, supporting end-to-end rapid simulation of airfoils.

## MindSpore Flow 0.1.0-alpha Release Notes

### Major Features and Improvements

#### Data Driven

- [STABLE] Various neural networks are supported, including fully connected networks, residual networks, Fourier neural operators and Vision Transformer. Dataset merging and multiple data formats are supported. High level API is provided for training and evaluation. Multiple learning rates and losses are supported.

#### Data-Mechanism Fusion

- [STABLE] [PDE-Net](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/data_mechanism_fusion/pde_net): A physics plus data driven deep learning method, PDE-Net, is provided for unsteady flow field prediction and regression of PDEs.

#### Physics Driven

- [STABLE] Solve partial differential equations (PDEs) based on physics informed neural network. PDEs and basic equations can be defined by sympy. Users can calculate the Hessian and Jacobian matrix of network output to input. Basic geometrics, time domains and their operations are supported, which can be used for sampling within the geometric region and on the boundary.

#### Differentiable CFD Solver

- [STABLE] An end-to-end differentiable compressible CFD solver, MindFlow-CFD, is introduced. WENO5 reconstruction, Rusanov flux, Runge-Kutta integrator are supported. Symmetry, periodic, solid wall and Neumann boundary conditions are supported.

### Contributors

Thanks goes to these wonderful people:

hsliu_ustc, Yi_zhang95, zwdeng, liulei277, chengzrz, liangjiaming2023, yanglin2023

Contributions to the project in any form are welcome!