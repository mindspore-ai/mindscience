# MindSpore Flow Release Notes

[查看中文](./RELEASE_CN.md)

MindSpore Flow is a flow simulation suite developed based on MindSpore. It supports AI flow simulation in industries such as aerospace, ship manufacturing, and energy and power under AI fluid simulation, AI aerodynamc design and AI flow control applications. It aims to provide efficient and easy-to-use AI computing flow simulation software for industrial research engineers, university professors, and students.

## MindSpore Flow 0.3.0 Release Notes

### Major Feature and Improvements

#### Data Driven

- [STABLE] [Burgers_SNO](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/data_driven/burgers/sno1d)/[Navier_Stokes_SNO2D](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/data_driven/navier_stokes/sno2d)/[Navier_Stokes_SNO3D](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/data_driven/navier_stokes/sno3d):  Applications sovling one-dimension Burgers Equation, two/three-dimension  Navier Stokes Equation by Spectral Neural Operator under data driven method are added.

- [STABLE] [API-SNO1D/2D/3D](https://gitee.com/mindspore/mindscience/blob/master/MindFlow/mindflow/cell/neural_operators/sno.py): Spectral Neural Operator (including SNO and U-SNO) APIs are added, utilizing polynomial transformations to transform computations into a spectral space similar to FNO architecture. Its advantage lies in effectively reducing system bias caused by aliasing errors.

- [STABLE] [API-Attention](https://gitee.com/mindspore/mindscience/blob/master/MindFlow/mindflow/cell/attention.py): Refactoring most commonly used  Transformer class networks such as Attention, MultiHeadAttention, AttentionBlock, and ViT network interfaces.

- [STABLE] [API-Diffusion](https://gitee.com/mindspore/mindscience/blob/master/MindFlow/mindflow/cell/diffusion.py): A complete set of training and inference interfaces for diffusion models are added with support of two mainstream diffusion methods of DDPM and DDIM. Meanwhile the entire process of diffusion model training and inference can be completed through the simple and easy-to-use interfaces of Diffusion Scheduler, Diffusion Trainer, Diffusion Pipeline, and Diffusion Transformer.

- [STABLE] [API-Refactor_Core](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/mindflow/core): Refactor of mindflow.core by fusion of mindflow.common, mindflow.loss and mindflow.operators.

- [RESEARCH] [CascadeNet](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/research/cascade_net): CascadeNet case is added, It uses surface pressure, Reynolds number, and a small number of wake velocity measurement points as inputs to predict the spatiotemporal field of cylinder wake pulsation velocity through a generative adversarial network with scale transfer topology structure.

- [RESEARCH] [MultiScaleGNN](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/research/multiscale_gnn): A multi-scale graph neural network case to solve the large-scale pressure Poisson equation is added, which supports the use of projection method (or fractional step method) to solve incompressible Navier Stokes equations.

- [RESEARCH] [TurbineUQ](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/research/turbine_uq): A case study of turbine stage flow field prediction and uncertainty optimization design is added with a combination of Monte Carlo method with deep learning methods to quantitative evaluation of uncertainty.

#### Data-Mechanism Fusion

- [STABLE] [PhyMPGN](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/data_mechanism_fusion/phympgn): An application of PhyMPGN, a physical equation solving model based on graph neural networks for the problem of flow around a cylinder is added. PhyMPGN can solve Burgers, FitzHugh-Nagumo, Gray-Scott and other equations in unstructured grids. Related [paper](http://arxiv.org/abs/2410.01337) has been received as ICLR 2025 Spotlight.

- [RESEARCH] [Heat_Conduction](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/research/heat_conduction):  A case study of steady-state heat conduction physics field prediction driven by data and physics is added.

- [RESEARCH] [SuperPosition](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/research/superposition): SDNO, an operator neural network based on the superposition principle, is added for predicting the temperature field of complex flow patterns in aircraft engine internal flow cascades.

#### Physics Driven

- [RESEARCH] [NSFNets](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/research/nsf_nets): Navier Stokes Flow Networks (NSFNets) are added. It is a [highly cited paper](https://www.sciencedirect.com/science/article/pii/S0021999120307257) for solving ill posed problems (such as partially missing boundary conditions or inversion problems.

#### Solver

- [STABLE] [CBS solver](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/cfd/acoustic): Application of CBS acoustic equation solver for solving two-dimensional acoustic equations in complex parameter fields is added. The CBS solver solves the acoustic equation in the frequency domain and has spectral accuracy in all spatial directions, with higher accuracy than the finite difference method. Reference: [Osnabrugge et al. 2016](https://linkinghub.elsevier.com/retrieve/pii/S0021999116302595)

#### Optimizer

- [STABLE] [API-AdaHessian second-order optimizer](https://gitee.com/mindspore/mindscience/blob/master/MindFlow/mindflow/core/optimizers.py): AdaHessian second-order optimizer based on the second-order information provided by the diagonal elements of the Hessian matrix for optimization calculations is added. Tests achieved a loss reduction over 20% compared with Adam under the same number of steps.

#### Foundation Model

- [RESEARCH] [PDEformer](https://github.com/functoreality/pdeformer-2): PDEformer supports to solve [one dimensional](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/pdeformer1d)/[two dimensional](https://github.com/functoreality/pdeformer-2) general partial differential equations with time with a superior of accuracy to domain model by foundation model under Zero-Shot occasions.

### Contributors

Thanks to the following developers for their contributions:

hsliu_ustc, gracezou, mengqinghe0909, Yi_zhang95, b_rookie, WhFanatic, xingzhongfan, juliagurieva, GQEm, chenruilin2024, ZYF00000, chenchao2024, wangqineng2024, BingyangWu-pkusms21, Bochengz, functoreality, huangxiang360729, ChenLeheng, juste_une_photo.

Contributions to the project in any form are welcome!

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