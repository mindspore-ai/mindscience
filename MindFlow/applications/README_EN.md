ENGLISH | [简体中文](README.md)

# **MindFlow-APPLICATIONS**

- [**Introduction**](#introduction)
- [**Contents**](#contents)

## **Introduction**

Application relies on the computational fluid toolkit MindFlow and MindSpore, aiming to provide efficient and easy-to-use AI computational fluid simulation cases for industrial research engineers, university teachers and students.
MindFlow covers physics-driven, data-driven, data-mechanism fusion AI fluid simulation, differentiable CFD solution and other directions.

Physics-driven AI fluid simulation, that is, by introducing physical equations into the loss function of neural networks to participate in network training, so that the learning results meet physical laws, this module is mainly oriented to forward solving of PDE equations, inverse problems based on data fusion, and data assimilation and other applications.

Data-driven AI fluid simulation relies on a large amount of fluid simulation data, by designing a suitable neural network to mine the physical laws between data samples, it has the advantages of efficient parallelism, fast reasoning, and certain parameter generalization capabilities. This module is mainly aimed at application scenarios such as fast inference with a large amount of label data and parameter space design optimization.

Data-mechanism fusion driven AI fluid simulation, such as PDENet and PeRCNN, can learn partial differential equations from data, and can accurately predict the dynamic characteristics of complex systems and reveal potential PDE models. This module is mainly oriented to the application scenarios of small scientific data samples and known control equations, and reduces the data requirements of neural networks and improves the generalization of networks through built-in flow field equation information.

The differentiable CFD solver mainly solves the control equation of fluid dynamics in the computer through numerical methods, so as to realize the analysis, prediction and control of flow, and the CFD differentiable solver based on the AI framework MindFlow has the advantages of JIT instant compilation, vmap automatic vectorization, autograd end-to-end automatic differentiation and support for different hardware, which is suitable for the solution of classical flow.

## **Contents**

- Physics Driven
    - [Burgers Equation](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/physics_driven/burgers)
    - [2D Cylinder Flow](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/physics_driven/navier_stokes/cylinder_flow_forward)
    - [2D Darcy](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/physics_driven/darcy)
    - [Poisson Equation](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/physics_driven/poisson/continuous)
    - [Boltzmann Equation](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/physics_driven/boltzmann)
    - [2D Taylor-Green Votex](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/physics_driven/navier_stokes/taylor_green)
    - [Navier-Stoken Inverse](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/physics_driven/navier_stokes/cylinder_flow_inverse)
    - [2D Poisson Equation with Point Source](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/physics_driven/poisson/point_source)
    - [CMA-ES&Multi-objective Gradient Descent Algorithm](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/research/cma_es_mgda)
    - [Kovasznay Flow](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/physics_driven/navier_stokes/kovasznay)
    - [Periodic Flow](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/physics_driven/navier_stokes/periodic_hill)
- Data Driven
    - [DongFang.YuFeng](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/data_driven/airfoil/2D_steady)
    - [FNO Solve Burgers Equation](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/data_driven/burgers/fno1d)
    - [KNO Solve Burgers Equation](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/data_driven/burgers/kno1d)
    - [FNO Solve Navier-Stokes Equation](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/data_driven/navier_stokes/fno2d)
    - [FNO3d Solve Navier-Stokes Equation](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/data_driven/navier_stokes/fno3d)
    - [KNO Solve Navier-Stokes Equation](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/data_driven/navier_stokes/kno2d)
    - [2D Riemann Problem](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/research/cae_lstm)
    - [shu-osher Problem](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/research/cae_lstm)
    - [sod Problem](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/research/cae_lstm)
    - [KH Problem](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/research/cae_lstm/)
    - [2D Airfoil Buffet](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/research/transonic_buffet_ehdnn)
    - [Move Boundary Hdnn](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/research/move_boundary_hdnn)
    - [3d Unsteady Flow](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/data_driven/flow_around_sphere)
    - [Low Reynolds 2D Cylinder Flow](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/research/cae_transformer)
- Data-Mechanism Fusion
    - [PDE-Net for Convection-Diffusion Equation](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/data_mechanism_fusion/pde_net)
    - [PeRCNN](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/data_mechanism_fusion/PeRCNN)
- CFD
    - [sod shock tube](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/cfd/sod)
    - [lax shock tube](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/cfd/lax)
    - [2D Riemann Problem](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/cfd/riemann2d)
    - [Couette Flow](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/cfd/couette)