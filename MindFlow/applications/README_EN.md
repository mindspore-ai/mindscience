ENGLISH | [简体中文](README.md)

# **MindFlow-APPLICATIONS**

- [**Introduction**](#introduction)
- [**Contents**](#contents)

## **Introduction**

MindFlow covers physics-driven, data-driven, data-mechanism fusion AI fluid simulation, differentiable CFD solution and other directions.

Physics-driven AI fluid simulation introduces physical equations into the neural network's loss function. These equations participate in network training, ensuring the learning results align with physical laws. The module is specially designed to handle forward solving of PDE equations. It also focuses on inverse problems based on data fusion and applications like data assimilation.

Data-driven AI fluid simulation relies on abundant fluid simulation data. With a purposefully designed neural network, the module is capable of extracting the physical laws among data samples. The simulation is recognized for its efficient parallelism and fast reasoning capabilities, and certain parameter generalization capabilities. This module is mainly aimed at application scenarios such as fast inference with a large amount of label data and parameter space design optimization.

Data-mechanism fusion driven AI fluid simulation, such as PDENet and PeRCNN. These simulation methods adept at learning partial differential equations from data, and can accurately predict the dynamic characteristics of complex systems and reveal potential PDE models. This module is mainly oriented to the application scenarios of small scientific data samples and known control equations. A significant benefit is the reduction in data requisites for neural networks, further enhancing the network's generalization due to embedded flow field equation information.

The differentiable CFD solver mainly solves the control equation of fluid dynamics in the computer through numerical methods. So this solver realizes the analysis, prediction and control of flow. Underpinned by the AI framework MindFlow, this solver boasts features such as JIT instant compilation and vmap automatic vectorization. Additionally, this solver offers autograd end-to-end automatic differentiation and compatibility with diverse hardware. Its proficiency makes it suitable for resolving classical flow problems.

## **Contents**

- Data Driven
    - [DongFang.YuFeng](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/data_driven/airfoil/2D_steady)
    - [Solve Burgers Equation by FNO](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/data_driven/burgers/fno1d)
    - [Solve Burgers Equation by KNO](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/data_driven/burgers/kno1d)
    - [Solve Burgers Equation by SNO](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/data_driven/burgers/sno1d)
    - [Solve Navier-Stokes Equation by FNO](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/data_driven/navier_stokes/fno2d)
    - [Solve Navier-Stokes Equation by SNO](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/data_driven/navier_stokes/sno2d)
    - [Solve Navier-Stokes Equation by FNO3D](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/data_driven/navier_stokes/fno3d)
    - [Solve Navier-Stokes Equation by KNO](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/data_driven/navier_stokes/kno2d)
    - [Solve Navier-Stokes Equation by SNO3D](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/data_driven/navier_stokes/sno3d)
    - [Solve 2D Riemann Problem by CAE-LSTM](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/research/cae_lstm)
    - [Solve Shu-Osher Problem by CAE-LSTM](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/research/cae_lstm)
    - [Solve 1D Sod Shock Tube Problem by CAE-LSTM](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/research/cae_lstm)
    - [Solve KH Problem by CAE-LSTM](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/research/cae_lstm/)
    - [Solve 2D Airfoil Buffet by eHDNN](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/research/transonic_buffet_ehdnn)
    - [Predict Unsteady Flow Fields with Move Boundary by eHDNN](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/research/move_boundary_hdnn)
    - [Solve 3D Unsteady Sphere Flow by ResUnet3D](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/data_driven/flow_around_sphere)
    - [Solve 2D Cylinder Flow by CAE-Transformer](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/research/cae_transformer)
    - [Predict Multi-timestep Complicated Transonic Airfoil by FNO2D and UNET2D](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/data_driven/airfoil/2D_unsteady)
    - [Predict Fluid-structure Interaction System by HDNN](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/research/fluid_structure_interaction)
- Data-Mechanism Fusion
    - [Solve Convection-Diffusion Equation by PDE-NET](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/data_mechanism_fusion/pde_net)
    - [Solve 2D Burgers Equation by PeRCNN](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/data_mechanism_fusion/percnn/burgers_2d)
    - [Solve 3D Reaction-Diffusion Equation by PeRCNN](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/data_mechanism_fusion/percnn/gsrd_3d)
    - [AI Turb Model](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/data_mechanism_fusion/ai_turbulence_modeling)
- Physics Driven
    - [Solve Burgers Equation by PINNs](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/physics_driven/burgers)
    - [Solve 2D Cylinder Flow by PINNs](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/physics_driven/navier_stokes/cylinder_flow_forward)
    - [Solve 2D Darcy Problem by PINNs](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/physics_driven/darcy)
    - [Solve Poisson Equation by PINNs](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/physics_driven/poisson/continuous)
    - [Solve Boltzmann Equation by PINNs](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/physics_driven/boltzmann)
    - [Solve 2D Taylor-Green Votex by PINNs](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/physics_driven/navier_stokes/taylor_green)
    - [Solve Inverse Navier-Stoken Problem by PINNs](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/physics_driven/navier_stokes/cylinder_flow_inverse)
    - [Solve 2D Poisson Equation with Point Source by PINNs](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/physics_driven/poisson/point_source)
    - [Solve Kovasznay Flow by PINNs](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/physics_driven/navier_stokes/kovasznay)
    - [Solve Periodic Hill Flow by PINNs](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/physics_driven/navier_stokes/periodic_hill)
    - [Solve Allen-Cahn Equation by PINNs](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/research/allen_cahn)
    - [CMA-ES&Multi-objective Gradient Descent Algorithm Accelerates PINNs Convergence](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/research/cma_es_mgda)
    - [META-PINNs Algorithm](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/research/meta_pinns)
    - [MOE-PINNs Algorithm](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/research/moe_pinns)
    - [R-DLGA Algorithm](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/research/r_dlga)
- CFD
    - [Sod Shock Tube](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/cfd/sod)
    - [Lax Shock Tube](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/cfd/lax)
    - [2D Riemann Problem](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/cfd/riemann2d)
    - [Couette Flow](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/cfd/couette)

How to contribute: Please refer to [Tutorial](https://gitee.com/mindspore/mindscience/blob/master/MindFlow/CONTRIBUTION_CN.md)