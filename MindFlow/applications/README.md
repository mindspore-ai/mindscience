[ENGLISH](README_EN.md) | 简体中文

# **MindFlow-应用**

- [简介](#简介)
- [目录](#目录)

## **简介**

MindFlow覆盖了物理驱动、数据驱动、数据机理融合的AI流体仿真、可微分CFD求解等多个方向。

物理驱动的AI流体仿真，即通过将物理方程引入到神经网络的损失函数中使其参与网络训练，从而使得学习的结果满足物理规律，该模块主要面向PDE方程的正向求解、基于数据融合的反问题以及数据同化等应用。

基于数据驱动的AI流体仿真，依赖大量的流体仿真数据，通过设计合适的神经网络挖掘数据样本间的物理规律，具备高效并行，快速推理的优势，具备一定的参数泛化能力。该模块主要面向具备大量标签数据的快速推理、参数空间设计优化等应用场景。

基于数据机理融合的AI流体仿真，以PDENet、PeRCNN等为典型代表，能从数据中学习偏微分方程，并且能够准确预测复杂系统的动力学特性和揭示潜在的PDE模型。该模块主要面向科学数据样本较少且已知控制方程的应用场景，通过内置流场方程信息，降低神经网络对于数据量的需求，提升网络的泛化性。

可微分CFD求解器主要通过数值方法在计算机中对流体力学的控制方程进行求解，从而实现流动的分析、预测和控制，基于AI框架昇思MindFlow的CFD可微求解器具备jit即时编译，vmap自动向量化，autograd端到端自动微分和支持不同硬件等优点，适用于经典流动的求解。

## **目录**

- 数据驱动
    - [东方.御风](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/data_driven/airfoil/2D_steady)
    - [FNO方法求解Burgers方程](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/data_driven/burgers/fno1d)
    - [KNO方法求解Burgers方程](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/data_driven/burgers/kno1d)
    - [FNO方法求解NS方程](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/data_driven/navier_stokes/fno2d)
    - [FNO3D方法求解NS方程](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/data_driven/navier_stokes/fno3d)
    - [KNO方法求解NS方程](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/data_driven/navier_stokes/kno2d)
    - [CAE-LSTM方法求解二维黎曼问题](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/research/cae_lstm)
    - [CAE-LSTM方法求解Shu-Osher问题](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/research/cae_lstm)
    - [CAE-LSTM方法求解Sod激波管问题](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/research/cae_lstm)
    - [CAE-LSTM方法求解KH问题](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/research/cae_lstm)
    - [eHDNN方法求解抖振流场](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/research/transonic_buffet_ehdnn)
    - [eHDNN方法预测非定常流场](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/research/move_boundary_hdnn)
    - [ResUnet3D方法求解三维圆球绕流](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/data_driven/flow_around_sphere)
    - [CAE-Transformer方法求解二维圆柱绕流问题](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/research/cae_transformer)
    - [FNO2D和UNET2D方法预测多时间步跨声速翼型复杂流场](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/data_driven/airfoil/2D_unsteady)
    - [HDNN方法预测流固耦合系统流场](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/research/fluid_structure_interaction)
- 数据-机理融合驱动
    - [PDE-NET方法求解对流扩散方程](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/data_mechanism_fusion/pde_net)
    - [PeRCNN方法求解二维Burgers方程](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/data_mechanism_fusion/percnn/burgers_2d)
    - [PeRCNN方法求解三维反应扩散方程](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/data_mechanism_fusion/percnn/gsrd_3d)
    - [AI湍流模型](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/data_mechanism_fusion/ai_turbulence_modeling)
- 物理驱动
    - [PINNs方法求解Burgers方程](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/physics_driven/burgers)
    - [PINNs方法求解圆柱绕流](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/physics_driven/navier_stokes/cylinder_flow_forward)
    - [PINNs方法求解Darcy流动](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/physics_driven/darcy)
    - [PINNs方法求解泊松方程](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/physics_driven/poisson/continuous)
    - [PINNs方法求解玻尔兹曼方程](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/physics_driven/boltzmann)
    - [PINNs方法求解泰勒-格林涡](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/physics_driven/navier_stokes/taylor_green)
    - [PINNs方法求解NS方程反问题](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/physics_driven/navier_stokes/cylinder_flow_inverse)
    - [PINNs方法求解二维带点源的泊松方程](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/physics_driven/poisson/point_source)
    - [PINNs方法求解Kovasznay流动](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/physics_driven/navier_stokes/kovasznay)
    - [PINNs方法求解周期山流动问题](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/physics_driven/navier_stokes/periodic_hill)
    - [PINNs方法求解Allen-Cahn方程](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/research/allen_cahn)
    - [CMA-ES&多目标梯度下降算法加速PINNs收敛](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/research/cma_es_mgda)
    - [META-PINNs算法](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/research/meta_pinns)
    - [MOE-PINNs算法](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/research/moe_pinns)
    - [R-DLGA算法](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/research/r_dlga)
- CFD
    - [Sod激波管](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/cfd/sod)
    - [Lax激波管](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/cfd/lax)
    - [二维黎曼问题](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/cfd/riemann2d)
    - [库埃特流动](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/cfd/couette)

代码贡献指导：请参考[教程](https://gitee.com/mindspore/mindscience/blob/master/MindFlow/CONTRIBUTION_CN.md)