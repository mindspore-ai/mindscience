# MindSpore Flow Release Notes

[View English](./RELEASE.md)

MindSpore Flow是基于昇思MindSpore开发的流体仿真领域套件，支持航空航天、船舶制造以及能源电力等行业领域的AI流场模拟，旨在于为广大的工业界科研工程人员、高校老师及学生提供高效易用的AI计算流体仿真软件。

## MindFlow 0.1.0 Release Notes

### 主要特性和增强

#### 数据驱动

- [STABLE] [`CAE-LSTM`](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/research/cae_lstm): 支持数据驱动的卷积自编码器–长短时记忆神经网络求解非定常可压缩流动。
- [STABLE] [`Move Boundary Hdnn`](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/research/move_boundary_hdnn): 支持数据驱动的HDNN网络求解动边界的非定常流场。

#### 数据-机理融合驱动

- [STABLE] [`PeRCNN`](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/data_mechanism_fusion/PeRCNN): 支持物理编码递归卷积神经网络(Physics-encoded Recurrent Convolutional Neural Network，PeRCNN)。

#### 物理驱动

- [STABLE] [`Boltzmann`](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/physics_driven/boltzmann): 支持PINNs方法求解玻尔兹曼方程。
- [STABLE] [`Poisson with Point Source`](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/physics_driven/poisson/point_source): 支持PINNs方法求解带点源的泊松方程。

## MindSpore Flow 0.1.0.rc1 Release Notes

### 主要特性和增强

#### 数据驱动

- [STABLE] [`KNO`](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/data_driven/navier_stokes/kno2d): 支持KNO神经算子，提升NS方程仿真精度
- [STABLE] [`东方.御风`](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/data_driven/airfoil/2D_steady): 东方御风大模型，支持翼型端到端快速仿真。

## MindFlow 0.1.0-alpha Release Notes

### 主要特性和增强

#### 数据驱动

- [STABLE] 提供了多种神经网络，包括全连接网络、残差网络、傅里叶神经算子、Vision Transformer，支持多种数据格式的读取和数据集的合并，MindFlow提供了模型训练和推理的高阶API，支持多种学习率和损失函数的使用。

#### 数据-物理融合驱动

- [STABLE] [`PDE-Net`](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/data_mechanism_fusion/pde_net): 提供数据-物理融合驱动的深度学习方法PDE-Net，用于流场的时序预测和偏微分方程的回归。

#### 物理驱动

- [STABLE] 支持物理信息神经网络求解偏微分方程，使用sympy定义微分方程及基本方程的求解，支持计算神经网络输出对输入的一阶和二阶导数矩阵，提供了基本几何形状、时域的定义及其操作，便于几何区域内和边界上的采样。

#### 可微分CFD求解器

- [STABLE] 我们推出了端到端可微分的可压缩CFD求解器MindFlow-CFD，支持WENO5重构、Rusanov通量以及龙格-库塔积分，支持对称、周期性、固壁及诺依曼边界条件。

### 贡献者

感谢以下开发者做出的贡献：

hsliu_ustc, Yi_zhang95, zwdeng, liulei277, chengzrz, liangjiaming2023, yanglin2023