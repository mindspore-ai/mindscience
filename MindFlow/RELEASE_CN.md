# MindFlow Release Notes

## MindFlow 0.1.0a0 Release Notes

### 主要特性和增强

#### Network Cells

- [STABLE] 支持多种神经网络，包括全连接网络，残差网络，傅里叶神经算子，Vision Transformer (ViT)，PDE-Net。

#### PDE

- [STABLE] 基于物理信息神经网络(PINNs)求解偏微分方程(PDE)。用户可以通过sympy定义PDE。支持常用的基本方程求解。

#### Geometry

- [STABLE] 基本几何、时域的定义及其相关操作。支持几何域和边界上的采样。

#### Data

- [STABLE] 支持多个数据集的组合，也支持加载npy格式文件。

#### Learning Rate

- [STABLE] 支持多项式衰减、升温余弦退火和多阶段的学习率。

#### Loss

- [STABLE] 支持相对RMSE损失、多级小波损失和加权多任务损失。

#### Operators

- [STABLE] 计算神经网络输出对输入的一阶导数和二阶导数矩阵。

#### Solver

- [STABLE] 支持神经网络的训练和推理。

#### CFD

- [STABLE] 我们推出了端到端可微分的可压缩CFD求解器MindFlow-CFD。支持WENO5重构、Rusanov通量、龙格-库塔积分。支持对称、周期、无滑移壁面和诺伊曼边界条件。

## Contributors

感谢以下开发者做出的贡献：

wangzidong, liuhongsheng, dengzhiwen, zhangyi, zhouhongye, libokai, yangge, liulei, longzichao

