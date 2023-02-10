# MindFlow Release Notes

[View English](./RELEASE.md)

## MindFlow 0.1.0-alpha Release Notes

### 主要特性和增强

#### 物理驱动

- [STABLE] 支持物理信息神经网络求解偏微分方程，使用sympy定义微分方程及基本方程的求解，支持计算神经网络输出对输入的一阶和二阶导数矩阵，提供了基本几何形状、时域的定义及其操作，便于几何区域内和边界上的采样。

#### 数据驱动

- [STABLE] 提供了多种神经网络，包括全连接网络、残差网络、傅里叶神经算子、Vision Transformer，支持多种数据格式的读取和数据集的合并，MindFlow提供了模型训练和推理的高阶API，支持多种学习率和损失函数的使用。

#### 数据-物理融合驱动

- [STABLE] 提供了数据-物理融合驱动的深度学习方法PDE-Net，用于流场的时序预测和偏微分方程的回归。

#### 可微分CFD求解器

- [STABLE] 我们推出了端到端可微分的可压缩CFD求解器MindFlow-CFD，支持WENO5重构、Rusanov通量以及龙格-库塔积分，支持对称、周期性、固壁及诺依曼边界条件。

### 贡献者

感谢以下开发者做出的贡献：

yufan, wangzidong, liuhongsheng, zhouhongye, zhangyi, dengzhiwen, liulei, libokai, yangge, longzichao, yqiuu, haojiwei, leiyixiang
