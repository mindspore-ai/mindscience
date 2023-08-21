# 二维纳维-斯托克斯方程

## 概述

纳维-斯托克斯方程（Navier-Stokes equation）是计算流体力学领域的经典方程，是一组描述流体动量守恒的偏微分方程，简称N-S方程。它在二维不可压缩流动中的涡度形式如下：

$$
\partial_t w(x, t)+u(x, t) \cdot \nabla w(x, t)=\nu \Delta w(x, t)+f(x), \quad x \in(0,1)^2, t \in(0, T]
$$

$$
\nabla \cdot u(x, t)=0, \quad x \in(0,1)^2, t \in[0, T]
$$

$$
w(x, 0)=w_0(x), \quad x \in(0,1)^2
$$

其中$u$表示速度场，$w=\nabla \times u$表示涡度，$w_0(x)$表示初始条件，$\nu$表示粘度系数，$f(x)$为外力合力项。

本案例利用Koopman Neural Operator学习某一个时刻对应涡度到下一时刻涡度的映射，实现二维不可压缩N-S方程的求解：

$$
w_t \mapsto w(\cdot, t+1)
$$

[详见](https://gitee.com/mindspore/mindscience/blob/master/MindFlow/applications/data_driven/navier_stokes/kno2d/KNO2D_CN.ipynb)

## 结果展示

取1个样本做连续10步预测，并可视化。

[推理误差](images/result.gif)

## 性能

|        参数         |        Ascend               |    GPU       |
|:----------------------:|:--------------------------:|:---------------:|
|     硬件资源         |     Ascend 910A, 显存32G；CPU: 2.6GHz, 192核      |      NVIDIA V100 显存32G       |
|     MindSpore版本   |        2.1            |      2.1       |
|        训练损失      |       0.17                |       0.16       |
|        验证损失      |        3e-2               |       3e-2    |
|        速度          |     25s/epoch        |    160s/epoch  |

## Contributor

gitee id：[dyonghan](https://gitee.com/dyonghan)

email: dyonghan@qq.com