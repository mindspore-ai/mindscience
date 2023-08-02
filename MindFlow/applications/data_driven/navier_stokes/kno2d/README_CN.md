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

## 训练

训练日志、训练Loss以及验证Loss如下。

```text
epoch: 1, time cost: 55.562426, recons loss: 0.467314, pred loss: 0.237300
epoch: 2, time cost: 32.804436, recons loss: 0.175188, pred loss: 0.050888
epoch: 3, time cost: 32.946971, recons loss: 0.167865, pred loss: 0.041778
epoch: 4, time cost: 33.064430, recons loss: 0.170181, pred loss: 0.038075
epoch: 5, time cost: 32.907211, recons loss: 0.171853, pred loss: 0.035849
epoch: 6, time cost: 33.799230, recons loss: 0.173322, pred loss: 0.034017
epoch: 7, time cost: 32.612255, recons loss: 0.174376, pred loss: 0.032719
epoch: 8, time cost: 32.896673, recons loss: 0.175445, pred loss: 0.031596
epoch: 9, time cost: 33.907305, recons loss: 0.176131, pred loss: 0.030644
epoch: 10, time cost: 33.175130, recons loss: 0.176701, pred loss: 0.029969
Eval epoch: 10, recons loss: 0.23137304687500002, relative pred loss: 0.03798459614068269

...

epoch: 41, time cost: 32.962233, recons loss: 0.185430, pred loss: 0.017872
epoch: 42, time cost: 33.296847, recons loss: 0.185595, pred loss: 0.017749
epoch: 43, time cost: 33.803700, recons loss: 0.185646, pred loss: 0.017651
epoch: 44, time cost: 32.776349, recons loss: 0.185723, pred loss: 0.017564
epoch: 45, time cost: 33.377666, recons loss: 0.185724, pred loss: 0.017497
epoch: 46, time cost: 33.228983, recons loss: 0.185827, pred loss: 0.017434
epoch: 47, time cost: 33.244342, recons loss: 0.185854, pred loss: 0.017393
epoch: 48, time cost: 33.211263, recons loss: 0.185912, pred loss: 0.017361
epoch: 49, time cost: 35.656644, recons loss: 0.185897, pred loss: 0.017349
epoch: 50, time cost: 33.527458, recons loss: 0.185899, pred loss: 0.017344
Eval epoch: 50, recons loss: 0.2389616699218751, relative pred loss: 0.03355878115445375
```

## 测试

取1个样本做连续10步预测，并可视化。

![](images/result.gif)

## Contributor

dyonghan
