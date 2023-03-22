# 一维伯格斯方程

## 概述

伯格斯方程（Burgers' equation）是一个模拟冲击波的传播和反射的非线性偏微分方程，被广泛应用于流体力学，非线性声学，
气体动力学等领域，它以约翰内斯·马丁斯汉堡（1895-1981）的名字命名。

一维伯格斯方程（1-d Burgers' equation）的应用包括一维粘性流体流动建模。它的形式如下：

$$
\partial_t u(x, t)+\partial_x (u^2(x, t)/2)=\nu \partial_{xx} u(x, t), \quad x \in(0,1), t \in(0, 1]
$$

$$
u(x, 0)=u_0(x), \quad x \in(0,1)
$$

其中$u$表示速度场，$u_0$表示初始条件，$\nu$表示粘度系数。

本案例利用Koopman Neural Operator学习初始状态到下一时刻状态的映射，实现一维Burgers'方程的求解：

$$
u_0 \mapsto u(\cdot, 1)
$$

[详见](https://gitee.com/mindspore/mindscience/blob/master/MindFlow/applications/data_driven/burgers_kno/KNO1D_CN.ipynb)

## 训练

训练日志、训练Loss以及验证Loss如下。

```text
Data preparation finished
input_path:  (1000, 1024, 1)
label_path:  (1000, 1024)
Data preparation finished
input_path:  (200, 1024, 1)
label_path:  (200, 1024)
name:KNO1D_channels:32_modes:64_depths:4_resolution:1024
./summary_dir/name:KNO1D_channels:32_modes:64_depths:4_resolution:1024
epoch: 1, time cost: 21.623394, recons loss: 0.295491, pred loss: 0.085095
epoch: 2, time cost: 2.527564, recons loss: 0.161590, pred loss: 0.002210
epoch: 3, time cost: 2.598942, recons loss: 0.027091, pred loss: 0.000967
epoch: 4, time cost: 2.517585, recons loss: 0.000775, pred loss: 0.000502
epoch: 5, time cost: 2.573697, recons loss: 0.000057, pred loss: 0.000282
epoch: 6, time cost: 2.562175, recons loss: 0.000048, pred loss: 0.000244
epoch: 7, time cost: 2.491402, recons loss: 0.000048, pred loss: 0.000214
epoch: 8, time cost: 2.530793, recons loss: 0.000048, pred loss: 0.000237
epoch: 9, time cost: 2.504641, recons loss: 0.000048, pred loss: 0.000231
epoch: 10, time cost: 2.544668, recons loss: 0.000049, pred loss: 0.000227
---------------------------start evaluation-------------------------
Eval epoch: 10, recons loss: 4.7650219457864295e-05, relative pred loss: 0.01156728882342577
---------------------------end evaluation---------------------------

...

epoch: 91, time cost: 2.539794, recons loss: 0.000042, pred loss: 0.000006
epoch: 92, time cost: 2.521379, recons loss: 0.000042, pred loss: 0.000007
epoch: 93, time cost: 3.142074, recons loss: 0.000042, pred loss: 0.000006
epoch: 94, time cost: 2.569737, recons loss: 0.000042, pred loss: 0.000006
epoch: 95, time cost: 2.545627, recons loss: 0.000042, pred loss: 0.000006
epoch: 96, time cost: 2.568123, recons loss: 0.000042, pred loss: 0.000006
epoch: 97, time cost: 2.547843, recons loss: 0.000042, pred loss: 0.000006
epoch: 98, time cost: 2.709663, recons loss: 0.000042, pred loss: 0.000006
epoch: 99, time cost: 2.529918, recons loss: 0.000042, pred loss: 0.000006
epoch: 100, time cost: 2.502929, recons loss: 0.000042, pred loss: 0.000006
---------------------------start evaluation-------------------------
Eval epoch: 100, recons loss: 4.1765865171328186e-05, relative pred loss: 0.004054672718048095
---------------------------end evaluation---------------------------
```

## 测试

取6个样本做连续10步预测，并可视化。

![](images/result.jpg)

## Contributor

dyonghan
