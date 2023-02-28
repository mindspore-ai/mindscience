[ENGLISH](README.md) | 简体中文

# 二维圆柱绕流

## 概述

圆柱绕流，是指二维圆柱低速定常绕流的流型只与`Re`数有关。在`Re`≤1时，流场中的惯性力与粘性力相比居次要地位，圆柱上下游的流线前后对称，阻力系数近似与`Re`成反比，此`Re`数范围的绕流称为斯托克斯区；随着Re的增大，圆柱上下游的流线逐渐失去对称性。这种特殊的现象反映了流体与物体表面相互作用的奇特本质，求解圆柱绕流则是流体力学中的经典问题。本案例利用PINNs求解圆柱绕流的尾流流场。

![flow](images/image-flow.png)

![Time Error](images/TimeError_epoch5000.png)

[详见](https://gitee.com/mindspore/mindscience/blob/master/MindFlow/applications/physics_driven/cylinder_flow/navier_stokes2D_CN.ipynb)

## 贡献者

liulei277