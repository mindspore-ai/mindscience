[ENGLISH](README.md) | 简体中文

# 二维Taylor Green涡流

## 概述

在流体力学中，Taylor-Green涡流动是一种不稳定的衰减的涡流，在二维周期性边界条件时存在精确解，该精确解与Navier-Stokes方程的解一致。本案例使用PINNs对2维的taylor green涡流进行仿真

![tatlo_green](images/taylor_green.gif)

![flow](images/mid_stage.png)

![Time Error](images/TimeError_30000.png)

[详见](https://gitee.com/mindspore/mindscience/blob/master/MindFlow/applications/physics_driven/navier_stokes/taylor_green/taylor_green_2D.ipynb)

## 性能

| 参数               | Ascend               | GPU                |
|:----------------------:|:--------------------------:|:---------------:|
| 硬件资源                | Ascend, 显存32G            | NVIDIA V100, 显存32G    |
| MindSpore版本           | >=2.0.0                 | >=2.0.0                   |
| 参数量                  | 1.3e5                   | 1.3e5                   |
| 训练参数                | batch_size=1024, steps_per_epoch=64, epochs=300 | batch_size=1024, steps_per_epoch=64, epochs=300 |
| 测试参数                | batch_size=1024      | batch_size=1024               |
| 优化器                  | Adam                 | Adam                   |
| 训练损失(MSE)           | 0.0004                | 0.0001             |
| 验证损失(RMSE)          | 0.06                | 0.01              |
| 速度(ms/step)           | 15                   | 50                |