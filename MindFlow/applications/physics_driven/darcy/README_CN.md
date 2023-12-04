[ENGLISH](README.md) | 简体中文

# 二维定常Darcy问题

## 概述

达西方程（Darcy equation）是一个描述了流体在多孔介质中低速流动时渗流规律的二阶椭圆型偏微分方程，被广泛应用于水利工程，石油工程等领域中。达西方程最初由亨利·达西根据沙土渗流实验的实验结果制定，后来由斯蒂芬·惠特克通过均质化方法从纳维-斯托克斯方程推导出来。本案例利用PINNs求解二维定常达西方程。

## 结果展示

![Darcy PINNs](images/result.png)

## 性能

| 参数               | Ascend               | GPU                |
|:----------------------:|:--------------------------:|:---------------:|
| 硬件资源                | Ascend, 显存32G            | NVIDIA V100, 显存32G    |
| MindSpore版本           | >=2.0.0                 | >=2.0.0                   |
| 参数量                  | 6e4                   | 6e4                   |
| 训练参数                | batch_size=8192, steps_per_epoch=8, epochs=4000 | batch_size=8192, steps_per_epoch=8, epochs=4000|
| 测试参数                | batch_size=8192       | batch_size=8192               |
| 优化器                  | Adam                 | Adam                   |
| 训练损失(MSE)           | 0.001                | 0.025             |
| 验证损失(RMSE)          | 0.01                 | 0.01              |
| 速度(ms/step)           |150                   | 400                |

## Contributor

haojiwei