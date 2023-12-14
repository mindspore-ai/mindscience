[ENGLISH](README.md) | 简体中文

# Navier Stokes反问题

## 概述

Navier-Stokes的反问题是指，在已知某些流体运动特征（如流量、速度等）的条件下，求解出能够产生这些运动特征的流体性质（如黏度、密度等）和流体边界条件（如壁面摩擦力等）的问题。与正问题（即已知流体性质和边界条件，求解流体的运动特征）不同，反问题的解决需要通过数值优化和逆推算法等方法进行求解。

![inv_flow](images/FlowField_10000.gif)

![Time Error](images/TimeError_10000.png)

![Parameter](images/Parameter.png)

|Correct PDE|Identified PDE|
|  ----  | ----  |
|$u_t + (u u_x + v u_x) = - p_x + 0.01(u_{xx} + u_{yy})$|$u_t + 0.9984444 (u u_x + v u_x) = - p_x + 0.01072927(u_{xx} + u_{yy})$|
|$v_t + (u v_x + v v_x) = - p_y + 0.01(v_{xx} + v_{yy})$|$v_t + 0.9984444 (u v_x + v v_x) = - p_y + 0.01072927(v_{xx} + v_{yy})$|

[详见](https://gitee.com/mindspore/mindscience/blob/master/MindFlow/applications/physics_driven/navier_stokes/cylinder_flow_inverse/navier_stokes_inverse_CN.ipynb)

## 性能

| 参数               | Ascend               | GPU                |
|:----------------------:|:--------------------------:|:---------------:|
| 硬件资源                | Ascend, 显存32G            | NVIDIA V100, 显存32G    |
| MindSpore版本           | >=2.0.0                 | >=2.0.0                   |
| 数据集                  | [NS方程反问题数据集](https://download-mindspore.osinfra.cn/mindscience/mindflow/dataset/applications/physics_driven/inverse_navier_stokes/)      | [NS方程反问题数据集](https://download-mindspore.osinfra.cn/mindscience/mindflow/dataset/applications/physics_driven/inverse_navier_stokes/)                   |
| 参数量                  | 3.5e3                   | 3.5e3                    |
| 训练参数                | batch_size=1024, steps_per_epoch=39, epochs=10000 | batch_size=1024, steps_per_epoch=39, epochs=10000 |
| 测试参数                | batch_size=1024      | batch_size=1024               |
| 优化器                  | Adam                 | Adam                   |
| 训练损失(MSE)           | 0.0001                | 0.0001             |
| 验证损失(RMSE)          | 0.01                  | 0.01              |
| 速度(ms/step)           | 20                   | 30                |
