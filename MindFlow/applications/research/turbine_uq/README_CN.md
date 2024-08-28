## 基于神经算子网络的涡轮级流场预测与不确定性优化设计

## 概述

流体机械在设计、加工制造和运行的过程中存在诸多的不确定性，会导致实际运行时流量、效率等总体参数偏离设计值，对运行性能造成影响。然而将几何与工况的不确定性纳入设计的考量又会使得设计过程中性能综合评估的成本与耗时急剧增加，当前流体机械领域缺乏低成本的不确定性优化设计方法。
将高精度神经算子网络应用到不确定性优化设计中，作为耗时的CFD计算的代替品是解决上述的问题的一类方法，可以有效提升不确定性优化设计的效率，增强设计对象的运行稳定性。

### 方法

直接使用蒙特卡洛方法计算特定设计方案在不确定性输入下的性能分布的成本极大。传统的不确定性优化设计往往采用基于代理模型的不确定性量化方法，例如混沌多项式来完成不确定性参数的评估，但即使如此在对一个样本进行评估时也需要完成数十上百次的数值计算。
本文采用了蒙特卡洛结合深度学习的方法来完成不确定性的量化评估。并基于此评估结果完成优化设计，具体流场如下图所示：

1. NSGA2算法输出需要计算的$n$个设计方案的变量信息；
2. 使用ITS方法获取不确定性输入的$m$个离散化表示；
3. 组合设计变量与不确定性变量，使用深度学习模型计算得到$m \times n$个物理场结果；
4. 对物理场结果进行后处理获得性能参数结果；
5. 完成不确定性量化计算，获得$n$个设计的不确定性评估并返回到算法模块；
6. 重复上述5个步骤，直到$h$次迭代结束输出最优解；

<div align="center">
<img alt="UQ_optimization" src="./images/UQ_optimization_EN.png" width="450"/>
<div align="left">

在上述过程中，对于设计方案的全物理场评估是关键，一次优化设计共需要完成$m \times n \times h$次评估，以传统的CFD数值计算的成本完全无法完成，因此需要计算快速，方便并行的神经网络来完成物理场的预测，同时要求有较高的预测精度。
基于神经网络的物理场预测是整个流程的核心步骤。其具体网络结构如下：

<div align="center">
<img alt="trans_operator_gvrb" src="./images/trans_operator_gvrb.png" width="550"/>
<div align="left">

1. 深度算子网络的输入包括：设计参数x、工况参数$\alpha$、坐标参数$p$，三者在连接后输入网络。
2. 本文所采用的网络结构为使用galerkin线性计算的transformer算子网络；
3. 网络的输出为基础物理场，包括有：压力$p$、温度$T$、速度$V$（三个分量）、密度$、\rho$这些N-S方程求解的基本特征量，每个特征量占据一个通道；
4. 当获取的结果具有所有N-S方向的特征物理量时，可以基于后处理程序计算任意的物理场或性能参数值。

### 数据集

本文以GE-E3第一级高压涡轮为研究对象建立性能评估模型，GE-E3高压级单通道模型如下所示。其中包含一列静叶S1和一列动叶R1。数值计算使用商业CFD软件Numeca完成。[数据集下载链接](https://gr.xjtu.edu.cn/web/songlm/1)

<div align="center">
<img alt="GEE3_blade" src="./images/GEE3_blade.png" width="450"/>
<div align="left">

数据集如下图所示，在数据准备过程中，采用拉丁超立方采样(LHS)方法，对共计100个几何与边界条件设计变量 的问题在设计空间中采集了4900个样本。随后，使用几何参数化方法生成相应的几何模型，并计算每个样本的三维流场。选取其中4000个样本为训练集，900个样本为验证集。

<div align="center">
<img alt="gvrb_data_prepare" src="./images/gvrb_data_prepare_EN.png" width="550"/>
<div align="left">

### 效果

在使用训练集数据完成网络训练后， 使用网络直接预测验证集数据的物理场，并计算部分叶轮机械设计常用性能指标发现，预测得到的性能指标的相对误差均小于1%，预测精度基本满足工程实践需求。

<div align="center">
<img alt="related_error" src="./images/related_error.png" width="550"/>
<div align="left">

## 快速开始

### 训练方式：在命令行中调用 `main.py` 脚本

```
python main.py --config_file_path 'configs/FNO_GVRB' --device_target 'GPU'
```

main.py脚本的可输入参数有：
`--config_file_path` 表示参数和路径控制文件，默认值'./config.yaml'；
`--mode` 表示运行的模式，'GRAPH'表示静态图模式, 'PYNATIVE'表示动态图模式， 默认值'GRAPH'；
`--device_target` 表示使用的计算平台类型，可以选择'Ascend'或'GPU'，默认值'Ascend'；
`--device_id` 表示使用的计算卡编号，可按照实际情况填写，默认值0

## 结果展示

开始训练后得到结果如下：

```
Epoch 495: loss 0.032
Train epoch time: 3.261 s, per step time: 51.755 ms
Epoch 496: loss 0.032
Train epoch time: 3.628 s, per step time: 57.592 ms
Epoch 497: loss 0.032
Train epoch time: 3.262 s, per step time: 51.785 ms
Epoch 498: loss 0.032
Train epoch time: 3.358 s, per step time: 53.305 ms
Epoch 499: loss 0.032
Train epoch time: 3.246 s, per step time: 51.531 ms
Epoch 500: loss 0.032
Train epoch time: 3.293 s, per step time: 52.275 ms
Epoch 500: train: 0.06596697171349167
Epoch 500: test: 0.09644165817184366
training done!
Total train time: 1777.7318394184113s
```

开始绘图后得到结果如下：

<div align="center">
<img alt="case_A" src="./images/case_A.png" width="450"/>
<br>
<img alt="case_A_d" src="./images/case_A_d.png" width="450"/>
<br>

开始优化后得到结果如下：

```
=================================================
n_gen  |  n_eval  |     f_avg     |     f_min
=================================================
...
    25 |      800 | -9.143245E-01 | -9.216238E-01
    26 |      832 | -9.143438E-01 | -9.216238E-01
    27 |      864 | -9.144188E-01 | -9.216238E-01
    28 |      896 | -9.144471E-01 | -9.216238E-01
    29 |      928 | -9.145065E-01 | -9.216238E-01
    30 |      960 | -9.145443E-01 | -9.216238E-01
optimal value of dtm task: [-0.92162382]
total optimization time of dtm task: 25.431s
```

<img alt="optimal_rst_hist" src="./images/optimal_rst_hist.png" width="350"/>
<div align="left">

## 性能

### UNet性能对比

| 参数               | Ascend               | GPU                |
|:----------------------:|:--------------------------:|:---------------:|
| 硬件资源                | Ascend, 显存32G            | NVIDIA V100, 显存32G    |
| MindSpore版本           | 2.2.14                | 2.2.12                 |
| 数据集                  |[涡轮级子午面流场数据集](https://download-mindspore.osinfra.cn/mindscience/mindflow/dataset/applications/research/turbine_uq/)     |[涡轮级子午面流场数据集](https://download-mindspore.osinfra.cn/mindscience/mindflow/dataset/applications/research/turbine_uq/)       |
| 参数量                  | 4.2e6                   | 4.2e6                    |
| 训练参数                | batch_size=32, <br>steps_per_epoch=70, <br>epochs=500 | batch_size=32, <br>steps_per_epoch=70, <br>epochs=500 |
| 测试参数                | batch_size=32          | batch_size=32               |
| 优化器                  | Adam                 | Adam                   |
| 动态图-训练损失(RL2)           | 0.0654                | 0.0652            |
| 动态图-验证损失(RL2)          | 0.0932                | 0.0941              |
| 动态图-训练step时间(ms)           | 52              |54              |
| 静态图-训练损失(RL2)           | 0.0667                | 0.0667           |
| 静态图-验证损失(RL2)          | 0.0956                | 0.0970              |
| 静态图-训练step时间(ms)           | 33                  | 44            |

### FNO性能对比

| 参数               | Ascend               | GPU                |
|:----------------------:|:--------------------------:|:---------------:|
| 硬件资源                | Ascend, 显存32G            | NVIDIA V100, 显存32G    |
| MindSpore版本           | 2.2.14                | 2.2.12                 |
| 数据集                  |[涡轮级子午面流场数据集](https://download-mindspore.osinfra.cn/mindscience/mindflow/dataset/applications/research/turbine_uq/)     |[涡轮级子午面流场数据集](https://download-mindspore.osinfra.cn/mindscience/mindflow/dataset/applications/research/turbine_uq/)       |
| 参数量                  | 3.1e6                   | 3.1e6                    |
| 训练参数                | batch_size=32, <br>steps_per_epoch=70, <br>epochs=500 | batch_size=32, <br>steps_per_epoch=70, <br>epochs=500 |
| 测试参数                | batch_size=32          | batch_size=32               |
| 优化器                  | Adam                 | Adam                   |
| 动态图-训练损失(RL2)           | 0.0371                | 0.0370             |
| 动态图-验证损失(RL2)          | 0.0378                | 0.0377             |
| 动态图-训练step时间(ms)           | 124                   | 116                |
| 静态图-训练损失(RL2)           | 0.0381                | 0.0371            |
| 静态图-验证损失(RL2)          | 0.0446               | 0.0371             |
| 静态图-训练step时间(ms)           | 35               | 68            |
