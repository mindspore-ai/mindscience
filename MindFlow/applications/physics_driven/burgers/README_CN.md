[ENGLISH](README.md) | 简体中文

# 一维Burgers问题

## 概述

伯格斯方程（Burgers' equation）是一个模拟冲击波的传播和反射的非线性偏微分方程，被广泛应用于流体力学，非线性声学，气体动力学等领域，它以约翰内斯·马丁斯汉堡（1895-1981）的名字命名。本案例采用MindFlow流体仿真套件，基于物理驱动的PINNs (Physics Informed Neural Networks)方法，求解一维有粘性情况下的Burgers方程。

## 快速开始

从[physics_driven/burgers_pinns/dataset](https://download.mindspore.cn/mindscience/mindflow/dataset/applications/physics_driven/burgers_pinns/dataset/) 中下载验证所需要的数据集，并保存在`./dataset`目录下。

### 训练方式一：在命令行中调用`train.py`脚本

```shell
python train.py --config_file_path ./configs/burgers.yaml --mode GRAPH --device_target Ascend --device_id 0
```

其中，

`--config_file_path`表示参数文件的路径，默认值'./configs/burgers.yaml'；

`--mode`表示运行的模式，'GRAPH'表示静态图模式, 'PYNATIVE'表示动态图模式，默认值'GRAPH'；

`--device_target`表示使用的计算平台类型，可以选择'Ascend'或'GPU'，默认值'Ascend'；

`--device_id`表示使用的计算卡编号，可按照实际情况填写，默认值0；

### 训练方式二：运行Jupyter Notebook

您可以使用[中文版](./burgers1D_CN.ipynb)和[英文版](./burgers1D.ipynb)Jupyter Notebook逐行运行训练和验证代码。

## 结果展示

![Burgers PINNs](images/result.jpg)

## 性能

| 参数               | Ascend               | GPU                |
|:----------------------:|:--------------------------:|:---------------:|
| 硬件资源                | Ascend, 显存32G            | NVIDIA V100, 显存32G    |
| MindSpore版本           | >=2.0.0                 | >=2.0.0                    |
| 数据集                  | [Burgers数据集](https://download.mindspore.cn/mindscience/mindflow/dataset/applications/physics_driven/burgers_pinns/)      | [Burgers数据集](https://download.mindspore.cn/mindscience/mindflow/dataset/applications/physics_driven/burgers_pinns/)                   |
| 参数量                  | 6e4                   | 6e4                   |
| 训练参数                | batch_size=8192, steps_per_epoch=1, epochs=15000 | batch_size=8192, steps_per_epoch=1, epochs=15000 |
| 测试参数                | batch_size=8192, steps=4      | batch_size=8192, steps=4           |
| 优化器                  | Adam                 | Adam                   |
| 训练损失(MSE)           | 0.001                | 0.0001             |
| 验证损失(RMSE)          | 0.010                | 0.008              |
| 速度(ms/step)           | 10                   | 40                |

## Contributor

gitee id：[liulei277](https://gitee.com/liulei277)

email: liulei2770919@163.com