[ENGLISH](README.md) | 简体中文

# PeRCNN求解二维Burgers方程

## 概述

伯格斯方程（Burgers' equation）是一个模拟冲击波的传播和反射的非线性偏微分方程，被广泛应用于流体力学，非线性声学，气体动力学等领域，它以约翰内斯·马丁斯汉堡（1895-1981）的名字命名。本案例基于PeRCNN方法，求解二维有粘性情况下的Burgers方程。

## 快速开始

从[链接](https://download.mindspore.cn/mindscience/mindflow/dataset/applications/data_mechanism_fusion/PeRCNN) 中下载验证所需要的数据集，并保存在`./dataset`目录下。需要安装`prettytable`和`imageio`库。

```shell
pip install prettytable
pip install imageio
```

### Stage 1：调用`train.py`脚本实现数据驱动的PeRCNN仿真

```shell
python train.py --config_file_path ./percnn_burgers.yaml --mode GRAPH --device_target Ascend --device_id 0
```

其中，

`--mode`表示运行的模式，'GRAPH'表示静态图模式, 'PYNATIVE'表示动态图模式，详见[MindSpore官网](https://www.mindspore.cn/docs/zh-CN/r2.0/design/dynamic_graph_and_static_graph.html?highlight=pynative)，默认值'GRAPH'；

`--save_graphs`表示是否保存计算图，默认值'False'；

`--save_graphs_path`表示计算图保存的路径，默认值'./graphs'

`--device_target`表示使用的计算平台类型，可以选择'Ascend'或'GPU'，默认值'Ascend'；

`--device_id`表示使用的计算卡编号，可按照实际情况填写，默认值0；

`--config_file_path`表示参数文件的路径，默认值'./configs/percnn_burgers.yaml'；

`--pattern`表示模型范式，该阶段应选择'data_driven'；

### Stage 2：调用`uncover_coef.py`脚本，使用稀疏回归，揭示潜在物理方程

```shell
python uncover_coef.py --mode GRAPH --save_graphs_path ./graphs --device_target Ascend --device_id 0 --config_file_path ./percnn_burgers.yaml --pattern data_driven
```

其中，

`--data_path`表示数据文件的路径，可选择stage1中推理得到的数据，默认值'./data/Burgers_2001x2x100x100_[dt=00025].mat'；

### Stage 3：调用`train.py`脚本，使用基于物理驱动的PeRCNN进行微调

```shell
python train.py --mode GRAPH --save_graphs_path ./graphs --device_target Ascend --device_id 0 --config_file_path ./percnn_burgers.yaml --pattern data_driven
```

其中，

`--config_file_path`表示参数文件的路径，默认值'./configs/percnn_burgers.yaml'；

`--pattern`表示模型范式，该阶段应选择'physics_driven'；

## 结果展示

![Burgers PINNs](images/results.gif)

## 性能

| 参数               | Ascend               | GPU                |
|:----------------------:|:--------------------------:|:---------------:|
| 硬件资源                | Ascend, 显存32G            | NVIDIA V100, 显存32G    |
| MindSpore版本           | >=2.1.0                 | >=2.1.0                   |
| 数据集                  | [PeRCNN数据集](https://download-mindspore.osinfra.cn/mindscience/mindflow/dataset/applications/data_mechanism_fusion/PeRCNN/)      | [PeRCNN数据集](https://download-mindspore.osinfra.cn/mindscience/mindflow/dataset/applications/data_mechanism_fusion/PeRCNN/)                   |
| 参数量                  | 4e3                   | 4e3                  |
| 训练参数                | batch_size=1, steps_per_epoch=1, epochs=15000 | batch_size=8, steps_per_epoch=1, epochs=15000 |
| 测试参数                | batch_size=1          | batch_size=1               |
| 优化器                  | Adam                 | Adam                   |
| 训练损失(MSE)           | 0.001                | 0.001             |
| 验证损失(RMSE)          | 0.06                | 0.10              |
| 速度(ms/step)           | 40                   | 150                |

## Contributor

gitee id：[yi-zhang95](https://gitee.com/yi-zhang95), [chengzrz](https://gitee.com/chengzrz)

email: zhang_yi_1995@163.com