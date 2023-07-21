[ENGLISH](README.md) | 简体中文

# 物理嵌入递归卷积神经网络（Physics-encoded Recurrent Convolutional Neural Network，PeRCNN）

## 概述

PDE方程在对物理系统的建模中占据着中心地位，但在流行病学、气象科学、流体力学和生物学等等领域中，很多的底层PDE仍未被完全发掘出来。而对于那些已知的PDE方程，比如Naiver-Stokes方程，对这些方程进行精确数值计算需要巨大的算力，阻碍了数值仿真了在大规模时空系统上的应用。目前，机器学习的进步提供了一种PDE求解和反演的新思路。

![PeRCNN](images/percnn.jpg)

近日，华为与中国人民大学孙浩教授团队合作，基于昇腾AI基础软硬件平台与昇思
MindSpore AI框架提出了一种[物理编码递归卷积神经网络（Physics-encoded Recurrent Convolutional Neural Network， PeRCNN）](https://www.nature.com/articles/s42256-023-00685-7)。相较于物理信息神经网络、ConvLSTM、PDE-NET等方法，模型泛化性和抗噪性明显提升，长期推理精度提升了
10倍以上，在航空航天、船舶制造、气象预报等领域拥有广阔的应用前景，目前该成果已在 nature machine intelligence 上发表。

PeRCNN神经网络强制嵌入物理结构，基于结合部分物理先验设计的π-卷积模块，通过特征图之间的元素乘积实现非线性逼近。该物理嵌入机制保证模型根据我们的先验知识严格服从给定的物理方程。所提出的方法可以应用于有关PDE系统的各种问题，包括数据驱动建模和PDE的发现，并可以保证准确性和泛用性。

伯格斯方程（Burgers' equation）是一个模拟冲击波的传播和反射的非线性偏微分方程，被广泛应用于流体力学，非线性声学，气体动力学等领域，它以约翰内斯·马丁斯汉堡（1895-1981）的名字命名。本案例基于PeRCNN方法，求解二维有粘性情况下的Burgers方程。

## 快速开始

从[链接](https://download.mindspore.cn/mindscience/mindflow/dataset/applications/data_mechanism_fusion/PeRCNN) 中下载验证所需要的数据集，并保存在`./data`目录下。

### Stage 1：调用`train.py`脚本实现数据驱动的PeRCNN仿真

```shell
python train.py --mode GRAPH --save_graphs_path ./graphs --device_target Ascend --device_id 0 --config_file_path ./percnn_burgers.yaml --pattern data_driven
```

其中，

`--mode`表示运行的模式，'GRAPH'表示静态图模式, 'PYNATIVE'表示动态图模式，详见[MindSpore官网](https://www.mindspore.cn/docs/zh-CN/r2.0/design/dynamic_graph_and_static_graph.html?highlight=pynative)，默认值'GRAPH'；

`--save_graphs`表示是否保存计算图，默认值'False'；

`--save_graphs_path`表示计算图保存的路径，默认值'./graphs'

`--device_target`表示使用的计算平台类型，可以选择'Ascend'或'GPU'，默认值'Ascend'；

`--device_id`表示使用的计算卡编号，可按照实际情况填写，默认值0；

`--config_file_path`表示参数文件的路径，默认值'./percnn_burgers.yaml'；

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

`--config_file_path`表示参数文件的路径，默认值'./percnn_burgers.yaml'；

`--pattern`表示模型范式，该阶段应选择'physics_driven'；

## 结果展示

![Burgers PINNs](images/results.gif)
