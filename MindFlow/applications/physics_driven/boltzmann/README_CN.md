[ENGLISH](README.md) | 简体中文

# Boltzmann方程求解

## 概述

玻尔兹曼方程或玻尔兹曼输运方程（Boltzmann transport equation，BTE）是由玻尔兹曼于1872年提出的一个方程，用于描述非平衡状态热力学系统的统计行为。本案例采用MindFlow流体仿真套件，基于神经网络稀疏表示的方法，求解一维的BGK以及二次碰撞项的玻尔兹曼方程。更多信息可参考[Solving Boltzmann equation with neural sparse representation](https://arxiv.org/abs/2302.09233).

## 快速开始

### 训练方式一：在命令行中调用`train.py`脚本

```shell
python train.py --mode GRAPH --device_target GPU --device_id 0 --config_file_path ./config/WaveD1V3_BGK.yaml
```

其中，
`--mode`表示运行的模式，'GRAPH'表示静态图模式, 'PYNATIVE'表示动态图模式，详见[MindSpore官网](https://www.mindspore.cn/docs/zh-CN/r2.0.0-alpha/design/dynamic_graph_and_static_graph.html?highlight=pynative)，默认值'GRAPH'；

`--device_target`表示使用的计算平台类型，可以选择'Ascend'或'GPU'，默认值'GPU'；

`--device_id`表示使用的计算卡编号，可按照实际情况填写，默认值0；

`--config_file_path`表示参数文件的路径，默认值'./WaveD1V3_BGK.yaml'；

我们这里针对不同的Boltzmann方程提供了不同的模型和方法的样例，分别对应不同的配置文件。

其中`WaveD1V3_BGK.yaml`与`WaveD1V3_LR.yaml`对应BGK碰撞项，分别是直接基于离散速度模型进行建模，以及使用神经网络稀疏表示进行建模的方法。

其中`WaveD1V3_FSM.yaml`与`WaveD1V3_LA.yaml`对应二次碰撞项，分别是直接基于快速谱方法模型进行建模，以及使用神经网络稀疏表示进行建模的方法。在运行`WaveD1V3_LA.yaml`对应的样例前，需要先运行`WaveD1V3_FBGK.yaml`样例生成对应的近似解。

### 训练方式二：运行Jupyter Notebook

您可以使用[中文版](https://gitee.com/mindspore/mindscience/blob/master/MindFlow/applications/physics_driven/boltzmann/boltzmannD1V3_CN.ipynb)和[英文版](https://gitee.com/mindspore/mindscience/blob/master/MindFlow/applications/physics_driven/boltzmann/boltzmannD1V3.ipynb)Jupyter Notebook逐行运行训练和验证代码。

## Results

### BGK model

#### WaveD1V3_BGK

![BGK](images/Wave_BGK_kn0.01.png)

#### WaveD1V3_LR

![LR](images/Wave_LR_kn0.01.png)

### Quadratic collision model

#### WaveD1V3_FSM

![FSM](images/Wave_FSM_kn0.01.png)

#### WaveD1V3_LA

![LA](images/Wave_LA_kn0.01.png)

## Contributor

gitee id：positive-one

email: lizhengyi@pku.edu.cn