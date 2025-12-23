[ENGLISH](https://atomgit.com/mindspore-lab/mindscience/blob/master/MindFlow/applications/data_mechanism_fusion/p2c2net/README.md) | 简体中文

# P2C2Net求解二维Burgers方程

## 背景简介

### 概述

**P2C2Net（PDE-Preserved Coarse Correction Network**是一种新型神经网络架构，旨在在粗网格和有限训练数据条件下高效求解时空偏微分方程（PDE）。其原始论文为[P2C2Net: PDE-Preserved Coarse Correction Network for Efficient Prediction of Spatiotemporal Dynamics](https://arxiv.org/pdf/2411.00040)。

![模型架构](https://raw.atomgit.com/mindspore-lab/mindscience/raw/master/MindFlow/applications/data_mechanism_fusion/p2c2net/images/model_architecture.png)

如上图所示，该模型由两个协同模块组成：(1) 可训练的PDE模块：基于高阶数值格式并结合边界条件编码，学习更新粗网格解；(2) 神经网络校正模块：在预测过程中对解进行动态一致的修正。特别地，P2C2Net采用了一种可学习的对称卷积滤波器，其权重在整个模型中共享，可基于神经网络校正后的系统状态精确估计PDE的空间导数。

### Burger's 方程

Burgers 方程是一类非线性偏微分方程，用于描述激波的传播与反射，广泛应用于流体力学、非线性声学、气体动力学等领域。

$$
\frac{\partial \mathbf{u}}{\partial t}=\nu \nabla^2 \mathbf{u}-\mathbf{u}\cdot \nabla \mathbf{u}, t\in [0,T], x\in [0,1]^2
$$

周期边界用于规避人工指定求解域带来的非物理反射/误差，适用于无界域问题和带有周期性的物理结构/现象。核心是要求物理量在求解区域的 “对应边界” 上满足数值相等、导数连续。

$$
\mathbf{u}(\mathbf{x}_1, t)=\mathbf{u}(\mathbf{x}_2, t), \nabla\mathbf{u}(\mathbf{x}_1, t)=\nabla\mathbf{u}(\mathbf{x}_2, t)
$$

其中$\mathbf{x}_1\in \partial \Omega_1, \mathbf{x}_2\in \partial \Omega_2$是边界上的周期对应点。

### 问题描述

在本案例中，我们重点研究如何利用 P2C2Net 高效求解周期边界**二维 Burgers**方程。

$$
\mathbf{u}_t \mapsto \mathbf{u}(\cdot, t+1)
$$

## 模型实现

### 硬件要求

NPU 显存>32G

### MindSpore和MindScience版本关系

MindSpore>=2.5.0
MindScience==0.8.0

### 安装

1. 确保环境已安装正确版本的MindSpore和MindScience；
2. 可能需要安装numpy、pandas、sympy、matplotlib的python包
3. 克隆MindScience仓或直接获取[MindFlow/applications/data_mechanism_fusion/p2c2net](https://atomgit.com/mindspore-lab/mindscience/tree/master/MindFlow/applications/data_mechanism_fusion/p2c2net)目录下的代码；

### 数据集

可以通过[MindFlow/applications/data_mechanism_fusion/p2c2net/src/data_gen.py](https://atomgit.com/mindspore-lab/mindscience/blob/master/MindFlow/applications/data_mechanism_fusion/p2c2net/src/data_gen.py)自动生成；

### 编码

MindFlow求解该问题的具体流程如下：

1. 创建数据集。
2. 训练模型。
3. 检查结果。

#### 1. 创建数据集

数据生成代码可以从此处下载[dataGen.py](https://atomgit.com/mindspore-lab/mindscience/blob/master/MindFlow/applications/data_mechanism_fusion/p2c2net/src/data_gen.py)。

首先运行以下命令以生成训练和测试数据：

```shell
cd src
python dataGen.py
```

#### 2. 模型训练

训练代码可以从此处下载[train_burgers.py](https://atomgit.com/mindspore-lab/mindscience/blob/master/MindFlow/applications/data_mechanism_fusion/p2c2net/train_burgers.py)。

运行以下命令在生成的数据上训练 P2C2Net：

```shell
python p2c2net/train_burgers.py --experiment p2c2net
```

其中：

`--experiment` 是实验目录，应包含位于'config/'下的实验配置文件；

`--mode` 是运行模式。'GRAPH' 表示静态图模式。'PYNATIVE' 表示动态图模式。详见[MindSpore官网](https://www.mindspore.cn/docs/zh-CN/r2.0/design/dynamic_graph_and_static_graph.html?highlight=pynative)。默认值'GRAPH'；

`--device_target` 表示所使用的计算平台类型，可选 'Ascend' 或 'GPU'，默认值为 'Ascend'；

`--device_id` 表示所使用的计算卡编号，默认值为 0；

`--continue` 表示是否从已有的检查点恢复训练，默认值为 False；

`--config_filename` 是配置文件的文件名 (位于 `configs/` 目录下) ，其中定义了实验设置，如模型参数、训练参数等，默认值为 'burgers.json'；

`--train_stage` 表示是否开启训练模式，默认值为 True；

`--test_stage` 表示是否开启测试模式，默认值为 True；

#### 3. 检查结果

训练完成后，实验输出（检查点和评估结果）将保存在你提供的 --experiment 目录下的 result 文件夹中。可使用保存的检查点进行复现评估或继续训练。

## 实验结果

![推理结果](https://raw.atomgit.com/mindspore-lab/mindscience/raw/master/MindFlow/applications/data_mechanism_fusion/p2c2net/images/inference.png)

## 许可证

- 开源协议：`Apache License 2.0`
- 许可证连接：`https://atomgit.com/mindspore-lab/mindscience/blob/master/LICENSE`

## 致谢

### 贡献者

gitee id：[liuguangyuu](https://gitee.com/liuguangyuu)

email：liuguangyuu@outlook.com

## 联系我们

如果您对MindSpore MindScience有任何建议，请通过[issue](https://atomgit.com/mindspore-lab/mindscience/issues)与我们联系，我们将及时处理。

## 引用

如果本项目对您的研究有帮助，请引用相关工作。

- Wang Q, Ren P, Zhou H, et al. P²C²Net: PDE-preserved coarse correction network for efficient prediction of spatiotemporal dynamics[C]//The Thirty-eighth Annual Conference on Neural Information Processing Systems. 2024.