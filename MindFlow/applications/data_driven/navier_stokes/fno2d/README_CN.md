[ENGLISH](https://atomgit.com/mindspore-lab/mindscience/blob/master/MindFlow/applications/data_driven/navier_stokes/fno2d/README.md) | 简体中文

# FNO求解二维纳维-斯托克斯方程

## 背景简介

### 概述

计算流体力学是21世纪流体力学领域的重要技术之一，其通过使用数值方法在计算机中对流体力学的控制方程进行求解，从而实现流动的分析、预测和控制。传统的有限元法（finite element method，FEM）和有限差分法（finite difference method，FDM）常用于复杂的仿真流程（物理建模、网格划分、数值离散、迭代求解等）和较高的计算成本，往往效率低下。因此，借助AI提升流体仿真效率是十分必要的。

近年来，随着神经网络的迅猛发展，为科学计算提供了新的范式。经典的神经网络是在有限维度的空间进行映射，只能学习与特定离散化相关的解。与经典神经网络不同，傅里叶神经算子（Fourier Neural Operator，FNO）是一种能够学习无限维函数空间映射的新型深度学习架构。该架构可直接学习从任意函数参数到解的映射，用于解决一类偏微分方程的求解问题，具有更强的泛化能力。更多信息可参考[Fourier Neural Operator for Parametric Partial Differential Equations](https://arxiv.org/abs/2010.08895)。

### Fourier Neural Operator

Fourier Neural Operator模型构架如下图所示。图中$w_0(x)$表示初始涡度，通过Lifting Layer实现输入向量的高维映射，然后将映射结果作为Fourier Layer的输入，进行频域信息的非线性变换，最后由Decoding Layer将变换结果映射至最终的预测结果$w_1(x)$。

Lifting Layer、Fourier Layer以及Decoding Layer共同组成了Fourier Neural Operator。

![Fourier Neural Operator模型构架](https://raw.atomgit.com/mindspore-lab/mindscience/raw/master/MindFlow/applications/data_driven/navier_stokes/fno2d/images/FNO.png)

Fourier Layer网络结构如下图所示。图中V表示输入向量，上框表示向量经过傅里叶变换后，经过线性变换R，过滤高频信息，然后进行傅里叶逆变换；另一分支经过线性变换W，最后通过激活函数，得到Fourier Layer输出向量。

![Fourier Layer网络结构](https://raw.atomgit.com/mindspore-lab/mindscience/raw/master/MindFlow/applications/data_driven/navier_stokes/fno2d/images/FNO-2.png)

本教程介绍了使用 FNO 求解纳维-斯托克斯方程的求解方法。

### 纳维-斯托克斯方程（Navier-Stokes equation）

纳维-斯托克斯方程（Navier-Stokes equation）是计算流体力学领域的经典方程，是一组描述流体动量守恒的偏微分方程，简称N-S方程。它在二维不可压缩流动中的涡度形式如下：

$$
\partial_t w(x, t)+u(x, t) \cdot \nabla w(x, t)=\nu \Delta w(x, t)+f(x), \quad x \in(0,1)^2, t \in(0, T]
$$

$$
\nabla \cdot u(x, t)=0, \quad x \in(0,1)^2, t \in[0, T]
$$

$$
w(x, 0)=w_0(x), \quad x \in(0,1)^2
$$

其中$u$表示速度场，$w=\nabla \times u$表示涡度，$w_0(x)$表示初始条件，$\nu$表示粘度系数，$f(x)$为外力合力项。

本案例利用Fourier Neural Operator学习某一个时刻对应涡度到下一时刻涡度的映射，实现二维不可压缩N-S方程的求解：

$$
w_t \mapsto w(\cdot, t+1)
$$

## 模型实现

### 硬件要求

NPU 显存>32G

### mindspore和mindscience版本关系

mindspore>=2.7.0
mindscience==0.8.0

### 安装

1. 确保环境已安装正确版本的MindSpore和MindScience；
2. 克隆MindScience仓或直接获取[MindFlow/applications/data_driven/navier_stokes/fno2d/](https://atomgit.com/mindspore-lab/mindscience/tree/master/MindFlow/applications/data_driven/navier_stokes/fno2d)目录下的代码；

### 数据集

下载训练与测试数据集：[data_driven/navier_stokes/dataset](https://download.mindspore.cn/mindscience/mindflow/dataset/applications/data_driven/navier_stokes/dataset/)。

### 编码

#### 快速开始

从[data_driven/navier_stokes/](https://download.mindspore.cn/mindscience/mindflow/dataset/applications/data_driven/navier_stokes/)中下载验证所需要的数据集，并保存在`./dataset`目录下。

#### 训练方式一：在命令行中调用`train.py`脚本

您可以从这里下载训练脚本[train.py](ttps://atomgit.com/mindspore-lab/mindscience/blob/master/MindFlow/applications/data_driven/navier_stokes/fno2d/train.py)。

```shell
export PYTHONPATH=$(cd ../../../../../ && pwd):$PYTHONPATH
python train.py --config_file_path ./configs/fno2d.yaml --mode GRAPH --device_target Ascend --device_id 0
```

其中，

`--config_file_path`表示参数文件的路径，默认值'./configs/fno2d.yaml'；

`--mode`表示运行的模式，'GRAPH'表示静态图模式，'PYNATIVE'表示动态图模式，默认值'GRAPH'；

`--device_target`表示使用的计算平台类型，可以选择'Ascend'或'GPU'，默认值'Ascend'；

`--device_id`表示使用的计算卡编号，可按照实际情况填写，默认值0；

#### 训练方式二：运行Jupyter Notebook

您可以使用[中文版](https://atomgit.com/mindspore-lab/mindscience/blob/master/MindFlow/applications/data_driven/navier_stokes/fno2d/FNO2D_CN.ipynb)和[英文版](https://atomgit.com/mindspore-lab/mindscience/blob/master/MindFlow/applications/data_driven/navier_stokes/fno2d/FNO2D.ipynb)Jupyter Notebook逐行运行训练和验证代码。

## 实验结果

取1个样本做连续10步预测，并可视化。

![推理误差](https://raw.atomgit.com/mindspore-lab/mindscience/raw/master/MindFlow/applications/data_driven/navier_stokes/fno2d/images/result.gif)

### 性能

| 参数               | Ascend               |
|:----------------------:|:--------------------------:|
| 硬件资源                | Ascend，显存32G            |
| MindSpore版本           | 2.7.0                |
| 数据集                  | [二维NS方程数据集](https://download-mindspore.osinfra.cn/mindscience/mindflow/dataset/applications/data_driven/navier_stokes/)      |
| 参数量                  | 9e5                   | 9e5                   |
| 训练参数                | batch_size=19, steps_per_epoch=1000, epochs=150 |
| 测试参数                | batch_size=1          |
| 优化器                  | Adam                 |
| 训练损失(MSE)           | 0.4                |
| 验证损失(RMSE)          | 0.06                |
| 速度(ms/step)           | 32                   |

## 许可证

- 开源协议：`Apache License 2.0`
- 许可证连接：`https://atomgit.com/mindspore-lab/mindscience/blob/master/LICENSE`

## 致谢

### 贡献者

gitee id：[yi-zhang95](https://gitee.com/yi-zhang95)，[huangwangwen2025](https://gitee.com/huangwangwen2025)

email：zhang_yi_1995@163.com，wangwen@isrc.iscas.ac.cn

## 联系我们

如果您对MindSpore Mindscience有任何建议，请通过[issue](https://atomgit.com/mindspore-lab/mindscience/issues)与我们联系，我们将及时处理。

## 引用

如果本项目对您的研究有帮助，请引用相关工作。