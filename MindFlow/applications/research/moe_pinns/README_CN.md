[ENGLISH](README.md) | 简体中文

# MOE-PINNs

## 概述

### 整体背景

MOE-PINNs的核心思想是将多个专家网络集成到一个整体网络中，以解决复杂问题。每个专家网络负责特定方面的预测，然后通过混合或加权它们的输出，得到最终的预测结果。MOE-PINNs通过在PINNs网络后面引入分路器（router）和聚合器（aggregator）来实现。分路器将输入数据分配给不同的专家网络，聚合器将专家网络的输出组合成最终的预测结果。分路器和聚合器可以是任何类型的网络，例如MLP、CNN、RNN等。本案例中，分路器和聚合器均为MLP网络。

### 模型结构

模型结构如下图所示：

![cylinder_flow](images/moe_pinns_structure.png)

首先将空间坐标和时间坐标输入到PINNs主干网络中，随后将得到的输出结果输入到分路器中，分路器将输入数据分配给不同的专家网络，每个专家网络负责特定方面的预测，然后通过聚合器将专家网络的输出组合成最终的预测结果。预测结果通过和真实值对比得到三种损失项，最终通过MGDA优化器实现多个优化目标的联合优化。

### 验证方程

本方法在Burgers、Cylinder Flow, Periodic Hill三个数据集上进行验证。

#### Burgers数据集简介

伯格斯方程（Burgers' equation）是一个模拟冲击波的传播和反射的非线性偏微分方程。

Burgers'方程的形式如下：

$$
u_t + uu_x = \epsilon u_{xx}, \quad x \in[-1,1], t \in[0, T],
$$

其中$\epsilon=0.01/\pi$，等号左边为对流项，右边为耗散项，本案例使用迪利克雷边界条件和正弦函数的初始条件，形式如下：

$$
u(t, -1) = u(t, 1) = 0,
$$

$$
u(0, x) = -sin(\pi x),
$$

本案例利用PINNs方法学习位置和时间到相应物理量的映射$(x, t) \mapsto u$，实现Burgers'方程的求解。

#### Cylinder Flow数据集简介

圆柱绕流，即二维圆柱低速非定常绕流，流动特性与雷诺数`Re`有关。

纳维-斯托克斯方程（Navier-Stokes equation），简称`N-S`方程，是流体力学领域的经典偏微分方程，在粘性不可压缩情况下，无量纲`N-S`方程的形式如下：

$$
\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} = 0
$$

$$
\frac{\partial u} {\partial t} + u \frac{\partial u}{\partial x} + v \frac{\partial u}{\partial y} = - \frac{\partial p}{\partial x} + \frac{1} {Re} (\frac{\partial^2u}{\partial x^2} + \frac{\partial^2u}{\partial y^2})
$$

$$
\frac{\partial v} {\partial t} + u \frac{\partial v}{\partial x} + v \frac{\partial v}{\partial y} = - \frac{\partial p}{\partial y} + \frac{1} {Re} (\frac{\partial^2v}{\partial x^2} + \frac{\partial^2v}{\partial y^2})
$$

其中，`Re`表示雷诺数。

本案例利用PINNs方法学习位置和时间到相应流场物理量的映射，实现`N-S`方程的求解：

$$
(x, y, t) \mapsto (u, v, p)
$$

#### Periodic Hill数据集简介

雷诺平均Navier-Stokes方程求解周期山流动问题是流体力学和气象学领域中的一个经典数值模拟案例，用于研究空气或流体在周期性山地地形上的流动行为。雷诺平均动量方程如下：

$$
\rho \bar{u}_j \frac{\partial \bar{u}_i}{\partial x_j}=\rho \bar{f}_i+\frac{\partial}{\partial x_j}\left[-\bar{p} \delta_{i j}+\mu\left(\frac{\partial \bar{u}_i}{\partial x_j}+\frac{\partial \bar{u}_j}{\partial x_i}\right)-\rho \overline{u_i^{\prime} u_j^{\prime}}\right]
$$

## 快速开始

从[physics_driven/burgers_pinns/dataset](https://download.mindspore.cn/mindscience/mindflow/dataset/applications/physics_driven/burgers_pinns/dataset/)、[physics_driven/cylinder_flow_pinns/dataset](https://download.mindspore.cn/mindscience/mindflow/dataset/applications/physics_driven/flow_past_cylinder/dataset/) 、[dataset/periodic_hill_2d](https://download.mindspore.cn/mindscience/mindflow/dataset/periodic_hill_2d/)中下载验证所需要的数据集，并保存在`./dataset`目录下。

### 训练方式一：在命令行中调用`train.py`脚本

```shell
python train.py --case burgers --mode GRAPH --device_target Ascend --device_id 0 --config_file_path ./configs/burgers.yaml
```

其中，
`--case`表示案例的选择，'burgers'表示选择burgers方程，'cylinder_flow'表示选择Navier-Stokes方程的圆柱绕流数据集，'periodic_hill'表示选择雷诺平均Navier-Stokes方程对周期山数据集进行训练。

`--mode`表示运行的模式，'GRAPH'表示静态图模式, 'PYNATIVE'表示动态图模式，详见[MindSpore官网](https://www.mindspore.cn/docs/zh-CN/r2.0.0-alpha/design/dynamic_graph_and_static_graph.html?highlight=pynative)，默认值'GRAPH'；

`--device_target`表示使用的计算平台类型，可以选择'Ascend'或'GPU'，默认值'Ascend'；

`--device_id`表示使用的计算卡编号，可按照实际情况填写，默认值0；

`--config_file_path`表示参数文件的路径，默认值'./configs/burgers.yaml'；

### 训练方式二：运行Jupyter Notebook

您可以使用[中文版](./moe_pinns.ipynb)和[英文版](./moe_pinns.ipynb)Jupyter Notebook逐行运行训练和验证代码。

## 结果展示

### Burgers方程上的验证效果

经过4000个epoch的训练后，模型预测值如下图所示：

![Burgers](images/burgers_4000-result.jpg)

### Navier-Stokes方程上的验证效果

经过4000个epoch的训练后，模型预测值与真实值对比如下图所示：

![cylinder flow](images/cylinder_FlowField_4000.gif)

### 雷诺平均Navier-Stokes方程上的验证效果

经过160个epoch的训练后，模型预测值与真实值对比如下图所示：

![periodic_hill](images/periodic_500.png)

### 消融实验

为了验证MOE架构的有效性，我们在burgers数据集上测试了专家模型数量分别为0、4、8时MOE-PINNs模型的效果。其中专家模型数量为0对应的为不加任何处理的PINNs网络。具体实验结果如下：

![train_loss](images/loss.png)

可以看到，MOE架构的加入可以有效降低模型损失，提高模型预测精度。

## 性能

### Burgers数据集

|        参数         |        NPU               |    GPU       |
|:----------------------:|:--------------------------:|:---------------:|
|     硬件资源         |     Ascend, 显存32G     |      NVIDIA V100, 显存32G       |
|     MindSpore版本   |        2.0.0             |        |
| 数据集 | [Burgers数据集](https://download.mindspore.cn/mindscience/mindflow/dataset/applications/physics_driven/burgers_pinns/dataset/) |  |
|  参数量 | 1908 |  |
|  训练参数 | batch_size=8192, epochs=20000, steps_per_epoch=1|  |
|  测试参数 | batch_size=8192 |  |
|  优化器 | MGDA| |
|        训练损失      |        7.8e-4               |       8.0e-4       |
|        验证损失      |        0.014              |       0.013       |
|        速度(ms/step)           |     93      |   103  |

### Cylinder Flow数据集

|        参数         |        NPU               |    GPU       |
|:----------------------:|:--------------------------:|:---------------:|
|     硬件资源         |     Ascend, 显存32G   |      NVIDIA V100, 显存32G       |
|     MindSpore版本   |        2.0.0             |    |
| 数据集 | [Cylinder-flow数据集](https://download.mindspore.cn/mindscience/mindflow/dataset/applications/physics_driven/flow_past_cylinder/dataset/) | |
|  参数量 | 2096 |  |
|  训练参数 | batch_size=8192, epochs=20000, steps_per_epoch=2| |
|  测试参数 | batch_size=8192 |  |
|  优化器 | MGDA |  |
|        训练损失      |        1.34e-4               |       1.31e-4     |
|        验证损失      |        0.037             |       0.033      |
|        速度(ms/step)           |     380      |   115 |

### Periodic Hill数据集

|        参数         |        NPU               |    GPU       |
|:----------------------:|:--------------------------:|:---------------:|
|     硬件资源         |     Ascend, 显存32G   |      NVIDIA V100, 显存32G       |
|     MindSpore版本   |        2.0.0             |        |
| 数据集 | [Periodic_hill数据集](https://download.mindspore.cn/mindscience/mindflow/dataset/periodic_hill_2d/) |  |
|  参数量 | 2328 |  |
|  训练参数 | batch_size=1000, epochs=1000, steps_per_epoch=200 | |
|  测试参数 | batch_size=1000 | |
|  优化器 | MGDA |  |
|        训练损失      |        5.12e-4          |   5.17e-4       |
|        验证损失      |           0.167           |   0.177           |
|        速度(ms/step)           |     150       |   96  |

## Contributor

gitee id: [Marc-Antoine-6258](https://gitee.com/Marc-Antoine-6258)

email: 775493010@qq.com
