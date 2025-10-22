# DAE-PINN

## 概述

* **电力网络动态安全评估需求** ：随着电力网络中分布式能源资源的整合、市场自由化以及复杂通信和控制算法的采用，电力网络的运行条件和潜在故障场景变得更加多样化，影响其安全性。为了评估电力网络的动态安全性，需要模拟其在面对单一故障时的动态响应，这需要求解一组非线性微分代数方程（DAE），而传统显式积分方案在求解 DAE 时会失败，商业求解器计算成本高、内存需求大，限制了动态安全评估的在线部署。
* **深度学习在科学和工程领域的潜力与挑战** ：尽管深度学习在计算机视觉和自然语言处理等领域取得了巨大成功，但在学习科学和工程动态系统方面应用有限，因为数据收集成本高昂，且大多数传统深度学习方法在数据量有限的情况下缺乏鲁棒性和泛化能力。

## 工作原理

* DAE-PINNs 框架结合了隐式龙格 - 库塔时间步进方案（专为求解 DAE 设计）和物理信息神经网络（PINN）。在时间步进过程中，假设已积分至 $(t_n, y_n, z_n)$，目标是推进至 $(t_{n+1}, y_{n+1}, z_{n+1})$，应用隐式龙格 - 库塔方案后，得到一系列方程，包括内部阶段的更新公式和最终状态的更新公式。
* 通过惩罚方法强制神经网络满足 DAE 作为近似硬约束。在训练过程中，将 DAE 的残差作为损失函数的一部分，使得网络在学习过程中不仅拟合数据，还能满足物理定律所描述的 DAE 方程，从而将物理信息融入到神经网络的学习过程中。

## 方法细节

* **问题设置** ：DAE 以半显式形式给出，包括动态状态 y 和代数变量 z，以及描述微分方程的 f 和代数方程的 g。假设 f 和 g 具有足够高的可微性，并且 DAE 的索引为 1，即雅可比矩阵 g_z 的逆存在且在精确解附近有界，这使得代数方程在局部有唯一解 $z = G(y)$，从而 DAE 可以转化为普通微分方程系统。
* **网络结构** ：与标准的 PINN 类似，DAE-PINNs 通常由输入层、多个隐藏层和输出层组成。输入层接收时间和动态状态等信息，隐藏层通过非线性激活函数进行特征提取和转换，输出层预测代数变量的值。
* **损失函数** ：损失函数由两部分组成，一部分是数据损失，用于拟合初始条件、边界条件等已知点的数据；另一部分是物理损失，即 DAE 的残差损失，通过自动微分计算网络输出对时间和状态变量的导数，代入 DAE 方程得到残差，并将其作为物理损失的一部分。通过优化这两个部分的损失函数，使得网络既能拟合数据，又能满足物理方程。

<p align = "center">
<img src="images/model.png" height="300" />
</p>

[DAE-PINN](https://arxiv.org/abs/2109.04304)的网络结构如上图。

与传统神经网络不同，DAE-PINN 在网络结构中融入了物理信息，通过构造特定的损失函数，使网络在学习过程中不仅拟合数据，还能满足物理定律所描述的 DAE 方程，从而提高了模型的准确性和泛化能力。整体的网络架构如上，分为两个网络分别处理动态状态和代数状态，网络的输入是包括时间信息和动态状态信息，网络的输出是对动态状态和代数状态的预测值，具体来说，会输出动态状态 y 和代数变量 z 的预测结果，如在电力网络案例中，输出动态状态和代数变量的预测以实现对电力网络动态行为的模拟。网络支持使用`fnn`、`attention`、`conv1d`3种backbone。`fnn`为多层感知机网络，`attention`为采用类似transformer attention形式的FFN网络，`conv1d`为使用了`Conv1D`的FFN网络。

## 优势与贡献

* **优势** ：DAE-PINNs 能够有效地学习和模拟具有一定程度刚性的通用电力系统的解轨迹，生成的模拟适用于长时间范围的 DAE 模拟，填补了深度学习方法在处理刚性动力学方面的空白，为解决复杂工程系统中的 DAE 问题提供了一种新的高效方法。
* **贡献** ：作者通过三节点电力网络的案例，验证了 DAE-PINN 在短时间内学习初始条件分布到解轨迹的映射以及长时间模拟 DAE 的能力，展示了其有效性和准确性，为电力系统动态安全评估提供了一种潜在的在线工具。

## 快速开始

### 准备数据集

从[数据下载地址](https://download.mindspore.cn/mindscience/mindenergy/dataset/applications/DAE-PINN/)获取数据集，并将其解压到`DAE-PINN/data`目录下，数据集目录结构如下：

```shell
DAE-PINN/data/
├── IRK_weights
│   ├── Butcher_IRK100.txt
│   ├── Gauss-Legendre-tableau.npz
│   ├── RK-3-8-tableau.npz
│   └── RK-4-tableau.npz
└── data.npz
```

### 训练方式一：在命令行中调用`train.py`脚本

```shell
python -u train.py --config_file ./configs/config.yaml --device_target Ascend --device_id 1 --mode PYNATIVE
```

其中，

`--config_file`表示配置文件的路径，默认值'./configs/vit.yaml'；

`--device_target`表示使用的计算平台类型，可以选择'Ascend'或'CPU'，默认值'Ascend'；

`--device_id`表示使用的计算卡编号，可按照实际情况填写，默认值 1；

`--mode`表示运行的模式，'GRAPH'表示静态图模式, 'PYNATIVE'表示动态图模式，默认值'PYNATIVE'。

### 训练方式二：运行 Jupyter Notebook

您可以使用[Jupyter Notebook](./DAE-PINN.ipynb)逐行运行训练和验证代码。

## 结果展示

训练的loss曲线如下图：
<p align = "center">
<img src="images/loss.png" height="400" />
</p>

网络对于4个动态和1个代数变量预测的L2相对损失如下图:
<p align = "center">
<img src="images/L2relative_error_0.png" height="200" />
<img src="images/L2relative_error_1.png" height="200" />
</p>
<p align = "center">
<img src="images/L2relative_error_2.png" height="200" />
<img src="images/L2relative_error_3.png" height="200" />
<img src="images/L2relative_error_4.png" height="200" />
</p>

## 性能

|     参数      |                       指标                        |
| :-----------: | :-----------------------------------------------: |
|   硬件资源    |                   Atlas 800T A2                   |
| MindSpore版本 |                      >=2.5.0                      |
|    数据集     |                     HyperCube                     |
|    优化器     |                       Adam                        |
|   训练参数    | batch_size=1048, steps_per_epoch=6, epochs=30000 |
|配置参数       |        [config.yaml](./configs/config.yaml)      |
|    参数量     |                        7e4                        |
| 训练损失(MSE) |                       5e-3                        |
| 验证损失(MSE) |                       5e-3                        |
| 速度(ms/step) |                        140                        |

## 贡献者

gitee id: [Brian-K](https://gitee.com/b_rookie)

email: brian_k2023@163.com
