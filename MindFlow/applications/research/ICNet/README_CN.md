# ICNet不变性约束偏微分控制方程发现

## 背景介绍

由偏微分方程描述的物理规律广泛存在于自然环境之中，物理系统的计算与模拟依赖于准确的基本方程和模型，传统方法推导控制方程主要基于第一性原理，例如Navier-Stokes方程基于动量守恒，传统方法难点在于复杂动力学的模型与方程常常难以推导，例如多相流、神经科学以及生物科学等，在大数据时代，通过人工智能的方法从数据中挖掘控制方程成为一种新的研究思路。已有的数据驱动发现方程的方法依然存在一定的局限性，目前构建过完备库的候选项时缺乏指导原则，无法保证发现的方程满足基本的物理要求，同时在处理复杂多维系统时候选库过大，而难以发现出简约准确的方程。考虑到基本的物理要求（不变性，守恒性等）是很多物理问题出发的基石，因此有必要研究如何在发现方程中施加物理约束。

## 模型框架

模型框架如下图所示：

![ICNet](images/ICNet.png)

图中：

A. 嵌入不变性约束至发现偏微分方程框架中的推导过程示意图；

B. 不变性约束发现偏微分方程的神经网络模块，利用神经网络自动微分求出构建不变性候选函数所需要的偏导数，损失函数包括数据损失Data loss，不变性损失Invariance loss以及增强稀疏性的正则化损失Regularization loss。

## 快速开始

数据集下载地址：[ICNet/dataset](https://download-mindspore.osinfra.cn/mindscience/mindflow/dataset/applications/research/ICNet/). 将数据集保存在`./dataset`路径下。

案例提供两种训练方式

- 训练方式一：在命令行中调用`train_GI_1D_KS.py`脚本

  ```python
  # 在命令行调用train.py进行训练示例
  python train.py --device_target GPU --device_id 0 --config_file_path ./config/ICNet_KS.yaml

  ```

  `--config_path`表示配置文件的路径，默认值为 "./config/config.yaml"。

  在 "./config/config.yaml" 配置文件中：
  'case' 表示案例名称；

  'device' 表示使用的计算平台类型，可以选择 'CPU'、'GPU' 或 'Ascend'，默认值 'GPU'；

  'device_id' 表示后端平台端口号，默认值为 0；

  'network_size' 表示网络大小；

  'learning_rate' 表示学习率；

  'epochs' 表示训练迭代次数；

- 训练方式二：运行Jupyter Notebook

  您可以使用[中文版](./ICNet_CN.ipynb) 和[英文版](./ICNet.ipynb) Jupyter Notebook逐行运行训练和验证代码。

## 案例说明

其中命名为ICNet的Jupyter Notebook 运行的案例为施加伽利略不变性约束的一维Kuramoto–Sivashinsky 方程，本次开源除了提供该方程的运行代码和数据以外，还分别提供了二维Burgers方程（伽利略不变性）、二维单变量 Klein-Gordon 方程（洛伦兹不变性）以及二维耦合变量的 Klein-Gordon 方程（洛伦兹不变性）的代码和数据，可以在命令行中调用`train_GI_2D_Burgers.py`、`train_LI_2D_SKG.py`以及`train_LI_2D_CKG.py` 脚本即可直接运行。

## 性能

|         参数          |           GPU           |        NPU         |
|:-------------------:|:-----------------------:|:------------------:|
|         硬件          | NVIDIA 1080Ti(memory 11G) | Ascend(memory 32G) |
|     MindSpore版本     |         2.2.14          |       2.2.14       |
|        数据大小       |          12800          |       12800        |
|       batch大小       |           1             |        1           |
|        训练步数       |           60w           |        60w         |
|         优化器        |         Adam            |      Adam          |
|  total loss 训练精度(MSE)  |         1.8e-3     |       9.4e-4       |
|  data loss 测试精度(MSE)  |         1.3e-3      |       3.7e-4       |
| invariance loss 训练精度(MSE)  |         5.5e-4          |       5.7e-4       |
| regularization loss 测试精度(MSE)  |         1.9e-7      |       1.8e-7       |
|     性能(s/epoch)      |          0.27           |        0.041        |

## 贡献者

gitee id: [chenchao2024](https://gitee.com/chenchao2024)
email: chenchao@isrc.iscas.ac.cn

## 参考文献

chen c, Li H, Jin X. An invariance constrained deep learning network
for partial differential equation discovery[J]. Physics of Fluids, 2043, 65: 471202.  https://doi.org10.1063/5.02026339
