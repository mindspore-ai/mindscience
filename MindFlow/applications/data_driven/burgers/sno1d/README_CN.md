# SNO神经算子求解一维伯格斯方程

## 概述

### 问题描述

伯格斯方程（Burgers' equation）是一个模拟冲击波的传播和反射的非线性偏微分方程，被广泛应用于流体力学，非线性声学，气体动力学等领域。

一维伯格斯方程（1-d Burgers' equation）的应用包括一维粘性流的建模。它的形式如下：

$$
\partial_t u(x, t)+\partial_x (u^2(x, t)/2)=\nu \partial_{xx} u(x, t), \quad x \in(0,1), t \in(0, 1]
$$

$$
u(x, 0)=u_0(x), \quad x \in(0,1)
$$

其中$u$表示速度场，$u_0$表示初始条件，$\nu$表示粘度系数。

本案例利用Spectral Neural Operator学习初始状态到下一时刻状态的映射，实现一维Burgers'方程的求解：

$$
u_0 \mapsto u(\cdot, 1)
$$

#### 技术路径

SNO神经算子（Spectral Neural Operator, SNO）的架构类似傅里叶神经算子FNO，利用多项式而不是FFT变换到谱空间（Chebyshev、Legendre等）。为了计算频谱卷积的正向和逆多项式变换矩阵，应在相应的Gauss正交节点（Chebyshev网格等）对输入进行插值。通过卷积编码器将插值后的输入提升到更高维度的信道空间。编码后的结果用来作为一系列SNO层的输入，每个层对其截断的频谱表示进行线性卷积。SNO层的输出通过卷积解码器投影回目标维度，最后插值回原始节点。

SNO层执行以下操作：将多项式变换$A$应用于光谱空间（Chebyshev，Legendre等）操作；多项式低阶模态上的线性卷积$L$操作，高阶模态上的过滤操作；而后，应用逆变换 $S={A}^{-1}$（回到物理空间）。然后添加输入层的线性卷积 $W$操作，并应用非线性激活层。

![SNO网络结构](images/SNO.png)

## 快速开始

数据集下载地址：[data_driven/burgers](https://download.mindspore.cn/mindscience/mindflow/dataset/applications/data_driven/burgers/) ；将数据集保存在`./dataset`路径下.

### 训练方式一：在命令行中调用`train.py`脚本

```shell
python train.py --config_file_path ./configs/sno1d.yaml --mode GRAPH --device_target Ascend --device_id 0
```

其中，

`--config_file_path`表示配置文件的路径，默认值'./configs/sno1d.yaml'；

`--device_target`表示使用的计算平台类型，可以选择'Ascend'或'GPU'，默认值'GPU'；

`--device_id`表示使用的计算卡编号，可按照实际情况填写，默认值 0；

`--mode`表示运行的模式，'GRAPH'表示静态图模式, 'PYNATIVE'表示动态图模式。

### 训练方式二：运行 Jupyter Notebook

您可以使用[中文版](./SNO1D_CN.ipynb)和[英文版](./SNO1D.ipynb)Jupyter Notebook 逐行运行训练和验证代码。

## 结果展示

取6个样本做连续10步预测，并可视化。

![SNO求解burgers方程](images/result.jpg)

## 性能

| 参数                | Ascend               | GPU                |
|:----------------------:|:--------------------------:|:---------------:|
| 硬件资源                | Ascend, 显存64G            | NVIDIA, 显存16G   |
|  MindSpore版本           | >=2.2.0                 | >=2.2.0                   |
| 数据集                  | [一维Burgers方程分辨率数据集](https://download-mindspore.osinfra.cn/mindscience/mindflow/dataset/applications/data_driven/burgers/)      | [一维Burgers方程分辨率数据集](https://download-mindspore.osinfra.cn/mindscience/mindflow/dataset/applications/data_driven/burgers/)                   |
| 参数量                  | 6e5                   | 6e5                   |
| 训练参数                | channels=128, modes=15, sno_layers=5, batch_size=50, steps_per_epoch=20, epochs=500 | channels=128, modes=15, sno_layers=5, batch_size=50, steps_per_epoch=20, epochs=500 |
| 测试参数                | batch_size=1          | batch_size=1               |
| 优化器                 | Adam                 | Adam                   |
| 训练损失(MSE)         | 3.8e-06                | 3.8e-06             |
| 验证损失(RMSE)          | 0.014               | 0.014             |
|  速度(ms/step)           | 8                  | 14.5               |

取不同分辨率下的数据集进行测试，根据以下结果可得出数据集分辨率对训练结果没有影响。

![SNO求解Burgers方程](images/resolution_test.png)

## Contributor

gitee id：[juliagurieva](https://gitee.com/JuliaGurieva)

email: gureva-yulya@list.ru
