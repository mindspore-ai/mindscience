# SNO求解二维纳维-斯托克斯方程

## 概述

纳维-斯托克斯方程（Navier-Stokes equation）是计算流体力学领域的经典方程，是一组描述流体动量守恒的偏微分方程，简称N-S方程。它在二维不可压缩流动中的涡量形式如下：

$$
\partial_t w(x, t)+u(x, t) \cdot \nabla w(x, t)=\nu \Delta w(x, t)+f(x), \quad x \in(0,1)^2, t \in(0, T]
$$

$$
\nabla \cdot u(x, t)=0, \quad x \in(0,1)^2, t \in[0, T]
$$

$$
w(x, 0)=w_0(x), \quad x \in(0,1)^2
$$

其中$u$表示速度场，$w=\nabla \times u$表示涡量，$w_0(x)$表示初始条件，$\nu$表示粘度系数，$f(x)$为外力合力项。

本案例利用Spectral Neural Operator学习某一个时刻对应涡量到下一时刻涡量的映射，实现二维不可压缩N-S方程的求解：

$$
w_t \mapsto w(\cdot, t+1)
$$

### 技术路径

SNO神经算子（Spectral Neural Operator, SNO）的架构类似傅里叶神经算子FNO，利用多项式而不是FFT变换到谱空间（Chebyshev、Legendre等）。为了计算频谱卷积的正向和逆多项式变换矩阵，应在相应的Gauss正交节点（Chebyshev网格等）对输入进行插值。通过卷积编码器将插值后的输入提升到更高维度的信道空间。编码后的结果用来作为一系列SNO层的输入，每个层对其截断的频谱表示进行线性卷积。SNO层的输出通过卷积解码器投影回目标维度，最后插值回原始节点。

SNO层执行以下操作：将多项式变换$A$应用于光谱空间（Chebyshev，Legendre等）操作；多项式低阶模态上的线性卷积$L$操作，高阶模态上的过滤操作；而后，应用逆变换 $S={A}^{-1}$（回到物理空间）。然后添加输入层的线性卷积 $W$操作，并应用非线性激活层。

![SNO网络结构](images/USNO.png)

## 快速开始

从 [data_driven/navier_stokes/dataset](https://download.mindspore.cn/mindscience/mindflow/dataset/applications/data_driven/navier_stokes/dataset/) 中下载验证所需要的数据集，并保存在`./dataset`目录下。

### 训练方式一：在命令行中调用`train.py`脚本

```shell
python train.py --config_file_path ./configs/sno2d.yaml --mode GRAPH --device_target Ascend --device_id 0
```

其中，

`--config_file_path`表示参数文件的路径，默认值'./configs/sno2d.yaml'；

`--mode`表示运行的模式，'GRAPH'表示静态图模式, 'PYNATIVE'表示动态图模式，默认值'GRAPH'；

`--device_target`表示使用的计算平台类型，可以选择'Ascend'或'GPU'，默认值'Ascend'；

`--device_id`表示使用的计算卡编号，可按照实际情况填写，默认值0；

### 训练方式二：运行Jupyter Notebook

您可以使用[中文版](./SNO2D_CN.ipynb)和[英文版](./SNO2D.ipynb)Jupyter Notebook逐行运行训练和验证代码。

## 结果展示

取1个样本做连续10步预测，并可视化。

![推理误差](./images/result.gif)

## 性能

| 参数            | Ascend               | GPU                |
|:----------------------:|:--------------------------:|:---------------:|
| 硬件资源                 | Ascend，显存64G          | NVIDIA, 显存16G    |
| MindSpore版本         | >=2.2.0                | >=2.2.0                   |
| 数据集                  | [二维NS方程数据集](https://download-mindspore.osinfra.cn/mindscience/mindflow/dataset/applications/data_driven/navier_stokes/)      | [二维NS方程数据集](https://download-mindspore.osinfra.cn/mindscience/mindflow/dataset/applications/data_driven/navier_stokes/)                    |
| 参数量                | 2.4e6                  | 2.4e6                   |
| 训练参数               | batch_size=19, steps_per_epoch=1000, epochs=90 | batch_size=19, steps_per_epoch=1000, epochs=90 |
| 测试参数       | batch_size=1      | batch_size=1               |
| 优化器                  | Adam                 | Adam                 |
| 训练损失(MSE)           | 0.0016                 | 0.0016             |
| 验证损失(RMSE)   | 0.056                | 0.056              |
| 速度(ms/step)           | 13                   | 29                |

## Contributor

gitee id：[juliagurieva](https://gitee.com/JuliaGurieva)

email: gureva-yulya@list.ru
