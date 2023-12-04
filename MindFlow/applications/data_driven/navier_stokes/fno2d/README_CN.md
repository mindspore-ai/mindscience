# FNO求解二维纳维-斯托克斯方程

## 概述

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

## 快速开始

从 [data_driven/navier_stokes/dataset](https://download.mindspore.cn/mindscience/mindflow/dataset/applications/data_driven/navier_stokes/dataset/) 中下载验证所需要的数据集，并保存在`./dataset`目录下。

### 训练方式一：在命令行中调用`train.py`脚本

```shell
python train.py --config_file_path ./configs/fno2d.yaml --mode GRAPH --device_target Ascend --device_id 0
```

其中，

`--config_file_path`表示参数文件的路径，默认值'./configs/fno2d.yaml'；

`--mode`表示运行的模式，'GRAPH'表示静态图模式, 'PYNATIVE'表示动态图模式，默认值'GRAPH'；

`--device_target`表示使用的计算平台类型，可以选择'Ascend'或'GPU'，默认值'Ascend'；

`--device_id`表示使用的计算卡编号，可按照实际情况填写，默认值0；

### 训练方式二：运行Jupyter Notebook

您可以使用[中文版](./FNO2D_CN.ipynb)和[英文版](./FNO2D.ipynb)Jupyter Notebook逐行运行训练和验证代码。

## 结果展示

取1个样本做连续10步预测，并可视化。

![推理误差](./images/result.gif)

## 性能

| 参数               | Ascend               | GPU                |
|:----------------------:|:--------------------------:|:---------------:|
| 硬件资源                | Ascend, 显存32G            | NVIDIA V100, 显存32G    |
| MindSpore版本           | >=2.1.0                 | >=2.1.0                   |
| 数据集                  | [二维NS方程数据集](https://download-mindspore.osinfra.cn/mindscience/mindflow/dataset/applications/data_driven/navier_stokes/)      | [二维NS方程数据集](https://download-mindspore.osinfra.cn/mindscience/mindflow/dataset/applications/data_driven/navier_stokes/)                   |
| 参数量                  | 9e5                   | 9e5                   |
| 训练参数                | batch_size=19, steps_per_epoch=1000, epochs=150 | batch_size=19, steps_per_epoch=1000, epochs=150 |
| 测试参数                | batch_size=1          | batch_size=1               |
| 优化器                  | Adam                 | Adam                   |
| 训练损失(MSE)           | 0.7                | 0.7             |
| 验证损失(RMSE)          | 0.06                | 0.06              |
| 速度(ms/step)           | 15                   | 45                |

## Contributor

gitee id：[yi-zhang95](https://gitee.com/yi-zhang95)

email: zhang_yi_1995@163.com
