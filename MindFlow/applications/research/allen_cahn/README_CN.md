[ENGLISH](README.md) | 简体中文

# Allen Cahn问题

## 概述

Allen-Cahn 方程（以 John W. Cahn 和 Sam Allen 命名）是数学物理的反应扩散方程，描述了多组分合金系统中的相分离过程，包括有序-无序转变。 该方程描述了域 $\Omega$ 上标量值状态变量 $\eta$ 在时间间隔 $T$ 内的时间演化。 本例中，MindFLow流体模拟套件用于基于物理驱动的PINNs（Physics INformed Neural Networks）方法求解Allen Cahn方程。

## 快速开始

从[dataset](https://download.mindspore.cn/mindscience/mindflow/dataset/applications/research/allen_cahn/) 中下载验证所需要的数据集，并保存在`./dataset`目录下。

### 训练方式一：在命令行中调用`train.py`脚本

```shell
python train.py --mode GRAPH --device_target Ascend --device_id 0 --config_file_path ./configs/allen_cahn_cfg.yaml
```

其中，
`--mode`表示运行的模式，'GRAPH'表示静态图模式, 'PYNATIVE'表示动态图模式，详见[MindSpore官网](https://www.mindspore.cn/docs/zh-CN/r2.0.0-alpha/design/dynamic_graph_and_static_graph.html?highlight=pynative)，默认值'GRAPH'；

`--device_target`表示使用的计算平台类型，可以选择'Ascend'或'GPU'，默认值'Ascend'；

`--device_id`表示使用的计算卡编号，可按照实际情况填写，默认值0；

`--config_file_path`表示参数文件的路径，默认值'./configs/allen_cahn_cfg.yaml'；

### 训练方式二：运行Jupyter Notebook

您可以使用[中文版](allen_cahn_CN.ipynb)和[英文版](allen_cahn.ipynb) Jupyter Notebook逐行运行训练和验证代码。

## 结果展示

![Allen Cahn PINNs](images/result.jpg)

## 性能

|        参数         |        Ascend               |    GPU       |
|:----------------------:|:--------------------------:|:---------------:|
|     硬件资源         |     Ascend      |      NVIDIA GeForce RTX 4070 Ti       |
|     MindSpore版本   |        2.0.0rc1            |      2.0.0rc1       |
|        训练损失      |       0.12                |       1.2e-06       |
|        验证损失      |        0.2               |       0.005    |
|        速度          |     0.6s/epoch        |    0.09s/epoch  |

## Contributor

gitee id：[yanglin2023](https://gitee.com/yanglin2023)

email: lucky@lucky9.cyou

