[ENGLISH](README.md) | 简体中文

# PeRCNN求解3D反应扩散方程

## 概述

反应扩散方程（reaction-diffusion equation）是非常重要且应用广泛的一类偏微分方程，它描述了物理学中的种种现象，也在化学反应中被广泛使用。

$$
u_t = \mu_u \Delta u - u{v*2} + F(1-v)
$$

$$
v_t = \mu_v \Delta v + u{v*2} + (F+\kappa)v
$$

其中，
$$
\mu_v = 0.1, \mu_u = 0.2, F = 0.025, \kappa = 0.055
$$

在本案例中，拟在$ \Omega \times \tau = {[-50,50]}^3 \times [0,500] $ 的物理域中求解100个时间步的流场演化（时间步长为0.5s），初始条件经历了高斯加噪，采取周期性边界条件。

## 快速开始

从[链接](https://download.mindspore.cn/mindscience/mindflow/dataset/applications/data_mechanism_fusion/PeRCNN) 中下载验证所需要的数据集，并保存在`./dataset`目录下。

### 调用`train.py`脚本实现数据驱动的PeRCNN仿真

```shell
python train.py --mode GRAPH --device_target Ascend --device_id 0 --config_file_path ./configs/percnn_3d_rd.yaml
```

其中，

`--mode`表示运行的模式，'GRAPH'表示静态图模式, 'PYNATIVE'表示动态图模式，详见[MindSpore官网](https://www.mindspore.cn/docs/zh-CN/r2.0/design/dynamic_graph_and_static_graph.html?highlight=pynative)，默认值'GRAPH'；

`--device_target`表示使用的计算平台类型，可以选择'Ascend'或'GPU'，默认值'Ascend'；

`--device_id`表示使用的计算卡编号，可按照实际情况填写，默认值0；

`--config_file_path`表示参数文件的路径，默认值'./configs/percnn_3d_rd.yaml'；

## 结果展示

![3d GS RD](images/result.jpg)

## 性能

|        参数         |        Ascend              |    GPU       |
|:----------------------:|:--------------------------:|:---------------:|
|     硬件资源         |     显存32G；CPU: 2.6GHz, 192核      |      NVIDIA V100 显存32G       |
|     MindSpore版本   |        2.1.0             |      2.1.0       |
|     数据集      |      [3DRD](https://download.mindspore.cn/mindscience/mindflow/dataset/applications/data_mechanism_fusion/PeRCNN)             |     [3DRD](https://download.mindspore.cn/mindscience/mindflow/dataset/applications/data_mechanism_fusion/PeRCNN)        |
|     参数量      |          10078         |         10078    |
|     训练参数      |        batch_size=1, steps_per_epoch=1, epochs=10000              |     batch_size=1, steps_per_epoch=1, epochs=10000        |
|     测试参数      |        batch_size=1,steps=1              |     batch_size=1,steps=1        |
|     优化器      |        Adam              |     Adam       |
|        训练损失(RMSE)      |        2e-3              |       2e-3      |
|        验证损失(RMSE)      |        6e-2               |       6e-2    |
|        速度(ms/step)          |     5300       |    2500 |

## Contributor

gitee id: [chengzrz](https://gitee.com/chengzrz)

email: czrzrichard@gmail.com