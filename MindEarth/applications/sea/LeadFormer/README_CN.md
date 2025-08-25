# LeadFormer: 北极海冰高分辨率智能预报

## 概述

冰间水道是海冰在海浪、风力和洋流作用下形成的线状断裂带，其形态特征能够反映海洋与大气之间物质能量交换的强度，影响着水道表面的湍流热通量。因此，冰间水道的形态及空间分布的准确刻画对研究北极的海冰变化和预测航道通航具有重要意义。冰间水道的形态特征包括长度、宽度和倾角等。冰间水道宽度在一定程度上决定了大气和海洋水热交换的强度，水道倾角反应且影响海冰动力学特征，水道总长度可以作为衡量冰间水道尺度变异及季节和年际变化的指标。高分辨率海冰冰间水道预测模型是当前应对全球气候变暖背景下北极海冰快速变化的关键技术工具。针对海冰变化机理的复杂性和海冰预报的不确定性，***LeadFormer***以北极高分辨率数值模式数据和基于transformer的人工智能模型为支撑，实现北极冰间水道的智能预报，区域覆盖泛北极，分辨率达到2km的高分辨率冰情预报体系。
模型框架图入下图所示

![LeadFormer](images/model.png)

该模型采用编码器-解码器框架，编码阶段通过重叠块嵌入和四级Transformer块实现特征压缩与深化；解码阶段通过MLP和上采样操作逐步重建空间维度；核心创新在于融合Transformer的全局建模能力与CNN的局部感知特性，适用于高精度图像处理任务。

本模型数据集暂不开源，仅开源代码。

## 快速开始

准备数据，然后在`./configs/2km_ice_config.yaml`中修改`data_path`路径（暂不开源数据）。

### 运行方式： 在命令行调用`main`脚本

```python

python main.py --device_id 0 --device_target Ascend --cfg ./configs/diffusion_cfg.yaml --mode train

```

其中， --device_target 表示设备类型，默认Ascend。 --device_id 表示运行设备的编号，默认值0。 --cfg 配置文件路径，默认值"./configs/2km_ice_config.yaml"。 --mode 运行模式，默认值train

### 推理

在`./configs/2km_ice_config.yaml`中设置`model_checkpoint`为diffusion模型ckpt地址。

```python

python main.py --device_id 0 --mode test

```

### 结果展示：

#### 预测结果可视化

下图展示了使用728条样本训练30个epoch后进行推理绘制的结果。
图中，黑色轮廓为地形，彩色条纹为预测结果。

![LeadFormer](images/result.jpg)

### 性能

|        Parameter         |        NPU              |
|:----------------------:|:--------------------------:|
|    硬件版本        |     Ascend， 64G     |
|     mindspore版本   |        2.5.0             |
|     数据集      |      极区图像             |
|     训练参数    |        batch_size=1, steps_per_epoch=728, epochs=30            |
|     测试参数      |        batch_size=1,steps=44              |
|     优化器      |        AdamW              |
|     训练损失(RMSE)      |        0.07727             |
|     冰间水道识别预报准确率(Acc)      |        98.90112%             |
|     冰间水道长度偏差      |        0.09848%             |
|     冰间水道角度偏差      |        6.27244°             |
|     冰间水道宽度偏差      |        1.21519%             |
|     训练资源      |        1Node 8NPU            |

## 贡献者

gitee id: Zhou Chuansai, funfunplus

email: chuansaizhou@163.com, funniless@163.com