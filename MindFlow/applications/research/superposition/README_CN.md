## 融合物理机理的复杂流动温度场预测模型

## 概述

在工程学领域内，复杂流态下温度场的预测受到了广泛的关注。特别是在那些空间中分布着众多离散的热源/冷源的情境下，这些热源与工质流动的耦合作用，既增加了研究对象设计的自由度，又显著提升了计算流体动力学（CFD）模拟的复杂性及其所需的时间成本。针对这类问题，纯粹基于数据驱动的模型构建往往难以达到高精度预测，且在外推泛化方面表现不佳，其训练过程需要依赖大量高成本的 CFD 数据。
本研究将传统的冷效叠加公式（Seller 公式）抽象化并集成入网络结构中，以此来增强神经算子网络在预测复杂流动下温度场方面的精度及泛化能力，为处理高度复杂的温度场预测问题提供了一种高效且可靠的解决方案。

### 方法

1. 搭建基于叠加原理的算子神经网络 SDNO。其整体架构如图所示。
   其架构由两个关键组件：一是负责基于气膜孔排布信息输入进行温度场计算的计算网络，二是依据 Seller 公式构造，实现物理场结果叠加的叠加网络。深度算子网络的融合叠加原理详细阐述了对同一样本进行前向计算时的两种不同处理方式。标准处理方式（如图所示的黑色箭头路径）仅涉及计算网络的使用，而在融合叠加处理方式中（如图所示的蓝色箭头路径），整个过程被分为“分解-计算-叠加”三个关键步骤。

<div align="center">
<img alt="super_net_training" src="./images/super_net_training.png" width="550"/>
<div align="left">

2. 基于提出的叠加训练策略开展网络训练。在一次前向计算中叠加网络可以被多次循环调用；而且同一个样本将以多种不同方式来参与到模型参数的训练中，叠加网络的这种特性在有效提升模型泛化性。并降低了所需的样本要求，这对于数据获取成本高昂或数据稀缺的领域尤为重要。

<div align="center">
<img alt="superposition_example" src="./images/superposition_example_2.png" width="450"/>
<div align="left">

3. 对于使用训练数据完成训练后的模型，使用少量的数据仅对叠加网络进行进一步的微调训练。

### 创建数据集

本研究聚焦于 Pak-B 叶型端壁上的气膜孔排布。在之前的端壁气膜冷却相关研究中，Pak-B 叶型因其结构和流动特性的典型性，已经被广泛采纳作为研究对象。如图 1 所示的 Pak-B 叶片部件、流道、端壁上的气膜孔及端壁下的冷气腔室。数值计算使用商业 CFD 软件 Star-CCM+完成。[数据集下载链接](https://gr.xjtu.edu.cn/web/songlm/1)

<div align="center">
<img alt="computation_domain" src="./images/computation_domain.png" width="450"/>
<div align="left">
本研究在数据集的组成上采取了非混合策略，即具有不同气膜孔数量的样本被严格区分。
使用所提出的高自由度气膜孔排布参数化方法生成了一系列不同气膜孔数量的几何模型。具体来说，对于气膜孔数量分别为1、2、3、5的配置，各自生成了600组数据样本。对于气膜孔数量较多的情况，即10、15、20孔的配置，各自生成了110组样本。这些数据样本通过所建立的数值计算方法以获取涡轮端壁上表面的温度场分布。
总体而言，所制备的2730组数据被划分为三个子集：一个由2000个样本组成的训练集，其中包含了气膜孔数量为1、2、3、5的样本，每种配置各500个；一个由700个样本组成的验证集，覆盖所有气膜孔数量配置，每种配置各100个；最后，一个包含30个样本的微调测试集，专门针对较高数量的气膜孔配置。
<div align="center">
<img alt="data_set" src="./images/data_set.png" width="550"/>
<div align="left">

## 快速开始

### 训练方式：在命令行中调用 `main.py` 脚本

```
python main.py --config_file_path 'configs/FNO_PAKB' --device_target 'GPU'
```

main.py 脚本的可输入参数有：
'''
`--config_file_path` 表示参数和路径控制文件，默认值'./config.yaml'；
`--batch_size` 表示每次训练送入网络的图片数量，默认值 32；
`--mode` 表示运行的模式，'GRAPH'表示静态图模式, 'PYNATIVE'表示动态图模式， 默认值'GRAPH'；
`--device_target` 表示使用的计算平台类型，可以选择'Ascend'或'GPU'，默认值'Ascend'；
`--device_id` 表示使用的计算卡编号，可按照实际情况填写，默认值 0
'''

## 结果展示

开始训练后得到结果如下：

```
19:46:55INFO Epoch 295: loss_0   0.034830 and loss_1   0.190902
19:46:55INFO Train epoch time: 23.483 s, per step time: 372.740 ms
19:47:19INFO Epoch 296: loss_0   0.034766 and loss_1   0.192154
19:47:19INFO Train epoch time: 23.587 s, per step time: 374.405 ms
19:47:42INFO Epoch 297: loss_0   0.034707 and loss_1   0.188068
19:47:42INFO Train epoch time: 23.563 s, per step time: 374.022 ms
19:48:06INFO Epoch 298: loss_0   0.034653 and loss_1   0.187970
19:48:06INFO Train epoch time: 23.355 s, per step time: 370.709 ms
19:48:29INFO Epoch 299: loss_0   0.034603 and loss_1   0.189416
19:48:29INFO Train epoch time: 23.531 s, per step time: 373.513 ms
```

开始绘图后得到结果如下：

<div align="center">
<img alt="case_A" src="./images/case_A.png" width="550"/>
<img alt="case_B" src="./images/case_B.png" width="550"/>
<img alt="case_C" src="./images/case_C.png" width="550"/>
<br>
<img alt="10_curves_1" src="./images/10_curves_1.jpg" width="550"/>
<img alt="10_curves_2" src="./images/10_curves_2.jpg" width="550"/>
<img alt="10_curves_3" src="./images/10_curves_3.jpg" width="550"/>
<br>
<div align="left">

## 性能

|           参数           |                                                                Ascend                                                                |                                                                 GPU                                                                  |
| :----------------------: | :----------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------: |
|         硬件资源         |                                                           Ascend, 显存 32G                                                           |                                                        NVIDIA V100, 显存 32G                                                         |
|      MindSpore 版本      |                                                                2.2.14                                                                |                                                                2.2.12                                                                |
|          数据集          | [端壁气膜温度场数据集](https://download-mindspore.osinfra.cn/mindscience/mindflow/dataset/applications/research/superposition_spno/) | [端壁气膜温度场数据集](https://download-mindspore.osinfra.cn/mindscience/mindflow/dataset/applications/research/superposition_spno/) |
|          参数量          |                                                                4.4e6                                                                 |                                                                4.4e6                                                                 |
|         训练参数         |                                        batch_size=32, <br>steps_per_epoch=70, <br>epochs=300                                         |                                        batch_size=32, <br>steps_per_epoch=70, <br>epochs=300                                         |
|         测试参数         |                                                            batch_size=32                                                             |                                                            batch_size=32                                                             |
|          优化器          |                                                                AdamW                                                                 |                                                                AdamW                                                                 |
|   动态图-训练损失(MSE)   |                                                                0.0333                                                                |                                                                0.0346                                                                |
|   动态图-验证损失(Rl2)   |                                                                0.1901                                                                |                                                                0.1763                                                                |
| 动态图-外推验证损失(Rl2) |                                                                0.2535                                                                |                                                                0.2320                                                                |
|     单步训练时间(ms)     |                                                                 1524                                                                 |                                                                 373                                                                  |
