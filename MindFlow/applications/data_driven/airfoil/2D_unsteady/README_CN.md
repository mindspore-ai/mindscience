# 数据驱动(FNO2D和UNET2D两种backbone)下跨声速翼型复杂流场的多时间步预测

## 背景介绍

高精度非定常流动模拟是计算流体力学中的关键课题，有着广泛的应用场景和广阔的市场价值。然而，传统方法存在着算不快、算不准、算不稳等问题，通过AI方法探索流场演化规律为此提供了新的视角。

本案例在二维跨声速翼型场景下提供了端到端的非定常复杂流场预测解决方案。案例中搭建了傅里叶神经算子(Fourier Neural Operator，FNO)和Unet两种网络结构。可以在保证一定精度的前提下利用*k*个时刻的流场学习接续的*m*个时刻的流场，实现二维可压缩非定常流场的预测，验证了深度学习方法在存在激波等复杂流动结构场景中，对多物理参数变化下非定常流场预测的有效性。

$$
u_{[t_0\sim t_{k-1}]} \mapsto u_{[t_k\sim t_{k+m}]}
$$

![Fig1](./images/img_1.png)
<center>图1. 不同方案下的流场流向速度对比((a-d)：CFD结果；(e-h)：FNO结果；(i-l)Unet结果)</center>

## 技术路径

airfoil2D-unsteady案例主要由两部分组成，即**模型与数据准备**部分和**核心模型**部分。

### 模型与数据准备

案例数据集维度为4，按照THWC排列，其中C(channel)包括流向速度*U*，周向速度*V*和静压*P*。数据准备时，需要将*T_in*个时间步合并作为核心模型的输入，则数据集的input尺寸为(*B, T_in, H, W, C*)，数据集的label尺寸为(*B, T_out, H, W, C*)。模型准备时，训练需要遍历*T_out*，则每个step数据集label尺寸为(*B, H, W, C*)并将数据集input合并成(*B \*T_in, H, W, C*)；完成一个step的训练后，更新input和label，直到*T_out*遍历结束。

### 核心模型

- FNO2D

   FNO2D模型架构如下图所示。*P*和*Q*均为全连接层，其中*P*为Lifting Layer，实现输入向量的高维映射。映射结果作为Fourier Layer的输入，进行频域信息的非线性变换，而Fourier Layer通常会嵌套多层。最后，由*Q*作为Decoding Layer将变换结果映射至最终的预测结果。

![Fig2](./images/FNO.png)

- Unet2D

   Unet2D模型架构如下图所示。其主要由上采样块和下采样块组成。下采样块通过卷积和池化操作，逐步减少数据维度，增加特征信息。上采样块则增加了反卷积层，以逐步增加数据维度，减少特征信息；同时上采样还包括跳跃连接，将下采样的输出与对应的上采样输出连接起来，作为上采样中卷积层的输入。

![Fig3](./images/Unet.png)

## 快速开始

数据集下载地址：[data_driven/airfoil/2D_unsteady](https://download.mindspore.cn/mindscience/mindflow/dataset/applications/data_driven/airfoil/2D_unsteady/). 将数据集保存在`./dataset`路径下.

案例提供两种训练方式

- 在命令行调用train.py

  ```python
  # 在命令行调用train.py进行训练示例
  python train.py --config_path ./config/config.yaml --device_target Ascend --device_id 0 --mode GRAPH --backbone UNET2D

  ```

  其中：

  --config_path 表示配置文件的路径，默认值为 "./config/config.yaml"

  --device_target 表示后端平台的选取，默认值为 "Ascend"

  --device_id 表示后端平台端口号，默认值为 0

  --mode 表示计算模式，默认为静态图模式 "GRAPH"

  --backbone 表示网络架构，默认UNET2D

- 运行Jupyter Notebook

  您可以使用[中文版](./2D_unsteady_CN.ipynb) 和[英文版](./2D_unsteady.ipynb) Jupyter Notebook逐行运行训练和验证代码。

## 性能

|        参数         |        NPU              |    GPU       |
|:----------------------:|:--------------------------:|:---------------:|
|     硬件            |     Ascend(memory 32G)      |   NVIDIA V100(memory 32G) |
|     MindSpore版本   |      2.0.0    |     2.0.0   |
|     T_in/T_out      |      8/32     |     8/32    |
|     数据大小        |      3600     |     3600    |
|     训练步数        |      200      |     200     |
|     优化器           |      AdamWeightDecay        |     AdamWeightDecay       |
|   FNO2D 训练精度(RMSE)  |     6.9e-3       |      6.8e-3      |
| **FNO2D 测试精度(RMSE)** |     **5.5e-3**     |     **5.4e-3**     |
| **FNO2D 性能(s/step)**   |     **0.68**      |     **1.07**      |
| Unet2D 训练精度(RMSE)   |     5.8e-3          |     5.1e-3        |
| **Unet2D 测试精度(RMSE)**|     **5.1e-3**     |     **4.7e-3**     |
| **Unet2D 性能(s/step)**  |     **0.49**      |     **1.46**      |

## 贡献者

gitee id: [mengqinghe0909](https://gitee.com/mengqinghe0909)
email: mengqinghe0909@126.com

## 参考文献

Deng Z, Liu H, Shi B, et al. Temporal predictions of periodic flows using a mesh transformation and deep learning-based strategy[J]. Aerospace Science and Technology, 2023, 134: 108081. https://doi.org/10.1016/j.ast.2022.108081