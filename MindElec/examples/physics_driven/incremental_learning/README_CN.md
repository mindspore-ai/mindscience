# 目录

- [目录](#目录)
    - [麦克斯韦方程组](#麦克斯韦方程组)
    - [AI求解点源麦克斯韦方程族](#ai求解点源麦克斯韦方程族)
    - [数据集](#数据集)
    - [环境要求](#环境要求)
    - [脚本说明](#脚本说明)
        - [脚本及样例代码](#脚本及样例代码)
        - [脚本参数](#脚本参数)
    - [模型描述](#模型描述)
    - [模型预训练](#模型预训练)
    - [预训练性能与精度](#预训练性能与精度)
    - [增量训练求解新方程](#增量训练求解新方程)
    - [增量训练性能与精度](#增量训练性能与精度)
    - [随机情况说明](#随机情况说明)
    - [MindScience主页](#mindscience主页)

## 麦克斯韦方程组

麦克斯韦方程组是一组描述电场、磁场与电荷密度、电流密度之间关系的偏微分方程，有激励源的控制方程具体描述如下：

$$
\nabla\times E=-\mu \dfrac{\partial H}{\partial t} + J(x, t),
$$
$$
\nabla\times H=\epsilon \dfrac{\partial E}{\partial t}
$$

其中$\epsilon,\mu$分别是介质的绝对介电常数、绝对磁导率。$J(x, t)$是电磁仿真过程中的激励源，常用的加源方式包括点源，线源和面源，本案例主要考虑点源的形式，其数学表示为：
$$
J(x, t)=\delta(x - x_0)g(t)
$$

## AI求解点源麦克斯韦方程族

在PINNs方法中，神经网络直接学习方程组的解函数与方程自变量之间的映射关系。当方程组中的特征参数发生变化时，需要重新进行网络训练来得到新方程的解，因此PINNs方法不具备求解同一类方程组的泛化能力。

在该模型中，我们提出了基于物理信息的自解码增量训练方法来克服PINNs不具备泛化能力的问题。该方法包含预训练和增量训练两个步骤，AI求解点源麦克斯韦方程族的整体网络架构如下：

![PINNs_for_Maxwell](./docs/pid_maxwell.png)

其中，$\lambda$为待求方程的可变参数，通常情况下可变参数$\lambda$的分布构成高维空间。为了降低模型复杂度以及训练成本，我们首先将高维可变参数空间映射到由低维向量$Z$表征的低维流形上。然后将流形的特征参数$Z$与方程的输入$X$融合作为点源问题求解网络的输入一起参与到PINNs的训练中，由此可以得到预训练模型。针对新给定的可变参数问题，对预训练模型进行微调即可以得到新方程的解。根据我们的经验，在微调阶段，对网络模型和随机初始化的隐向量同时进行增量训练，或者冻结网络模型参数仅对隐向量进行增量训练可以快速收敛到新方程的解。相比于PINNs方法，该方法针对新问题的求解效率可以提升一个量级。

以二维点源麦克斯韦方程组为例，预训练方程的个数为$9$(即$[\mu/\mu_0, \epsilon/\epsilon_0] = [1,3,5] \times [1,3,5]$)个，$9$个$16$维的隐向量$Z$与自变量$\Omega=(x, y, t)\in [0,1]^3$向拼接作为网络输入, 输出为方程的解$u=(E_x, E_y, H_z)$。基于网络的输出和MindSpore框架的自动微分功能，训练损失函数来自于控制方程(PDE loss)，初始条件(IC loss)，边界条件(BC loss)以及针对隐向量的正则项(reg loss)四部分。这里我们采用电磁场为0的初始值，边界采用二阶Mur吸收边界条件。由于激励源的存在，我们将PDE loss的计算分为两部分：激励源附近区域$\Omega_0$与非激励源区域$\Omega_1$。最终我们整体的损失函数可以表示为：
$$L_{total} = \lambda_{src}L_{src} + \lambda_{src_ic}L_{src_ic} + \lambda_{no_src}L_{no_src} + \lambda_{no_src_ic}L_{no_src_ic} +  \lambda_{bc}L_{bc} + \lambda_{reg}\left \| Z \right \| ^2 $$
其中$\lambda$s表示各项损失函数的权重。为了降低权重选择的难度，我们采用了自适应加权的算法，具体参见我们的论文。

## 数据集

- 预训练数据与增量训练数据：基于五个损失函数，我们分别对有源区域，无源区域，边界，初始时刻进行随机采点，作为网络的输入。
- 评估数据：我们基于传统的时域有限差分算法生成高精度的电磁场。
    - 注：数据在src/dataset.py中处理。

## 环境要求

- 硬件（Ascend）
    - 准备Ascend处理器搭建硬件环境。
- 框架
    - [MindSpore Elec](https://gitee.com/mindspore/mindscience/tree/master/MindElec)
- 如需查看详情，请参见如下资源：
    - [MindSpore Elec教程](https://www.mindspore.cn/mindelec/docs/zh-CN/master/intro_and_install.html)
    - [MindSpore Elec Python API](https://www.mindspore.cn/mindelec/docs/zh-CN/master/mindelec.architecture.html)

## 脚本说明

### 脚本及样例代码

```path
.
└─Maxwell
  ├─README.md
  ├─docs                              # README示意图
  ├─src
    ├──dataset.py                     # 数据集配置
    ├──maxwell.py                     # 点源麦克斯韦方程定义
    ├──lr_scheduler.py                # 学习率下降方式
    ├──callback.py                    # 回调函数
    ├──sampling_config.py             # 随机采样数据集的参数配置文件
    ├──utils.py                       # 功能函数
  ├─config
    ├──pretrain.json                  # 预训练参数配置
    ├──reconstruct.json               # 增量训练参数配置
  ├──mad.py                          # 预训练和增量训练脚本
```

### 脚本参数

数据集采样控制参数在`src/sampling_config.py`文件中配置：

```python
src_sampling_config = edict({         # 有源区域的采样配置
    'domain': edict({                 # 内部点空间采样配置
        'random_sampling': True,      # 是否随机采样
        'size': 262144,               # 采样样本数目
        'sampler': 'uniform'          # 随机采样方式
    }),
    'IC': edict({                     # 初始条件样本采样配置
        'random_sampling': True,      # 是否随机采样
        'size': 262144,               # 采样样本数目
        'sampler': 'uniform',         # 随机采样方式
    }),
    'time': edict({                   # 时间采样配置
        'random_sampling': True,      # 是否随机采样
        'size': 262144,               # 采样样本数目
        'sampler': 'uniform',         # 随机采样方式
    }),
})

no_src_sampling_config = edict({      # 无源区域的采样配置
    'domain': edict({                 # 内部点空间采样配置
        'random_sampling': True,      # 是否随机采样
        'size': 262144,               # 采样样本数目
        'sampler': 'uniform'          # 随机采样方式
    }),
    'IC': edict({                     # 初始条件样本采样配置
        'random_sampling': True,      # 是否随机采样
        'size': 262144,               # 采样样本数目
        'sampler': 'uniform',         # 随机采样方式
    }),
    'time': edict({                   # 时间采样配置
        'random_sampling': True,      # 是否随机采样
        'size': 262144,               # 采样样本数目
        'sampler': 'uniform',         # 随机采样方式
    }),
})

bc_sampling_config = edict({          # 边界区域的采样配置
    'BC': edict({                     # 边界点空间采样配置
        'random_sampling': True,      # 是否随机采样
        'size': 262144,               # 采样样本数目
        'sampler': 'uniform',         # 随机采样方式
        'with_normal': False          # 是否需要边界法向向量
    }),
    'time': edict({                   # 时间采样配置
        'random_sampling': True,      # 是否随机采样
        'size': 262144,               # 采样样本数目
        'sampler': 'uniform',         # 随机采样方式
    }),
})
```

预模型训练及控制参数在`config/pretrain.json`文件中配置：

```python
{
    "Description" : ["PINNs for solve Maxwell's equations"],                # 案例描述
    "Case" : "2D_Mur_Src_Gauss_Mscale_MTL_Physical-Informed Auto Decoder",  # 案例标记
    "coord_min" : [0.0, 0.0],                                               # 矩形计算域x和y轴最小坐标
    "coord_max" : [1.0, 1.0],                                               # 矩形计算域x和y轴最大坐标
    "src_pos" : [0.4975, 0.4975],                                           # 点源位置坐标
    "SrcFrq": 1e+9,                                                         # 激励源主频率
    "range_t" : 4e-9,                                                       # 模拟时长
    "input_center": [0.5, 0.5, 2.0e-9],                                     # 网络输入坐标平移距离
    "input_scale": [2.0, 2.0, 5.0e+8],                                      # 网络输入坐标的缩放系数
    "output_scale": [37.67303, 37.67303, 0.1],                              # 网络输出物理量的缩放系数
    "src_radius": 0.01,                                                     # 高斯平滑后的点源范围半径大小
    "input_size" : 3,                                                       # 网络输入维度
    "output_size" : 3,                                                      # 网络输出维度
    "residual" : true,                                                      # 网络结构是否包含残差模块
    "num_scales" : 4,                                                       # 多尺度网络的子网数目
    "layers" : 7,                                                           # 全连接网络层数(输入、输出加隐藏层)
    "neurons" : 64,                                                         # 隐层神经元个数
    "amp_factor" : 2,                                                       # 网络输入的放大因子
    "scale_factor" : 2,                                                     # 多尺度网络的各子网放大系数
    "save_ckpt" : true,                                                     # 训练中是否保存checkpoint信息
    "load_ckpt" : false,                                                    # 是否加载权重进行增量训练
    "save_ckpt_path" : "./ckpt",                                            # 权重保存路径
    "load_ckpt_path" : "",                                                  # 权重加载路径
    "train_with_eval": false,                                               # 是否边训练边验证
    "test_data_path" : "",                                                  # 加载离线测试数据集的路径
    "lr" : 0.001,                                                           # 初始学习率
    "milestones" : [2000],                                                  # 学习率衰减的里程碑
    "lr_gamma" : 0.25,                                                      # 学习率衰减系数
    "train_epoch" : 3000,                                                   # 迭代训练数据集的次数
    "train_batch_size" : 1024,                                              # 网络训练的批数据大小
    "test_batch_size" : 8192,                                               # 网络推理的批数据大小
    "predict_interval" : 500,                                               # 边训练边推理的迭代间隔步数
    "vision_path" : "./vision",                                             # 可视化结果保存路径
    "summary_path" : "./summary",                                           # mindinsight summary结果保存路径

    "EPS_candidates": [1, 3, 5],                                            # 麦克斯韦方程族电导率取值列表
    "MU_candidates": [1, 3, 5],                                             # 麦克斯韦方程族磁导率取值列表
    "num_scenarios": 9,                                                     # 预训练方程个数
    "latent_vector_size": 16,                                               # 隐向量维度
    "latent_reg": 1.0,                                                      # 隐向量损失函数正则化系数
    "latent_init_std": 1.0                                                  # 随机初始化隐向量方差
}
```

增量训练微调及控制参数在`config/pretrain.json`文件中配置如下：

```python
{
    "Description" : ["PINNs for solve Maxwell's equations"],                # 案例描述
    "Case" : "2D_Mur_Src_Gauss_Mscale_MTL_Physical-Informed Auto Decoder",  # 案例标记
    "coord_min" : [0, 0],                                                   # 矩形计算域x和y轴最小坐标
    "coord_max" : [1, 1],                                                   # 矩形计算域x和y轴最大坐标
    "src_pos" : [0.4975, 0.4975],                                           # 点源位置坐标
    "SrcFrq": 1e+9,                                                         # 激励源主频率
    "range_t" : 4e-9,                                                       # 模拟时长
    "input_center": [0.5, 0.5, 2.0e-9],                                     # 网络输入坐标平移距离
    "input_scale": [2.0, 2.0, 5.0e+8],                                      # 网络输入坐标的缩放系数
    "output_scale": [37.67303, 37.67303, 0.1],                              # 网络输出物理量的缩放系数
    "src_radius": 0.01,                                                     # 高斯平滑后的点源范围半径大小
    "input_size" : 3,                                                       # 网络输入维度
    "output_size" : 3,                                                      # 网络输出维度
    "residual" : true,                                                      # 网络结构是否包含残差模块
    "num_scales" : 4,                                                       # 多尺度网络的子网数目
    "layers" : 7,                                                           # 全连接网络层数(输入、输出加隐藏层)
    "neurons" : 64,                                                         # 隐层神经元个数
    "amp_factor" : 2,                                                       # 网络输入的放大因子
    "scale_factor" : 2,                                                     # 多尺度网络的各子网放大系数
    "save_ckpt" : true,                                                     # 训练中是否保存checkpoint信息
    "save_ckpt_path" : "./ckpt",                                            # 权重保存路径
    "load_ckpt_path" : "",                                                  # 权重加载路径
    "train_with_eval": true,                                                # 是否边训练边验证
    "test_data_path" : "./benchmark/",                                      # 加载离线测试数据集的路径
    "lr" : 0.001,                                                           # 初始学习率
    "milestones" : [100],                                                  # 学习率衰减的里程碑
    "lr_gamma" : 0.1,                                                      # 学习率衰减系数
    "train_epoch" : 120,                                                   # 迭代训练数据集的次数
    "train_batch_size" : 8192,                                              # 网络训练的批数据大小
    "test_batch_size" : 8192,                                               # 网络推理的批数据大小
    "predict_interval" : 10,                                               # 边训练边推理的迭代间隔步数
    "vision_path" : "./vision",                                             # 可视化结果保存路径
    "summary_path" : "./summary",                                           # mindinsight summary结果保存路径

    "EPS_candidates": [2],                                                  # 麦克斯韦方程族电导率取值列表
    "MU_candidates": [2],                                                   # 麦克斯韦方程族磁导率取值列表
    "num_scenarios": 2,                                                     # 预训练方程个数
    "latent_vector_size": 16,                                               # 隐向量维度
    "latent_reg": 1.0,                                                      # 隐向量损失函数正则化系数
    "latent_init_std": 1.0                                                  # 随机初始化隐向量方差
    "finetune_model": true                                                  # 增量训练是否更新模型权重
    "enable_mtl" : true,                                                    # 增量训练是否采用自适用加权
}
```

## 模型描述

本案例采用多通道残差网络结合Sin激活函数的网络架构。

![network_architecture](./docs/multi-scale-NN.png)

## 模型预训练

您可以通过mad.py脚本训练参数化电磁仿真模型，训练过程中模型参数会自动保存：

```shell
python mad.py --mode=pretrain
```

## 预训练性能与精度

脚本提供了边训练边评估的功能，网络训练的损失函数、性能数据以及精度评估结果如下：

```log
epoch: 1 step: 28, loss is 4.332097
epoch time: 198849.624 ms, per step time: 7101.772 ms
epoch: 2 step: 28, loss is 4.150775
epoch time: 2697.448 ms, per step time: 96.337 ms
epoch: 3 step: 28, loss is 4.062408
epoch time: 2697.567 ms, per step time: 96.342 ms
epoch: 4 step: 28, loss is 3.942519
epoch time: 2695.202 ms, per step time: 96.257 ms
epoch: 5 step: 28, loss is 3.7573988
epoch time: 2684.708 ms, per step time: 95.882 ms
epoch: 6 step: 28, loss is 3.6898723
epoch time: 2688.102 ms, per step time: 96.004 ms
epoch: 7 step: 28, loss is 3.5447907
epoch time: 2685.811 ms, per step time: 95.922 ms
epoch: 8 step: 28, loss is 3.462135
epoch time: 2686.360 ms, per step time: 95.941 ms
epoch: 9 step: 28, loss is 3.33505
epoch time: 2678.746 ms, per step time: 95.669 ms
epoch: 10 step: 28, loss is 3.227073
epoch time: 2684.541 ms, per step time: 95.876 ms
......
epoch time: 2681.309 ms, per step time: 95.761 ms
epoch: 2991 step: 28, loss is 0.062745415
epoch time: 2680.290 ms, per step time: 95.725 ms
epoch: 2992 step: 28, loss is 0.0728458
epoch time: 2677.612 ms, per step time: 95.629 ms
epoch: 2993 step: 28, loss is 0.103519976
epoch time: 2680.193 ms, per step time: 95.721 ms
epoch: 2994 step: 28, loss is 0.086214304
epoch time: 2680.334 ms, per step time: 95.726 ms
epoch: 2995 step: 28, loss is 0.064058885
epoch time: 2681.828 ms, per step time: 95.780 ms
epoch: 2996 step: 28, loss is 0.06439945
epoch time: 2687.958 ms, per step time: 95.998 ms
epoch: 2997 step: 28, loss is 0.06600211
epoch time: 2684.817 ms, per step time: 95.886 ms
epoch: 2998 step: 28, loss is 0.08579312
epoch time: 2676.644 ms, per step time: 95.594 ms
epoch: 2999 step: 28, loss is 0.07677732
epoch time: 2677.484 ms, per step time: 95.624 ms
epoch: 3000 step: 28, loss is 0.061393328
epoch time: 2691.284 ms, per step time: 96.117 ms
==========================================================================================
l2_error, Ex:  0.06892983792636541 , Ey:  0.06803824510149464 , Hz:  0.07061244131423149
==========================================================================================
```

## 增量训练求解新方程

给定一组新的方程参数，您可以通过mad.py脚本加载预训练模型与测试数据集增量训练，从而快速得到新问题的解：

```shell
python mad.py --mode=reconstruct
```

## 增量训练性能与精度

``` log
epoch: 1 step: 32, loss is 3.4485734
epoch time: 207.005 s, per step time: 6468.899 ms
epoch: 2 step: 32, loss is 3.2356246
epoch time: 2.859 s, per step time: 89.334 ms
epoch: 3 step: 32, loss is 3.0757806
epoch time: 2.873 s, per step time: 89.787 ms
epoch: 4 step: 32, loss is 2.9055781
epoch time: 2.864 s, per step time: 89.515 ms
epoch: 5 step: 32, loss is 2.7547212
epoch time: 2.865 s, per step time: 89.530 ms
epoch: 6 step: 32, loss is 2.5957384
epoch time: 2.849 s, per step time: 89.040 ms
epoch: 7 step: 32, loss is 2.4329371
epoch time: 2.848 s, per step time: 88.992 ms
epoch: 8 step: 32, loss is 2.3149633
epoch time: 2.845 s, per step time: 88.894 ms
epoch: 9 step: 32, loss is 2.153984
epoch time: 2.855 s, per step time: 89.222 ms
epoch: 10 step: 32, loss is 2.0002615
epoch time: 2.855 s, per step time: 89.224 ms
......
epoch: 120 step: 32, loss is 0.03491814
epoch time: 2.847 s, per step time: 88.966 ms
epoch: 121 step: 32, loss is 0.034486588
epoch time: 2.856 s, per step time: 89.254 ms
epoch: 122 step: 32, loss is 0.033784416
epoch time: 2.849 s, per step time: 89.044 ms
epoch: 123 step: 32, loss is 0.03252008
epoch time: 2.869 s, per step time: 89.657 ms
epoch: 124 step: 32, loss is 0.031876195
epoch time: 2.860 s, per step time: 89.364 ms
epoch: 125 step: 32, loss is 0.03133876
epoch time: 2.852 s, per step time: 89.128 ms
==================================================================================================
Prediction total time: 53.52936124801636 s
l2_error, Ex:  0.06008509896061373 , Ey:  0.06597097288551895 , Hz:  0.059188475323901625
==================================================================================================
```

## 随机情况说明

train.py中设置了随机种子，网络输入通过均匀分布随机采样。

## MindScience主页

请浏览官网[主页](https://gitee.com/mindspore/mindscience)。
