# 目录

- [目录](#目录)
    - [频域麦克斯韦方程(Maxwell's Equation in Frequency Domain)](#频域麦克斯韦方程maxwells-equation-in-frequency-domain)
    - [AI求解频域麦克斯韦方程](#ai求解频域麦克斯韦方程)
    - [数据集](#数据集)
    - [环境要求](#环境要求)
    - [脚本说明](#脚本说明)
        - [脚本及样例代码](#脚本及样例代码)
        - [脚本参数](#脚本参数)
    - [模型训练](#模型训练)
    - [随机情况说明](#随机情况说明)
    - [MindScience主页](#mindscience主页)

## 频域麦克斯韦方程(Maxwell's Equation in Frequency Domain)

频域麦克斯韦方程是一个描述电磁波的椭圆偏微分方程，其基本形式如下：

$$(\nabla^2 + k^2)u=0$$

其中$k=\omega c$是分离常数波数, $\omega$是频率, $c$是光速。

## AI求解频域麦克斯韦方程

AI求解频域麦克斯韦方程的整体网络架构如下：

![network_architecture](./docs/pinns_for_frequency_domain_maxwell.png)

以二维的频域麦克斯韦方程为例，网络输入为$\Omega=(x, y)\in [0,1]^2$, 输出为方程的解$u(x, y)$。基于网络的输出和MindSpore框架的自动微分功能可以构建网络的训练损失函数，该损失函数分为PDE和BC两部分：
$$L_{pde}= \dfrac{1}{N_1}\sum_{i=1}^{N_1} ||(\nabla^2 + k^2)u(x_i, y_i)||^2$$
$$L_{bc} = \dfrac{1}{N_2}\sum_{i=1}^{N_2} ||u(x_i, y_i)||^2$$
为了保证上述方程解的唯一性，我们给定方程的边界条件为$u_{|\partial \Omega}=\sin(kx)$。用户可以自定义分离常数波数$k$，本案例中取值为$k=2$。

## 数据集

AI求解频域麦克斯韦方程时使用自监督方式训练，数据集在运行过程中实时生成，训练与推理数据生成的方式如下：

- 训练数据：每次迭代中，在可行域内部从101*101的均匀网格中选取128个样本点计算损失函数的PDE部分$L_{pde}$；在边界上随机生成128个样本点计算损失函数的BC部分$L_{bc}$。
- 评估数据：在可行域内生成101*101的均匀网格点，其对应的label为方程解析解$u=\sin(kx)$
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
└─FrequencyDomainMaxwell
  ├─README.md
  ├─docs                              # README示意图
  ├─src
    ├──callback.py                    # 回调函数
    ├──config.py                      # 参数配置
    ├──dataset.py                     # 数据集配置
    ├──model.py                       # 网络模型
  ├──solve.py                         # 训练和评估网络
```

### 脚本参数

在`src/config.py`里面可以设置训练的参数和采样参数

```python
Helmholtz2D_config = ed({
    "name": "Helmholtz2D",                  # 方程名称
    "columns_list": ["input", "label"],     # 评估数据集名称
    "epochs": 10,                           # 训练周期
    "batch_size": 128,                      # 单次训练样本数目
    "lr": 0.001,                            # 学习率
    "coord_min": [0.0, 0.0],                # 求解域下限
    "coord_max": [1.0, 1.0],                # 求解域上限
    "axis_size": 101,                       # 网格粗细度
    "wave_number": 2                        # 分离常数波数
})

rectangle_sampling_config = ed({
    'domain' : ed({                         # 定义内部采样率
        'random_sampling' : False,          # 是否使用随机
        'size' : [100, 100],                # 不使用随机采样时的网格分辨率
    }),
    'BC' : ed({                             # 定义边界采样率
        'random_sampling' : True,           # 是否使用随机
        'size' : 128,                       # 使用随机采样时的总采样数
        'with_normal' : False,              # 是否返回边界法向量
    })
})
```

## 模型训练

您可以通过solve.py脚本训练求解频域麦克斯韦方程，训练过程中模型参数会自动保存为检查点文件：

```shell
python solve.py
```

训练过程中会实时显示损失函数值：

```log
epoch: 1 step: 79, loss is 630.0
epoch time: 26461.205 ms, per step time: 334.952 ms
epoch: 2 step: 79, loss is 196.4
epoch time: 278.594 ms, per step time: 3.527 ms
epoch: 3 step: 79, loss is 191.4
================================Start Evaluation================================
Total prediction time: 10.388108491897583 s
l2_error:  0.1875916075312643
=================================End Evaluation=================================
epoch time: 10678.531 ms, per step time: 135.171 ms
epoch: 4 step: 79, loss is 3.998
epoch time: 277.924 ms, per step time: 3.518 ms
epoch: 5 step: 79, loss is 3.082
epoch time: 274.681 ms, per step time: 3.477 ms
epoch: 6 step: 79, loss is 2.469
================================Start Evaluation================================
Total prediction time: 0.009278535842895508 s
l2_error:  0.019952444820775538
=================================End Evaluation=================================
epoch time: 292.866 ms, per step time: 3.707 ms
epoch: 7 step: 79, loss is 1.934
epoch time: 275.578 ms, per step time: 3.488 ms
epoch: 8 step: 79, loss is 2.162
epoch time: 274.334 ms, per step time: 3.473 ms
epoch: 9 step: 79, loss is 1.744
================================Start Evaluation================================
Total prediction time: 0.0029311180114746094 s
l2_error:  0.017332553759497542
=================================End Evaluation=================================
epoch time: 277.262 ms, per step time: 3.510 ms
epoch: 10 step: 79, loss is 1.502
epoch time: 272.946 ms, per step time: 3.455 ms
l2 error: 0.0173325538
per step time: 3.4550081325
```

## 随机情况说明

dataset.py中设置了“create_dataset”函数内的种子，同时还使用了train.py中的随机种子。

## MindScience主页

请浏览官网[主页](https://gitee.com/mindspore/mindscience)。
