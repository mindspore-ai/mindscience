# 目录

- [目录](#目录)
- [波动方程](#波动方程)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [脚本参数](#脚本参数)
- [模型训练](#模型训练)
    - [训练性能与精度](#训练性能与精度)
- [模型推理](#模型推理)
- [随机情况说明](#随机情况说明)
- [MindScience主页](#mindscience主页)

# [波动方程](#目录)

波动方程或称波方程（Wave equation）是用来描述自然界中各种波动现象的微分方程，例如声波、光波和水波等。方程具体描述如下：
$$
\alpha^2 \nabla^2 \phi + f = \dfrac{\partial^2 \phi}{\partial t^2} ,
$$
其中$\alpha$是波的传播速率，$\phi$是波势，$f$是源项。声波反演是地球成像的重要工具。本案例采用物理驱动神经网络（PINN）方法求解二维波动方程全波反演（FWI）问题，即根据有限的观测波势值反演得到传播速率。在此案例中，不失一般性，令$f=0$，通过早期时刻的波势场来施加外力。训练上，相比参考论文【1】，本案例使用了多尺度技巧【2】，进而加速收敛。

参考论文：

1. Rasht‐Behesht M, Huber C, Shukla K, et al. Physics‐Informed Neural Networks (PINNs) for Wave Propagation and Full Waveform Inversions[J]. Journal of Geophysical Research: Solid Earth, 2022, 127(5): e2021JB023120.

2. Ziqi Liu, Wei Cai, Zhi-Qin John Xu* , Multi-scale Deep Neural Network (MscaleDNN) for Solving Poisson-Boltzmann Equation in Complex Domains, Communications in Computational Physics (CiCP)

# [数据集](#目录)

数据集链接：[Data Link](https://repository.library.brown.edu/studio/item/bdr:4c3sezqg/)

- 训练数据：我们分别对求解区域的边界、内部进行随机采样，同时使用早期时刻的波势值快照和监视器最终观测到的部分波势值。本案例中用到的快照及观测数据是通过SpecFem2D仿真模拟得到的（具体设置参考论文【1】）。

注：从数据集相应的链接中下载至指定目录（默认是'./datasets'，可通过dafault_config.yaml中data_dir修改）后可运行相关脚本。文档后边会介绍如何使用相关脚本。

# [环境要求](#目录)

- 硬件（Ascend）
    - 准备Ascend处理器搭建硬件环境。
- 框架
    - [MindSpore](<https://gitee.com/mindspore>）
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/r1.8/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/r1.8/index.html)

# [脚本说明](#目录)

## [脚本及样例代码](#目录)

```path
└─PINNFWI
    ├── README.md                           # 模型说明文档
    ├── requirements.txt                    # 依赖说明文件
    ├── environment.yaml                    # conda环境配置
    ├── eval.py                      # 精度验证脚本
    ├── src                                 # 模型定义源码目录
    │   ├── model.py                       # 模型结构定义
    │   ├── data_gen.py                     # 数据集处理定义
    │   ├── dafault_config.yaml           # 模型配置参数文件
    │   └── customloss.py                   # 损失函数定义
    ├── utils                                #工具定义
    │   └── plot.py                         # 画图程序
    ├── datasets                           #用于存放训练用到的数据
    └── train.py                            # 训练脚本
```

## [脚本参数](#目录)

部分参数如下:
batch_size : 40000
epoch : 200000
learning rate : 1e-4
optimizer: Adam
更多超参数见src/dafault_config.yaml

# [模型训练](#目录)

您可以通过train.py脚本训练该模型，训练过程中模型参数会自动保存：

```shell
python train.py
```

## [训练性能与精度](#目录)

![our_result_phi](Figs/fig1.png "结果")

![our_result_alpha](Figs/fig2.png "结果")

# [模型推理](#目录)

您可以通过eval.py脚本加载测试数据集进行推理，并获取推理精度：

```shell
python eval.py
```

# [随机情况说明](#目录)

train.py中设置了随机种子，网络输入通过均匀分布随机采样。

# [MindScience主页](#目录)

请浏览官网[主页](https://gitee.com/mindspore/mindscience)。
