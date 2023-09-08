[ENGLISH](README.md) | 简体中文

# 目录

- [Sympnets 描述](#sympnets-描述)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [脚本说明](#脚本说明)
    - [脚本和示例代码](#脚本和示例代码)
    - [脚本参数](#脚本参数)
    - [训练流程](#训练流程)
    - [推理流程](#推理流程)

## [Sympnets 描述](#目录)

网络模型New symplectic networks（SympNets）被提出用于在数据中辨识哈密顿系统。
共有两种SympNets定义，一种是由线性和激活模块组成的LA-SympNets, 另一种是由梯度模块组成的G-SympNets。
SympNets可以很好地估计辛同胚（symplectic maps）并且有着很好的普适性。
其表现优于基线模型，例如哈密顿神经网络，并且可以更快速地训练和推理。
SympNets也可以通过拓展去学习不规则采样数据中的动态系统。

> [论文](https://www.sciencedirect.com/science/article/pii/S0893608020303063):
> Jin P, Zhang Z, Zhu A, et al. SympNets:Intrinsic structure-preserving symplectic networks for identifying Hamiltonian
> systems[J]. Neural Networks, 2020, 132:166-179.

案例详情: 共研究三个问题场景，分别为单摆问题、双摆问题和三体问题

## [数据集](#目录)

训练数据集在运行时生成。
数据集大小可以在每个问题场景的`init_data`函数中进行设置。

预训练checkpoints文件将会在首次启动时自动下载。
您如果需要手动下载checkpoints文件，
请访问[此链接](https://download.mindspore.cn/mindscience/SciAI/sciai/model/sympnets/)。

## [环境要求](#目录)

- 硬件（Ascend/GPU）
    - 使用 Ascend 或 GPU 处理器准备硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install)
- 欲了解更多信息，请查看以下资源:
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

## [快速开始](#目录)

通过官网安装好MindSpore后，就可以开始训练和验证如下:

- 在 Ascend 或 GPU 上运行

默认:

```bash
python train.py
```

完整命令:

```bash
python train.py \
    --problem pendulum \
    --layers 2 50 50 50 50 3 \
    --save_ckpt true \
    --save_data true \
    --save_fig true \
    --load_ckpt false \
    --save_ckpt_path ./checkpoints \
    --load_ckpt_path ./checkpoints/model_pendulum_LASympNet_iter50000.ckpt \
    --save_data_path ./data \
    --figures_path ./figures \
    --log_path ./logs \
    --print_interval 1000 \
    --lr 1e-3 \
    --batch_size null \
    --epochs 50000 \
    --net_type LA \
    --la_layers 3 \
    --la_sublayers 2 \
    --g_layers 5 \
    --g_width 30 \
    --activation sigmoid \
    --h_layers 4 \
    --h_width 30 \
    --h_activation tanh \
    --download_data sympnets \
    --force_download false \
    --amp_level O3 \
    --device_id 0 \
    --mode 0
```

## [脚本说明](#目录)

### [脚本和示例代码](#目录)

文件结构如下:

```text
├── sympnets
│   ├── checkpoints                  # checkpoint文件
│   ├── data                         # 数据文件
│   ├── figures                      # 结果图片
│   ├── logs                         # 日志文件
│   ├── src                          # 源代码
│   │   ├── cases                    # 不同场景源代码
│   │   │   ├── double_pendulum.py   # 双摆问题定义
│   │   │   ├── pendulum.py          # 单摆问题定义
│   │   │   ├── problem.py           # 问题基类
│   │   │   └── three_body.py        # 三体问题
│   │   ├── nn                       # 神经网络源代码
│   │   │   ├── fnn.py               # 全连接神经网络
│   │   │   ├── hnn.py               # Hamiltonian神经网络
│   │   │   ├── module.py            # 标准模块
│   │   │   └── symnets.py           # symplectic模块
│   │   ├── brain.py                 # 基于mindspore的训练流程
│   │   ├── data.py                  # 数据处理
│   │   ├── stormer_verlet.py        # Stormer-Verlet方法
│   │   └── utils.py                 # 辅助函数
│   ├── config.yaml                  # 超参数配置
│   ├── README.md                    # 英文模型说明
│   ├── README_CN.md                 # 中文模型说明
│   ├── train.py                     # python训练脚本
│   └── eval.py                      # python评估脚本
```

### [脚本参数](#目录)

train.py中的重要参数如下:

| 参数             | 描述                         | 默认值                                                   |
|----------------|----------------------------|-------------------------------------------------------|
| problem        | 问题场景                       | pendulum                                              |
| layers         | 神经网络各层层宽                   | 2 50 50 50 50 3                                       |
| save_ckpt      | 是否保存checkpoint             | true                                                  |
| save_data      | 是否保存data                   | true                                                  |
| save_fig       | 是否绘制和保存图片                  | true                                                  |
| load_ckpt      | 是否加载checkpoint             | false                                                 |
| save_ckpt_path | checkpoint保存路径             | ./checkpoints                                         |
| load_ckpt_path | checkpoint加载路径             | ./checkpoints/model_pendulum_LASympNet_iter50000.ckpt |
| save_data_path | data保存路径                   | ./data                                                |
| figures_path   | 图片保存路径                     | ./figures                                             |
| log_path       | 日志保存路径                     | ./logs                                                |
| print_interval | 时间与loss打印间隔                | 1000                                                  |
| lr             | 学习率                        | 1e-3                                                  |
| batch_size     | 批尺寸                        | null                                                  |
| epochs         | 时期（迭代次数）                   | 50000                                                 |
| net_type       | 神经网络类型                     | LA                                                    |
| la_layers      | LA 神经网络深度                  | 3                                                     |
| la_sublayers   | LA 神经网络子层深度                | 2                                                     |
| g_layers       | G 神经网络深度                   | 5                                                     |
| g_width        | G 神经网络层宽                   | 30                                                    |
| activation     | 神经网络激活函数                   | sigmoid                                               |
| h_layers       | H 神经网络深度                   | 4                                                     |
| h_width        | H 神经网络层宽                   | 30                                                    |
| h_activation   | H 神经网络激活函数                 | tanh                                                  |
| download_data  | 模型所需数据集与(或)checkpoints     | sympnets                                              |
| force_download | 是否强制下载数据                   | false                                                 |
| amp_level      | MindSpore自动混合精度等级          | O3                                                    |
| device_id      | 需要设置的设备号                   | None                                                  |
| mode           | MindSpore静态图模式（0）或动态图模式（1） | 0                                                     |

### [训练流程](#目录)

- 在 GPU/Ascend 上运行

  ```bash
  python train.py
  ```

  训练期间的损失值将打印在控制台中， 也可以训练后在日志文件中查看。

  ```bash
  # grep "loss:" log
  step: 0, loss: 0.006594808, interval: 1.4325690269470215s, total: 1.4325690269470215s
  step: 1000, loss: 3.4384914e-06, interval: 4.685465097427368s, total: 6.11803412437439s
  step: 2000, loss: 3.2273747e-06, interval: 3.522109031677246s, total: 9.640143156051636s
  step: 3000, loss: 3.0768356e-06, interval: 3.4496490955352783s, total: 13.089792251586914s
  step: 4000, loss: 2.8382028e-06, interval: 3.485715389251709s, total: 16.575507640838623s
  step: 5000, loss: 2.4878047e-06, interval: 3.4817137718200684s, total: 20.05722141265869s
  step: 6000, loss: 2.0460955e-06, interval: 3.4582290649414062s, total: 23.515450477600098s
  step: 7000, loss: 1.9280903e-06, interval: 3.470597505569458s, total: 26.986047983169556s
  step: 8000, loss: 1.2088091e-06, interval: 3.4948606491088867s, total: 30.480908632278442s
  step: 9000, loss: 9.309894e-07, interval: 3.5296313762664795s, total: 34.01054000854492s
  step: 10000, loss: 6.1760164e-07, interval: 3.5044443607330322s, total: 37.514984369277954s
  ...
  ```

- 训练结束后，您仍然可以通过保存在`log_path`下面的日志文件回顾训练过程，默认为`./logs`目录中。

- 模型checkpoint将保存在 `save_ckpt_path`中，默认为`./checkpoints` 目录中。

### [推理流程](#目录)

在运行下面的命令之前，请检查使用的`config.yaml` 中的checkpoint加载路径`load_ckpt_path`
进行推理。

- 在 GPU/Ascend 上运行

   ```bash
   python eval.py
   ```

  您可以通过日志文件`log_path`查看过程与结果，默认位于`./logs` 。
  结果图片存放于`figures_path`中，默认位于[`./figures`](./figures)。