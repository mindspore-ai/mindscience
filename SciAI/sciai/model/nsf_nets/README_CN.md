[ENGLISH](README.md) | 简体中文

# 目录

- [Navier-Stokes Flow Nets 描述](#navier-stokes-flow-nets-描述)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [脚本说明](#脚本说明)
    - [脚本和示例代码](#脚本和示例代码)
    - [脚本参数](#脚本参数)
    - [训练流程](#训练流程)
    - [推理流程](#推理流程)

## [Navier-Stokes Flow Nets 描述](#目录)

Navier-Stokes Flow Net（NSFNets）通过训练神经网络求解维纳-斯托克斯方程中涡度-速度（VV）公式和速度-压力（VP）公式。
该模型的训练流程是无监督学习，训练的损失由涡度-速度公式或速度-压力公式的残差以及边界条件得出。
在问题条件缺失或是求解反问题的情况下，使用PINNs比传统计算流体力学方法能够得到更优的解。

> [论文](https://www.sciencedirect.com/science/article/pii/S0021999120307257): Xiaowei Jin, Shengze Cai, Hui Li, George
> Em Karniadakis, NSFnets (Navier-Stokes flow nets): Physics-informed neural networks for the incompressible
> Navier-Stokes equations, Journal of Computational Physics, Volume 426, 2021, 109951, ISSN 0021-9991.

案例详情: 空间域二维Kovasnay流仿真。

## [数据集](#目录)

训练数据集在运行时随机生成。
数据集的大小由`config.yaml`中的问题域内采样参数`n_train`和问题边界采样参数`n_bound`控制，
默认值分别为2601和50。

预训练checkpoints文件将会在首次启动时自动下载。
您如果需要手动下载checkpoints文件，
请访问[此链接](https://download.mindspore.cn/mindscience/SciAI/sciai/model/nsf_nets/)。

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
    --layers 2 50 50 50 50 3 \
    --save_ckpt true \
    --load_ckpt false \
    --save_ckpt_path ./checkpoints \
    --load_ckpt_path ./checkpoints/model_final_float32.ckpt \
    --log_path ./logs \
    --print_interval 10 \
    --n_train 2601 \
    --n_bound 100 \
    --lr 1e-3 1e-4 1e-5 1e-6 \
    --epochs 5000 5000 50000 50000 \
    --download_data nsf_nets \
    --force_download false \
    --amp_level O2 \
    --device_id 0 \
    --mode 0
```

## [脚本说明](#目录)

### [脚本和示例代码](#目录)

文件结构如下:

```text
├── nsf_nets
│   ├── checkpoints                  # checkpoint文件
│   ├── data                         # 数据文件
│   ├── figures                      # 结果图片
│   ├── logs                         # 日志文件
│   ├── src                          # 源代码
│   │   ├── network.py               # 网络架构
│   │   └── process.py               # 数据处理
│   ├── config.yaml                  # 超参数配置
│   ├── README.md                    # 英文模型说明
│   ├── README_CN.md                 # 中文模型说明
│   ├── train.py                     # python训练脚本
│   └── eval.py                      # python评估脚本
```

### [脚本参数](#目录)

train.py中的重要参数如下:

| 参数             | 描述                         | 默认值                                    |
|----------------|----------------------------|----------------------------------------|
| layers         | 神经网络结构                     | 2 50 50 50 50 3                        |
| save_ckpt      | 是否保存checkpoint             | true                                   |
| load_ckpt      | 是否加载checkpoint             | false                                  |
| save_ckpt_path | checkpoint保存路径             | ./checkpoints                          |
| load_ckpt_path | checkpoint加载路径             | ./checkpoints/model_final_float32.ckpt |
| log_path       | 日志保存路径                     | ./logs                                 |
| print_interval | 时间与loss打印间隔                | 10                                     |
| n_train        | 问题域内数据集采样数                 | 2601                                   |
| n_bound        | 问题边界数据集采样数                 | 100                                    |
| lr             | 学习率                        | 1e-3 1e-4 1e-5 1e-6                    |
| epochs         | 时期（迭代次数）                   | 5000 5000 50000 50000                  |
| download_data  | 模型所需数据集与(或)checkpoints     | nsf_nets                               |
| force_download | 是否强制下载数据                   | false                                  |
| amp_level      | MindSpore自动混合精度等级          | O2                                     |
| device_id      | 需要设置的设备号                   | None                                   |
| mode           | MindSpore静态图模式（0）或动态图模式（1） | 0                                      |

### [训练流程](#目录)

- 在 GPU/Ascend 上运行

   ```bash
   python train.py
   ```

  训练期间的损失值将打印在控制台中， 也可以训练后在日志文件中查看。

   ```bash
   # grep "loss:" log
   step: 0, loss: 3.0267472, interval: 10.88703203201294s, total: 10.88703203201294s
   step: 10, loss: 1.9014359, interval: 0.2849254608154297s, total: 11.1719574928283697s
   step: 20, loss: 0.9572897, interval: 0.24947023391723633s, total: 11.42142772674560603s
   step: 30, loss: 0.6608443, interval: 0.24956488609313965s, total: 11.67099261283874568s
   step: 40, loss: 0.61762005, interval: 0.2589101791381836s, total: 11.92990279197692928s
   step: 50, loss: 0.61856925, interval: 0.2607557773590088s, total: 12.19065856933593808s
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