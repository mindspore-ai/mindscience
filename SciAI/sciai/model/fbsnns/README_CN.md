[ENGLISH](README.md) | 简体中文

# 目录

- [正反向随机神经网络](#正反向随机神经网络)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [脚本说明](#脚本说明)
    - [脚本和示例代码](#脚本和示例代码)
    - [脚本参数](#脚本参数)
    - [训练流程](#训练流程)
    - [推理流程](#推理流程)

## [正反向随机神经网络](#目录)

用于求解偏微分方程（PDE）的经典数值方法存在维度的灾难，因为它们依赖于时空网格。
受到应用于解决与偏微分方程相关的正向和反向问题的现代深度学习技术的启发，Raissi通过深度神经网络逼近未知解，
并利用了大家所熟知的高维偏微分方程和前向-反向随机微分方程之间的关系。

在这个repo库中，我们使用Raissi的算法求解了两个偏微分方程：100维的Black-Scholes-Barenblatt方程和20维的Allen-Cahn方程。

> [论文](https://arxiv.org/abs/1804.07010):
> Raissi, Maziar. Forward-Backward Stochastic Neural Networks: Deep Learning of High-dimensional Partial Differential
> Equations. ArXiv preprint arXiv:1804.07010 (2018).

## [数据集](#目录)

训练数据由`class Problem`中的`fetch_minibatch`方法生成，
维度由 `config.yaml`中的以下参数控制:

- batch_size（m）: 轨迹数
- num_snapshots（n）: 时间快照数
- layers[0]（dim）: 维度数

生成的训练数据:

- t: （m, n+1, 1）
- w: （m, n+1, dim-1）

预训练checkpoints文件将会在首次启动时自动下载。
您如果需要手动下载checkpoints文件，
请访问[此链接](https://download.mindspore.cn/mindscience/SciAI/sciai/model/fbsnns/)。

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
    --problem allen_cahn_20D \
    --layers 21 256 256 256 256 1 \
    --save_ckpt true \
    --load_ckpt false \
    --save_fig true \
    --save_ckpt_path ./checkpoints/ac \
    --load_ckpt_path ./checkpoints/ac/model_100000_float16.ckpt \
    --figures_path ./figures/ac \
    --lr 1e-3 1e-4 1e-5 1e-6 \
    --epochs 20000 30000 30000 20000 \
    --batch_size 100 \
    --num_snapshots 15 \
    --terminal_time 0.3 \
    --log_path ./logs/ac \
    --download_data fbsnns \
    --force_download false \
    --amp_level O3 \
    --device_id 0 \
    --mode 0
```

如果您想对BlackScholesBarenblatt100D案例运行完整命令，请更改`config.yaml`中的`problem`字段。

## [脚本说明](#目录)

### [脚本和示例代码](#目录)

文件结构如下:

```text
├── fbsnns
│   ├── checkpoints                               # checkpoint文件
│   ├── data                                      # 数据文件
│   ├── figures                                   # 结果图片
│   ├── logs                                      # 日志文件
│   ├── src                                       # 源代码目录
│   │   ├── allen_cahn_20d.py                     # AllenChan20D案例
│   │   ├── black_scholes_barenblatt_100_d.py     # BlackScholesBarenblatt100D案例
│   │   ├── problem.py                            # 基本问题定义
│   │   ├── network.py                            # 网络架构
│   │   └── utils.py                              # 通用方法
│   ├── config.yaml                               # 超参数配置
│   ├── README.md                                 # 英文模型说明
│   ├── README_CN.md                              # 中文模型说明
│   ├── train.py                                  # python训练脚本
│   └── eval.py                                   # python评估脚本
```

### [脚本参数](#目录)

总共两个案例. 在 `config.yaml` 或命令参数中, 可以通过参数 `problem` 来选择案例.

| 参数名     | 含义                                                       | 默认值            |
|---------|----------------------------------------------------------|----------------|
| problem | 用于解决的案例，`allen_cahn_20D`或`black_scholes_barenblatt_100D` | allen_cahn_20D |

对于每个问题案例，参数如下:

| 参数名            | 含义                         | 默认值                                               |
|----------------|----------------------------|---------------------------------------------------|
| layers         | 网络的层宽                      | 21 256 256 256 256 1                              |
| save_ckpt      | 是否保存checkpoint             | true                                              |
| save_fig       | 是否保存和绘制图片                  | true                                              |
| load_ckpt      | 是否加载checkpoint             | false                                             |
| save_ckpt_path | checkpoint保存路径             | ./checkpoints/[problem]                           |
| load_ckpt_path | checkpoint加载路径             | ./checkpoints/[problem]/model_100000_float16.ckpt |
| figures_path   | 图片保存路径                     | ./figures/[problem]                               |
| log_path       | 日志保存路径                     | ./logs/[problem]                                  |
| lr             | 学习率                        | 1e-3 1e-4 1e-5 1e-6                               |
| epochs         | 时期（迭代次数）                   | 20000 30000 30000 20000                           |
| batch_size     | 批次大小                       | 100                                               |
| num_snapshots  | 时间快照数                      | 15                                                |
| terminal_time  | 终止时间                       | 0.3                                               |
| download_data  | 模型所需数据集与(或)checkpoints     | fbsnns                                            |
| force_download | 是否强制下载数据                   | false                                             |
| amp_level      | MindSpore自动混合精度等级          | O3                                                |
| device_id      | 需要设置的设备号                   | None                                              |
| mode           | MindSpore静态图模式（0）或动态图模式（1） | 0                                                 |

### [训练流程](#目录)

- 在 GPU/Ascend 上运行

   ```bash
   python train.py
   ```

  经过训练，loss值会输出如下:

   ```bash
  # grep "loss:" log
  step: 0, loss: 0.0629, interval: 89.23714256286621s, total: 89.23714256286621s
  step: 500, loss: 0.002146, interval: 6.8948588371276855s, total: 96.1320013999939s
  step: 1000, loss: 0.002295, interval: 6.947064161300659s, total: 103.07906556129456s
  step: 1500, loss: 0.001376, interval: 6.927499055862427s, total: 110.00656461715698s
  step: 2000, loss: 0.00161, interval: 7.066746950149536s, total: 117.07331156730652s
  step: 2500, loss: 0.0009484, interval: 7.003252267837524s, total: 124.07656383514404s
  step: 3000, loss: 0.001128, interval: 6.8707005977630615s, total: 130.9472644329071s
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

  您可以通过日志文件`log_path`查看过程与结果，默认位于`./logs/[case]` 。
  结果图片存放于`figures_path`中，默认位于`./figures/[case]`。