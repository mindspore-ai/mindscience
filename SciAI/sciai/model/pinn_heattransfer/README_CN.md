[ENGLISH](README.md) | 简体中文

# 目录

- [Heat Transfer PINNs 描述](#heat-transfer-pinns-描述)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [脚本说明](#脚本说明)
    - [脚本和示例代码](#脚本和示例代码)
    - [脚本参数](#脚本参数)
    - [训练流程](#训练流程)
    - [推理流程](#推理流程)

## [Heat Transfer PINNs 描述](#目录)

Raissi等人通过为标准全连接深度神经网络设计自定义损失函数，来强制应用于不同场景的已知物理定律，他们的研究表明，我们可以从噪声大且稀疏的数据中，
以惊人的准确度解决或探索偏微分方程。这类方法在现实生活中的应用非常广泛。

本项目通过物理信息神经网络(PINN)再现了热传导的场景。

<div align="center">

![](./figures/results_200adam_200lbfgs/graph.png)

Figure 1. 3分割的模拟场结果
</div>

> [论文](https://arxiv.org/abs/1711.10561):Raissi M, Perdikaris P, Karniadakis G E.
> Physics informed deep learning (part i): Data-driven solutions of nonlinear partial differential equations[J].
> arXiv preprint arXiv:1711.10561, 2017.

## [数据集](#目录)

用于训练的数据集和预训练checkpoints文件将会在首次启动时自动下载。

文件`./data/1d_transient_100.mat`提供了训练模型所需要的数据集，包含如下变量：

- 数据集大小
    - tau:（100, 1） in [0, 1]
    - eta:（256, 1） in [-1, 1]
    - theta:（100, 256）
- 数据格式: `.mat`文件
    - 注: 数据会在`process.py`中处理
- 数据集在`./data`目录下，目录结构如下:

```text
├── data
│   └── 1d_transient_100.mat
```

您如果需要手动下载数据集或checkpoints文件，
请访问[此链接](https://download.mindspore.cn/mindscience/SciAI/sciai/model/pinn_heattransfer/)。

## [环境要求](#目录)

- 硬件（Ascend/GPU）
    - 使用 Ascend 或 GPU 处理器准备硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install)
- 欲了解更多信息，请查看以下资源:
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

## [快速开始](#目录)

通过官网安装好MindSpore和上面需要的[数据集](#数据集)后，就可以开始训练和验证如下:

- 在 Ascend 或 GPU 上运行

默认:

```bash
python train.py
```

完整命令:

```bash
python train.py \
    --layers 2 20 20 20 20 20 20 20 20 1 \
    --save_ckpt true \
    --save_fig true \
    --load_ckpt false \
    --save_ckpt_path ./checkpoints \
    --load_data_path ./data \
    --load_ckpt_path ./checkpoints/model_200adam_float32.ckpt \
    --figures_path ./figures \
    --log_path ./logs \
    --print_interval 10 \
    --ckpt_interval 10 \
    --lr 0.03 \
    --n_t 200 \
    --n_f 10000 \
    --b1 0.9 \
    --epochs 200 \
    --lbfgs false \
    --nt_epochs 200 \
    --download_data pinn_heattransfer \
    --force_download false \
    --amp_level O0 \
    --device_id 0 \
    --mode 0
```

## [脚本说明](#目录)

### [脚本和示例代码](#目录)

文件结构如下:

```text
├── pinn_heattransfer
│   ├── checkpoints                  # checkpoint文件
│   ├── data                         # 数据文件
│   │   └── 1d_transient_100.mat     # 1d transient 100 matlab数据集
│   ├── figures                      # 结果图片
│   ├── logs                         # 日志文件
│   ├── src                          # 源代码
│   │   ├── network.py               # 网络架构
│   │   ├── plot.py                  # 绘制结果
│   │   └── process.py               # 数据处理
│   ├── config.yaml                  # 超参数配置
│   ├── README.md                    # 英文模型说明
│   ├── README_CN.md                 # 中文模型说明
│   ├── train.py                     # python训练脚本
│   └── eval.py                       # python评估脚本
```

### [脚本参数](#目录)

train.py中的重要参数如下:

| parameter      | description                | default value                            |
|----------------|----------------------------|------------------------------------------|
| layers         | 神经网络宽度                     | 2 20 20 20 20 20 20 20 20 1              |
| save_ckpt      | 是否保存checkpoint             | true                                     |
| save_fig       | 是否保存和绘制图片                  | true                                     |
| load_ckpt      | 是否加载checkpoint             | false                                    |
| save_ckpt_path | checkpoint保存路径             | ./checkpoints                            |
| load_data_path | 加载数据的路径                    | ./data                                   |
| load_ckpt_path | checkpoint加载路径             | ./checkpoints/model_200adam_float32.ckpt |
| figures_path   | 图片保存路径                     | ./figures                                |
| log_path       | 日志保存路径                     | ./logs                                   |
| print_interval | 时间与loss打印间隔                | 10                                       |
| ckpt_interval  | checkpoint保存间隔             | 10                                       |
| lr             | 学习率                        | 0.03                                     |
| n_t            | 训练数据采样点数                   | 200                                      |
| n_f            | 域内采样位置个数                   | 10000                                    |
| b1             | 一阶衰变率                      | 0.9                                      |
| epochs         | 时期（迭代次数）                   | 200                                      |
| lbfgs          | 是否使用L-BFGS                 | false                                    |
| nt_epochs      | L-BFGS时期（迭代次数）             | 200                                      |
| download_data  | 模型所需数据集与(或)checkpoints     | pinn_heattransfer                        |
| force_download | 是否强制下载数据                   | false                                    |
| amp_level      | MindSpore自动混合精度等级          | O0                                       |
| device_id      | 需要设置的设备号                   | None                                     |
| mode           | MindSpore静态图模式（0）或动态图模式（1） | 0                                        |

### [训练流程](#目录)

- 在 GPU/Ascend 上运行

   ```bash
   python train.py
   ```

  训练期间的损失值将打印在控制台中， 也可以训练后在日志文件中查看。

   ```bash
  # grep "loss:" log
  step: 0, loss: 1.1074585, interval: 23.024836778640747s, total: 23.024836778640747s
  step: 10, loss: 0.24028176, interval: 29.892443895339966s, total: 52.91728067398071s
  step: 20, loss: 0.2183091, interval: 29.853577613830566s, total: 82.77085828781128s
  step: 30, loss: 0.14730139, interval: 29.850199937820435s, total: 112.62105822563171s
  step: 40, loss: 0.068295084, interval: 29.85150957107544s, total: 142.47256779670715s
  step: 50, loss: 0.045247488, interval: 29.851693391799927s, total: 172.32426118850708s
  step: 60, loss: 0.093925714, interval: 29.878837823867798s, total: 202.20309901237488s
  step: 70, loss: 0.04949145, interval: 29.850360870361328s, total: 232.0534598827362s
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