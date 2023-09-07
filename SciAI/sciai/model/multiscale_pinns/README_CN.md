[ENGLISH](README.md) | 简体中文

# 目录

- [Multiscale PINNs 描述](#multiscale-pinns-描述)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [脚本说明](#脚本说明)
    - [脚本和示例代码](#脚本和示例代码)
    - [脚本参数](#脚本参数)
    - [训练流程](#训练流程)
    - [推理流程](#推理流程)

## [Multiscale PINNs 描述](#目录)

在这项研究中，研究人员探讨了物理信息神经网络(PINNs)
在近似具有高频率或多尺度特征的函数方面的限制。他们提出了新颖的架构，这些架构使用时空和多尺度随机傅立叶特征，以实现稳健和准确的PINN模型。
对于一些在传统的PINN模型无法处理的挑战性问题，例如波的传播和反应扩散动态，他们提供了数值示例。

> [论文](https://www.sciencedirect.com/science/article/abs/pii/S0045782521002759): Wang S, Wang H, Perdikaris P. On the
> eigenvector bias of Fourier feature networks: From regression to solving multiscale PDEs with physics-informed neural
> networks[J]. Computer Methods in Applied Mechanics and Engineering, 2021, 384: 113938.

## [数据集](#目录)

训练数据集在运行时随机生成。
数据集的大小由`config.yaml`中的参数`nnum`控制，默认值为1000。

预训练checkpoints文件将会在首次启动时自动下载。
您如果需要手动下载checkpoints文件，
请访问[此链接](https://download.mindspore.cn/mindscience/SciAI/sciai/model/multiscale_pinns/)。

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

- 在 Ascend 或 GPU 上运行、

默认:

```bash
python train.py
```

完整命令:

```bash
python train.py \
    --layers 2 100 100 100 1  \
    --save_ckpt true  \
    --save_fig true  \
    --load_ckpt false  \
    --save_ckpt_path ./checkpoints \
    --figures_path ./figures \
    --load_ckpt_path ./checkpoints/model_10000.ckpt \
    --log_path ./logs \
    --lr 1e-3  \
    --epochs 40000  \
    --batch_size 128  \
    --net_type net_st_ff \
    --print_interval 100 \
    --nnum 1000 \
    --download_data multiscale_pinns \
    --force_download false \
    --amp_level O3 \
    --device_id 0 \
    --mode 0
```

## [脚本说明](#目录)

### [脚本和示例代码](#目录)

文件结构如下:

```text
├── multiscale_pinns
│   ├── checkpoints                  # checkpoint文件
│   ├── data                         # 数据文件
│   ├── figures                      # 结果图片
│   ├── logs                         # 日志文件
│   ├── src                          # 源代码
│   │   ├── network.py               # 网络架构
│   │   ├── plot.py                  # 绘制结果
│   │   └── process.py               # 数据处理
│   ├── config.yaml                  # 超参数配置
│   ├── README.md                    # 模型说明
│   ├── train.py                     # python训练脚本
│   └── eval.py                      # python评估脚本
```

### [脚本参数](#目录)

train.py中的重要参数如下:

| parameter      | description                       | default value                  |
|----------------|-----------------------------------|--------------------------------|
| layers         | 神经网络宽度                            | 2 100 100 100 1                |
| save_ckpt      | 是否保存checkpoint                    | true                           |
| save_fig       | 是否保存和绘制图片                         | true                           |
| load_ckpt      | 是否加载checkpoint                    | false                          |
| save_ckpt_path | checkpoint保存路径                    | ./checkpoints                  |
| figures_path   | 图片保存路径                            | ./figures                      |
| load_ckpt_path | checkpoint加载路径                    | ./checkpoints/model_10000.ckpt |
| log_path       | 日志保存路径                            | ./logs                         |
| lr             | 学习率                               | 1e-3                           |
| epochs         | 时期（迭代次数）                          | 40000                          |
| batch_size     | 批尺寸                               | 128                            |
| net_type       | 网络类型，可以是net_nn, net_ff, net_st_ff | net_st_ff                      |
| print_interval | 时间与loss打印间隔                       | 100                            |
| nnum           | 采样空间数据点数量                         | 1000                           |
| download_data  | 模型所需数据集与(或)checkpoints            | multiscale_pinns               |
| force_download | 是否强制下载数据                          | false                          |
| amp_level      | MindSpore自动混合精度等级                 | O3                             |
| device_id      | 需要设置的设备号                          | None                           |
| mode           | MindSpore静态图模式（0）或动态图模式（1）        | 0                              |

### [训练流程](#目录)

- 在 GPU/Ascend 上运行

   ```bash
   python train.py
   ```

  训练期间的损失值将打印在控制台中， 也可以训练后在日志文件中查看。

  ```bash
  # grep "loss:" log
  step: 0, total loss: 0.50939494, bcs_loss: 0.003596314, ics_loss: 0.49960062, res_loss: 0.0061979764, interval: 1.1743102073669434s, total: 1.1743102073669434s
  step: 100, total loss: 0.46047255, bcs_loss: 5.7089237e-05, ics_loss: 0.4603598, res_loss: 5.562951e-05, interval: 1.724724292755127s, total: 2.8990345001220703s
  step: 200, total loss: 0.55632657, bcs_loss: 6.789516e-05, ics_loss: 0.55621916, res_loss: 3.951962e-05, interval: 1.4499413967132568s, total: 4.348975896835327s
  step: 300, total loss: 0.51157826, bcs_loss: 2.2257213e-05, ics_loss: 0.5115249, res_loss: 3.1086667e-05, interval: 1.463547706604004s, total: 5.812523603439331s
  step: 400, total loss: 0.50273365, bcs_loss: 0.00047580304, ics_loss: 0.50186944, res_loss: 0.00038838215, interval: 1.4236555099487305s, total: 7.2361791133880615s
  step: 500, total loss: 0.5403254, bcs_loss: 6.049385e-05, ics_loss: 0.5401609, res_loss: 0.0001040334, interval: 1.5674817562103271s, total: 8.803660869598389s
  step: 600, total loss: 0.43764904, bcs_loss: 0.00024841435, ics_loss: 0.43689, res_loss: 0.0005106426, interval: 1.2663447856903076s, total: 10.070005655288696s
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