[ENGLISH](README.md) | 简体中文

# 目录

- [Adversarial Uncertainty Quantification PINNs 描述](#adversarial-uncertainty-quantification-pinns-描述)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [脚本说明](#脚本说明)
    - [脚本和示例代码](#脚本和示例代码)
    - [脚本参数](#脚本参数)
    - [训练流程](#训练流程)
    - [推理流程](#推理流程)

## [Adversarial Uncertainty Quantification PINNs 描述](#目录)

Adversarial Uncertainty Quantification PINNs （auq_pinns）模型再现了一个深度学习框架，
该框架使用了物理驱动神经网络（PINNs）来量化和传播由非线性微分方程控制的系统中的不确定性。
具体来说，该模型使用潜在变量来构建系统状态的概率表示，
并提出了一种对抗推理的网络训练流程，训练过程中同时要求神经网络的预测值满足由偏微分方程表示的物理定律。
该模型为表征物理系统输出的不确定性提供了一个灵活的框架。

> [论文](https://www.sciencedirect.com/science/article/pii/S0021999119303584):
> Yibo Yang, Paris Perdikaris, Adversarial uncertainty quantification in physics-informed neural networks, Journal of
> Computational Physics, 2019, ISSN 0021-9991

案例详情: 使用Adversarial Uncertainty Quantification PINNs和带有噪声的数据，求解一维微分方程。

## [数据集](#目录)

在训练流程中，训练数据集将会随机生成，无需外部数据。用于验证的数据集和预训练checkpoints文件将会在首次启动时自动下载。
数据集的大小由`config.yaml`中的参数`n_col`和`n_bound`控制，默认值分别为100和20。

在验证阶段，文件`./data/ODE2000.mat`提供了与训练时相同的概率分布进行Monte-Carlo随机采样的结果。
在输入为某一特定值时，可以使用该数据与神经网络预测值的分布进行对比。

- 数据集大小
    - U: [-1, 1] 中的 （2000, 201）

您如果需要手动下载验证数据集或checkpoints文件，
请访问[此链接](https://download.mindspore.cn/mindscience/SciAI/sciai/model/auq_pinns/)。

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
    --layers_p 2 50 50 50 50 1 \
    --layers_q 2 50 50 50 50 1 \
    --layers_t 2 50 50 1 \
    --print_interval 100 \
    --save_fig true \
    --save_ckpt true \
    --load_ckpt false \
    --save_ckpt_path ./checkpoints \
    --load_ckpt_path \
        ./checkpoints/discriminator/model_iter_30000_float32.ckpt \
        ./checkpoints/generator/model_iter_150000_float32.ckpt \
    --ckpt_interval 400 \
    --figures_path ./figures \
    --load_data_path ./data \
    --log_path ./logs \
    --lam 1.5 \
    --beta 1 \
    --n_col 100 \
    --n_bound 20 \
    --epochs 30001 \
    --lr 1e-4 \
    --term_t 1 \
    --term_kl 5 \
    --download_data auq_pinns \
    --force_download false \
    --amp_level O2 \
    --device_id 0 \
    --mode 0
```

## [脚本说明](#目录)

### [脚本和示例代码](#目录)

文件结构如下:

```text
├── auq_pinns
│   ├── checkpoints                  # checkpoint文件
│   ├── data                         # 数据文件
│   │   └── ODE2000.mat              # Monte-Carlo随机采样验证数据集
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
│   └── eval.py                      # python评估脚本
```

### [脚本参数](#目录)

`train.py`中的重要参数如下:

| 参数             | 描述                         | 默认值                                                                                                                   |
|----------------|----------------------------|-----------------------------------------------------------------------------------------------------------------------|
| layers_p       | decoder 神经网络宽度             | 2 50 50 50 50 1                                                                                                       |
| layers_q       | encoder 神经网络宽度             | 2 50 50 50 50 1                                                                                                       |
| layers_t       | discriminator 神经网络宽度       | 2 50 50 1                                                                                                             |
| print_interval | 时间与loss打印间隔                | 100                                                                                                                   |
| save_fig       | 是否保存和绘制图片                  | true                                                                                                                  |
| save_ckpt      | 是否保存checkpoint             | true                                                                                                                  |
| load_ckpt      | 是否加载checkpoint             | false                                                                                                                 |
| save_ckpt_path | checkpoint保存路径             | ./checkpoints                                                                                                         |
| load_ckpt_path | checkpoint加载路径             | ./checkpoints/discriminator/model_iter_30000_float32.ckpt <br/>./checkpoints/generator/model_iter_150000_float32.ckpt |
| ckpt_interval  | checkpoint保存周期             | 400                                                                                                                   |
| figures_path   | 图片保存路径                     | ./figures                                                                                                             |
| load_data_path | 加载验证数据的路径                  | ./data                                                                                                                |
| log_path       | 日志保存路径                     | ./logs                                                                                                                |
| lam            | generator 损失函数系数           | 1.5                                                                                                                   |
| beta           | pde 损失函数系数                 | 1                                                                                                                     |
| n_col          | 研究域内训练数据量                  | 100                                                                                                                   |
| n_bound        | 研究域边界训练数据量                 | 20                                                                                                                    |
| epochs         | 训练周期                       | 30001                                                                                                                 |
| lr             | 学习率                        | 1e-4                                                                                                                  |
| term_t         | 每个周期discriminator训练次数      | 1                                                                                                                     |
| term_kl        | 每个周期generator训练次数          | 5                                                                                                                     |
| download_data  | 模型所需数据集与(或)checkpoints     | auq_pinns                                                                                                             |
| force_download | 是否强制下载数据                   | false                                                                                                                 |
| amp_level      | MindSpore自动混合精度等级          | O2                                                                                                                    |
| device_id      | 需要设置的设备号                   | None                                                                                                                  |
| mode           | MindSpore静态图模式（0）或动态图模式（1） | 0                                                                                                                     |

### [训练流程](#目录)

- 在 GPU/Ascend 上运行

   ```bash
   python train.py
   ```

  训练期间的损失值将打印在控制台中， 也可以训练后在日志文件中查看。

  ```bash
  # grep "G_loss:" log
  step: 0, G_loss: 56.84, KL_loss: -0.00947, recon_loss: 0.6465, pde_loss: 56.22, interval: 30.575496435165405s, total: 30.575496435165405s
  step: 100, G_loss: 35.56, KL_loss: 0.00891, recon_loss: 0.2842, pde_loss: 35.28, interval: 0.31897568702697754s, total: 30.894472122192383s
  step: 200, G_loss: 12.83, KL_loss: 0.10864, recon_loss: 0.5396, pde_loss: 12.18, interval: 0.2951545715332031s, total: 31.189626693725586s
  step: 300, G_loss: 10.03, KL_loss: 0.1841, recon_loss: 0.393, pde_loss: 9.45, interval: 0.28347206115722656s, total: 31.473098754882812s
  step: 400, G_loss: 8.9, KL_loss: 0.2262, recon_loss: 0.3416, pde_loss: 8.33, interval: 0.28740811347961426s, total: 31.760506868362427s
  step: 500, G_loss: 6.773, KL_loss: 0.2179, recon_loss: 0.4636, pde_loss: 6.094, interval: 0.28986334800720215s, total: 32.05037021636963s
  step: 600, G_loss: 3.385, KL_loss: -0.09106, recon_loss: 0.4062, pde_loss: 3.07, interval: 0.289898157119751s, total: 32.34026837348938s
  step: 700, G_loss: 1.025, KL_loss: -0.254, recon_loss: 0.3726, pde_loss: 0.9067, interval: 0.28998303413391113s, total: 32.63025140762329s
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