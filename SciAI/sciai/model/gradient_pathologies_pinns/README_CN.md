[ENGLISH](README.md) | 简体中文

# 目录

- [理解并减轻PINNs中的病态梯度问题](#理解并减轻pinns中的病态梯度问题)
- [改进的全连接网络架构](#改进的全连接网络架构)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [脚本说明](#脚本说明)
    - [脚本和示例代码](#脚本和示例代码)
    - [脚本参数](#脚本参数)
    - [模型训练](#模型训练)
    - [推理评估](#推理评估)

## [理解并减轻PINNs中的病态梯度问题](#目录)

本工作回顾了AI科学计算的最新进展，特别侧重于PINNs网络在带噪声的数据中对物理系统结果的预测和发现其中隐藏物理的有效性。
我们确定了与数值刚度相关的基本失效模式，这种刚度会导致模型训练期间反向传播中的梯度不平衡。为了解决这一限制，我们提出了一种学习率annealing算法，
该算法利用模型训练期间的梯度统计，以平衡复合损失函数中不同loss项的贡献程度。另外，我们还提出了一种新的神经网络架构，它对这种病态梯度更具抗性。

> [论文](https://arxiv.org/pdf/2001.04536.pdf): Wang, S., Teng, Y. & Perdikaris, P. Understanding and mitigating
> gradient pathologies in physics-informed neural networks. arXiv:2001.04536 [cs.LG] (2020).

案例详情: 二维Helmholtz方程。

## [数据集](#目录)

无外部数据集，随机采样domain边界和内部数据点用于训练和推理。
数据集的大小由`config.yaml`中的参数`batch_size`控制，默认值为128。

预训练checkpoints文件将会在首次启动时自动下载。
您如果需要手动下载checkpoints文件，
请访问[此链接](https://download.mindspore.cn/mindscience/SciAI/sciai/model/gradient_pathologies_pinns/)。

## [改进的全连接网络架构](#目录)

![改进后架构](./figures/improved_fc_arch.png)

一种改进的PINNs神经网络全连接结构: 引入残差连接（residual connections）和考虑输入之间的乘法相互作用可以提升模型的预测能力。

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
    --method M4 \
    --layers 2 50 50 50 1 \
    --save_ckpt true \
    --save_fig true \
    --load_ckpt false \
    --save_ckpt_path ./checkpoints \
    --load_ckpt_path ./checkpoints/helmholtz_M4_float32.ckpt \
    --figures_path ./figures \
    --log_path ./logs \
    --lr 1e-3 \
    --epochs 40001 \
    --batch_size 128 \
    --download_data gradient_pathologies_pinns \
    --force_download false \
    --amp_level O3 \
    --device_id 0 \
    --mode 0
```

## [脚本说明](#目录)

### [脚本和示例代码](#目录)

文件结构如下:

```text
├── gradient_pathologies_pinns
│   ├── checkpoints              # checkpoint文件
│   ├── data                     # 数据文件
│   ├── figures                  # 结果图片
│   ├── logs                     # 日志文件
│   ├── src                      # 源代码
│   │   ├── network.py           # 二维helmholtz网络架构
│   │   ├── plot.py              # 绘制结果
│   │   └── process.py           # 数据处理
│   ├── config.yaml              # 超参数配置
│   ├── README.md                # 英文模型说明
│   ├── README_CN.md             # 中文模型说明
│   ├── train.py                 # python训练脚本
│   └── eval.py                   # python评估脚本
```

### [脚本参数](#目录)

train.py中的重要参数如下:

| parameter      | description                | default value                           |
|----------------|----------------------------|-----------------------------------------|
| method         | 原论文中的model（M1-M4）          | M4                                      |
| layers         | 神经网络每层的宽度                  | 2 50 50 50 1                            |
| save_ckpt      | 是否保存checkpoint             | true                                    |
| save_fig       | 是否保存和绘制图片                  | true                                    |
| load_ckpt      | 是否加载checkpoint             | false                                   |
| save_ckpt_path | checkpoint保存路径             | ./checkpoints                           |
| load_ckpt_path | checkpoint加载路径             | ./checkpoints/helmholtz_M4_float32.ckpt |
| figures_path   | 图片保存路径                     | ./figures                               |
| log_path       | 日志保存路径                     | ./logs                                  |
| lr             | 学习率                        | 1e-3                                    |
| epochs         | 时期（迭代次数）                   | 40001                                   |
| batch_size     | 每个batch的数据点采样个数            | 128                                     |
| download_data  | 模型所需数据集与(或)checkpoints     | gradient_pathologies_pinns              |
| force_download | 是否强制下载数据                   | false                                   |
| amp_level      | MindSpore自动混合精度等级          | O3                                      |
| device_id      | 需要设置的设备号                   | None                                    |
| mode           | MindSpore静态图模式（0）或动态图模式（1） | 0                                       |

### [模型训练](#目录)

- 在 GPU/Ascend 上运行

   ```bash
   python train.py
   ```

  经过训练，loss值会输出如下:

   ```bash
  It: 10, Loss: 6.098e+03, Loss_bcs: 1.628e+00, Loss_res: 6.096e+03, Adaptive_Constant: 15.26 ,Time: 0.39
  It: 20, Loss: 6.325e+03, Loss_bcs: 1.391e+00, Loss_res: 6.324e+03, Adaptive_Constant: 16.20 ,Time: 0.23
  It: 30, Loss: 8.256e+03, Loss_bcs: 1.173e+00, Loss_res: 8.255e+03, Adaptive_Constant: 20.49 ,Time: 0.22
  It: 40, Loss: 6.699e+03, Loss_bcs: 1.749e+00, Loss_res: 6.697e+03, Adaptive_Constant: 22.41 ,Time: 0.22
  It: 50, Loss: 6.904e+03, Loss_bcs: 2.115e+00, Loss_res: 6.902e+03, Adaptive_Constant: 25.56 ,Time: 0.21
  It: 60, Loss: 6.896e+03, Loss_bcs: 3.768e+00, Loss_res: 6.892e+03, Adaptive_Constant: 28.76 ,Time: 0.22
  It: 70, Loss: 6.982e+03, Loss_bcs: 8.551e+00, Loss_res: 6.974e+03, Adaptive_Constant: 29.38 ,Time: 0.22
  It: 80, Loss: 7.389e+03, Loss_bcs: 1.350e+01, Loss_res: 7.375e+03, Adaptive_Constant: 30.37 ,Time: 0.22
  ...
   ```

- 训练结束后，您仍然可以通过保存在`log_path`下面的日志文件回顾训练过程，默认为`./logs`目录中。

- 模型checkpoint将保存在 `save_ckpt_path`中，默认为`./checkpoints` 目录中。

### [推理评估](#目录)

在运行下面的命令之前，请检查使用的`config.yaml` 中的checkpoint加载路径`load_ckpt_path`
进行推理。

- 在 GPU/Ascend 上运行

   ```bash
   python eval.py
   ```

  您可以通过日志文件`log_path`查看过程与结果，默认位于`./logs` 。
  结果图片存放于`figures_path`中，默认位于[`./figures`](./figures)。