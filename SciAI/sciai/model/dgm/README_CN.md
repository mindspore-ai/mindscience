[ENGLISH](README.md) | 简体中文

# 目录

- [Deep Galerkin 方法](#deep-galerkin-方法)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [脚本说明](#脚本说明)
    - [脚本和示例代码](#脚本和示例代码)
    - [脚本参数](#脚本参数)
    - [训练流程](#训练流程)
    - [推理流程](#推理流程)

## [Deep Galerkin 方法](#目录)

深度伽辽金方法（DGM）使用深度神经网络，取代了伽辽金方法（Galerkin method）中基函数的线性组合，来获得偏微分方程简约解。
DGM算法可以精确求解一类自由边界偏微分方程，展示了神经网络求解偏微分方程的能力。

此模型展示了一个平流方程的示例，使用具有`tanh`激活函数的四层密集神经网络来求解方程。
该[动画](figures/animation.gif)说明了方程的学习过程。

> [论文](https://arxiv.org/abs/1708.07469):
> Sirignano J, Spiliopoulos K. DGM: A deep learning algorithm for solving partial differential equations[J].
> Journal of computational physics, 2018, 375: 1339-1364.

## [数据集](#目录)

训练数据集在每个时期（训练迭代）生成。
域中的数据集大小由`config.yaml`中的`batch_size`控制，默认为256。

预训练checkpoints文件将会在首次启动时自动下载。
您如果需要手动下载checkpoints文件，
请访问[此链接](https://download.mindspore.cn/mindscience/SciAI/sciai/model/dgm/)。

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

输出的信息：
<br>
1- 给定方程的神经网络解  <br>
2- 损失函数值（对于微分算子、边界条件等） <br>
3- 神经网络的逐层平均激活值（训练期间） <br>
<br>

[checkpoints](checkpoints)文件夹包含预先训练的网络，通过设置`config.yaml`中的`load_ckpt`为`True`并设置`load_ckpt_path`
为你想要的预训练网络，可以将它们加载到网络中。然后，我们可以通过运行以下命令继续训练：

```bash
python train.py
```

完整命令:

```bash
python train.py \
    --layers 1 10 10 10 1 \
    --save_ckpt true \
    --save_fig true \
    --load_ckpt false \
    --save_fig true \
    --save_anim true \
    --save_ckpt_path ./checkpoints \
    --load_ckpt_path ./checkpoints/model_iter_2000_float32.ckpt \
    --log_path ./logs \
    --figures_path ./figures \
    --print_interval 20 \
    --lr 0.01 \
    --epochs 2001 \
    --batch_size 256 \
    --download_data dgm \
    --force_download false \
    --amp_level float32 \
    --device_id 0 \
    --mode 0
```

如果想生成动画，显示学习过程，必须打开`./config.yaml`中的`save_fig`和`save_anim`配置，并确保脚本以`PYNATIVE`模式运行。
可以通过在调用`init_project`时更改参数来调整运行模式。

```python
init_project(ms.PYNATIVE_MODE)  # PYNATIVE 模式
init_project(ms.GRAPH_MODE)  # GRAPH 模式
```

此外，软件包`ImageMagick`必须在环境中可用。
如果没有，可以通过以下方式安装命令。

```bash
sudo yum install ImageMagick
```

## [脚本说明](#目录)

## [脚本和示例代码](#目录)

文件结构如下:

```text
├── dgm
│   ├── checkpoints                         # checkpoint文件
│   ├── data                                # 数据文件
│   ├── figures                             # 结果图片
│   ├── logs                                # 日志文件
│   ├── src                                 # 源代码
│   │   ├── advection.py                    # advection定义
│   │   ├── plot.py                         # 绘制结果
│   │   └── network.py                      # 网络架构
│   ├── config.yaml                         # 超参数配置
│   ├── README.md                           # 英文模型说明
│   ├── README_CN.md                        # 中文模型说明
│   ├── train.py                            # python训练脚本
│   └── eval.py                             # python评估脚本
```

### [脚本参数](#目录)

train.py中的重要参数如下:

| 参数名            | 描述                         | 默认值                                        |
|----------------|----------------------------|--------------------------------------------|
| layers         | 神经网络层定义                    | 1 10 10 10 1                               |
| save_ckpt      | 是否保存checkpoint             | true                                       |
| load_ckpt      | 是否加载checkpoint             | false                                      |
| save_fig       | 是否保存和绘制图片                  | true                                       |
| save_anim      | 是否保存生成的动画图片                | true                                       |
| save_ckpt_path | checkpoint保存路径             | ./checkpoints                              |
| load_ckpt_path | checkpoint加载路径             | ./checkpoints/model_iter_2000_float32.ckpt |
| figures_path   | 图片保存路径                     | ./figures                                  |
| log_path       | 日志保存路径                     | ./logs                                     |
| print_interval | 时间与loss打印间隔                | 20                                         |
| lr             | 学习率                        | 0.01                                       |
| epochs         | 时期（迭代次数）                   | 2001                                       |
| batch_size     | 数据集的大小                     | 256                                        |
| download_data  | 模型所需数据集与(或)checkpoints     | dgm                                        |
| force_download | 是否强制下载数据                   | false                                      |
| amp_level      | MindSpore自动混合精度等级          | O0                                         |
| device_id      | 需要设置的设备号                   | None                                       |
| mode           | MindSpore静态图模式（0）或动态图模式（1） | 0                                          |

### [训练流程](#目录)

- 在 GPU/Ascend 上运行

  ```bash
  python train.py
  ```

  经过训练，loss值会输出如下:

  ```bash
  # cat DGM_[your_time].log
  ...
  step: 320, total loss: 47.46501, loss_domain: 47.353966, loss_ic: 0.11104686, interval: 3.9606783390045166s, total: 105.76785326004028s
  step: 340, total loss: 42.394558, loss_domain: 42.363266, loss_ic: 0.031290982, interval: 3.5947396755218506s, total: 109.36259293556213s
  step: 360, total loss: 40.42996, loss_domain: 40.41625, loss_ic: 0.013711907, interval: 3.7165727615356445s, total: 113.07916569709778s
  step: 380, total loss: 33.631718, loss_domain: 33.61124, loss_ic: 0.020477751, interval: 3.7209231853485107s, total: 116.80008888244629s
  step: 400, total loss: 25.643202, loss_domain: 25.643173, loss_ic: 2.944071e-05, interval: 3.7022390365600586s, total: 120.50232791900635s
  step: 420, total loss: 29.891747, loss_domain: 29.88154, loss_ic: 0.010205832, interval: 3.708503007888794s, total: 124.21083092689514s
  step: 440, total loss: 28.60983, loss_domain: 28.609776, loss_ic: 5.462918e-05, interval: 3.9715089797973633s, total: 128.1823399066925s
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