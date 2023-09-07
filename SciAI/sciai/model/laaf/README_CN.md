[ENGLISH](README.md) | 简体中文

# 目录

- [LAAF 描述](#laaf-描述)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [脚本说明](#脚本说明)
    - [脚本和示例代码](#脚本和示例代码)
    - [脚本参数](#脚本参数)
    - [训练流程](#训练流程)
    - [推理流程](#推理流程)

## [LAAF 描述](#目录)

该模型提出分层和神经元局部自适应激活函数（LAAF），用于深度和物理信息神经网络，这些代码使用随机梯度下降变体优化可扩展参数。
该方法加速收敛，降低训练成本，并且避免了局部最优解。

> [论文](https://doi.org/10.1016/j.jcp.2019.109136):  A.D. Jagtap, K.Kawaguchi, G.E.Karniadakis, Adaptive activation
> functions accelerate convergence in deep and physics-informed neural networks, Journal of Computational Physics, 404
> (2020) 109136.

样例细节:

```bash
f = 0.2*np.sin(6*x) if x < 0 else 0.1*x*np.cos(18*x) + 1
```

## [数据集](#目录)

训练数据集在运行时随机生成。
数据集的大小由`config.yaml`中的参数`num_grid`控制，默认值为300。

预训练checkpoints文件将会在首次启动时自动下载。
您如果需要手动下载checkpoints文件，
请访问[此链接](https://download.mindspore.cn/mindscience/SciAI/sciai/model/laaf/)。

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
    --layers 1 50 50 50 50 1 \
    --save_ckpt true \
    --load_ckpt false \
    --save_fig true \
    --save_data true \
    --save_ckpt_path ./checkpoints \
    --figures_path ./figures \
    --load_ckpt_path ./checkpoints/model_15001.ckpt \
    --save_data_path ./data \
    --log_path ./logs \
    --lr 2e-4 \
    --epochs 15001 \
    --num_grid 300 \
    --sol_epochs 2000 8000 15000 \
    --download_data laaf \
    --force_download false \
    --amp_level O3 \
    --device_id 0 \
    --mode 0
```

## [脚本说明](#目录)

## [脚本和示例代码](#目录)

文件结构如下:

```text
├── laaf
│   ├── checkpoints                                 # checkpoint文件
│   ├── data                                        # 数据文件
│   ├── figures                                     # 结果图片
│   ├── logs                                        # 日志文件
│   ├── src                                         # 源代码
│   │   ├── network.py                              # 网络架构
│   │   ├── plot.py                                 # 绘制结果
│   │   └── process.py                              # 数据处理
│   ├── config.yaml                                 # 超参数配置
│   ├── README.md                                   # 英文模型说明
│   ├── README_CN.md                                # 中文模型说明
│   ├── train.py                                    # python训练脚本
│   └── eval.py                                     # python评估脚本
```

### [脚本参数](#目录)

train.py中的重要参数如下:

| 参数名            | 描述                         | 默认值                            |
|----------------|----------------------------|--------------------------------|
| layers         | 神经网络层定义                    | 1 50 50 50 50 1                |
| save_ckpt      | 是否保存checkpoint             | true                           |
| load_ckpt      | 是否加载checkpoint             | false                          |
| save_fig       | 是否保存和绘制图片                  | true                           |
| save_data      | 是否保存生成数据                   | true                           |
| save_ckpt_path | checkpoint保存路径             | ./checkpoints                  |
| figures_path   | 图片保存路径                     | ./figures                      |
| load_ckpt_path | checkpoint加载路径             | ./checkpoints/model_15001.ckpt |
| save_data_path | 保存数据的路径                    | ./data                         |
| log_path       | 日志保存路径                     | ./logs                         |
| lr             | 学习率                        | 2e-4                           |
| epochs         | 时期（迭代次数）                   | 15001                          |
| num_grid       | 网格数量                       | 300                            |
| sol_epochs     | 抓取快照的时期（迭代次数）              | 2000 8000 15000                |
| download_data  | 模型所需数据集与(或)checkpoints     | laaf                           |
| force_download | 是否强制下载数据                   | false                          |
| amp_level      | MindSpore自动混合精度等级          | O3                             |
| device_id      | 需要设置的设备号                   | None                           |
| mode           | MindSpore静态图模式（0）或动态图模式（1） | 0                              |

### [训练流程](#目录)

- 在 GPU/Ascend 上运行

   ```bash
   python train.py
   ```

  经过训练，loss值会输出如下:

  ```bash
  # grep "loss:" log
  step: 0, loss: 1.0621891, interval: 15.865238428115845s, total: 15.865238428115845s
  step: 10, loss: 0.8333796, interval: 0.27651143074035645s, total: 16.1417498588562s
  step: 20, loss: 0.6490651, interval: 0.24263739585876465s, total: 16.384387254714966s
  step: 30, loss: 0.49252713, interval: 0.24282169342041016s, total: 16.627208948135376s
  step: 40, loss: 0.37449843, interval: 0.24222493171691895s, total: 16.869433879852295s
  step: 50, loss: 0.317139, interval: 0.24213695526123047s, total: 17.111570835113525s
  step: 60, loss: 0.31154847, interval: 0.24191784858703613s, total: 17.35348868370056s
  step: 70, loss: 0.3132628, interval: 0.24203085899353027s, total: 17.595519542694092s
  step: 80, loss: 0.31056988, interval: 0.24201083183288574s, total: 17.837530374526978s
  step: 90, loss: 0.3099324, interval: 0.2420203685760498s, total: 18.079550743103027s
  step: 100, loss: 0.30981177, interval: 0.24202728271484375s, total: 18.32157802581787s
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
