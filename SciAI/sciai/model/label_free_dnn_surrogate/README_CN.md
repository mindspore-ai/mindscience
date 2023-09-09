[ENGLISH](README.md) | 简体中文

# 目录

- [LabelFree DNN Surrogate 描述](#labelfree-dnn-surrogate-描述)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [脚本说明](#脚本说明)
    - [脚本和示例代码](#脚本和示例代码)
    - [脚本参数](#脚本参数)
    - [训练流程](#训练流程)
    - [推理流程](#推理流程)

## [LabelFree DNN Surrogate 描述](#目录)

流体动力学问题的数值模拟在计算上是昂贵的，所以开发具有成本效益的替代模型非常重要。
该模型提出了一种物理约束的深度学习方法，用于在不依赖仿真数据的情况下对流体流动进行建模。
该方法将控制方程纳入损失函数，在数值实验中表现良好。

> [论文](https://www.sciencedirect.com/science/article/pii/S004578251930622X): Luning Sun, Han Gao, Shaowu Pan, Jian-Xun
> Wang. Surrogate modeling for fluid flows based on physics-constrained deep learning without simulation data. Computer
> Methods in Applied Mechanics and Engineering, Volume 361, 2020, 112732, ISSN 0045-7825.

参数化管道流
<p >
    <img align = 'center' height="200" src="figures/pipe_uProfiles_nuIdx_.png?raw=true">
</p>

|                                      Small Aneurysm                                       |                                     Middle Aneurysm                                      |                                    Large Aneurysm                                     |
|:-----------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------:|
| <img height="200" src="figures/486scale-0.0032346553257729654uContour_test.png?raw=true"> | <img height="200" src="figures/151scale-0.011815133162025654uContour_test.png?raw=true"> | <img height="200" src="figures/1scale-0.02267951024095881uContour_test.png?raw=true"> |

## [数据集](#目录)

训练数据集在运行时随机生成。
数据集的大小由`config.yaml`中的参数`batch_size`控制，默认值为50。

在验证阶段，文件夹`./data/`提供了不同case count情况下的CFD与NN结果。
用于验证的数据集和预训练checkpoints文件将会在首次启动时自动下载。

您如果需要手动下载验证数据集或checkpoints文件，
请访问[此链接](https://download.mindspore.cn/mindscience/SciAI/sciai/model/label_free_dnn_surrogate/)。

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
    --layers 3 20 20 20 1 \
    --save_ckpt true \
    --save_fig  true \
    --load_ckpt false \
    --save_ckpt_path ./checkpoints \
    --load_ckpt_path \
        ./checkpoints/geo_para_axisy_sigma0.1_epoch500hard_u.ckpt \
        ./checkpoints/geo_para_axisy_sigma0.1_epoch500hard_v.ckpt \
        ./checkpoints/geo_para_axisy_sigma0.1_epoch500hard_P.ckpt \
    --load_data_path ./data \
    --save_data_path ./data \
    --figures_path ./figures \
    --log_path ./logs \
    --lr 1e-3 \
    --epochs_train 500 \
    --epochs_val 400 \
    --batch_size 50 \
    --print_interval 100 \
    --nu 1e-3 \
    --download_data label_free_dnn_surrogate \
    --force_download false \
    --amp_level O0 \
    --device_id 0 \
    --mode 0
```

## [脚本说明](#目录)

## [脚本和示例代码](#目录)

文件结构如下:

```text
├── label_free_dnn_surrogate
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
│   └── eval.py                                      # python评估脚本
```

### [脚本参数](#目录)

train.py中的重要参数如下:

| 参数名            | 描述                         | 默认值                                                                                                                                                                                     |
|----------------|----------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| layers         | 神经网络层定义                    | 3 20 20 20 1                                                                                                                                                                            |
| save_ckpt      | 是否保存checkpoint             | true                                                                                                                                                                                    |
| save_fig       | 是否保存和绘制图片                  | true                                                                                                                                                                                    |
| load_ckpt      | 是否加载checkpoint             | false                                                                                                                                                                                   |
| save_ckpt_path | checkpoint保存路径             | ./checkpoints                                                                                                                                                                           |
| load_ckpt_path | checkpoint加载路径             | ./checkpoints/geo_para_axisy_sigma0.1_epoch500hard_u.ckpt <br/>./checkpoints/geo_para_axisy_sigma0.1_epoch500hard_v.ckpt <br/>./checkpoints/geo_para_axisy_sigma0.1_epoch500hard_P.ckpt |
| load_data_path | 加载原始数据的路径                  | ./data                                                                                                                                                                                  |
| save_data_path | 保存生成数据与用于画图数据的路径           | ./data                                                                                                                                                                                  |
| figures_path   | 图片保存路径                     | ./figures                                                                                                                                                                               |
| log_path       | 日志保存路径                     | ./logs                                                                                                                                                                                  |
| lr             | 学习率                        | 1e-3                                                                                                                                                                                    |
| epochs_train   | 训练的时期（迭代次数）                | 500                                                                                                                                                                                     |
| epochs_val     | 验证和绘图的时期（迭代次数）             | 400                                                                                                                                                                                     |
| batch_size     | 训练数据集的大小                   | 50                                                                                                                                                                                      |
| print_interval | 时间与loss打印间隔                | 100                                                                                                                                                                                     |
| nu             | loss函数的nu参数                | 1e-3                                                                                                                                                                                    |
| download_data  | 模型所需数据集与(或)checkpoints     | label_free_dnn_surrogate                                                                                                                                                                |
| force_download | 是否强制下载数据                   | false                                                                                                                                                                                   |
| amp_level      | MindSpore自动混合精度等级          | O0                                                                                                                                                                                      |
| device_id      | 需要设置的设备号                   | None                                                                                                                                                                                    |
| mode           | MindSpore静态图模式（0）或动态图模式（1） | 0                                                                                                                                                                                       |

### [训练流程](#目录)

- 在 GPU/Ascend 上运行

   ```bash
   python train.py
   ```

  经过训练，loss值会输出如下:

  ```bash
  # grep "loss:" log
  epoch:0, step: 0/2000, loss: 0.010065492, interval: 64.38495588302612s, total: 64.38495588302612s
  epoch:0, step: 100/2000, loss: 0.009565717, interval: 16.189687490463257s, total: 80.57464337348938s
  epoch:0, step: 200/2000, loss: 0.009905259, interval: 16.150275468826294s, total: 96.72491884231567s
  epoch:0, step: 300/2000, loss: 0.009798448, interval: 16.015777587890625s, total: 112.7406964302063s
  epoch:0, step: 400/2000, loss: 0.010146898, interval: 15.762168169021606s, total: 128.5028645992279s
  epoch:0, step: 500/2000, loss: 0.009967192, interval: 15.626747369766235s, total: 144.12961196899414s
  epoch:0, step: 600/2000, loss: 0.010065671, interval: 15.571009874343872s, total: 159.700621843338s
  epoch:0, step: 700/2000, loss: 0.010013511, interval: 16.07706379890442s, total: 175.77768564224243s
  epoch:0, step: 800/2000, loss: 0.0097869225, interval: 15.972872018814087s, total: 191.75055766105652s
  epoch:0, step: 900/2000, loss: 0.009805476, interval: 16.053072452545166s, total: 207.80363011360168s
  epoch:0, step: 1000/2000, loss: 0.009835069, interval: 16.04354953765869s, total: 223.84717965126038s
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

