[ENGLISH](README.md) | 简体中文

# 目录

- [PINNs Neural Tangent Kernel 描述](#pinns-neural-tangent-kernel-描述)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [脚本说明](#脚本说明)
    - [脚本和示例代码](#脚本和示例代码)
    - [脚本参数](#脚本参数)
    - [训练流程](#训练流程)
    - [推理流程](#推理流程)

## [PINNs Neural Tangent Kernel 描述](#目录)

Neural Tangent Kernel（NTK）是描述无限宽度全连接神经网络在梯度下降训练流程中如何演化的核。
在适当的条件下，在无限宽度限制下，物理驱动神经网络（PINNs）的NTK将在训练期间决定性地收敛常量核。
这使我们能够通过其极限NTK的视角分析PINNs的训练动态，并发现训练流程中，不同损失成分收敛速率存在显著的差异。

> [论文](https://www.sciencedirect.com/science/article/pii/S002199912100663X): Sifan Wang, Xinling Yu, Paris Perdikaris,
> When and why PINNs fail to train: A neural tangent kernel perspective,
> Journal of Computational Physics, Volume 449, 2022, 110768, ISSN 0021-9991.

案例详情: 训练PINNs求解一维柏松方程，并记录NTK。

## [数据集](#目录)

训练数据集在运行时生成。
数据集的大小由`config.yaml`中的参数`num`控制，默认值为100。

预训练checkpoints文件将会在首次启动时自动下载。
您如果需要手动下载checkpoints文件，
请访问[此链接](https://download.mindspore.cn/mindscience/SciAI/sciai/model/pinns_ntk/)。

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
    --layers 1 512 1 \
    --save_ckpt true \
    --save_fig true \
    --load_ckpt false \
    --save_ckpt_path ./checkpoints \
    --load_ckpt_path ./checkpoints/model_final_float32.ckpt \
    --figures_path ./figures \
    --log_path ./logs \
    --print_interval 10 \
    --num 100 \
    --lr 1e-4 \
    --epochs 40001 \
    --download_data pinns_ntk \
    --force_download false \
    --amp_level O2 \
    --device_id 0 \
    --mode 0
```

## [脚本说明](#目录)

### [脚本和示例代码](#目录)

文件结构如下:

```text
├── pinns_ntk
│   ├── checkpoints                  # checkpoint文件
│   ├── data                         # 数据文件
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

train.py中的重要参数如下:

| 参数             | 描述                         | 默认值                                    |
|----------------|----------------------------|----------------------------------------|
| layers         | 神经网络结构                     | 1 512 1                                |
| save_ckpt      | 是否保存checkpoint             | true                                   |
| save_fig       | 是否保存和绘制图片                  | true                                   |
| load_ckpt      | 是否加载checkpoint             | false                                  |
| save_ckpt_path | checkpoint保存路径             | ./checkpoints                          |
| load_ckpt_path | checkpoint加载路径             | ./checkpoints/model_final_float32.ckpt |
| figures_path   | 图片保存路径                     | ./figures                              |
| log_path       | 日志保存路径                     | ./logs                                 |
| print_interval | 时间与loss打印间隔                | 10                                     |
| num            | 数据集采样点数                    | 100                                    |
| lr             | 学习率                        | 1e-4                                   |
| epochs         | 时期（迭代次数）                   | 40001                                  |
| download_data  | 模型所需数据集与(或)checkpoints     | pinns_ntk                              |
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
  step: 0, total loss: 12353.368, loss_bcs: 9.613263, loss_res: 12343.755, interval: 0.41385459899902344s, total: 0.41385459899902344s, checkpoint saved at: ./checkpoints/model_iter_0_2023-04-23-07-42-46.ckpt
  Compute NTK...
  Weigts stored...
  step: 10, total loss: 11993.846, loss_bcs: 8.270422, loss_res: 11985.224, interval: 1.02224523987624589s, total: 1.43609983887526933s
  step: 20, total loss: 11339.517, loss_bcs: 6.435134, loss_res: 11333.08, interval: 0.024523986245987602s, total: 1.460623825121256932s
  step: 30, total loss: 11287.906, loss_bcs: 6.306574, loss_res: 11281.6, interval: 0.0191900713459287945s, total: 1.4798138964671857265s
  step: 40, total loss: 6723.454, loss_bcs: 2.0566676, loss_res: 6721.3975, interval: 0.01975234587509485s, total: 1.4995662423422805765s
  step: 50, total loss: 8277.567, loss_bcs: 0.0453923, loss_res: 8277.522, interval: 0.01824523876245972s, total: 1.5178114811047402965s
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