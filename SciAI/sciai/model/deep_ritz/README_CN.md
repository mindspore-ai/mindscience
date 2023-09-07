[ENGLISH](README.md) | 简体中文

# 目录

- [Deep Ritz 描述](#deep-ritz-描述)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [脚本说明](#脚本说明)
    - [脚本和示例代码](#脚本和示例代码)
    - [脚本参数](#脚本参数)
    - [训练流程](#训练流程)
    - [推理流程](#推理流程)

## [Deep Ritz 描述](#目录)

Deep Ritz方法是一种深度学习方法，该方法使用数值求解变分问题，尤其是偏微分方程引出的变分问题。

本项目中使用Deep Ritz方法解决了两个与泊松方程有关的问题。

> [论文](https://arxiv.org/abs/1710.00211): W E, B Yu.
> The Deep Ritz method: A deep learning-based numerical algorithm for solving variational problems.
> Communications in Mathematics and Statistics 2018, 6:1-12.
> e problems[J]. Computer Methods in Applied Mechanics and Engineering, 2020, 365: 113028.

## [数据集](#目录)

训练数据集在运行时随机生成。
数据集的大小由`config.yaml`中的参数`body_batch`和`bdry_batch`控制，默认值分别为1024和1024。
用于验证的数据集和预训练checkpoints文件将会在首次启动时自动下载。

您如果需要手动下载验证数据集或checkpoints文件，
请访问[此链接](https://download.mindspore.cn/mindscience/SciAI/sciai/model/deep_ritz/)。

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

poisson-hole案例的完整命令如下:

```bash
python train.py \
    --layers 2 8 8 8 1 \
    --save_ckpt true \
    --load_ckpt false \
    --save_ckpt_path ./checkpoints/hole \
    --load_ckpt_path ./checkpoints/hole/model_50000_float32.ckpt \
    --save_fig true \
    --figures_path ./figures \
    --save_data true \
    --save_data_path ./data/hole \
    --log_path ./logs \
    --lr 0.01 \
    --train_epoch 50000 \
    --train_epoch_pre 0 \
    --body_batch 1024 \
    --bdry_batch 1024 \
    --write_step 50 \
    --sample_step 10 \
    --step_size 5000 \
    --num_quad 40000 \
    --radius 1 \
    --penalty 500 \
    --diff 0.001 \
    --gamma 0.3 \
    --decay 0.00001 \
    --autograd true \
    --download_data deep_ritz \
    --force_download false \
    --amp_level O2 \
    --device_id 0 \
    --mode 0
```

如果您想运行poisson-ls案例的完整的命令，请在`config.yaml`中切换`problem`。

## [脚本说明](#目录)

### [脚本和示例代码](#目录)

文件结构如下:

```text
├── deep_ritz
│   ├── checkpoints                  # checkpoint文件
│   ├── data                         # 数据文件
│   ├── figures                      # 结果图片
│   ├── logs                         # 日志文件
│   ├── src                          # 源代码
│   │   ├── process.py               # 数据处理
│   │   ├── poisson_hole.py          # hole案例问题定义
│   │   ├── poisson_ls.py            # ls案例问题定义
│   │   ├── network.py               # 网络架构
│   │   └── plot.py                  # 绘制结果
│   ├── config.yaml                  # 超参数配置
│   ├── README.md                    # 英文模型说明
│   ├── README_CN.md                 # 中文模型说明
│   ├── train.py                     # python训练脚本
│   └── eval.py                      # python评估脚本
```

### [脚本参数](#目录)

train.py中的重要参数如下:

| 参数              | 描述                         | 默认值                                         |
|-----------------|----------------------------|---------------------------------------------|
| layers          | 神经网络宽度                     | 2 8 8 8 1                                   |
| save_ckpt       | 是否保存checkpoint             | true                                        |
| load_ckpt       | 是否加载checkpoint             | false                                       |
| save_ckpt_path  | checkpoint保存路径             | ./checkpoints/hole                          |
| load_ckpt_path  | checkpoint加载路径             | ./checkpoints/hole/model_50000_float32.ckpt |
| save_fig        | 是否保存和绘制图片                  | true                                        |
| figures_path    | 图片保存路径                     | ./figures                                   |
| save_data       | 是否保存数据                     | true                                        |
| save_data_path  | 保存数据的路径                    | ./data/hole                                 |
| log_path        | 日志保存路径                     | ./logs                                      |
| lr              | 学习率                        | 1e-2                                        |
| train_epoch     | 时期（迭代次数）                   | 50001                                       |
| train_epoch_pre | 预训练时期（迭代次数）                | 0                                           |
| body_batch      | 盘内每批采样个数                   | 1024                                        |
| bdry_batch      | 盘面每批采样个数                   | 1024                                        |
| write_step      | 时间与loss打印间隔                | 50                                          |
| sample_step     | 训练中重采样步长                   | 10                                          |
| step_size       | 学习率的指数衰变步长                 | 5000                                        |
| num_quad        | 验证集采点个数                    | 40000                                       |
| radius          | 圆盘半径                       | 1                                           |
| penalty         | 训练期间loss2的惩罚因子             | 500                                         |
| diff            | 差异步长                       | 1e-3                                        |
| gamma           | 学习率指数衰变率                   | 0.3                                         |
| decay           | 权重衰减                       | 1e-5                                        |
| autograd        | 是否使用自动微分                   | true                                        |
| download_data   | 模型所需数据集与(或)checkpoints     | deep_ritz                                   |
| force_download  | 是否强制下载数据                   | false                                       |
| amp_level       | MindSpore自动混合精度等级          | O2                                          |
| device_id       | 需要设置的设备号                   | None                                        |
| mode            | MindSpore静态图模式（0）或动态图模式（1） | 0                                           |

### [训练流程](#目录)

- 在 GPU/Ascend 上运行

   ```bash
   python train.py
   ```

  训练期间的损失值将打印在控制台中， 也可以训练后在日志文件中查看。

   ```bash
   # grep "loss:" log
  step: 0, total loss: 166.09909, loss: 165.08899, error: 1.0101029, interval: 29.70781683921814s, total: 29.70781683921814s
  step: 50, total loss: 5.871787, loss: 5.261068, error: 0.6107192, interval: 1.2001934051513672s, total: 30.908010244369507s
  step: 100, total loss: 0.80151683, loss: 0.43523002, error: 0.3662868, interval: 1.1730225086212158s, total: 32.08103275299072s
  step: 150, total loss: 0.5899545, loss: 0.36189145, error: 0.22806305, interval: 1.1766719818115234s, total: 33.257704734802246s
  step: 200, total loss: 0.5207778, loss: 0.3336542, error: 0.18712364, interval: 1.1791396141052246s, total: 34.43684434890747s
  step: 250, total loss: 0.5430529, loss: 0.36813667, error: 0.17491627, interval: 1.1709723472595215s, total: 35.60781669616699s
  step: 300, total loss: 0.554542, loss: 0.39627352, error: 0.1582685, interval: 1.1721374988555908s, total: 36.77995419502258s
  step: 350, total loss: 0.42904806, loss: 0.28422767, error: 0.14482038, interval: 1.167961597442627s, total: 37.94791579246521s
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