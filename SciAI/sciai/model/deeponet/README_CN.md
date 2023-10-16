[ENGLISH](README.md) | 简体中文

# 目录

- [DeepONet 描述](#Deeponet-描述)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [脚本说明](#脚本说明)
    - [脚本和示例代码](#脚本和示例代码)
    - [脚本参数](#脚本参数)
    - [训练流程](#训练流程)
    - [推理流程](#推理流程)

## [DeepONet 描述](#目录)

DeepONet是一种泛化误差小的新网络，由用于编码输入函数空间和另一个用于编码输出函数域的DNN组成。
它可以学习显式和隐式运算符，并且已经在16个不同的应用场景通过验证。

本项目中使用DeepONet方法解决了1D Caputo和2D fractional Laplacian问题。

> [论文](https://www.nature.com/articles/s42256-021-00302-5): Lu L, Jin P, Pang G, et al. Learning nonlinear operators
> via DeepONet based on the universal approximation theorem of operators[J].
> Nature machine intelligence, 2021, 3(3): 218-229.

## [数据集](#目录)

用于训练/验证的数据集和预训练checkpoints文件将会在首次启动时自动下载。

使用的数据集: [1D Caputo fractional derivative] 和 [2D fractional Laplacian]

- 数据集大小:
    - 1D Caputo fractional derivative
        - u向量的长度: m = 15
        - (y, alpha)的维度: d = 2
        - train数据量: 1e6
        - test数据量: 1e6
        - test0数据量: 1e2
    - 2D fractional Laplacian
        - u向量的长度: m = 225
        - (x, y, alpha)的维度: d = 3
        - train数据量: 1e6
        - test数据量: 1e6
        - test0数据量: 1e2
- 数据格式: `.mat`文件
    - 注: 数据会在`process.py`中处理
- 数据集在 `./data` 目录下，目录结构如下:

```text
├── data
│   ├── 1d_caputo
│   │   ├── test.npz
│   │   ├── test0.npz
│   │   └── train.npz
│   ├── 2d_fractional_laplacian
│   │   ├── test.npz
│   │   ├── test0.npz
│   │   └── train.npz
```

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

`1D Caputo fractional derivative`案例的完整命令如下:

```bash
python train.py \
    --problem 1d_caputo \
    --layers_u 15 80 80 80 \
    --layers_y 2 80 80 80 \
    --save_ckpt true \
    --load_ckpt false \
    --save_ckpt_path ./checkpoints/1d_caputo \
    --load_ckpt_path ./checkpoints/1d_caputo/1d_caputo.ckpt \
    --save_fig true \
    --figures_path ./figures/1d_caputo \
    --save_data true \
    --load_data_path ./data/1d_caputo \
    --save_data_path ./data/1d_caputo \
    --log_path ./logs \
    --lr 1e-3 \
    --epochs 20001 \
    --batch_size 100000 \
    --print_interval 10 \
    --download_data deeponet \
    --force_download false \
    --amp_level O3 \
    --device_id 0 \
    --mode 0
```

如果您想运行 `2D fractional Laplacian` 案例的完整的命令，请在`config.yaml`或命令参数中切换`problem`。

## [脚本说明](#目录)

### [脚本和示例代码](#目录)

文件结构如下:

```text
├── deeponet
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
│   └── eval.py                      # python验证脚本
```

### [脚本参数](#目录)

总共两个案例. 在 `config.yaml` 或命令参数中, 可以通过参数 `problem` 来选择案例.

| parameter | description                                   | default value |
|-----------|-----------------------------------------------|---------------|
| problem   | 用于解决的案例，`1d_caputo`或`2d_fractional_laplacian` | 1d_caputo     |

train.py中的重要参数如下:

| 参数             | 描述                         | 默认值                                    |
|----------------|----------------------------|----------------------------------------|
| layers_u       | 神经网络d的宽度                   | 15 80 80 80                            |
| layers_y       | 神经网络y的宽度                   | 2 80 80 80                             |
| save_ckpt      | 是否保存checkpoint             | true                                   |
| load_ckpt      | 是否加载checkpoint             | false                                  |
| save_ckpt_path | checkpoint保存路径             | ./checkpoints/1d_caputo                |
| load_ckpt_path | checkpoint加载路径             | ./checkpoints/1d_caputo/1d_caputo.ckpt |
| save_fig       | 是否保存和绘制图片                  | true                                   |
| figures_path   | 图片保存路径                     | ./figures/1d_caputo                    |
| save_data      | 是否保存数据                     | true                                   |
| load_data_path | 加载数据的路径                    | ./data/1d_caputo                       |
| save_data_path | 保存数据的路径                    | ./data/1d_caputo                       |
| log_path       | 日志保存路径                     | ./logs                                 |
| lr             | 学习率                        | 1e-3                                   |
| epoch          | 时期（迭代次数）                   | 20001                                  |
| batch_size     | 批尺寸                        | 100000                                 |
| print_interval | 损失与时间打印间隔                  | 10                                     |
| download_data  | 必要的数据集与checkpoint          | deeponet                               |
| force_download | 是否强制下载数据集与checkpoint       | false                                  |
| amp_level      | MindSpore自动混合精度等级          | O3                                     |
| device_id      | 需要设置的设备号                   | None                                   |
| mode           | MindSpore静态图模式（0）或动态图模式（1） | 0                                      |

### [训练流程](#目录)

- 在 GPU/Ascend 上运行

   ```bash
   python train.py
   ```

  训练期间的损失值将打印在控制台中，也可以训练后在日志文件中查看。

   ```bash
   # grep "loss:" log
    epoch:0, step: 0/10, loss: 0.9956, interval: 33.33840894699097s, total: 33.33840894699097s
    Epoch: 1, Training loss: 0.8623, Test loss: 0.853619, Test loss0: 1.567865, RelErr: 0.923920750617981,  RelErr0: 1.25214421749115
    epoch:1, step: 0/10, loss: 0.853, interval: 18.432061195373535s, total: 51.7704701423645s
    epoch:2, step: 0/10, loss: 0.8345, interval: 0.27780890464782715s, total: 52.04827904701233s
    epoch:3, step: 0/10, loss: 0.818, interval: 0.2761566638946533s, total: 52.32443571090698s
    epoch:4, step: 0/10, loss: 0.816, interval: 0.2772941589355469s, total: 52.60172986984253s
    epoch:5, step: 0/10, loss: 0.8013, interval: 0.278522253036499s, total: 52.88025212287903s
    epoch:6, step: 0/10, loss: 0.795, interval: 0.2778182029724121s, total: 53.15807032585144s
    epoch:7, step: 0/10, loss: 0.794, interval: 0.2756061553955078s, total: 53.43367648124695s
    epoch:8, step: 0/10, loss: 0.791, interval: 0.272977352142334s, total: 53.70665383338928s
    epoch:9, step: 0/10, loss: 0.7837, interval: 0.2748894691467285s, total: 53.98154330253601s
    epoch:10, step: 0/10, loss: 0.7754, interval: 0.2739229202270508s, total: 54.25546622276306s
  ...
   ```

  模型checkpoint将保存在 `save_ckpt_path`中, 默认为`./checkpoints` 目录中。

### [推理流程](#目录)

  在运行下面的命令之前，请检查使用的`config.yaml` 中的checkpoint加载路径`load_ckpt_path`进行推理。

- 在 GPU/Ascend 上运行

   ```bash
   python eval.py
   ```

  您可以通过日志文件`log_path`查看过程与结果，默认位于`./logs` 。
  结果图片存放于`figures_path`中，默认位于[`./figures`](./figures)。