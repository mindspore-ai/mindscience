[ENGLISH](README.md) | 简体中文

# 目录

- [Fractional PINNs 描述](#fractional-pinns-描述)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [脚本说明](#脚本说明)
    - [脚本和示例代码](#脚本和示例代码)
    - [脚本参数](#脚本参数)
    - [训练流程](#训练流程)
    - [推理流程](#推理流程)

## [Fractional PINNs 描述](#目录)

Fractional PINNs（FPINNs）模型主要解决的问题是时空域中带有分数阶偏导的对流-扩散方程（ADE）。
该模型在计算残差时，对整数阶算子和分数阶算子采取了不同的计算方法。
对于整数阶算子，仍然采用自动微分的方式（AD），而对于分数阶算子则采用数值离散的方法计算。
该方法克服了自动微分技术无法应用于分数阶算子的困难。
该困难来源于整数微积分中的链式法则在分数微积分中并不成立。
另外，在合适的初始条件下，通过该方法得到的模型在较大的噪声下，仍然能够得到准确的预测值。

> [论文](https://arxiv.org/abs/1811.08967): Pang G, Lu L, Karniadakis G E. fPINNs: Fractional physics-informed neural
> networks[J]. SIAM Journal on Scientific Computing, 2019, 41(4): A2603-A2626.

案例详情: 使用PINNs解决一维分数阶偏导扩散模型。

## [数据集](#目录)

用于每个案例的数据集在训练过程中随机生成。
数据集的大小取决于样本的数量，这些样本由`config.yaml`中的`num_domain`， `num_boundary`和`num_initial`控制。

预训练checkpoints文件将会在首次启动时自动下载。
您如果需要手动下载checkpoints文件，
请访问[此链接](https://download.mindspore.cn/mindscience/SciAI/sciai/model/fpinns/)。

## [环境要求](#目录)

- 硬件(Ascend/GPU)
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
    --problem fractional_diffusion_1d \
    --layers 2 20 20 20 20 1 \
    --x_range 0 1 \
    --t_range 0 1 \
    --num_domain 400 \
    --num_boundary 0 \
    --num_initial 0 \
    --num_test 400 \
    --lr 1e-3 \
    --save_ckpt true \
    --save_fig true \
    --load_ckpt false \
    --save_ckpt_path ./checkpoints/fractional_diffusion_1d \
    --load_ckpt_path ./checkpoints/fractional_diffusion_1d/model_iter_10000_float32.ckpt \
    --figures_path ./figures/fractional_diffusion_1d \
    --log_path ./logs \
    --print_interval 100 \
    --epochs 10001 \
    --download_data fpinns \
    --force_download false \
    --amp_level O0 \
    --device_id 0 \
    --mode 0
```

## [脚本说明](#目录)

### [脚本和示例代码](#目录)

文件结构如下:

```text
├── fpinns
│   ├── checkpoints                  # checkpoint文件
│   ├── data                         # 数据文件
│   ├── figures                      # 结果图片
│   ├── logs                         # 日志文件
│   ├── src                          # 源代码
│   │   ├── network.py               # 网络架构
│   │   ├── plot.py                  # 绘制结果
│   │   ├── problem.py               # 抽象问题类
│   │   └── process.py               # 流程处理
│   ├── config.yaml                  # 超参数配置
│   ├── README.md                    # 英文模型说明
│   ├── README_CN.md                 # 中文模型说明
│   ├── train.py                     # python训练脚本
│   └── eval.py                      # python评估脚本
```

### [脚本参数](#目录)

train.py中的重要参数如下:

| 参数             | 描述                         | 默认值                                                                 |
|----------------|----------------------------|---------------------------------------------------------------------|
| problem        | 问题场景                       | fractional_diffusion_1d                                             |
| layers         | 神经网络结构                     | 2 20 20 20 20 1                                                     |
| x_range        | 空间范围                       | 0 1                                                                 |
| t_range        | 时间范围                       | 0 1                                                                 |
| num_domain     | 域内数据数量                     | 400                                                                 |
| num_boundary   | 边界数据数量                     | 0                                                                   |
| num_initial    | 初始条件数据数量                   | 0                                                                   |
| num_test       | 测试数据数量                     | 400                                                                 |
| lr             | 学习率                        | 1e-3                                                                |
| save_ckpt      | 是否保存checkpoint             | true                                                                |
| save_fig       | 是否保存和绘制图片                  | true                                                                |
| load_ckpt      | 时间与loss打印间隔                | false                                                               |
| save_ckpt_path | checkpoint保存路径             | ./checkpoints/fractional_diffusion_1d                               |
| load_ckpt_path | checkpoint加载路径             | ./checkpoints/fractional_diffusion_1d/model_iter_10000_float32.ckpt |
| figures_path   | 图片保存路径                     | ./figures/fractional_diffusion_1d                                   |
| log_path       | 日志保存路径                     | ./logs                                                              |
| print_interval | 时间与loss打印间隔                | 100                                                                 |
| epochs         | 时期（迭代次数）                   | 10001                                                               |
| download_data  | 模型所需数据集与(或)checkpoints     | fpinns                                                              |
| force_download | 是否强制下载数据                   | false                                                               |
| amp_level      | MindSpore自动混合精度等级          | O0                                                                  |
| device_id      | 需要设置的设备号                   | None                                                                |
| mode           | MindSpore静态图模式（0）或动态图模式（1） | 0                                                                   |

### [训练流程](#目录)

- 在 GPU/Ascend 上运行

   ```bash
   python train.py
   ```

  训练期间的损失值将打印在控制台中， 也可以训练后在日志文件中查看。

  ```bash
  # grep "loss:" log
  step: 0, loss: 0.1720775, interval: 0.515510082244873s, total: 0.515510082244873s
  step: 100, loss: 0.004217985, interval: 0.15668702125549316s, total: 0.6721971035003662s
  step: 200, loss: 0.0026953542, interval: 0.14049434661865234s, total: 0.8126914501190186s
  step: 300, loss: 0.002297479, interval: 0.13532018661499023s, total: 0.9480116367340088s
  step: 400, loss: 0.0018170077, interval: 0.13717007637023926s, total: 1.085181713104248s
  step: 500, loss: 0.0009912008, interval: 0.1338338851928711s, total: 1.2190155982971191s
  step: 600, loss: 0.00050001504, interval: 0.14569568634033203s, total: 1.3647112846374512s
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