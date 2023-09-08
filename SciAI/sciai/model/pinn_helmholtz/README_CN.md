[ENGLISH](README.md) | 简体中文

# 目录

- [Helmholtz PINNs 描述](#helmholtz-pinns-描述)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [脚本说明](#脚本说明)
    - [脚本和示例代码](#脚本和示例代码)
    - [脚本参数](#脚本参数)
    - [训练流程](#训练流程)
    - [推理流程](#推理流程)

## [Helmholtz PINNs 描述](#目录)

以下的论文应用了物理信息神经网络（PINNs）来求解各向同性和各向异性介质中的Helmholtz方程。网络使用正弦函数作为激活函数，因其在求解时间和频率域波动方程中效果显著。
该项目采用了固定正弦激活函数的神经网络，以解决从各向同性的Marmoussi地质模型的Helmholtz方程。

> [论文](https://academic.oup.com/gji/article-abstract/228/3/1750/6409132):
> Song C, Alkhalifah T, Waheed U B.
> A versatile framework to solve the Helmholtz equation using physics-informed neural networks[J].
> Geophysical Journal International, 2022, 228(3): 1750-1762.

## [数据集](#目录)

用于训练的数据集和预训练checkpoints文件将会在首次启动时自动下载。

数据集是使用各向同性的Marmousi模型收集的，该模型与原始论文中的模型完全相同。单个源位于表面上，位置为4.625公里。
无论是垂直方向还是水平方向，采样间隔都是25米。

- 数据格式: `.mat`文件
    - 注: 数据会在`process.py`中处理
- 数据集在`./data`目录下，目录结构如下:

```text
├── data
│   └── Marmousi_3Hz_singlesource_ps.mat
```

您如果需要手动下载数据集或checkpoints文件，
请访问[此链接](https://download.mindspore.cn/mindscience/SciAI/sciai/model/pinn_helmholtz/)。

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
    --layers 2 40 40 40 40 40 40 40 40 2 \
    --save_ckpt true \
    --load_ckpt false \
    --save_ckpt_path ./checkpoints \
    --load_ckpt_path ./checkpoints/model_adam_100000_float32.ckpt \
    --load_data_path ./data \
    --save_fig true \
    --figures_path ./figures \
    --save_results true \
    --results_path ./data/results \
    --print_interval 20 \
    --log_path ./logs \
    --lr 0.001 \
    --epochs 100000 \
    --num_batch 1 \
    --lbfgs false \
    --epochs_lbfgs 50000 \
    --download_data pinn_helmholtz \
    --force_download false \
    --amp_level O0 \
    --device_id 0 \
    --mode 0
```

## [脚本说明](#目录)

### [脚本和示例代码](#目录)

文件结构如下:

```text
├── pinn_helmholtz
│   ├── checkpoints                          # checkpoint文件
│   ├── data                                 # 数据文件
│   │   ├── results                          # 训练结果
│   │   └── Marmousi_3Hz_singlesource_ps.mat # Marmousi 3Hz单源数据集
│   ├── figures                              # 结果图片
│   ├── logs                                 # 日志文件
│   ├── src                                  # 源代码
│   │   ├── network.py                       # 网络架构
│   │   ├── plot.py                          # 绘制结果
│   │   └── process.py                       # 数据处理
│   ├── config.yaml                          # 超参数配置
│   ├── README.md                            # 英文模型说明
│   ├── README_CN.md                         # 中文模型说明
│   ├── train.py                             # python训练脚本
│   └── eval.py                              # python评估脚本
```

### [脚本参数](#目录)

train.py中的重要参数如下:

| parameter      | description                | default value                                |
|----------------|----------------------------|----------------------------------------------|
| layers         | 神经网络宽度                     | 2 40 40 40 40 40 40 40 40 2                  |
| save_ckpt      | 是否保存checkpoint             | true                                         |
| load_ckpt      | 是否加载checkpoint             | false                                        |
| save_ckpt_path | checkpoint保存路径             | ./checkpoints                                |
| load_ckpt_path | checkpoint加载路径             | ./checkpoints/model_adam_100000_float32.ckpt |
| load_data_path | 加载数据的路径                    | ./data                                       |
| save_fig       | 是否保存和绘制图片                  | true                                         |
| figures_path   | 图片保存路径                     | ./figures                                    |
| save_results   | 是否保存预测结果与loss              | true                                         |
| results_path   | 预测结果与loss保存路径              | ./data/results                               |
| print_interval | 时间与loss打印间隔                | 20                                           |
| log_path       | 日志保存路径                     | ./logs                                       |
| lr             | 学习率                        | 1e-3                                         |
| epochs         | 时期(迭代次数)                   | 100000                                       |
| num_batch      | 批数量                        | 1                                            |
| lbfgs          | 是否使用l-bfgs                 | false                                        |
| epochs_lbfgs   | l-bfgs时期数(迭代次数)            | 50000                                        |
| download_data  | 模型所需数据集与(或)checkpoints     | pinn_helmholtz                               |
| force_download | 是否强制下载数据                   | false                                        |
| amp_level      | MindSpore自动混合精度等级          | O0                                           |
| device_id      | 需要设置的设备号                   | None                                         |
| mode           | MindSpore静态图模式（0）或动态图模式（1） | 0                                            |

### [训练流程](#目录)

- 在 GPU/Ascend 上运行

   ```bash
   python train.py
   ```

  训练期间的损失值将打印在控制台中， 也可以训练后在日志文件中查看。

  ```bash
  # grep "loss:" log
  ...
  step: 40, loss: 2.0659282, interval: 16.741257190704346s, total: 63.9092378616333s
  step: 60, loss: 1.27979, interval: 16.732836723327637s, total: 80.64207458496094s
  step: 80, loss: 1.0679382, interval: 16.754377841949463s, total: 97.3964524269104s
  step: 100, loss: 0.96829647, interval: 16.702229022979736s, total: 114.09868144989014s
  step: 120, loss: 0.9059235, interval: 16.710976123809814s, total: 130.80965757369995s
  step: 140, loss: 0.86077166, interval: 16.749966621398926s, total: 147.55962419509888s
  step: 160, loss: 0.825172, interval: 16.73036813735962s, total: 164.2899923324585s
  step: 180, loss: 0.7951189, interval: 16.77035140991211s, total: 181.0603437423706s
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