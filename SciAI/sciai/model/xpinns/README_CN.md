[ENGLISH](README.md) | 简体中文

# 目录

- [Extended PINNs (XPINNs) 描述](#extended-pinns--xpinns--描述)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [脚本说明](#脚本说明)
    - [脚本和示例代码](#脚本和示例代码)
    - [脚本参数](#脚本参数)
    - [训练流程](#训练流程)
    - [推理流程](#推理流程)

## [Extended PINNs (XPINNs) 描述](#目录)

模型Extended PINNs（XPINNs）是一种通过分解时空域来求解任意复杂域中非线性偏微分方程（PDE）的物理感知深度学习方法。
该模型是对PINNs模型和Conservative PINNs模型的一种范化。
这种范化性同时表现在适用性和域分解方法，使得XPINNs可以高效地进行并行计算。

> [论文](https://doi.org/10.4208/cicp.OA-2020-0164): A.D.Jagtap, G.E.Karniadakis, Extended Physics-Informed Neural
> Networks (XPINNs): A Generalized Space-Time Domain Decomposition Based Deep Learning Framework for Nonlinear Partial
> Differential Equations, Commun. Comput. Phys., Vol.28, No.5, 2002-2041, 2020.

案例详情: 使用三个子域的XPINNs神经网络求解二维泊松方程。

## [模型架构](#目录)

![XPINN schematic](./figures/xpinn_arch.png)

在上图顶部展示了XPINNs在子域中的子网示意图，其中显示了Viscous Burgers Equation的神经网络部分和物理感知部分。
在上图底部展示了在一个“X”形研究域中不规则的子域分割，并且在每个子域中都有一个子神经网络。这些子神经网络由边界条件相缝合。
在这个案例中，研究域边界由连续的黑色实线定义，子域相邻边界由黑色虚线确定。

## [数据集](#目录)

用于训练的数据集和预训练checkpoints文件将会在首次启动时自动下载。

使用的数据集: [XPINN 2D_PoissonEqn]

- 数据集大小
    - u_exact:（1, 22387）
    - u_exact1:（1, 18211）
    - u_exact2:（1, 2885）
    - u_exact3:（1, 1291）
    - ub:（1, 6284）
    - x_f1:（1, 18211）
    - x_f2:（1, 2885）
    - x_f3:（1, 1291）
    - x_total:（1, 22387）
    - xb:（1, 6284）
    - xi1:（1, 6284）
    - xi2:（1, 6284）
    - y_f1:（1, 18211）
    - y_f2:（1, 2885）
    - y_f3:（1, 1291）
    - y_total:（1, 22387）
    - yb:（1, 6284）
    - yi1:（1, 6284）
    - yi1:（1, 6284）
- 数据集格式: `.mat` files
    - 注: 数据会在`process.py`中处理
- 数据集在`./data`目录下，目录结构如下:

```text
├── data
│   └─ XPINN_2D_PoissonEqn.mat
```

您如果需要手动下载数据集或checkpoints文件，
请访问[此链接](https://download.mindspore.cn/mindscience/SciAI/sciai/model/xpinns/)。

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
    --layers1 2 30 30 1 \
    --layers2 2 20 20 20 20 1 \
    --layers3 2 25 25 25 1 \
    --save_ckpt true \
    --save_fig true \
    --load_ckpt false \
    --save_ckpt_path ./checkpoints \
    --load_data_path ./data \
    --load_ckpt_path ./checkpoints/model_final_float32.ckpt \
    --figures_path ./figures \
    --log_path ./logs \
    --print_interval 20 \
    --ckpt_interval 20 \
    --lr 8e-4 \
    --epochs 501 \
    --download_data xpinns \
    --force_download false \
    --amp_level O3 \
    --device_id 0 \
    --mode 0
```

## [脚本说明](#目录)

### [脚本和示例代码](#目录)

文件结构如下:

```text
├── xpinns
│   ├── checkpoints                  # checkpoint文件
│   ├── data                         # 数据文件
│   │   └── XPINN_2D_PoissonEqn.mat  # 2-D泊松方程数据集
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

| parameter      | description                | default value                          |
|----------------|----------------------------|----------------------------------------|
| layers1        | 子域一中神经网络层宽                 | 2 30 30 1                              |
| layers2        | 子域二中神经网络层宽                 | 2 20 20 20 20 1                        |
| layers3        | 子域三中神经网络层宽                 | 2 25 25 25 1                           |
| save_ckpt      | 是否保存checkpoint             | true                                   |
| save_fig       | 是否保存和绘制图片                  | true                                   |
| load_ckpt      | 是否加载checkpoint             | false                                  |
| save_ckpt_path | checkpoint保存路径             | ./checkpoints                          |
| load_data_path | 加载数据的路径                    | ./data                                 |
| load_ckpt_path | checkpoint加载路径             | ./checkpoints/model_final_float32.ckpt |
| figures_path   | 图片保存路径                     | ./figures                              |
| log_path       | 日志保存路径                     | ./logs                                 |
| print_interval | 时间与loss打印间隔                | 20                                     |
| ckpt_interval  | checkpoint保存间隔             | 20                                     |
| lr             | 学习率                        | 8e-4                                   |
| epochs         | 时期（迭代次数）                   | 501                                    |
| download_data  | 模型所需数据集与(或)checkpoints     | xpinns                                 |
| force_download | 是否强制下载数据                   | false                                  |
| amp_level      | MindSpore自动混合精度等级          | O3                                     |
| device_id      | 需要设置的设备号                   | None                                   |
| mode           | MindSpore静态图模式（0）或动态图模式（1） | 0                                      |

### [训练流程](#目录)

- 在 GPU/Ascend 上运行

  ```bash
  python train.py
  ```

  训练期间的损失值将打印在控制台中， 也可以训练后在日志文件中查看。

  ```bash
  # grep "loss:" log
  step: 0, total loss: 302.36923, Loss1: 284.04446, Loss2: 1.6917802, Loss3: 16.633005, interval: 220.22029280662537s, total: 220.22029280662537s
  step: 20, total loss: 172.39247, Loss1: 155.68967, Loss2: 1.4727805, Loss3: 15.230027, interval: 11.906857013702393s, total: 232.12714982032776s
  step: 40, total loss: 49.625393, Loss1: 34.234493, Loss2: 2.4619372, Loss3: 12.928962, interval: 7.346828460693359s, total: 239.47397828102112s
  step: 60, total loss: 31.988087, Loss1: 21.279911, Loss2: 2.394319, Loss3: 8.313857, interval: 7.111770391464233s, total: 246.58574867248535s
  step: 80, total loss: 28.279648, Loss1: 19.259611, Loss2: 2.5380166, Loss3: 6.4820194, interval: 6.57047700881958s, total: 253.15622568130493s
  step: 100, total loss: 25.35678, Loss1: 17.901184, Loss2: 2.6389856, Loss3: 4.816611, interval: 6.642438173294067s, total: 259.798663854599s
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