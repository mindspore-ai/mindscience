[ENGLISH](README.md) | 简体中文

# 目录

- [PhyGeoNet 描述](#PhyGeoNet-描述)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [脚本说明](#脚本说明)
    - [脚本和示例代码](#脚本和示例代码)
    - [脚本参数](#脚本参数)
    - [训练流程](#训练流程)
    - [推理流程](#推理流程)

## [PhyGeoNet 描述](#目录)

PhyGeoNet作者提出了一种新颖的物理约束CNN学习架构，旨在学习没有任何标签数据的不规则域上的含参偏微分方程的解，
并通过求解一些不规则域上稳态偏微分方程（包括热力方程、NS方程和带参数化边界条件的，具有不同且变化几何和空间源场的泊松方程），
来评估该模型的表现。

> [论文](https://www.sciencedirect.com/science/article/pii/S0021999120308536): Gao H, Sun L, Wang J X. PhyGeoNet:
> Physics-informed geometry-adaptive convolutional neural networks for solving parameterized steady-state PDEs on
> irregular domain[J]. Journal of Computational Physics, 2021, 428: 110079.

## [数据集](#目录)

用于训练的数据集和预训练checkpoints文件将会在首次启动时自动下载。

使用的数据集: OpenFOAM 边界数据

- 数据格式: FoamFile
- 数据示例:  
  Geometry-Adaptive
  :-----:

<p align="center">
    <img align = 'center' height="200" src="figures/mesh.png?raw=true">
</p>

- 注意: 数据会在`process.py`中被处理
- 数据集存在于`./data`目录中，目录结构如下所示:

```text
├── data
│   ├── case0
│   ├── TemplateCase
│   │   ├── 0
│   │   ├── 30
│   │   ├── 60
            ...
```

您如果需要手动下载数据集或checkpoints文件，
请访问[此链接](https://download.mindspore.cn/mindscience/SciAI/sciai/model/phygeonet/)。

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
    --save_ckpt true \
    --load_ckpt false \
    --save_fig true \
    --save_ckpt_path ./checkpoints \
    --load_ckpt_path ./checkpoints/model_final.ckpt \
    --log_path ./logs \
    --figures_path ./figures \
    --load_data_path ./data/case0 \
    --save_data_path ./data/case0 \
    --lr 1e-3 \
    --epochs 1501 \
    --batch_size 1 \
    --download_data phygeonet \
    --force_download false \
    --amp_level O0 \
    --device_id 0 \
    --mode 0
```

## [脚本说明](#目录)

### [脚本和示例代码](#目录)

文件结构如下:

```text
├── phygeonet
│   ├── checkpoints                  # checkpoints文件
│   ├── data                         # 数据文件
│   ├── figures                      # 结果图片
│   ├── logs                         # 日志文件
│   ├── src                          # 源代码目录
│   │   ├── dataset.py               # 数据集类
│   │   ├── foam_ops.py              # openfoam处理函数
│   │   ├── network.py               # 网络架构
│   │   ├── plot.py                  # 绘制结果
│   │   ├── py_mesh.py               # mesh可视化
│   │   └── process.py               # 数据处理
│   ├── config.yaml                  # 超参数配置
│   ├── README.md                    # 英文模型说明
│   ├── README_CN.md                 # 中文模型说明
│   ├── train.py                     # python训练脚本
│   └── eval.py                      # python评估脚本
```

### [脚本参数](#目录)

train.py中的重要参数如下:

| 参数名            | 含义                         | 默认值                            |
|----------------|----------------------------|--------------------------------|
| save_ckpt      | 是否保存checkpoint             | true                           |
| save_fig       | 是否保存和绘制图片                  | true                           |
| load_ckpt      | 是否加载checkpoint             | false                          |
| save_ckpt_path | checkpoint保存路径             | ./checkpoints                  |
| load_ckpt_path | checkpoint加载路径             | ./checkpoints/model_final.ckpt |
| figures_path   | 图片保存路径                     | ./figures                      |
| log_path       | 日志保存路径                     | ./logs                         |
| load_data_path | 加载数据的路径                    | ./data/case0                   |
| save_data_path | 保存数据的路径                    | ./data/case0                   |
| lr             | 学习率                        | 1e-3                           |
| epochs         | 时期（迭代次数）                   | 1501                           |
| batch_size     | 批次大小                       | 1                              |
| download_data  | 模型所需数据集与(或)checkpoints     | phygeonet                      |
| force_download | 是否强制下载数据                   | false                          |
| amp_level      | MindSpore自动混合精度等级          | O0                             |
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
  step: 0, loss: 43195.062, interval: 27.112539768218994s, total: 27.112539768218994s
  m_res Loss:43195.062, e_v Loss:0.9002858788030269
  step: 1, loss: 33180.41, interval: 13.531551599502563s, total: 40.64409136772156s
  m_res Loss:33180.41, e_v Loss:0.9047985325837214
  step: 2, loss: 26929.658, interval: 13.711315870285034s, total: 54.35540723800659s
  m_res Loss:26929.658, e_v Loss:0.8957896098853404
  step: 3, loss: 25561.246, interval: 13.139239072799683s, total: 67.49464631080627s
  m_res Loss:25561.246, e_v Loss:0.873105561166751
  step: 4, loss: 22892.932, interval: 13.660631895065308s, total: 81.15527820587158s
  m_res Loss:22892.932, e_v Loss:0.8364832182670862
  step: 5, loss: 20156.662, interval: 13.264018535614014s, total: 94.4192967414856s
  m_res Loss:20156.662, e_v Loss:0.8054152573595449
  step: 6, loss: 18716.941, interval: 13.6557936668396s, total: 108.0750904083252s
  m_res Loss:18716.941, e_v Loss:0.7725231596138828
  step: 7, loss: 17887.295, interval: 13.549290895462036s, total: 121.62438130378723s
  m_res Loss:17887.295, e_v Loss:0.7317578269591556
  step: 8, loss: 16525.012, interval: 13.712275266647339s, total: 135.33665657043457s
  m_res Loss:16525.012, e_v Loss:0.6834196610680607
  step: 9, loss: 15021.678, interval: 13.676922798156738s, total: 149.0135793685913s
  m_res Loss:15021.678, e_v Loss:0.6367062855788159
  step: 10, loss: 14102.34, interval: 13.010124444961548s, total: 162.02370381355286s
  m_res Loss:14102.34, e_v Loss:0.5960404029235946
  ...
   ```

- 训练结束后，您仍然可以通过保存在`log_path`下面的日志文件回顾训练过程，默认为`./logs`目录中。

- 模型checkpoint将保存在 `save_ckpt_path`中，默认为`./checkpoints` 目录中。

### [推理流程](#目录)

在使用下面的命令进行推理之前，请检查`config.yaml` 中的checkpoint加载路径`load_ckpt_path`。

- 在 GPU/Ascend 上运行

   ```bash
   python eval.py
   ```

  您可以通过日志文件`log_path`查看过程与结果，默认位于`./logs` 。
  结果图片存放于`figures_path`中，默认位于[`./figures`](./figures)。