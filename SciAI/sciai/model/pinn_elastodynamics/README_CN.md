[ENGLISH](README.md) | 简体中文

# 目录

- [PINN elastodynamics 描述](#pinn-elastodynamics-描述)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [脚本说明](#脚本说明)
    - [脚本和示例代码](#脚本和示例代码)
    - [脚本参数](#脚本参数)
    - [训练流程](#训练流程)
    - [推理流程](#推理流程)

## [PINN elastodynamics 描述](#目录)

本文提出了一种具有混合变量输出的物理信息神经网络（PINN）来模拟弹性动力学问题，无需诉诸标签数据，其中将初始/边界条件作为硬性约束条件。

> [论文](https://arxiv.org/abs/2006.08472): Rao C, Sun H, Liu Y. Physics-informed deep learning for computational
> elastodynamics without labeled data[J]. Journal of Engineering Mechanics, 2021, 147(8): 04021043.

范例详情:

- **ElasticWaveInfinite**: 对于无限域中弹性波传播的训练脚本和数据集。

## [数据集](#数据集)

用于训练的数据集和预训练checkpoints文件将会在首次启动时自动下载。

数据集: [burgers shock]

- 数据大小
    - x:（256, 1） in [-1, 1]
    - t:（100, 1） in [0, 1]
- 数据格式: `.mat` files
    - 注: Data will be processed in `process.py`
- 数据集在 `./data` 目录下, 目录结构如下:

```text
├── data
│   ├── FEM_result
│   └── burgers_shock.mat
```

您如果需要手动下载数据集或checkpoints文件，
请访问[此链接](https://download.mindspore.cn/mindscience/SciAI/sciai/model/pinn_elastodynamics/)。

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
    --uv_layers 3 140 140 140 140 140 140 7 \
    --save_ckpt true \
    --save_fig true \
    --load_ckpt false \
    --save_ckpt_path ./checkpoints \
    --load_data_path ./data \
    --load_ckpt_path ./checkpoints/uv_NN_14s_float32_new.pickle \
    --figures_path ./figures/output \
    --log_path ./logs \
    --print_interval 1 \
    --lr 1e-3 \
    --epochs 100000 \
    --use_lbfgs false \
    --max_iter_lbfgs 10000 \
    --download_data pinn_elastodynamics \
    --force_download false \
    --amp_level O3 \
    --device_id 0 \
    --mode 0
```

## [脚本说明](#目录)

### [脚本和示例代码](#目录)

文件结构如下:

```text
├── pinn_elastodynamics
│   ├── checkpoints                                 # checkpoints文件
│   ├── data                                        # 数据文件
│   │   ├── FEM_result                              # 模型结果
│   │   └── burgers_shock.mat                       # burgers shock matlab 数据集
│   ├── figures                                     # 图片目录
│   │   ├── output                                  # 结果图片
│   │   └── GIF_uv.gif                              # gif动图
│   ├── logs                                        # 日志文件
│   ├── src                                         # 源代码目录
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

| 参数名            | 含义                         | 默认值                                        |
|----------------|----------------------------|--------------------------------------------|
| uv_layers      | uv网络的层宽                    | 3 140 140 140 140 140 140 7                |
| save_ckpt      | 是否保存checkpoint             | true                                       |
| save_fig       | 是否保存和绘制图片                  | true                                       |
| load_ckpt      | 是否加载checkpoint             | false                                      |
| save_ckpt_path | checkpoint保存路径             | ./checkpoints                              |
| load_ckpt_path | checkpoint加载路径             | ./checkpoints/uv_NN_14s_float32_new.pickle |
| load_data_path | 数据加载路径                     | ./data                                     |
| figures_path   | 图片保存路径                     | ./figures/output                           |
| log_path       | 日志保存路径                     | ./logs                                     |
| print_interval | 时间与loss打印间隔                | 1                                          |
| ckpt_interval  | checkpoint保存间隔             | 1000                                       |
| lr             | 学习率                        | 1e-3                                       |
| epochs         | 时期（迭代次数）                   | 100000                                     |
| use_lbfgs      | 是否在adam后使用L-BFGS           | false                                      |
| max_iter_lbfgs | L-BFGS最大迭代次数               | null                                       |
| download_data  | 模型所需数据集与(或)checkpoints     | pinn_elastodynamics                        |
| force_download | 是否强制下载数据                   | false                                      |
| amp_level      | MindSpore自动混合精度等级          | O3                                         |
| device_id      | 需要设置的设备号                   | None                                       |
| mode           | MindSpore静态图模式（0）或动态图模式（1） | 0                                          |

### [训练流程](#目录)

- 在 GPU/Ascend 上运行

   ```bash
   python train.py
   ```

  经过训练，loss值会输出如下:

  ```bash
  # grep "loss:" log
  step: 0, loss: 13.111078, interval: 34.97392821311951s, total: 34.97392821311951s
  step: 1, loss: 200.82831, interval: 0.08405542373657227s, total: 35.05798363685608s
  step: 2, loss: 25.90921, interval: 0.0743703842163086s, total: 35.13235402107239s
  step: 3, loss: 14.451968, interval: 0.08811116218566895s, total: 35.22046518325806s
  step: 4, loss: 48.904766, interval: 0.0770270824432373s, total: 35.297492265701294s
  step: 5, loss: 34.188297, interval: 0.08126688003540039s, total: 35.378759145736694s
  step: 6, loss: 6.9077187, interval: 0.07222247123718262s, total: 35.45098161697388s
  step: 7, loss: 3.6523025, interval: 0.07272839546203613s, total: 35.52371001243591s
  step: 8, loss: 17.293848, interval: 0.0725715160369873s, total: 35.5962815284729s
  step: 9, loss: 20.209349, interval: 0.0718080997467041s, total: 35.668089628219604s
  step: 10, loss: 11.190135, interval: 0.07178211212158203s, total: 35.73987174034119s
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