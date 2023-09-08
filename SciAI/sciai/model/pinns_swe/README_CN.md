[ENGLISH](README.md) | 简体中文

# 目录

- [浅水波方程PINNs描述](#浅水波方程PINNs描述)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [脚本说明](#脚本说明)
    - [脚本和示例代码](#脚本和示例代码)
    - [脚本参数](#脚本参数)
    - [训练流程](#训练流程)
    - [推理流程](#推理流程)

## [浅水波方程PINNs描述](#目录)

浅水波方程（SWE）是开发天气和气候预测模型动态核心的新算法的典型测试基础。
Bihlo提出了一个物理信息神经网络（PINNs）来解决在旋转球体上的SWE。
PINNs的一个缺点是，随着问题领域的扩大，训练数据也会增加。
为了克服测试案例中的大时间域，作者建议将大时间域分割成几个不重叠的子间隔，并连续地在每个子间隔中解决SWE，通过为每个间隔训练一个新的神经网络。

本项目展示了余弦钟绕球平流场景。

> [论文](https://arxiv.org/abs/2104.00615):Bihlo, Alex, and Roman O. Popovych.
> Physics-Informed Neural Networks for the Shallow-Water Equations on the Sphere. arXiv.org, February 12, 2022.

## [数据集](#目录)

用于训练的数据集是由函数`collocation_points`在`./src/process.py`中随机生成的。用户应该给出数据的范围，由days、lambda 和
theta 的边界定义。默认的范围是:

- days: [0, 12]
- lambda: [- pi, pi]
- theta: [- pi / 2, pi / 2]

数据集的大小取决于样本的数量，这些样本由`config.yaml`中的`n_pde`和`n_iv`控制，默认值分别为100000和10000。

预训练checkpoints文件将会在首次启动时自动下载。
您如果需要手动下载checkpoints文件，
请访问[此链接](https://download.mindspore.cn/mindscience/SciAI/sciai/model/pinns_swe/)。

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
    --layers 4 20 20 20 20 1 \
    --load_ckpt false \
    --save_ckpt_path ./checkpoints \
    --load_ckpt_path ./checkpoints \
    --save_fig true \
    --figures_path ./figures \
    --log_path ./logs \
    --lr 1e-3 \
    --epochs 18000 \
    --n_pde 100000 \
    --n_iv 10000 \
    --u 1 \
    --h 1000 \
    --days 12 \
    --download_data pinns_swe \
    --force_download false \
    --amp_level O3 \
    --device_id 0 \
    --mode 0
```

## [脚本说明](#目录)

### [脚本和示例代码](#目录)

文件结构如下:

```text
├── pinns_swe
│   ├── checkpoints                     # checkpoint文件
│   ├── data                            # 数据文件
│   ├── figures                         # 结果图片
│   ├── logs                            # 日志文件
│   ├── src                             # 源代码
│   │   ├── network.py                  # 网络架构
│   │   ├── plot.py                     # 绘制结果
│   │   ├── process.py                  # 数据处理
│   │   ├── problem.py                  # 训练流程定义
│   │   └── linear_advection_sphere.py  # 平流方程loss定义
│   ├── config.yaml                     # 超参数配置
│   ├── README.md                       # 英文模型说明
│   ├── README_CN.md                    # 中文模型说明
│   ├── train.py                        # python训练脚本
│   └── eval.py                         # python评估脚本
```

### [脚本参数](#目录)

train.py中的重要参数如下:

| parameter      | description                | default value   |
|----------------|----------------------------|-----------------|
| layers         | 神经网络宽度                     | 4 20 20 20 20 1 |
| load_ckpt      | 是否加载checkpoint             | false           |
| save_ckpt_path | checkpoint保存路径             | ./checkpoints   |
| load_ckpt_path | checkpoint加载路径             | ./checkpoints   |
| save_fig       | 是否保存和绘制图片                  | true            |
| figures_path   | 图片保存路径                     | ./figures       |
| log_path       | 日志保存路径                     | ./logs          |
| lr             | 学习率                        | 1e-3            |
| epochs         | 时期（迭代次数）                   | 18000           |
| n_pde          | 数据点数量                      | 100000          |
| n_iv           | 初始点数量                      | 10000           |
| u              | 问题尺度                       | 1               |
| h              | 问题尺度                       | 1000            |
| days           | 总共天数                       | 12              |
| download_data  | 模型所需数据集与(或)checkpoints     | pinns_swe       |
| force_download | 是否强制下载数据                   | false           |
| amp_level      | MindSpore自动混合精度等级          | O3              |
| device_id      | 需要设置的设备号                   | None            |
| mode           | MindSpore静态图模式（0）或动态图模式（1） | 0               |

### [训练流程](#目录)

- 在 GPU/Ascend 上运行

   ```bash
   python train.py
   ```

  训练期间的损失值将打印在控制台中， 也可以训练后在日志文件中查看。

   ```bash
   # grep "loss:" log
  PDE loss, IC loss in 0th epoch: 0.04082385, 0.13227281, interval 28.69731688, total: 28.69731688
  PDE loss, IC loss in 1th epoch: 0.02216472, 0.05938588, interval 3.24713469, total: 31.94445157
  PDE loss, IC loss in 2th epoch: 0.01156821, 0.02318317, interval 3.31807733, total: 35.26252890
  PDE loss, IC loss in 3th epoch: 0.00694417, 0.00913251, interval 3.22263527, total: 38.48516417
  PDE loss, IC loss in 4th epoch: 0.00577628, 0.00795174, interval 3.32371068, total: 41.80887485
  PDE loss, IC loss in 5th epoch: 0.00556142, 0.01145195, interval 3.30852318, total: 45.11739802
  PDE loss, IC loss in 6th epoch: 0.00492313, 0.01358479, interval 3.31264329, total: 48.43004131
  PDE loss, IC loss in 7th epoch: 0.00375959, 0.01274938, interval 3.32251096, total: 51.75255227
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