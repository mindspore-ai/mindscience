[ENGLISH](README.md) | 简体中文

# 目录

- [ENSO 描述](#enso-描述)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [脚本说明](#脚本说明)
    - [脚本和示例代码](#脚本和示例代码)
    - [脚本参数](#脚本参数)
    - [训练流程](#训练流程)
    - [推理流程](#推理流程)

## [ENSO 描述](#目录)

厄尔尼诺与南方涛动（ENSO）现象对区域生态系统影响较大，因此，准确的ENSO的预测带来了巨大的区域效益。
然而，对ENSO超过一年的预测仍然存在问题。最近，卷积神经网络（CNN）已被证明是预测ENSO的有效工具。

在这个模型中，我们实现了CNN的训练和评估过程，用于用气象数据预测ENSO。

> [论文](https://doi.org/10.1038/s41586-019-1559-7): Ham, Y.-G., J.-H. Kim, and J.-J. Luo, 2019:
> Deep learning for multi-year ENSO forecasts. Nature, 573, 568–572.

## [数据集](#目录)

用于训练的数据集和预训练checkpoints文件将会在首次启动时自动下载。

- 数据格式: `.npy`文件
    - 注: 数据会在`process.py`中处理
- 数据集在`./data`目录下，目录结构如下:

```text
├── data
│   ├── htmp_data
│   ├── train_data
│   │   ├── ACCESS-CM2
│   │   ├── CCSM4
│   │   ├── CESM1-CAM5
│   │   ├── ...
│   │   └── obs
│   └── var_data
```

您如果需要手动下载数据集或checkpoints文件，
请访问[此链接](https://download.mindspore.cn/mindscience/SciAI/sciai/model/enso/)。

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
    --save_ckpt_path ./checkpoints \
    --load_ckpt_path ./checkpoints/exp2_aftertrain/enso_float16.ckpt \
    --save_data true\
    --load_data_path ./data \
    --save_data_path ./data \
    --save_figure true \
    --figures_path ./figures \
    --log_path ./logs \
    --print_interval 10 \
    --lr 0.01 \
    --epochs 20 \
    --batch_size 400 \
    --skip_aftertrain false \
    --epochs_after 5 \
    --batch_size_after 30 \
    --lr_after 1e-6 \
    --download_data enso \
    --force_download false \
    --amp_level O3 \
    --device_id 0 \
    --mode 0
```

## [脚本说明](#目录)

## [脚本和示例代码](#目录)

```text
├── enso
│   ├── checkpoints                       # checkpoint文件
│   ├── data                              # 数据文件
│   │   ├── htmp_data                     # 验证结果的保存路径
│   │   ├── var_data                      # 验证数据集
│   │   └── train_data                    # 训练数据集
│   ├── figures                           # 结果图片
│   ├── logs                              # 日志文件
│   ├── src                               # 源代码
│   │   ├── network.py                    # 网络架构
│   │   ├── plot.py                       # 绘制结果
│   │   └── process.py                    # 数据处理
│   ├── config.yaml                       # 超参数配置
│   ├── README.md                         # 英文模型说明
│   ├── README_CN.md                      # 中文模型说明
│   ├── train.py                          # python训练脚本
│   └── eval.py                           # python评估脚本
```

### [脚本参数](#目录)

train.py中的重要参数如下:

| 参数名              | 描述                         | 默认值                                             |
|------------------|----------------------------|-------------------------------------------------|
| save_ckpt        | 是否保存checkpoint             | true                                            |
| load_ckpt        | 是否加载checkpoint             | false                                           |
| save_ckpt_path   | checkpoint保存路径             | ./checkpoints                                   |
| load_ckpt_path   | checkpoint加载路径             | ./checkpoints/exp2_aftertrain/enso_float16.ckpt |
| save_data        | 是否保存数据                     | true                                            |
| load_data_path   | 加载数据的路径                    | ./data                                          |
| save_data_path   | 保存数据的路径                    | ./data                                          |
| save_figure      | 是否保存和绘制图片                  | true                                            |
| figures_path     | 图片保存路径                     | ./figures                                       |
| log_path         | 日志保存路径                     | ./logs                                          |
| print_interval   | 时间与loss打印间隔                | 10                                              |
| lr               | 学习率                        | 0.01                                            |
| epochs           | 时期（迭代次数）                   | 20                                              |
| batch_size       | 数据集的大小                     | 400                                             |
| skip_aftertrain  | 是否跳过训练后的流程                 | false                                           |
| epochs_after     | 训练后流程的时期（迭代次数）             | 5                                               |
| batch_size_after | 训练后流程的数据集大小                | 30                                              |
| lr_after         | 训练后流程的学习率                  | 1e-6                                            |
| download_data    | 模型所需数据集与(或)checkpoints     | enso                                            |
| force_download   | 是否强制下载数据                   | false                                           |
| amp_level        | MindSpore自动混合精度等级          | O3                                              |
| device_id        | 需要设置的设备号                   | None                                            |
| mode             | MindSpore静态图模式（0）或动态图模式（1） | 0                                               |

### [训练流程](#目录)

  ```bash
  # python train.py
  ...
  epoch: 1 step: 1, loss is 0.9130635857582092
  epoch: 1 step: 2, loss is 1.0354164838790894
  epoch: 1 step: 3, loss is 0.8914494514465332
  epoch: 1 step: 4, loss is 0.9377754330635071
  epoch: 1 step: 5, loss is 1.0472232103347778
  epoch: 1 step: 6, loss is 1.0421113967895508
  epoch: 1 step: 7, loss is 1.100639820098877
  epoch: 1 step: 8, loss is 0.9849204421043396
  ...
  ```

- 训练结束后，您仍然可以通过保存在`log_path`下面的日志文件回顾训练过程，默认为`./logs`目录中。

- 模型checkpoint将保存在 `save_ckpt_path`中，默认为`./checkpoints` 目录中。

### [推理流程](#目录)

在运行下面的命令之前，请检查使用的`config.yaml` 中的checkpoint加载路径`load_ckpt_path`
进行推理。

```bash
python eval.py
```

您可以通过日志文件`log_path`查看过程与结果，默认位于`./logs` 。
结果图片存放于`figures_path`中，默认位于[`./figures`](./figures)。
