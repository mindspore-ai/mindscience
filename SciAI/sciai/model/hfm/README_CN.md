[ENGLISH](README.md) | 简体中文

# 目录

- [Hidden Fluid Mechanics 描述](#hidden-fluid-mechanics-描述)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [脚本说明](#脚本说明)
    - [脚本和示例代码](#脚本和示例代码)
    - [脚本参数](#脚本参数)
    - [训练流程](#训练流程)
    - [推理流程](#推理流程)

## [Hidden Fluid Mechanics 描述](#目录)

隐流体力学（HFM）是一个物理信息深度学习框架，通过将纳维-斯托克斯方程（Navier-Stokes equations）编码到神经网络中，能够提取隐藏的流体运动量，
例如速度和压力场。以下论文展示了详细的研究。

> [论文](https://www.science.org/doi/abs/10.1126/science.aaw4741):
> Raissi M, Yazdani A, Karniadakis G E. Hidden fluid mechanics:
> Learning velocity and pressure fields from flow visualizations[J]. Science, 2020, 367(6481): 1026-1030.

该模型重新构建了神经网络的训练流程，并预测了给定时间和位置的速度场和压力场。
模型使用的场景是：在雷诺数`Re=100`和佩克莱数`Pe=100`时，二维流体穿过的圆形圆柱体。

## [数据集](#目录)

用于训练的数据集和预训练checkpoints文件将会在首次启动时自动下载。

使用的数据集: [Cylinder2D_flower]

- 数据集大小
    - 时间:（201, 1）
    - 位置 x:（1500, 201）
    - 位置 y:（1500, 201）
- 数据格式: `.mat`文件
    - 注: 数据会在`process.py`中处理
- 数据集在`./data`目录下，目录结构如下:

```text
├── data
│   └── Cylinder2D_flower.mat
```

您如果需要手动下载数据集或checkpoints文件，
请访问[此链接](https://download.mindspore.cn/mindscience/SciAI/sciai/model/hfm/)。

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
    --layers 3 200 200 200 200 200 200 200 200 200 200 4 \
    --save_ckpt true \
    --load_ckpt false \
    --save_result true \
    --save_ckpt_path ./checkpoints \
    --load_ckpt_path ./checkpoints/model_iter_300000_float32.ckpt \
    --load_data_path ./data \
    --save_data_path ./data \
    --log_path ./logs \
    --print_interval 10 \
    --lr 1e-3 \
    --t 1500 \
    --n 1500 \
    --total_time 40 \
    --epochs 100001 \
    --batch_size 1000 \
    --download_data hfm \
    --force_download false \
    --amp_level O0 \
    --device_id 0 \
    --mode 0
```

## [脚本说明](#目录)

## [脚本和示例代码](#目录)

文件结构如下:

```text
├── hfm
│   ├── checkpoints                       # checkpoint文件
│   ├── data                              # 数据文件
│   ├── figures                           # 结果图片
│   ├── logs                              # 日志文件
│   ├── src                               # 源代码
│   │   ├── network.py                    # 网络架构
│   │   └── process.py                    # 数据处理
│   ├── case_studies_1.sh                 # 供多案例同时执行的脚本
│   ├── case_studies_2.sh                 # 供多案例同时执行的脚本
│   ├── config.yaml                       # 超参数配置
│   ├── README.md                         # 英文模型说明
│   ├── README_CN.md                      # 中文模型说明
│   ├── train.py                          # python训练脚本
│   └── eval.py                           # python评估脚本
```

### [脚本参数](#目录)

train.py中的重要参数如下:

| 参数名            | 描述                         | 默认值                                          |
|----------------|----------------------------|----------------------------------------------|
| layers         | 神经网络层定义                    | 3 200 200 200 200 200 200 200 200 200 200 4  |
| save_ckpt      | 是否保存checkpoint             | true                                         |
| load_ckpt      | 是否加载checkpoint             | false                                        |
| save_result    | 是否保存训练结果                   | true                                         |
| save_ckpt_path | checkpoint保存路径             | ./checkpoints                                |
| load_ckpt_path | checkpoint加载路径             | ./checkpoints/model_iter_300000_float32.ckpt |
| load_data_path | 训练数据路径                     | ./data                                       |
| save_data_path | 生成数据路径                     | ./data                                       |
| log_path       | 日志保存路径                     | ./logs                                       |
| print_interval | 时间与loss打印间隔                | 10                                           |
| lr             | 学习率                        | 1e-3                                         |
| t              | 时间样本的大小                    | 1500                                         |
| n              | 位置样本的大小                    | 1500                                         |
| total_time     | 最大训练时间, 单位: 小时             | 40                                           |
| epochs         | 最大训练时期（迭代次数）               | 100001                                       |
| batch_size     | 每个时期（迭代次数）的数据集大小           | 1000                                         |
| download_data  | 模型所需数据集与(或)checkpoints     | hfm                                          |
| force_download | 是否强制下载数据                   | false                                        |
| data_type      | MindSpore自动混合精度等级          | O0                                           |
| device_id      | 需要设置的设备号                   | None                                         |
| mode           | MindSpore静态图模式（0）或动态图模式（1） | 0                                            |

### [训练流程](#目录)

- 在 GPU/Ascend 上运行

   ```bash
   python train.py
   ```

  经过训练，loss值会输出如下:

  ```bash
  step: 0, loss: 0.8028861, interval: 210.2149157524109s, total: 210.2149157524109s
  step: 10, loss: 0.66187924, interval: 13.888249158859253s, total: 224.10316491127014s
  step: 20, loss: 0.45909613, interval: 13.550164461135864s, total: 237.653329372406s
  step: 30, loss: 0.21840161, interval: 13.551252603530884s, total: 251.2045819759369s
  step: 40, loss: 0.043125667, interval: 13.55091643333435s, total: 264.75549840927124s
  step: 50, loss: 0.04197544, interval: 13.552476167678833s, total: 278.3079745769501s
  step: 60, loss: 0.017915843, interval: 13.578445672988892s, total: 291.88642024993896s
  ...
  ```

- 训练结束后，您仍然可以通过保存在`log_path`下面的日志文件回顾训练过程，默认为`./logs`目录中.

- 模型checkpoint将保存在`save_ckpt_path`中, 默认为`./checkpoints`目录中。

### [推理流程](#目录)

在运行下面的命令之前，请检查使用的`config.yaml` 中的checkpoint加载路径`load_ckpt_path`
进行推理。checkpoint文件可以使用[数据集章节](#数据集)中的命令进行下载。

- 在 GPU/Ascend 上运行

  ```bash
  python eval.py
  ```

  训练过程将会输出如下的（c ,u, v, p）误差：

  ```bash
  # grep "Error" log
  ...
  Error c: 2.638576e-02, Error u: 4.955575e-02, Error v: 3.927004e-02, Error p: 1.061887e-01
  Error c: 2.622087e-02, Error u: 4.833636e-02, Error v: 3.940436e-02, Error p: 1.045989e-01
  Error c: 2.596794e-02, Error u: 4.727550e-02, Error v: 3.953079e-02, Error p: 1.030758e-01
  Error c: 2.543095e-02, Error u: 4.638828e-02, Error v: 3.969464e-02, Error p: 1.016249e-01
  Error c: 2.459827e-02, Error u: 4.566803e-02, Error v: 3.989391e-02, Error p: 1.002600e-01
  Error c: 2.360513e-02, Error u: 4.509190e-02, Error v: 4.006676e-02, Error p: 9.900088e-02
  Error c: 2.243761e-02, Error u: 4.463641e-02, Error v: 4.011014e-02, Error p: 9.788122e-02
  ...
  ```

  您可以通过日志文件`log_path`查看过程与结果，默认位于`./logs`。
