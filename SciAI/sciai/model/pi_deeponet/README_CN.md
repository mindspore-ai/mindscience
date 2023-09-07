[ENGLISH](README.md) | 简体中文

# 目录

- [Physics-informed DeepONets 描述](#physics-informed-deeponets-描述)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [脚本说明](#脚本说明)
    - [脚本和示例代码](#脚本和示例代码)
    - [脚本参数](#脚本参数)
    - [训练流程](#训练流程)
    - [推理流程](#推理流程)

## [Physics-informed DeepONets 描述](#目录)

本项目学习无穷维运算符，改算子可以将随机初始条件映射到短时间间隔的PDE解。
可以通过以下方式获得跨一系列初始条件的全局长时间预测:
使用每个预测作为下一个评估步骤的初始条件，迭代评估训练后的模型。
这介绍了一种新的时间域分解方法，该方法已被证明对于执行各种参数化偏微分方程系统的长时间模拟非常有效，从波动传播到反应-扩散动力学和刚性化学动力学，
所有这些都只需要经典数值求解器所需计算成本的一部分。

> [论文](https://www.sciencedirect.com/science/article/abs/pii/S0021999122009184): Wang S, Perdikaris P. Long-time
> integration of parametric evolution equations with physics-informed deeponets[J]. Journal of Computational Physics,
> 2023, 475: 111855.

## [数据集](#目录)

训练数据集在运行时随机生成。
数据集的大小由`config.yaml`中的参数`batch_size`控制，默认值为10000。

用于验证的数据集和预训练checkpoints文件将会在首次启动时自动下载。
您如果需要手动下载验证数据集或checkpoints文件，
请访问[此链接](https://download.mindspore.cn/mindscience/SciAI/sciai/model/pi_deeponet/)。

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
    --branch_layers 100 100 100 100 100 100 \
    --trunk_layers 2 100 100 100 100 100 \
    --save_ckpt true \
    --save_data true \
    --save_fig true \
    --load_ckpt false \
    --save_ckpt_path ./checkpoints \
    --load_ckpt_path ./checkpoints/dr_float32_final.ckpt \
    --save_data_path ./data \
    --figures_path ./figures \
    --log_path ./logs \
    --print_interval 100 \
    --lr 8e-4 \
    --epochs 200001 \
    --n_train 10000 \
    --batch_size 10000 \
    --download_data pi_deeponet \
    --force_download false \
    --amp_level O3 \
    --device_id 0 \
    --mode 0
```

## [脚本说明](#目录)

### [脚本和示例代码](#目录)

文件结构如下:

```text
├── pi_deeponet
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
│   └── eval.py                      # python评估脚本
```

### [脚本参数](#目录)

train.py中的重要参数如下:

| parameter      | description                | default value                       |
|----------------|----------------------------|-------------------------------------|
| branch_layers  | 分支神经网络深度                   | 100 100 100 100 100 100             |
| trunk_layers   | 主干神经网络宽度                   | 2 100 100 100 100 100               |
| save_ckpt      | 是否保存checkpoint             | true                                |
| save_data      | 是否保存loss数据                 | true                                |
| save_fig       | 是否保存和绘制图片                  | true                                |
| load_ckpt      | 是否加载checkpoint             | false                               |
| save_ckpt_path | checkpoint保存路径             | ./checkpoints                       |
| load_ckpt_path | checkpoint加载路径             | ./checkpoints/dr_float32_final.ckpt |
| save_data_path | 保存数据的路径                    | ./data                              |
| figures_path   | 图片保存路径                     | ./figures                           |
| log_path       | 日志保存路径                     | ./logs                              |
| print_interval | 时间与loss打印间隔                | 100                                 |
| lr             | 学习率                        | 8e-4                                |
| epochs         | 时期（迭代次数）                   | 200001                              |
| n_train        | 数据集生成次数                    | 10000                               |
| batch_size     | 批尺寸                        | 10000                               |
| download_data  | 模型所需数据集与(或)checkpoints     | pi_deeponet                         |
| force_download | 是否强制下载数据                   | false                               |
| amp_level      | MindSpore自动混合精度等级          | O3                                  |
| device_id      | 需要设置的设备号                   | None                                |
| mode           | MindSpore静态图模式（0）或动态图模式（1） | 0                                   |

### [训练流程](#目录)

- 在 GPU/Ascend 上运行

   ```bash
   python train.py
   ```

  经过训练，loss值会输出如下:

   ```bash
   # grep "loss:" log
  step: 0, total loss: 0.0971143, ic_loss: 0.0706561, bc_loss: 0.0047654584, res_loss: 0.021692745, interval: 9.959319114685059s, total: 9.959319114685059s
  step: 100, total loss: 0.0612279, ic_loss: 0.056949332, bc_loss: 0.0030672695, res_loss: 0.0012112984, interval: 9.085834741592407s, total: 19.045153856277466s
  step: 200, total loss: 0.059222076, ic_loss: 0.05515627, bc_loss: 0.0030886163, res_loss: 0.0009771917, interval: 9.108893632888794s, total: 28.15404748916626s
  step: 300, total loss: 0.05733742, ic_loss: 0.052310925, bc_loss: 0.003073115, res_loss: 0.0019533802, interval: 9.576531648635864s, total: 37.730579137802124s
  step: 400, total loss: 0.055052415, ic_loss: 0.049479727, bc_loss: 0.0032956824, res_loss: 0.0022770043, interval: 10.003910541534424s, total: 47.73448967933655s
  step: 500, total loss: 0.051897146, ic_loss: 0.047461353, bc_loss: 0.0025362624, res_loss: 0.0018995304, interval: 9.252656698226929s, total: 56.98714637756348s
  step: 600, total loss: 0.047137313, ic_loss: 0.04395392, bc_loss: 0.0014622104, res_loss: 0.0017211806, interval: 9.413921594619751s, total: 66.40106797218323s
  step: 700, total loss: 0.050823156, ic_loss: 0.044430587, bc_loss: 0.0040090764, res_loss: 0.002383494, interval: 9.160758018493652s, total: 75.56182599067688s
  step: 800, total loss: 0.029433459, ic_loss: 0.026467426, bc_loss: 0.00096403103, res_loss: 0.0020020034, interval: 8.86798882484436s, total: 84.42981481552124s
  step: 900, total loss: 0.0065431646, ic_loss: 0.0051204017, bc_loss: 0.0005367383, res_loss: 0.00088602427, interval: 9.333975076675415s, total: 93.76378989219666s
  step: 1000, total loss: 0.004916694, ic_loss: 0.0040391637, bc_loss: 0.00033330295, res_loss: 0.00054422737, interval: 9.447664737701416s, total: 103.21145462989807s
  ...
   ```

  模型checkpoint将保存在 `save_ckpt_path`中, 默认为`./checkpoints` 目录中。

### [推理流程](#目录)

在运行下面的命令之前，请检查使用的`config.yaml` 中的checkpoint加载路径`load_ckpt_path`
进行推理。

- 在 GPU/Ascend 上运行

   ```bash
   python eval.py
   ```

  您可以通过日志文件`log_path`查看过程与结果，默认位于`./logs` 。
  结果图片存放于`figures_path`中，默认位于[`./figures`](./figures)。