[ENGLISH](README.md) | 简体中文

# 目录

- [Parareal PINNs 描述](#parareal-pinns-描述)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [脚本说明](#脚本说明)
    - [脚本和示例代码](#脚本和示例代码)
    - [脚本参数](#脚本参数)
    - [训练流程](#训练流程)
    - [推理流程](#推理流程)

## [parareal PINNs 描述](#目录)

Parareal PINNs（PPINNs）将长时间域问题分解为多个短时间域的神经网络训练问题，并且伴随有一个粗粒度、快速的求解器作为监督。
粗粒度求解器可以快速计算长时间域内的粗略解，作为精细粒度的神经网络（PINNs）训练数据基础。
精细的神经网络将对粗略解进行更正，并将更正值反馈至粗粒度求解器，进行下一轮训练循环，以此得到长时间域内的预测值。
此模型相比于普通PINNs模型，在长时间域问题和训练速度上有着明显优势。

> [论文](https://www.sciencedirect.com/science/article/pii/S0045782520304357):
> Meng X, Li Z, Zhang D, et al. PPINN: Parareal physics-informed neural network for time-dependent PDEs[J]. Computer
> Methods in Applied Mechanics and Engineering, 2020, 370: 113250.

案例详情: 使用并行PINNs求解Burgers Equation。

## [数据集](#目录)

训练数据集在运行时随机生成。
数据集的大小由`config.yaml`中的参数`n_train`控制，默认值为10000。

预训练checkpoints文件将会在首次启动时自动下载。
您如果需要手动下载checkpoints文件，
请访问[此链接](https://download.mindspore.cn/mindscience/SciAI/sciai/model/ppinns/)。

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
mpiexec -n 8 python train.py
```

完整命令:

```bash
mpiexec -n 8 python train.py \
    --t_range 0 10 \
    --nt_coarse 1001 \
    --nt_fine 200001 \
    --n_train 10000 \
    --layers 1 20 20 1 \
    --save_ckpt true \
    --save_fig true \
    --load_ckpt false \
    --save_ckpt_path ./checkpoints \
    --save_output true \
    --save_data_path ./data \
    --load_ckpt_path ./checkpoints/fine_solver_4_float16/result_iter_1.ckpt \
    --figures_path ./figures \
    --log_path ./logs \
    --print_interval 10 \
    --epochs 50000 \
    --lbfgs false \
    --lbfgs_epochs 50000 \
    --amp_level O3 \
    --device_id 0 \
    --mode 0
```

## [脚本说明](#目录)

### [脚本和示例代码](#目录)

文件结构如下:

```text
├── ppinns
│   ├── checkpoints                  # checkpoint文件
│   ├── data                         # 数据文件
│   ├── figures                      # 结果图片
│   ├── logs                         # 日志文件
│   ├── src                          # 源代码
│   │   ├── coarsesolver.py          # coarse solver处理流程
│   │   ├── dataset.py               # 数据集创建
│   │   ├── finesolver.py            # fine solver处理流程
│   │   ├── model.py                 # 模型与损失函数定义
│   │   └── net.py                   # 神经网络结构
│   ├── config.yaml                  # 超参数配置
│   ├── README.md                    # 英文模型说明
│   ├── README_CN.md                 # 中文模型说明
│   ├── train.py                     # python训练脚本
│   └── eval.py                      # python评估脚本
```

### [脚本参数](#目录)

train.py中的重要参数如下:

| 参数             | 描述                         | 默认值                                                    |
|----------------|----------------------------|--------------------------------------------------------|
| t_range        | 时间范围                       | 0 10                                                   |
| nt_coarse      | Coarse Solver采样数量          | 1001                                                   |
| nt_fine        | Fine Solver采样数量            | 200001                                                 |
| n_train        | 训练数据集数量                    | 10000                                                  |
| layers         | 神经网络结构                     | 1 20 20 1                                              |
| save_ckpt      | 是否保存checkpoint             | true                                                   |
| save_fig       | 是否保存和绘制图片                  | true                                                   |
| load_ckpt      | 是否加载checkpoint             | false                                                  |
| save_ckpt_path | checkpoint保存路径             | ./checkpoints                                          |
| save_output    | 是否保存输出结果                   | true                                                   |
| save_data_path | 输出结果数据保存路径                 | ./data                                                 |
| load_ckpt_path | checkpoint加载路径             | ./checkpoints/fine_solver_4_float16/result_iter_1.ckpt |
| figures_path   | 图片保存路径                     | ./figures                                              |
| log_path       | 日志保存路径                     | ./logs                                                 |
| print_interval | 时间与loss打印间隔                | 10                                                     |
| epochs         | 时期（迭代次数）                   | 50000                                                  |
| lbfgs          | 是否使用L-BFGS优化器              | false                                                  |
| lbfgs_epochs   | L-BFGS优化迭代次数               | 50000                                                  |
| amp_level      | MindSpore自动混合精度等级          | O3                                                     |
| device_id      | 需要设置的设备号                   | None                                                   |
| mode           | MindSpore静态图模式（0）或动态图模式（1） | 0                                                      |

### [训练流程](#目录)

- 在 GPU/Ascend 上运行

  ```bash
  mpiexec -n 8 python train.py
  ```

  训练期间的损失值将打印在控制台中， 也可以训练后在日志文件中查看。

  ```bash
  Fine solver for chunk#:2
  Fine solver for chunk#:3
  Fine solver for chunk#:1
  Fine solver for chunk#:4
  Fine solver for chunk#:5
  Fine solver for chunk#:6
  Fine solver for chunk#:7
  step: 0, loss: 35.401, interval: 4.902739763259888s, total: 4.902739763259888s
  step: 0, loss: 3.0025306, interval: 4.05675196647644s, total: 4.05675196647644s
  step: 10, loss: 31.377882, interval: 0.057642224568323s, total: 4.960381987828211s
  step: 10, loss: 2.124565, interval: 0.05868268013559839s, total: 4.11543464661203839s
  step: 20, loss: 28.006842, interval: 0.0367842587620457s, total: 4.9971662465902567s
  step: 20, loss: 1.7020686, interval: 0.03674263405928059s, total: 4.15217728067131898s
  step: 30, loss: 25.339191, interval: 0.0367320495820349s, total: 5.0338982961722916s
  step: 30, loss: 1.5006942, interval: 0.0364089623498762s, total: 4.18858624302119518s
  step: 40, loss: 23.387045, interval: 0.0379562045760954s, total: 5.071854500748387s
  step: 40, loss: 1.3562441, interval: 0.03872304529386204s, total: 4.22730928831505722s
  step: 50, loss: 22.027771, interval: 0.03325230498620495s, total: 5.10510680573459195s
  step: 50, loss: 1.2255007, interval: 0.03410502934802956s, total: 4.26141431766308678s
  ...
  ```

- 训练结束后，您仍然可以通过保存在`log_path`下面的日志文件回顾训练过程，默认为`./logs`目录中。

- 模型checkpoint将保存在 `save_ckpt_path`中，默认为`./checkpoints` 目录中。

### [推理流程](#目录)

在运行下面的命令之前，请检查使用的`config.yaml` 中的checkpoint加载路径`load_ckpt_path`
进行推理。

- 在 GPU/Ascend 上运行

   ```bash
   mpiexec -n 8 python eval.py
   ```

  您可以通过日志文件`log_path`查看过程与结果，默认位于`./logs` 。
  结果图片存放于`figures_path`中，默认位于[`./figures`](./figures)。