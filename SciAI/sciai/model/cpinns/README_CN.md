[ENGLISH](README.md) | 简体中文

# 目录

- [Conservative PINNs 描述](#conservative-pinns-描述)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [脚本说明](#脚本说明)
    - [脚本和示例代码](#脚本和示例代码)
    - [脚本参数](#脚本参数)
    - [训练流程](#训练流程)
    - [推理流程](#推理流程)

## [Conservative PINNs 描述](#目录)

为离散域上的非线性守恒定律提出conservative PINN（cPINN），
此方法要求满足通量连续性，提供优化自由度，并调整激活函数以加快训练速度。
其有效地实现了并行计算，解决了各种演示案例并适应了复杂的结构。

> [论文](https://www.sciencedirect.com/science/article/abs/pii/S0045782520302127): Jagtap A D, Kharazmi E, Karniadakis G
> E. Conservative physics-informed neural networks on discrete domains for conservation laws: Applications to forward
> and inverse problems[J]. Computer Methods in Applied Mechanics and Engineering, 2020, 365: 113028.

案例详情: 具有4个空间子域的一维Burgers方程。

## [数据集](#目录)

用于训练的数据集和预训练checkpoints文件将会在首次启动时自动下载。

使用的数据集: [burgers shock]

- 数据集大小
    - x: [-1, 1] 中的（256, 1）
    - t: [0, 1] 中的（100, 1）
- 数据格式: `.mat`文件
    - 注: 数据会在`process.py`中处理
- 数据集在`./data`目录下，目录结构如下:

```text
├── data
│   ├── burgers_shock.mat
│   └── L2error_Bur4SD_200Wi.mat
```

您如果需要手动下载数据集或checkpoints文件，
请访问[此链接](https://download.mindspore.cn/mindscience/SciAI/sciai/model/cpinns/)。

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
    --nn_depth 4 6 6 4 \
    --nn_width 20 20 20 20 \
    --save_ckpt true \
    --save_fig true \
    --load_ckpt false \
    --save_ckpt_path ./checkpoints \
    --load_data_path ./data \
    --save_data_path ./data \
    --load_ckpt_path ./checkpoints/model_final.ckpt \
    --figures_path ./figures \
    --log_path ./logs \
    --print_interval 10 \
    --ckpt_interval 1000 \
    --lr 8e-4 \
    --epochs 15001 \
    --download_data cpinns \
    --force_download false \
    --amp_level O3 \
    --device_id 0 \
    --mode 0
```

## [脚本说明](#目录)

### [脚本和示例代码](#目录)

文件结构如下:

```text
├── cpinns
│   ├── checkpoints                  # checkpoint文件
│   ├── data                         # 数据文件
│   │   ├── burgers_shock.mat        # burgers shock matlab 数据集
│   │   └── L2error_Bur4SD_200Wi.mat # l2误差案例的结果数据集
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

| parameter      | description                | default value                  |
|----------------|----------------------------|--------------------------------|
| nn_depth       | 神经网络深度                     | 4 6 6 4                        |
| nn_width       | 神经网络宽度                     | 20 20 20 20                    |
| save_ckpt      | 是否保存checkpoint             | true                           |
| save_fig       | 是否保存和绘制图片                  | true                           |
| load_ckpt      | 是否加载checkpoint             | false                          |
| save_ckpt_path | checkpoint保存路径             | ./checkpoints                  |
| load_data_path | 加载数据的路径                    | ./data                         |
| save_data_path | 保存数据的路径                    | ./data                         |
| load_ckpt_path | checkpoint加载路径             | ./checkpoints/model_final.ckpt |
| figures_path   | 图片保存路径                     | ./figures                      |
| log_path       | 日志保存路径                     | ./logs                         |
| print_interval | 时间与loss打印间隔                | 10                             |
| ckpt_interval  | checkpoint保存间隔             | 1000                           |
| lr             | 学习率                        | 8e-4                           |
| epochs         | 时期（迭代次数）                   | 15001                          |
| download_data  | 模型所需数据集与(或)checkpoints     | cpinns                         |
| force_download | 是否强制下载数据                   | false                          |
| amp_level      | MindSpore自动混合精度等级          | O3                             |
| device_id      | 需要设置的设备号                   | None                           |
| mode           | MindSpore静态图模式（0）或动态图模式（1） | 0                              |

### [训练流程](#目录)

- 在 GPU/Ascend 上运行

   ```bash
   python train.py
   ```

  训练期间的损失值将打印在控制台中， 也可以训练后在日志文件中查看。

   ```bash
   # grep "loss1:" log
  step: 0, loss1: 2.1404986, loss2: 8.205103, loss3: 37.23588, loss4: 3.56359, interval: 50.85803508758545s, total: 50.85803508758545s
  step: 10, loss1: 2.6560388, loss2: 3.869413, loss3: 9.323585, loss4: 2.1194165, interval: 5.159524917602539s, total: 56.01756000518799s
  step: 20, loss1: 1.7885156, loss2: 4.470225, loss3: 3.3072894, loss4: 1.5674783, interval: 1.8615927696228027s, total: 57.87915277481079s
  step: 30, loss1: 1.8574346, loss2: 3.8972874, loss3: 2.103153, loss4: 1.2108151, interval: 1.7992112636566162s, total: 59.67836403846741s
  step: 40, loss1: 1.8863815, loss2: 2.7914107, loss3: 1.4245809, loss4: 0.94769603, interval: 1.8828914165496826s, total: 61.56125545501709s
  step: 50, loss1: 1.1929171, loss2: 1.5765706, loss3: 0.758412, loss4: 0.6086196, interval: 1.6731781959533691s, total: 63.23443365097046s
  step: 60, loss1: 0.7861989, loss2: 1.5977213, loss3: 0.56675017, loss4: 0.39846048, interval: 1.6708331108093262s, total: 64.90526676177979s
  step: 70, loss1: 0.33681053, loss2: 1.0673326, loss3: 0.5887743, loss4: 0.3366256, interval: 1.8425297737121582s, total: 66.74779653549194s
  step: 80, loss1: 0.29425326, loss2: 0.9776688, loss3: 0.5781496, loss4: 0.28926677, interval: 1.829559564590454s, total: 68.5773561000824s
  step: 90, loss1: 0.16654292, loss2: 0.9878452, loss3: 0.5724378, loss4: 0.2396864, interval: 1.784325122833252s, total: 70.36168122291565s
  step: 100, loss1: 0.11878409, loss2: 0.9726932, loss3: 0.5552572, loss4: 0.21440478, interval: 1.912705659866333s, total: 72.27438688278198s
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