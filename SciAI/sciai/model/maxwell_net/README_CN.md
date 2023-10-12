[ENGLISH](README.md) | 简体中文

# 目录

- [Maxwell Net 描述](#Maxwell-net-描述)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [脚本说明](#脚本说明)
    - [脚本和示例代码](#脚本和示例代码)
    - [脚本参数](#脚本参数)
    - [训练流程](#训练流程)
    - [推理流程](#推理流程)

## [Maxwell Net 描述](#目录)

本工作利用物理驱动的损失求解麦克斯韦方程，使用Maxwell方程的残差作为损失函数来训练MaxwellNet，因而训练无需Ground Truth。
此外，使用了新颖的反向方案设计MaxwellNet，可以参考主要文章了解详细信息。
<br />

![Scheme](/figures/scheme.png)

 <br />

> [论文](https://arxiv.org/abs/2107.06164):Lim J, Psaltis D. MaxwellNet: Physics-driven deep neural network training
> based on Maxwell’s equations[J]. Apl Photonics, 2022, 7(1).

## [数据集](#目录)

用于训练的数据集和预训练checkpoints文件将会在首次启动时自动下载。

使用的数据集:

- 数据集大小
    - scat_pot: (1, 1, 160, 192)
    - ri: (1,)
- 数据格式: `.npz`文件
    - 注: 数据会在`process.py`中处理
- 数据集在`./data`目录下，目录结构如下:

```text
├── data
│   ├── spheric_te
│   │   ├── sample.npz
│   │   └── train.npz
│   ├── spheric_tm
│   │   ├── sample.npz
│   │   └── train.npz
```

## [环境要求](#目录)

- 硬件（Ascend/GPU）
    - 使用 Ascend 或 GPU 处理器准备硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install)
- 欲了解更多信息，请查看以下资源:
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

## [快速开始](#目录)

> 注：`tm` 案例不支持动态图模式。

通过官网安装好MindSpore后，就可以开始训练和验证如下:

- 在 Ascend 或 GPU 上运行

默认:

```bash
python train.py
```

完整命令:

```bash
python train.py \
    --problem te \
    --in_channels 1 \
    --out_channels 2 \
    --depth 6 \
    --filter 16 \
    --norm weight \
    --up_mode upconv \
    --wavelength 1 \
    --dpl 20 \
    --nx 160 \
    --nz 192 \
    --pml_thickness 30 \
    --symmetry_x true \
    --high_order 4 \
    --lr 0.0005 \
    --lr_decay 0.5 \
    --lr_decay_step 50000 \
    --epochs 250001 \
    --ckpt_interval 50000 \
    --save_ckpt true \
    --load_ckpt false \
    --save_ckpt_path ./checkpoints \
    --load_ckpt_path ./checkpoints/te_latest.ckpt \
    --load_data_path ./data/spheric_te \
    --save_fig true \
    --figures_path ./figures \
    --log_path ./logs \
    --download_data maxwell_net \
    --force_download false \
    --amp_level O0 \
    --device_id 0 \
    --mode 0
```

## [脚本说明](#目录)

## [脚本和示例代码](#目录)

文件结构如下:

```text
├── maxwell_net
│   ├── checkpoints         # checkpoints文件
│   ├── data                # 数据文件
│   │   ├── spheric_te      # spheric te 案例数据集
│   │   └── spheric_tm      # spheric tm 案例数据集
│   ├── figures             # 结果图片
│   ├── logs                # 日志文件
│   ├── src                 # 源代码目录
│   │   ├── network.py      # 网络架构
│   │   ├── plot.py         # 绘制结果
│   │   └── process.py      # 数据处理
│   ├── config.yaml         # 超参数配置
│   ├── README.md           # 英文模型说明
│   ├── README_CN.md        # 中文模型说明
│   ├── train.py            # python训练脚本
│   └── eval.py             # python评估脚本
```

### [脚本参数](#目录)

总共两个案例. 在 `config.yaml` 或命令参数中, 可以通过参数 `problem` 来选择案例.

| parameter | description                 | default value |
|-----------|-----------------------------|---------------|
| problem   | 用于解决的案例，`te`（横向电）或`tm`（横向磁） | te            |

对于每个问题案例，参数如下:

| 参数名            | 描述                                                                            | 默认值                          |
|----------------|-------------------------------------------------------------------------------|------------------------------|
| in_channels    | UNet输入通道维度                                                                    | 1                            |
| out_channels   | UNet输出通道维度                                                                    | 2                            |
| depth          | UNet降采样或上采样的深度                                                                | 6                            |
| filter         | UNet第一层的通道数量                                                                  | 16                           |
| norm           | UNet的归一化类型. 权重归一化：'weight'；批归一化：'batch'；无归一化：'no'                             | weight                       |
| up_mode        | UNet的上采样模式. 转置卷积：'upcov'；上采样：'upsample'。                                      | upconv                       |
| wavelength     | 波长                                                                            | 1                            |
| dpl            | 单像素点尺寸为 wavelength / dpl                                                      | 20                           |
| nx             | 沿x轴的像素数，等效于沿x轴的散射样本的像素数                                                       | 160                          |
| nz             | 沿z轴的像素数(光传播方向)，等效于沿z轴的散射样本的像素数                                                | 192                          |
| pml_thickness  | 以像素数表示的完全匹配层（PML）厚度。'pml_thickness * wavelength/ dpl' 是PML层的实际厚度，单位为微米        | 30                           |
| symmetry_x     | 输入散射样本是否沿x轴对称。本项为True时, 若Nx=100, Nz=200且沿x轴对称，则在train.npz中仅一半(Nx=50,Nz=200)即可 | true                         |
| high_order     | 2或4。它决定计算梯度的阶数（2阶或4阶）。4比2更准确                                                  | 4                            |
| lr             | 学习率                                                                           | 0.0005                       |
| lr_decay       | 学习率衰减率                                                                        | 0.5                          |
| lr_decay_step  | 学习率衰减步长                                                                       | 50000                        |
| epochs         | 时期数（迭代次数）                                                                     | 250001                       |
| print_interval | 时间与损失的打印间隙                                                                    | 100                          |
| ckpt_interval  | checkpoint保存间隙                                                                | 50000                        |
| save_ckpt      | 是否保存checkpoint                                                                | true                         |
| load_ckpt      | 是否加载checkpoint                                                                | false                        |
| save_ckpt_path | checkpoint保存路径                                                                | ./checkpoints                |
| load_ckpt_path | checkpoint加载路径                                                                | ./checkpoints/te_latest.ckpt |
| load_data_path | 数据加载路径                                                                        | ./data/spheric_te            |
| save_fig       | 是否保存和绘制图片                                                                     | true                         |
| figures_path   | 图片保存路径                                                                        | ./figures                    |
| log_path       | 日志保存路径                                                                        | ./logs                       |
| download_data  | 必要的数据集与checkpoint                                                             | maxwell_net                  |
| force_download | 是否强制下载数据集与checkpoint                                                          | false                        |
| amp_level      | MindSpore 自动混合精度等级                                                            | O0                           |
| device_id      | 需要设置的设备号                                                                      | None                         |
| mode           | MindSpore静态图模式（0）或动态图模式（1）                                                    | 0                            |

### [训练流程](#目录)

- 在 GPU/Ascend 上运行

   ```bash
   python train.py
   ```

  经过训练，loss值会输出如下:

  ```bash
  # grep "loss:" log
  step: 0, loss: 446.1874, interval: 89.43078088760376s, total: 89.43078088760376s, checkpoint saved at: ./checkpoints/model_iter_0_2000-12-31-23-59-59te.ckpt
  'latest' checkpoint saved at 0 epoch.
  step: 10, loss: 149.06134, interval: 1.5497097969055176s, total: 90.98049068450928s
  step: 20, loss: 83.69271, interval: 1.2006518840789795s, total: 92.18114256858826s
  step: 30, loss: 43.22249, interval: 1.1962628364562988s, total: 93.37740540504456s
  step: 40, loss: 33.38814, interval: 1.1976008415222168s, total: 94.57500624656677s
  step: 50, loss: 26.913471, interval: 1.1968715190887451s, total: 95.77187776565552s
  step: 60, loss: 20.579971, interval: 1.1951792240142822s, total: 96.9670569896698s
  step: 70, loss: 17.35663, interval: 1.197744369506836s, total: 98.16480135917664s
  step: 80, loss: 15.115046, interval: 1.2009501457214355s, total: 99.36575150489807s
  step: 90, loss: 12.830681, interval: 1.206284999847412s, total: 100.57203650474548s
  step: 100, loss: 11.197462, interval: 1.2086222171783447s, total: 101.78065872192383s
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
