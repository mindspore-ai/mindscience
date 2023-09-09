[ENGLISH](README.md) | 简体中文

# 目录

- [Deep Hpms 描述](#Deep-Hpms-描述)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [脚本说明](#脚本说明)
    - [脚本和示例代码](#脚本和示例代码)
    - [脚本参数](#脚本参数)
    - [训练流程](#训练流程)
    - [评估流程](#评估流程)

## [Deep Hpms 描述](#目录)

本工作提出了一种深度学习方法，用于从空间和时间的噪声中观测非线性偏微分方程。该方法使用两个深度神经网络来近似未知解和非线性动力学。
该方法的有效性在几个基准案例上进行了测试，跨越多个科学领域。

在当前目录中，使用Deep Hpms解决了两个问题: Burgers方程与kdv方程。

> [论文](https://www.jmlr.org/papers/volume19/18-046/18-046.pdf):
> Raissi M. Deep hidden physics models: Deep learning of nonlinear partial differential equations[J]. The Journal of
> Machine Learning Research, 2018, 19(1): 932-955.

## [数据集](#目录)

用于训练的数据集和预训练checkpoints文件将会在首次启动时自动下载。

-数据格式: `.mat`文件
-注意: 数据将在`process.py`中处理
-数据集位于`./data`目录下，目录结构如下:

```text
├── data
│   ├── matlab
│   ├── burgers.mat
│   ├── burgers_sine.mat
│   ├── cylinder.mat
│   ├── cylinder_vorticity.mat
│   ├── KdV_cos.mat
│   ├── KdV_sine.mat
│   ├── KS.mat
│   ├── KS_chaotic.mat
│   └── NLS.mat
```

您如果需要手动下载数据集或checkpoints文件，
请访问[此链接](https://download.mindspore.cn/mindscience/SciAI/sciai/model/deep_hpms/)。

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
    --problem burgers_different \
    --u_layers 2 50 50 50 50 1 \
    --pde_layers 3 100 100 1 \
    --layers 2 50 50 50 50 1 \
    --save_ckpt true \
    --load_ckpt false \
    --save_ckpt_path ./checkpoints \
    --load_ckpt_path ./checkpoints/burgers_diff_final.ckpt \
    --save_fig true \
    --figures_path ./figures \
    --load_data_idn_path ./data/burgers_sine.mat \
    --load_data_sol_path ./data/burgers.mat \
    --log_path ./logs \
    --lr 1e-3 \
    --train_epoch 30001 \
    --train_epoch_lbfgs 100 \
    --print_interval 100 \
    --lb_idn 0.0 -8.0 \
    --ub_idn 10.0 8.0 \
    --lb_sol 0.0 -8.0 \
    --ub_sol 10.0 8.0 \
    --download_data deep_hpms \
    --force_download false \
    --data_type float32 \
    --amp_level O3 \
    --device_id 0 \
    --mode 0
```

如果要运行其他案例，请切换`config.yaml`中的`problem`或在命令参数中指定`--problem`。此网络暂不支持PYNATIVE模式。

## [脚本说明](#目录)

### [脚本和示例代码](#目录)

文件结构如下:

```text
├── deep_hpms
│   ├── checkpoints                      # checkpoint文件
│   ├── data                             # 数据文件
│   │   ├── matlab                       # 用于数据生成的matlab代码
│   │   ├── burgers.mat                  # burgers案例数据集
│   │   └── ...                          # 其他数据集
│   ├── figures                          # 结果图片
│   ├── logs                             # 日志文件
│   ├── src                              # 源代码
│   │   ├── network_burgers_different.py # burgers different 网络架构
│   │   ├── network_common.py            # 公共网络架构
│   │   ├── network_kdv.py               # kdv same 网络架构
│   │   ├── plot.py                      # 绘制结果
│   │   └── process.py                   # 数据处理
│   ├── config.yaml                      # 超参数配置
│   ├── README.md                        # 英文模型说明
│   ├── README_CN.md                     # 中文模型说明
│   ├── train.py                         # python训练脚本
│   └── eval.py                          # python评估脚本
```

### [脚本参数](#目录)

总共两个案例. 在 `config.yaml` 或命令参数中, 可以通过参数 `problem` 来选择案例.

| parameter | description                            | default value     |
|-----------|----------------------------------------|-------------------|
| problem   | 用于解决的案例，`burgers_different`或`kdv_same` | burgers_different |

train.py中的重要参数如下:

| parameter          | description                                   | default value                         |
|--------------------|-----------------------------------------------|---------------------------------------|
| u_layers           | 神经网络U宽度                                       | 2 50 50 50 50 1                       |
| pde_layers         | 神经网络PDE宽度                                     | 3 100 100 1                           |
| layers             | 神经网络Solution宽度                                | 2 50 50 50 50 1                       |
| save_ckpt          | 是否保存checkpoint                                | true                                  |
| load_ckpt          | 是否加载checkpoint                                | false                                 |
| save_ckpt_path     | checkpoint保存路径                                | ./checkpoints                         |
| load_ckpt_path     | checkpoint加载路径                                | ./checkpoints/burgers_diff_final.ckpt |
| save_fig           | 是否保存和绘制图片                                     | true                                  |
| figures_path       | 图片保存路径                                        | ./figures                             |
| load_data_idn_path | 加载idn数据和保存数据的路径                               | ./data/burgers_sine.mat               |
| load_data_sol_path | 加载sol数据和保存数据的路径                               | ./data/burgers.mat                    |
| log_path           | 日志保存路径                                        | ./logs                                |
| lr                 | 学习率                                           | 1e-3                                  |
| train_epoch        | adam 时期（迭代次数）                                 | 30001                                 |
| train_epoch_lbfgs  | l-bfgs 时期（迭代次数）                               | 100                                   |
| print_interval     | 时间与loss打印间隔                                   | 100                                   |
| lb_idn             | idn 下界                                        | 0.0, -8.0                             |
| ub_idn             | idn 上界                                        | 10.0, 8.0                             |
| lb_sol             | sol 下界                                        | 0.0, -8.0                             |
| ub_sol             | sol 上界                                        | 10.0, 8.0                             |
| download_data      | 模型所需数据集与(或)checkpoints                        | deep_hpms                             |
| force_download     | 是否强制下载数据                                      | false                                 |
| amp_level          | MindSpore自动混合精度等级                             | O3                                    |
| device_id          | 需要设置的设备号                                      | None                                  |
| mode               | MindSpore静态图模式（0）或动态图模式（1）。此网络暂不支持PYNATIVE模式。 | 0                                     |

### [训练流程](#目录)

- 在 GPU/Ascend 上运行

   ```bash
   python train.py
   ```

  经过训练，loss值会输出如下，也可训练完成后在log文件中查看。

   ```bash
   # grep "loss1:" log
  step: 0, loss: 3308.352, interval: 3.1490728855133057s, total: 3.1490728855133057s
  step: 100, loss: 1074.0432, interval: 0.4218735694885254s, total: 3.570946455001831s
  step: 200, loss: 181.29312, interval: 0.36736583709716797s, total: 3.938312292098999s
  step: 300, loss: 87.94882, interval: 0.36727356910705566s, total: 4.305585861206055s
  step: 400, loss: 33.567818, interval: 0.365675687789917s, total: 4.671261548995972s
  step: 500, loss: 15.378567, interval: 0.36209774017333984s, total: 5.0333592891693115s
  step: 600, loss: 14.30908, interval: 0.3638172149658203s, total: 5.397176504135132s
  step: 700, loss: 10.322, interval: 0.3609278202056885s, total: 5.75810432434082s
  step: 800, loss: 13.931234, interval: 0.36093950271606445s, total: 6.119043827056885s
  step: 900, loss: 5.209699, interval: 0.3612406253814697s, total: 6.4802844524383545s
  step: 1000, loss: 4.2461824, interval: 0.3610835075378418s, total: 6.841367959976196s
  ...
   ```

- 训练结束后，您仍然可以通过保存在`log_path`下面的日志文件回顾训练过程，默认为`./logs`目录中。

- 模型checkpoint将保存在 `save_ckpt_path`中，默认为`./checkpoints` 目录中。

### [评估流程](#目录)

在运行下面的命令之前，请检查使用的`config.yaml` 中的checkpoint加载路径`load_ckpt_path`
进行推理。

- 在 GPU/Ascend 上运行

   ```bash
   python eval.py
   ```

  您可以通过日志文件`log_path`查看过程与结果，默认位于`./logs` 。
  结果图片存放于`figures_path`中，默认位于[`./figures`](./figures)。