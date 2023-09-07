[ENGLISH](README.md) | 简体中文

# 目录

- [hp-VPINNs 描述](#hp-vpinns-描述)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [脚本说明](#脚本说明)
    - [脚本和示例代码](#脚本和示例代码)
    - [脚本参数](#脚本参数)
    - [训练流程](#训练流程)
    - [推理流程](#推理流程)

## [hp-VPINNs 描述](#目录)

该研究提出了一个用于hp变分的PINNs网络（hp-VPINNs）框架，该框架使用全局神经网络试验空间和分段多项式测试空间，并通过域分解和投影到高阶多项式上进行hp细化。
这种方法提高了精度，降低了函数近似和求解微分方程的训练成本。

> [论文](https://arxiv.org/abs/2003.05385): Kharazmi, Ehsan, Zhongqiang Zhang, and George E. Karniadakis. hp-VPINNs:
> Variational Physics-Informed Neural Networks With Domain Decomposition, arXiv preprint arXiv:2003.05385 (2020).

## [数据集](#目录)

训练数据集在运行时生成。
数据集的大小由`config.yaml`中的quadrature积分点个数`n_quad`与每个因子的采样数`n_f`控制，默认值分别为80和500。

预训练checkpoints文件将会在首次启动时自动下载。
您如果需要手动下载checkpoints文件，
请访问[此链接](https://download.mindspore.cn/mindscience/SciAI/sciai/model/hp_vpinns/)。

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
    --layers 1 20 20 20 20 1 \
    --save_ckpt true \
    --load_ckpt false \
    --save_fig true \
    --save_ckpt_path ./checkpoints \
    --load_ckpt_path ./checkpoints/model_40001.ckpt \
    --figures_path ./figures \
    --log_path ./logs \
    --lr 1e-3 \
    --epochs 40001 \
    --early_stop_loss 2e-32 \
    --var_form 1 \
    --n_element 4 \
    --n_testfcn 60 \
    --n_quad 80 \
    --n_f 500 \
    --lossb_weight 1 \
    --font 24 \
    --download_data hp_vpinns \
    --force_download false \
    --amp_level O3 \
    --device_id 0 \
    --mode 1
```

## [脚本说明](#目录)

### [脚本和示例代码](#目录)

文件结构如下:

```text
├── hp_vpinns
│   ├── checkpoints                  # checkpoints文件
│   ├── data                         # 数据文件
│   ├── figures                      # 结果图片
│   ├── logs                         # 日志文件
│   ├── src                          # 源代码目录
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

| 参数名             | 含义                                      | 默认值                            |
|-----------------|-----------------------------------------|--------------------------------|
| layers          | 网络的层宽                                   | 1 20 20 20 20 1                |
| save_ckpt       | 是否保存checkpoint                          | true                           |
| save_fig        | 是否保存和绘制图片                               | true                           |
| load_ckpt       | 是否加载checkpoint                          | false                          |
| save_ckpt_path  | checkpoint保存路径                          | ./checkpoints                  |
| load_ckpt_path  | checkpoint加载路径                          | ./checkpoints/model_40001.ckpt |
| figures_path    | 图片保存路径                                  | ./figures                      |
| log_path        | 日志保存路径                                  | ./logs                         |
| lr              | 学习率                                     | 1e-3                           |
| epochs          | 时期（迭代次数）                                | 40001                          |
| early_stop_loss | 提前停止的loss阈值                             | 2e-32                          |
| var_form        | 变分形式                                    | 1                              |
| n_element       | 元素数量                                    | 4                              |
| n_testfcn       | 测试函数的点个数                                | 60                             |
| n_quad          | quadrature积分点个数                         | 80                             |
| n_f             | 每个因子的采样数                                | 500                            |
| lossb_weight    | lossb的权重系数                              | 1                              |
| font            | 画图字体大小                                  | 24                             |
| download_data   | 模型所需数据集与(或)checkpoints                  | hp_vpinns                      |
| force_download  | 是否强制下载数据                                | false                          |
| amp_level       | MindSpore自动混合精度等级                       | O3                             |
| device_id       | 需要设置的设备号                                | None                           |
| mode            | MindSpore静态图模式（0）或动态图模式（1）。此网络暂不支持静态图模式 | 1                              |

### [训练流程](#目录)

- 在 GPU/Ascend 上运行

   ```bash
   python train.py
   ```

  经过训练，loss值会输出如下:

  ```bash
  # grep "loss:" log
  step: 0, loss: 142.92845, lossb: 0.7792306, lossv: 142.14922, interval: 9.788227081298828s, total: 9.788227081298828s
  step: 10, loss: 142.28386, lossb: 0.15564875, lossv: 142.1282, interval: 34.56686305999756s, total: 44.35509014129639s
  step: 20, loss: 142.10016, lossb: 0.003990522, lossv: 142.09616, interval: 34.076114892959595s, total: 78.43120503425598s
  step: 30, loss: 142.12141, lossb: 0.04219491, lossv: 142.07922, interval: 34.79131770133972s, total: 113.2225227355957s
  step: 40, loss: 142.09048, lossb: 0.004216266, lossv: 142.08627, interval: 34.17010521888733s, total: 147.39262795448303s
  step: 50, loss: 142.08542, lossb: 5.215431e-05, lossv: 142.08537, interval: 34.377405881881714s, total: 181.77003383636475s
  step: 60, loss: 142.07175, lossb: 0.0014050857, lossv: 142.07034, interval: 34.402318477630615s, total: 216.17235231399536s
  step: 70, loss: 142.05489, lossb: 0.0055830916, lossv: 142.0493, interval: 34.186588525772095s, total: 250.35894083976746s
  step: 80, loss: 142.02605, lossb: 0.0036653157, lossv: 142.02238, interval: 34.496164321899414s, total: 284.85510516166687s
  step: 90, loss: 141.97115, lossb: 0.0032817253, lossv: 141.96786, interval: 34.388723611831665s, total: 319.24382877349854s
  step: 100, loss: 141.83458, lossb: 0.0045128292, lossv: 141.83006, interval: 34.6738965511322s, total: 353.91772532463074s
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