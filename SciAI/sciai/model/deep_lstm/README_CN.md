[ENGLISH](README.md) | 简体中文

# 目录

- [DeepLSTM 描述](#DeepLSTM-描述)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [脚本说明](#脚本说明)
    - [脚本和示例代码](#脚本和示例代码)
    - [脚本参数](#脚本参数)
    - [训练流程](#训练流程)
    - [推理流程](#推理流程)

## [DeepLSTM 描述](#目录)

<br />

![Scheme](/figures/scheme.png)

 <br />

> [论文](https://www.sciencedirect.com/science/article/abs/pii/S0045794919302263):Ruiyang Zhang, Zhao Chen, Su Chen, Jingwei Zheng, Oral Büyüköztürk, Hao Sun,
> Deep long short-term memory networks for nonlinear structural seismic response prediction, Computers & Structures,
> Volume 220, 2019, Pages 55-68, ISSN 0045-7949.

## [数据集](#目录)

用于训练的数据集文件下载链接：
[下载链接](https://www.dropbox.com/sh/xyh9595l79fbaer/AABnAqV_WdhVHgPAav73KX8oa?dl=0).

使用的数据集:

- 数据集大小
    - data_BoucWen.mat（44MB）
    - data_MRFDBF.mat （564MB）
    - data_SanBernardino.mat （2.82MB）
- 数据格式: `.mat`文件
- 数据集在`./data`目录下，目录结构如下:

```text
├── data
│   ├── data_BoucWen.mat
│   ├── data_MRFDBF.mat
│   └── data_SanBernardino.mat
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

通过官网安装好MindSpore后，就可以开始训练和验证如下:

- 在 Ascend 或 GPU 上运行

默认:

```bash
python train.py
```

完整命令:

```bash
python train.py \
    --dataset data_BoucWen.mat \
    --model lstm-s
```

## [脚本说明](#目录)

## [脚本和示例代码](#目录)

文件结构如下:

```text
├── deep_lstm
│   ├── data                # 数据文件
│   ├── figures             # 日志文件
│   ├── src                 # 源代码目录
│   │   ├── network.py      # 网络架构
│   │   └── utils.py        # 数据处理
│   ├── README.md           # 英文模型说明
│   ├── README_CN.md        # 中文模型说明
│   ├── train.py            # python训练脚本
│   └── eval.py             # python评估脚本
```

### [脚本参数](#目录)

论文中共有3个问题，对于每个问题案例，参数如下:

| 参数名     | 描述         | 默认值              |
|---------|------------|------------------|
| dataset | 所使用的数据集文件名 | data_BoucWen.mat |
| model   | 所使用的模型版本   | lstm-s           |

### [训练流程](#目录)

- 在 GPU/Ascend 上运行

   ```bash
   python train.py
   ```

  经过训练，loss值会输出如下:

  ```bash
  Train total epoch:  50000
  ---------------train start---------------
  step: 0 epoch: 0 batch: 0 loss: 0.08907777
  step time is 4.832167148590088
  step: 1 epoch: 0 batch: 1 loss: 0.09467506
  step time is 0.03872847557067871
  step: 2 epoch: 0 batch: 2 loss: 0.09016898
  step time is 0.04083538055419922
  step: 3 epoch: 0 batch: 3 loss: 0.08822981
  step time is 1.1140174865722656
  train_mse: 0.090537906  test_mse: 0.081995904
  step: 4 epoch: 1 batch: 0 loss: 0.0851737
  step time is 0.03679323196411133
  step: 5 epoch: 1 batch: 1 loss: 0.09027029
  step time is 0.02897500991821289
  ...
  ```

- 模型checkpoint将保存在 `save_dir`中，默认为`./results` 目录中。

### [推理流程](#目录)

- 在 GPU/Ascend 上运行

   ```bash
   python eval.py
   ```

结果图片存放于`save_dir`中，默认位于`./results` 。
