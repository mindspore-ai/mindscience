# 目录

# UFold描述

对于许多RNA分子来说，二级结构对于RNA的正确功能至关重要。从核苷酸序列预测RNA二级结构是基因组学中一个长期存在的问题，但随着时间的推移，预测性能已经达到了稳定水平。传统的RNA二级结构预测算法主要基于热力学模型，通过自由能最小化，这强加了很强的先验假设，而且运行速度很慢。UFold作为一种基于深度学习的方法，用于RNA二级结构预测，直接根据注释数据和碱基配对规则进行训练。UFold提出了一种新的RNA序列的类图像表示方法，它可以通过完全卷积网络(FCNs)进行有效的处理。

## 模型架构

模型的输入是通过取One-Hot Encoding的四个基本通道的所有组合的外积生成的，这产生了16个通道。然后，表示配对概率的附加信道与16信道序列表示串联，并一起作为模型的输入。UFold模型是U-Net的一个变体，它将17通道张量作为输入，并通过连续卷积和最大池运算转换数据。

## 数据集

使用了几个基准数据集：

- RNAStralign，包含来自8个RNA家族的30 451个独特序列；

- ArchiveII，包含来自10个RNA家族的3975个序列，是最广泛使用的RNA结构预测性能基准数据集；

- bpRNA-1m，包含来自2588个家族的102 318个序列，是可用的最全面的RNA结构数据集之一；

- bpRNA new，源自Rfam 14.2，包含来自1500个新RNA家族的序列。

为了方便数据集的使用，我们将bpseq格式的数据文件处理成pickle文件。UFold模型使用的数据文件可以在[drive](https://drive.google.com/drive/folders/1Sq7MVgFOshGPlumRE_hpNXadvhJKaryi)中下载，使用时需将文件放到data文件夹。

## 环境要求

- 硬件（昇腾处理器/GPU）
    - 采用昇腾处理器/GPU搭建硬件环境。
- 框架
    - [MindSpore](https://gitee.com/mindspore/mindspore)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)
- 第三方库

```bash
pip install -r requirements.txt
```

## 快速入门

1. 下载pickle文件并放入data文件夹。

2. 在`config.json`中修改first_epoch以修改训练轮数。

3. 执行训练脚本。
  数据集准备完成后，按照如下步骤开始训练：

```text
python train.py --train_files dataset_A dataset_B
    --train_files: 可从(['ArchiveII','TS0','bpnew','TS1','TS2','TS3'])中选择一个或多个数据集进行训练。
    --device_target： 可从(['GPU', 'Ascend'])选择。
    --device_id: 可按照环境选择。
```

4. 执行推理脚本。
  训练结束后，按照如下步骤启动推理：

```text
python ufold_test.py --test_files TS2
    --test_files: 可从(['ArchiveII','TS0','bpnew','TS1','TS2','TS3'])中选择一个或多个数据集进行推理。
    --ckpt_file: 选择加载的ckpt文件。
    --device_target： 可从(['GPU', 'Ascend'])选择。
    --device_id: 可按照环境选择。
```

# 脚本说明

## 脚本和样例代码

```shell
.
└─UFold
  ├─README.md                             # README
  ├─README_CN.md                          # README_CN
  ├─ckpt_models                           # 用来存放训练好的/下载的ckpt文件
  ├─data                                  # 数据集预处理得到的pickle文件
  ├─src
    ├─config.json                         # 参数设置
    ├─config.py                           # 加载设置
    ├─data_generator.py                   # 自定义数据集类
    ├─Network.py                          # UFold网络定义
    ├─utils.py                            # 零散函数
    └─postprocess.py                      # 后处理进行优化
  ├─eval.py                               # 评估脚本
  └─train.py                              # 训练脚本
```

### 脚本参数

```bash
"BATCH_SIZE":1,
"epoch_first":100
```

## 训练过程

- 在`config.json`中设置配置项，包括BATCH_SIZE和训练epoch。

### 训练

- 运行`train.py`开始UFold的训练

```shell
    python train.py --train_files dataset_A
        --train_files: 可从(['ArchiveII','TS0','bpnew','TS1','TS2','TS3'])中选择一个或多个数据集进行训练。
        --device_target： 可从(['GPU', 'Ascend'])选择。
        --device_id: 可按照环境选择。
```

- 训练打印结果如下：

```log
# grep "loss is " train.log
epoch:1 epoch: 1, loss: 1.4842823
epcoh:2 epoch: 2, loss: 1.0897788
```

## 评估

### 评估过程

- 运行`eval.py`进行评估。

注：可以使用训练出来的ckpt进行评估，或者使用已训练好的提供的[文件](https://drive.google.com/drive/folders/1XmeoRaFS3iXa9kZwte99e7usB7mPLV4z?usp=sharing)进行评估。 (ufold_train_pdbfinetune.ckpt用于TS1, TS2, TS3; ufold_train.ckpt用于bpnew, ArchiveII和TS0; ufold_train_alldata.ckpt由于是使用全部数据训练出来的所以可以用于所有的测试数据。)

```bash
# 推理
python ufold_test.py --test_files TS2
  --test_files: 可从(['ArchiveII','TS0','bpnew','TS1','TS2','TS3'])中选择一个或多个数据集进行推理。
  --ckpt_file: 选择加载的ckpt文件。
  --device_target： 可从(['GPU', 'Ascend'])选择。
  --device_id: 可按照环境选择。
```

### 评估结果

测试数据集的准确性如下：

```log
Average testing precision with pure post-processing: 0.781516432132
```

