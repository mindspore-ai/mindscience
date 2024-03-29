
# 模型名称

> Steerable-E3-GNN（SEGNN）

## 介绍

> SEGNN是基于可控E(3)等变图神经网络构建的SOTA模型，拓展了等变图神经网络，使得图神经网络中节点和边的属性可以包含协变量，并验证了SEGNN模型在计算物理和化学领域多个任务中的优越表现

## 数据集

> QM9数据集下载地址：[数据文件1](https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/molnet_publish/qm9.zip)；[数据文件2](https://ndownloader.figshare.com/files/3195404)

## 环境要求

> 1. 安装`mindspore（2.2.12）`
> 2. 将`mindscience/MindChemistry`目录下的`mindchemistry`子文件夹放置于项目根目录，以便`segnn`目录下的文件直接调用（即非安装mindchemistry方式）

## 快速入门

> 训练命令： `python train.py`

## 脚本说明

### 代码目录结构

```txt
└─segnn
    │  README.md            README文件
    │  requirements.txt     依赖文件
    │  qm9.yaml             配置文件
    │  train.py             训练启动脚本
    │  
    └─src
            balanced_irreps.py      Irreps生成
            dataset.py              数据集处理
            inspector.py            函数参数提取模块
            instance_norm.py        特征归一化模块
            o3_building_blocks.py   张量积计算及后处理模块
            segnn.py                SEGNN模型
            trainer.py              训练脚本
```

## 训练过程

### 训练

直接训练

```txt
python train.py
```

训练过程日志

```log
Loading data...
train_set's mean and mad:  75.26605 6.2882
Initializing model...
Determined irrep type: 36x0e+36x1o+36x2e
Initializing train...
epoch:   0, step:   0, loss: 0.95143652, train MAE: 5.9828  , time: 206.24
epoch:   0, step: 100, loss: 0.58375533, train MAE: 3.6708  , time: 49.01
epoch:   0, step: 200, loss: 0.45873420, train MAE: 2.8846  , time: 48.29
epoch:   0, step: 300, loss: 0.43487877, train MAE: 2.7346  , time: 48.11
epoch:   0, step: 400, loss: 0.39558753, train MAE: 2.4875  , time: 48.97
epoch:   0, step: 500, loss: 0.39054746, train MAE: 2.4558  , time: 49.82
epoch:   0, step: 600, loss: 0.35352476, train MAE: 2.2230  , time: 48.81
epoch:   0, step: 700, loss: 0.36744346, train MAE: 2.3106  , time: 48.94
epoch:   0, train loss: 0.42016299  , time used: 588.49
eval MAE:2.2511  , time used: 204.01

```
