# 模型名称

> DeephE3nn

## 介绍

> DeephE3nn是一个基于E3的等变神经网络，利用晶体中的原子结构去预测体系的电子哈密顿量。

## 数据集

> 从https://zenodo.org/records/7553640  下载  Bilayer_graphene_dataset.zip 到当前目录并解压， 不要修改其文件名。

## 环境要求

> 1. 安装`mindspore`
> 2. 安装`numpy`
> 3. 安装`mindchemistry`

## 快速入门

> 1. 将数据集下载到当前目录
> 2. 训练命令： `python train.py configs/Bilayer_graphene_train.ini`

## 脚本说明

> 1. train.py 包括了图数据的生成和模型的训练
> 2. predict.py 模型推理脚本

### 代码目录结构

```txt
deephe3nn
    │  README.md    README文件
    │  train.py     训练启动脚本
    │  predictor.py     推理启动脚本
    │  
    └─data
            data.py  数据集处理
            graph.py   图数据结构
    │  
    └─models
            kernel.py 主执行流程
            parse_configs.py  config处理文件
    └─configs
            Bilayer_graphene_train_numpy.ini  模型config文件
```

## 训练推理过程

### 训练

```txt
pip install -r requirements.txt
python train.py configs/Bilayer_graphene_train.ini
```

### 推理

将权重的path写入config文件的checkpoint_dir中

```txt
pip install -r requirements.txt
python predict.py configs/Bilayer_graphene_train.ini
```

### 训练推理过程日志

```log
INFO:root:Starting new training process
INFO:root:-------Begin training-------
INFO:root:=================================epoch: 0
.
.
.
INFO:root:----------------------eval epoch: 916-------step: 19
INFO:root:evaluating time: 0.25410914421081543
INFO:root:learning rate: 3.159372e-10
INFO:root:val mse loss: 7.4168706e-06
INFO:root:epoch: 916

INFO:root:last train loss: 7.4168706e-06
INFO:root:average eval loss: 6.1306587e-06
INFO:root:Train finished, cost 63180.765609025955 s
INFO:root:best loss: 6.1306587e-06
```