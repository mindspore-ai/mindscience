
# 模型名称

> Matformer

## 介绍

> Matformer是基于图神经网络和Transformer架构的SOTA模型，用于预测晶体材料的各种性质。

## 数据集

> 从https://figshare.com/articles/dataset/jdft_3d-7-7-2018_json/6815699 下载 jdft_3d-12-12-2022.json 到当前目录，不需要修改其文件名。

## 环境要求

> 1. 安装`mindspore（2.2.12）`

## 快速入门

> 将数据集下载到当前目录
> 将Mindchemistry/mindchemistry文件包下载到当前目录
> 训练命令： `python train.py`

## 脚本说明

> train.py包括了图数据的生成和模型的训练

### 代码目录结构

```txt
matformer
    │  README.md    README文件
    │  config.yaml    配置文件
    │  train.py     训练启动脚本
    │  predict.py     推理启动脚本
    │  
    └─data
            data.py  数据集处理
            feature.py   特性处理
            generate.py  图数据生成
            graph.py   图数据结构
    └─models
            matformer.py  模型主架构
            transformer.py   transformer卷积架构模块
            utils.py  工具模块
```

## 训练过程

### 训练

直接训练

```txt
pip install -r requirements.txt
python train.py
```

### 推理

将权重的path写入config文件的predictor.checkpoint_path中

```txt
python predict.py
```

### 训练过程日志

```log
INFO:root:Loading from saved file...
INFO:root:The model you built has 2786689 parameters.
INFO:root:Starting new training process
INFO:root:Start to initialise train_loader
INFO:root:Start to initialise eval_loader
INFO:root:+++++++++++++++ start traning +++++++++++++++++++++
INFO:root:==============================step: 0 ,epoch: 0
INFO:root:learning rate: 4e-05
INFO:root:train mse loss: 0.97237825
INFO:root:is_finite: True
INFO:root:traning time: 60.6885781288147
.
.
.
INFO:root:epoch 360 running time: 224.05996918678284
INFO:root:epoch 360 average train mse loss: 0.0011379468
INFO:root:epoch 360 average validation mse loss: 0.00875873
INFO:root:epoch 360 average validation mae loss: 0.07289803
```
