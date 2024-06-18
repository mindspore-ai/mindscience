# 模型名称

> DeephE3nn

## 介绍

> DeephE3nn是一个基于E3的等变神经网络，利用晶体中的原子结构去预测体系的电子哈密顿量。

## 数据集

> 从https://zenodo.org/records/7553640  下载 jdft_3d-12-12-2022.json 下载  Bilayer_graphene_dataset.zip 到当前目录并解压， 不要修改其文件名。

## 环境要求

> 1. 安装`mindspore（2.2.12）`
> 2. 安装`numpy`

## 快速入门

> 将数据集下载到当前目录
> 将Mindchemistry/mindchemistry文件包下载到当前目录
> 训练命令： `python train.py configs/Bilayer_graphene_train.ini`

## 脚本说明

> train.py 包括了图数据的生成和模型的训练
> predictor.py 模型推理脚本

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
            model.py 主模型代码
            e3modules.py e3相关模型代码
            utils.py 工具模块
            parse_configs.py  config处理文件
    └─configs
            Bilayer_graphene_train_numpy.ini  模型config文件
```

## 训练推理过程

### 训练

```txt
pip install -r requirements.txt
python train.py configs/Bilayer_graphene_train_numpy.ini
```

### 推理

将权重的path写入config文件的checkpoint_dir中

```txt
pip install -r requirements.txt
python predict.py configs/Bilayer_graphene_train_numpy.ini
```

### 训练推理过程日志

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