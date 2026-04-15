
# 模型名称

> Matformer

## 介绍

> Matformer是基于图神经网络和Transformer架构的SOTA模型，用于预测晶体材料的各种性质。

## 数据集

> 从https://figshare.com/articles/dataset/jdft_3d-7-7-2018_json/6815699 下载 jdft_3d-12-12-2022.json 到当前目录，不需要修改其文件名。

## 环境要求

> 1. 安装`mindspore`
> 2. 安装`mindchemistry`

## 快速入门

> 1. 将数据集下载到当前目录
> 2. 训练命令： `python train.py`

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
INFO:root:The model you built has 2786689 parameters.
INFO:root:Starting new training process
INFO:root:Start to initialise train loader
INFO:root:Start to initialise eval loader
INFO:root:+++++++++++++++ start traning +++++++++++++++++++++
INFO:root:==============================step: 0 ,epoch: 0
INFO:root:learning rate: 4e-05
INFO:root:train mse loss: 0.8999285
INFO:root:is_finite: True
INFO:root:training time: 51.66963744163513
.
.
.
INFO:root:step:117, epoch: 499
INFO:root:validation mse loss: 0.004059551
INFO:root:validation mae loss: 0.034488887
INFO:root:validation time: 0.041112422943115234
INFO:root:epoch 499 running time: 137.772692
INFO:root:epoch 499 average train mse loss: 0.0003474082
INFO:root:epoch 499 average validation mse loss: 0.00414170
INFO:root:epoch 499 average validation mae loss: 0.03259226
```
