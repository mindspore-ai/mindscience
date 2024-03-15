
# 模型名称

> Allegro

## 介绍

> Allegro是基于等变图神经网络构建的SOTA模型，可以在大规模材料体系中进行高精度预测，相关论文作已经发表在期刊Nature Communications上，该案例验证了Allegro在分子势能预测中的有效性，具有较高的应用价值

## 数据集

> rmd数据集下载地址：[Revised MD17 dataset (rMD17)](https://figshare.com/articles/dataset/Revised_MD17_dataset_rMD17_/12672038)；默认配置文件读取路径为`dataset/rmd17/npz_data/`

## 环境要求

> 1. 安装`mindspore（2.2.12）`
> 2. 将`mindscience/MindChemistry`目录下的`mindchemistry`子文件夹放置于项目根目录，以便`allegro`目录下的文件直接调用（即非安装mindchemistry方式）

## 快速入门

> 训练命令： `python train.py`

## 脚本说明

### 代码目录结构

```txt
└─allegro
    │  .gitignore   git忽略文件
    │  README.md    README文件
    │  requirements.txt 依赖文件
    │  rmd.yaml  配置文件
    │  run.sh   并行训练脚本
    │  train.py 训练启动脚本
    │  
    └─src
            allegro_embedding.py    embedding模块
            dataset.py  数据集处理
            potential.py    势能网络模块
            reduce_lr_on_plateau.py LR衰减模块
            trainer.py  训练脚本
```

## 训练过程

### 训练

直接训练

```txt
python train.py
```

在昇腾上使用分布式训练运行下面的命令

```shell
bash run.sh
```

训练过程日志

```log
2024-03-08 07:37:03 (INFO): Dump config file to: log/input.yaml
2024-03-08 07:37:03 (INFO): Loading data...
2024-03-08 07:37:13 (INFO): Initializing model...
2024-03-08 07:37:13 (INFO): Parameter (name=model.latents.0.layers.0.weight, shape=(128, 16), dtype=Float32, requires_grad=True)
.
.
.
2024-03-08 07:37:13 (INFO): Total parameters: 15108232
2024-03-08 07:37:14 (INFO): Initializing train...
2024-03-08 07:37:14 (INFO): seed is: 123
2024-03-08 07:37:14 (INFO): Epoch 1
-------------------------------
2024-03-08 07:37:43 (INFO): loss: 468775552.0000000000000000  [  0/190]
.
.
.
2024-03-08 07:37:59 (INFO): loss: 1245489.8750000000000000  [180/190]
2024-03-08 07:38:00 (INFO): train loss: 160203749.5373766422271729, time gap: 46.4496
2024-03-08 07:38:17 (INFO): Test: mse loss: 661061.1500000000232831
2024-03-08 07:38:17 (INFO): Test: mae metric: 685.2471028645833258
2024-03-08 07:38:17 (INFO): lr: 0.002
.
.
.

```
