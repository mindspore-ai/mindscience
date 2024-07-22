
# 模型名称

> NequIP

## 介绍

> NequIP是基于等变图神经网络构建的SOTA模型，相关论文已经发表在期刊Nature Communications上，该案例验证了NequIP在分子势能与力的预测中的有效性，具有较高的应用价值

## 数据集

> rmd数据集下载地址：[Revised MD17 dataset (rMD17)](https://figshare.com/articles/dataset/Revised_MD17_dataset_rMD17_/12672038)；默认配置文件读取数据集的路径为`dataset/MD17/`

## 环境要求

> 1. 安装`mindspore`
> 2. 安装`mindchemistry`

## 快速入门

> 训练命令： `python train.py`

## 脚本说明

### 代码目录结构

```txt
└─nequip
    │  README.md    README文件
    │  rmd.yaml     配置文件
    │  train.py     训练启动脚本
    │  predict.py     推理评估启动脚本
    │  
    └─src
            dataset.py  数据集处理
            plot.py     结果作图
            trainer.py  训练脚本
            predicter.py  推理评估脚本
            utils.py    工具模块
```

## 训练过程

### 训练

直接训练

```txt
python train.py
```

训练过程日志

```log
2024-03-25 21:49:49 (INFO): ---- Configuration Summary -----
.
.
.
2024-03-25 21:49:49 (INFO): --------------------------------
2024-03-25 21:49:49 (INFO): Loading data...
2024-03-25 21:50:13 (INFO): Initializing model...
2024-03-25 21:50:37 (INFO): Initializing train...
2024-03-25 22:01:58 (INFO): epoch 1:  train loss: 1000.02729235, time gap: 680.55, total time used: 680.55
.
.
.
```

## 推理评估过程

### 推理评估

```txt
1.将权重checkpoint文件保存至 `/checkpoints/`目录下（默认读取目录）
2.执行推理脚本：python predict.py
```

推理评估结果

```txt
可以通过 predict.log 文件查看结果; 推理输出文件为 pred.npy
```
