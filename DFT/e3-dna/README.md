# 模型名称

> equivariant electron density model

## 介绍

> EEDM是一个基于E3的等变神经网络，用于预测DNA结构的电子密度。

## 数据集

> 从 http://aisccc.cn/database/data-details?id=171&type=resource 下载 dataset.zip 到并解压，将`trainset_np.pkl`和`testset_np.pkl`放在`data`目录下。

## 环境要求

> 1. 安装`mindspore`
> 2. 安装`numpy`
> 3. 安装`mindchemistry`

## 快速入门

> 1. 将数据集下载到当前目录
> 2. 训练命令： `python train_dna.py`

## 脚本说明

> 1. train.py 包括了图数据的生成和模型的训练推理

### 代码目录结构

```txt
EEDM
    │  README.md        README文件
    │  train_dna.py     训练启动脚本
    │  
    └─data   训练数据
            *.out
            *.pkl
    │  
    └─src
            __init__.py
            data.py     数据集构建
            models.py   模型结构
            utils.py    辅助函数
    └─configs
            config.yaml   模型config文件
```

## 训练推理过程

### 训练

```bash
python train.py
```

### 推理

将权重的path写入config文件的`ckpt_path`中，并将`is_train`参数设置为`False`

```bash
python predict.py
```

### 训练推理过程日志

```log
# 推理
2024-12-19 10:30:11,490 [INFO ]  Number of parameters: 987595
2024-12-19 10:30:11,572 [INFO ]  Load checkpoint from ./checkpoints/best_model.ckpt
2024-12-19 10:40:28,048 [INFO ]        Test Loss:   0.00138848
2024-12-19 10:40:28,050 [INFO ]        Test MAE:    3.53494
2024-12-19 10:40:28,050 [INFO ]        Test MUE:    3.53494
2024-12-19 10:40:28,050 [INFO ]        Test STDEV:  9.55096
2024-12-19 10:40:28,051 [INFO ]        Test time:   616.457
2024-12-19 10:40:28,051 [INFO ]        Test electron difference: 20.4457
2024-12-19 10:40:28,051 [INFO ]        Test big I:               0.172658
2024-12-19 10:40:28,051 [INFO ]        Test epsilon:             42.0982
```

```log
# 训练
2024-12-13 17:22:44,225 [INFO ]  Number of parameters: 987595
2024-12-13 18:14:17,706 [INFO ]  Epoch: 0 - Train loss: 11.060080685615539
2024-12-13 18:14:17,711 [INFO ]        MAE:         277.343
2024-12-13 18:14:17,715 [INFO ]        MUE:         -42.5077
2024-12-13 18:14:17,720 [INFO ]        Train STDEV: 1619.83
2024-12-13 18:14:17,723 [INFO ]        Train tot:   25
2024-12-13 18:14:17,727 [INFO ]        Load time: 64.1685061454773 s, Train time: 2851.595867395401 s, Post time: 177.58487606048584 s
2024-12-13 18:24:54,266 [INFO ]        Test Loss:   0.0449148
2024-12-13 18:24:54,267 [INFO ]        Test MAE:    17.5987
2024-12-13 18:24:54,267 [INFO ]        Test MUE:    17.5987
2024-12-13 18:24:54,267 [INFO ]        Test STDEV:  258.26
2024-12-13 18:24:54,267 [INFO ]        Test time:   636.508
2024-12-13 18:24:54,267 [INFO ]        Test electron difference: 20.4728
2024-12-13 18:24:54,267 [INFO ]        Test big I:               0.931161
2024-12-13 18:24:54,267 [INFO ]        Test epsilon:             425.668
2024-12-13 18:24:54,289 [INFO ]        Model saved with min loss at epoch 0
```