[ENGLISH](README.md) | 简体中文

# 基于深度生成模型的雷达数据气象短临预报

## 概述

DgmrNet（雷达数据深度生成模型）是由 DeepMind 的研究人员开发的用于雷达降水概率短临预报的深度生成模型。它可以对面积达 1,536 公里 × 1,280 公里的区域进行现实且时空一致的预测，并且提前时间为 5 至 90 分钟。

![dgmr](images/dgmr_DgmrNet.png)

本教程介绍了DgmrNet的研究背景和技术路径，并展示了如何通过MindEarth训练和快速推断模型。更多信息可以在[论文](https://arxiv.org/abs/2104.00954)中找到。

## 快速开始

在`dgmr/dataset`下载数据并保存在`./dataset`。

### 运行方式一：在命令行调用`main.py`脚本

```shell
python -u ./main.py \
  --device_target Ascend \
  --device_id 0 \
  --output_dir ./summary
```

其中，
--device_target 表示设备类型，默认Ascend。
--device_id 表示运行设备的编号，默认值0。
--output_dir 输出文件的路径，默认值"./summary"。

### 运行方式二: 运行Jupyter Notebook

使用'[Chinese](DgmrNet_CN.ipynb)'或'[English](DgmrNet.ipynb)' Jupyter Notebook可以逐行运行训练和推理代码

## 结果展示

下图显示了使用训练结果第100个epoch的真值、预测及其误差。

![epoch 100](images/dgmr_pre_image.png)

评估分数CRPS_max如下图所示。

![image_earth](images/dgmr_crps_max.png)

## 贡献者

gitee id: alancheng511

email: alanalacheng@gmail.com