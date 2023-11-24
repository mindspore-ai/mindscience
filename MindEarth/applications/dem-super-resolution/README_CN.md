[ENGLISH](README.md) | 简体中文

# DEM-SRNet: 全球3弧秒（90m）海陆高分辨率数字高程模型

## 概述

DEM-SRNet是一个预训练地面DEM数据的深度残差网络，设计的预训练结构源自增强型深度超分辨率网络（EDSR）。

![dem](images/dem_DEM-SRNet.png)

本教程介绍了DEM-SRNet的研究背景和技术路径，并展示了如何通过MindEarth训练和快速推理模型。更多信息可以在[论文](https://pubmed.ncbi.nlm.nih.gov/36604030/)中找到。

## 快速开始

从[dataset](https://download.mindspore.cn/mindscience/mindearth/dataset/dem_dataset.zip)下载数据并保存在`./dataset`。

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

使用'[Chinese](DEM-SRNet_CN.ipynb)'或'[English](DEM-SRNet.ipynb)'Jupyter Notebook可以逐行运行训练和推理代码

## 结果展示

陆地和海洋区域测试结果的均方根误差 (RMSE) 值

|            |               | Bicubic | SRCNN  | VDSR   | ESRGAN | EDSR   | DEM-SRNet | ΔRMSE (%) |
| :--------- | ------------- | ------- | ------ | ------ | ------ | ------ | --------- | --------- |
| Land area  | A1            | 57.99   | 35.76  | 36.66  | 38.44  | 35.92  | 35.23     | 39.24     |
|            | A2            | 51.99   | 44.83  | 37.36  | 43.22  | 40.66  | 36.43     | 29.92     |
|            | A3            | 20.85   | 13.60  | 20.44  | 13.89  | 14.81  | 13.56     | 34.96     |
|            | A4            | 67.22   | 46.56  | 47.01  | 49.68  | 53.36  | 45.28     | 32.63     |
| Ocean area | B             | 25.97   | 25.52  | 30.62  | 25.18  | 25.38  | 24.87     | 4.24      |
|            | C             | 27.59   | 22.65  | 24.44  | 21.18  | 20.90  | 17.95     | 34.94     |
|            | D             | 117.16  | 214.95 | 115.87 | 118.71 | 115.63 | 112.97    | 3.58      |
|            | E             | 35.18   | 104.68 | 37.97  | 74.02  | 38.74  | 29.25     | 16.86     |
|            | F             | 4.54    | 17.75  | 17.52  | 4.55   | 4.54   | 3.35      | 26.21     |
|            | G             | 57.63   | 66.08  | 58.11  | 56.61  | 52.10  | 49.01     | 14.96     |
|            | Total average |         |        |        |        |        |           | 23.75     |

## 贡献者

gitee id: alancheng511

email: alanalacheng@gmail.com