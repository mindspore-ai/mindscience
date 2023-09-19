[ENGLISH](README.md) | 简体中文

# 基于图神经网络的全球中期天气预报

## 概述

GraphCast是由DeepMind和Google的研究人员开发的一个基于数据驱动的全球天气预报模型。它提供了关键全球天气指标的中期预报，分辨率为0.25°。相当于赤道附近约25公里x25公里的空间分辨率和大小为721 x 1440像素的全球网格。与以前的基于ML的天气预报模型相比，该模型将252个目标的准确率提高到99.2%。

![winde_quiver](images/wind_quiver_0.25.png)

本教程介绍了GraphCast的研究背景和技术路径，并展示了如何通过MindEarth训练和快速推理模型。 更多信息参见[文章] (https://arxiv.org/abs/2212.12794)。本教程中使用分辨率为1.4°的部分数据集，结果如下所示。

## 快速开始

在`graphcast/dataset`下载数据并保存在`./dataset`。

### 运行方式一: 在命令行调用`main.py`脚本

```shell
python -u ./main.py \
  --device_target Ascend \
  --device_id 0 \
  --processing_steps 16\
  --latent_dims 512 \
  --mesh_level 4 \
  --grid_resolution 1.4
  --output_dir ./summary \
```

其中，
`--device_target` 表示设备类型，默认Ascend。

`--device_id` 表示运行设备的编号，默认值0。

`--processing_steps` 处理模块运行的次数， 默认值16。

`--latent_dims` 隐藏层的维度，默认值512。

`--mesh_level` 网格分别率等级，默认值4。

`--grid_resolution` 网格分辨率，默认值1.4。

`--output_dir` 输出文件的路径，默认值"./summary"。

### 运行方式二: 运行Jupyter Notebook

使用[中文](https://gitee.com/mindspore/mindscience/raw/master/MindEarth/applications/medium-range/graphcast/graphcast_CN.ipynb)或[英文](https://gitee.com/mindspore/mindscience/raw/master/MindEarth/applications/medium-range/graphcast/graphcast.ipynb) Jupyter Notebook可以逐行运行训练和推理代码

### 结果展示

下图展示了使用训练结果的第100个epoch进行推理绘制的地表、预测值和他们之间的误差。

![epoch100](images/key_info_comparison.png)

6小时至5天的天气预报关键指标见下图。

![image_earth](images/Eval_RMSE_epoch100.png)
![image_earth](images/Eval_ACC_epoch100.png)

## 贡献者

gitee id: liulei277, email: liulei2770919@163.com

gitee id: Bokai Li, email: 1052173504@qq.com