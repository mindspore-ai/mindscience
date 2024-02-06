ENGLISH | [简体中文](README_CN.md)

# CTEFNet: ENSO Forecast Based on CNN and Transfer Learning

## Overview

CTEFNet is an ENSO forecast model based on deep learning. It uses 2D CNN to extract features from climate data. Multiple time point features are merged into time series and then input into Transformer Encoder for time series analysis and ENSO prediction.
Compared with previous deep learning models, CTEFNet's effective prediction time is extended to 19 months.

![ctefnet](images/CTEFNet.png)

This tutorial introduces the research background and technical path of CTEFNet, and shows how to train and fast infer the model through MindEarth.

## QuickStart

You can download dataset from [mindearth/dataset](https://download-mindspore.osinfra.cn/mindscience/mindearth/dataset/enso_dataset.zip) for model training and evaluation. Save these dataset at `./dataset`.

### Run Option 1: Call `main.py` from command line

```shell
python -u ./main.py \
  --config_file_path ./configs/pretrain.yaml \
  --device_target GPU \
  --device_id 0
```

where:
`--config_file_path` the path of config file, default "./configs/pretrain.yaml".

`--device_target` device type, default 'GPU'.

`--device_id` NPU id, default 0.

### Run Option 2: Run Jupyter Notebook

You can use [Chinese](https://gitee.com/mindspore/mindscience/raw/master/MindEarth/applications/medium-range/ensoforecast/ctefnet_CN.ipynb) or [English](https://gitee.com/mindspore/mindscience/raw/master/MindEarth/applications/medium-range/ensoforecast/ctefnet.ipynb) Jupyter Notebook to run the training and evaluation code line-by-line.

## Performance

The figure below shows the predicted correlation coefficient after pre-training and fine-tuning of the model.

![epoch100](images/Forecast_Correlation_Skill.png)

## Contributor

gitee id: YingHaoCui, Charles_777

email: 1332350708@qq.com, 1332715137@qq.com