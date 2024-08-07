# FourCastNet

## Overview

FourCastNet is a data-driven global weather forecast model. It provides medium-term forecasts of key global weather indicators with a resolution of 0.25°，Equivalent to a spatial resolution of approximately 30 km x 30 km near the equator and a global grid of 721 x 1440 pixels in size.

In order to achieve high resolution prediction, FourCastNet uses AFNO model. The model network architecture is designed for high-resolution input, uses ViT as the backbone network, and incorporates Fourier Neural Operator.

## QuickStart

You can download dataset from [dataset](https://download.mindspore.cn/mindscience/mindearth/dataset/WeatherBench_1.4_69/) for model evaluation. Save these dataset at `./dataset`.

### Run Option 1: Call `main.py` from command line

```shell
bash ./scripts/run_standalone_train.sh $device_id $device_target $config_file_path
```

where:
`--device_id` NPU id.

`--device_target` device type, default 'Ascend'.

`--config_file_path` the path of config file, default "./configs/FourCastNet.yaml".

If running a 0.25° resolution training task, change  grid_resolution to 0.25 in FourCastNet.yaml .

### Run Option 2: Run Jupyter Notebook

You can use 'Chinese' or 'English' Jupyter Notebook to run the training and evaluation code line-by-line.

### Analysis

The following figure shows the ground truth, predicion and their errors using the checkpoint of training epoch 10.

![epoch10](images/pred_result.png)

Summary of skill score for 6-hours to 5-days lead time is shown below.

![epoch10](images/Eval_RMSE_epoch10.png)
![epoch10](images/Eval_ACC_epoch10.png)

## Contributor

gitee id: Bokai Li
email: 1052173504@qq.com