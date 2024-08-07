# ViT-KNO

## Overview

ViT-KNO is a deep learning model that leverages the Vision Transformer structure and Koopman theory to efficiently learn Koopman operators for predicting dynamics in nonlinear systems. By embedding the complicated dynamics in a linear structure through a constraint reconstruction process, ViT-KNO is able to capture complex nonlinear behaviors while remaining lightweight and computationally efficient. The model has clear mathematical theory and has the potential to enable breakthroughs in fields such as meteorology, fluid dynamics, and computational physics.

## QuickStart

You can download dataset from [dataset](https://download.mindspore.cn/mindscience/mindearth/dataset/WeatherBench_1.4_69/) for model evaluation. Save these dataset at `./dataset`.

### Run Option 1: use command line

```shell
bash ./scripts/run_standalone_train.sh $device_id $device_target $config_file_path
```

where:

`--device_id` NPU id.

`--device_target` device type, default 'Ascend'.

`--config_file_path` the path of config file, default "./configs/vit_kno_1.4.yaml".

If running a 0.25° resolution training task, simply pass in "./configs/vit_kno_0.25.yaml" for `config_file_path`.

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