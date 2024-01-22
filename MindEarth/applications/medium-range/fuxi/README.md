ENGLISH | [简体中文](README_CN.md)

# FuXi: Medium-range Global Weather Forecasting Based on Cascade Architecture

## Overview

FuXi is a data-driven global weather forecast model developed by researchers from Fudan University. It provides medium-term forecasts of key global weather indicators with a resolution of 0.25°. Equivalent to a spatial resolution of approximately 25 km x 25 km near the equator and a global grid of 720 x 1440 pixels in size. Compared with the previous ML-based weather forecast model, the FuXi model using cascade architecture achieved excellent results in [ECMWF](https://charts.ecmwf.int/products/plwww_3m_fc_aimodels_wp_mean?area=Northern%20Extra-tropics&parameter=Geopotential%20500hPa&score=Root%20mean%20square%20error).

![winde_quiver](images/wind_quiver_0_25.png)

This tutorial introduces the research background and technical path of FuXi, and shows how to train and fast infer the model through MindEarth. More information can be found in [paper](https://www.nature.com/articles/s41612-023-00512-1). The partial dataset with a resolution of 1.4° is used in this tutorial, and the results is shown below.

## QuickStart

You can download dataset from [mindearth/dataset](https://download.mindspore.cn/mindscience/mindearth/dataset/) for model evaluation. Save these dataset at `./dataset`.

### Run Option 1: Call `main.py` from command line

```shell
python -u ./main.py \
  --config_file_path \
  --device_target Ascend \
  --device_id 0 \
```

where:
`--config_file_path` the path of config file, default "./FuXi.yaml".

`--device_target` device type, default 'Ascend'.

`--device_id` NPU id, default 0.

### Run Option 2: Run Jupyter Notebook

You can use [Chinese](https://gitee.com/mindspore/mindscience/raw/master/MindEarth/applications/medium-range/graphcast/graphcast_CN.ipynb) or [English](https://gitee.com/mindspore/mindscience/raw/master/MindEarth/applications/medium-range/graphcast/graphcast.ipynb) Jupyter Notebook to run the training and evaluation code line-by-line.

### Analysis

The following figure shows the ground truth, predicion and their errors using the checkpoint of training epoch 100.

![epoch100](images/key_info_comparison.png)

Summary of skill score for 6-hours to 5-days lead time is shown below.

![image_earth](images/Eval_RMSE_epoch100.png)
![image_earth](images/Eval_ACC_epoch100.png)

## Performance

|        Parameter         |        Ascend               |                                         GPU                                         |
|:----------------------:|:--------------------------:|:-----------------------------------------------------------------------------------:|
|     Hardware         |     Ascend 910A, 32G；CPU: 2.6GHz, 192 cores      |                                   NVIDIA V100 32G                                   |
|     MindSpore   |        2.2.0             |                                        2.2.0                                        |
|        Dataset      | [ERA5_0_25](https://download.mindspore.cn/mindscience/mindearth/dataset/) | [ERA5_0_25](https://download.mindspore.cn/mindscience/mindearth/dataset/) |
|        Parameters      |           61161472    |                                      61161472                                       |
|        Train parameters      |        batch_size=1,steps_per_epoch=67,epochs=100              |                     batch_size=1,steps_per_epoch=67,epochs=100                      |
|        Test parameters      |        batch_size=1,steps=20              |                                batch_size=1,steps=20                                |
|        Optimizer      |        Adam               |                                        Adam                                         |
|        Train loss(RMSE)      |        0.12           |                                        0.12                                         |
|        Valid WeightedRMSE(z500/5days)      |           1200           |                                        1240                                         |
|        Valid WeightedRMSE(t850/5days)      |           6.5           |                                         6.8                                         |
|        Speed(ms/step)          |     3386     |                                        8071                                         |

Training with more data from [ERA5_0.25](https://github.com/pangeo-data/WeatherBench) can get the following results:

|        RMSE      |     Z500(3 / 5 days)      |     T850(3 / 5 days)     |    U10(3 / 5 days)      |    T2m(3 / 5 days)     |
|:----------------:|:--------------:|:---------------:|:--------------:|:---------------:|
|        Operational IFS     |     152.2 / 331.38     |     1.34 / 2.01     |    1.92 / 2.89      |    1.3 / 1.71     |
|        ours(16yr)     |     179 / 347     |     1.37 / 2.00     |    1.88 / 2.87    |    1.38 / 1.89    |

## Contributor

gitee id: alancheng511

email: alanalacheng@gmail.com