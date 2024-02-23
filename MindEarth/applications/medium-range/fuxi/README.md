ENGLISH | [简体中文](README_CN.md)

# FuXi: Medium-range Global Weather Forecasting Based on Cascade Architecture

## Overview

FuXi is a data-driven global weather forecast model developed by researchers from Fudan University. It provides medium-term forecasts of key global weather indicators with a resolution of 0.25°. Equivalent to a spatial resolution of approximately 25 km x 25 km near the equator and a global grid of 720 x 1440 pixels in size. Compared with the previous ML-based weather forecast model, the FuXi model using cascade architecture achieved excellent results in [ECMWF](https://charts.ecmwf.int/products/plwww_3m_fc_aimodels_wp_mean?area=Northern%20Extra-tropics&parameter=Geopotential%20500hPa&score=Root%20mean%20square%20error).

This tutorial introduces the research background and technical path of FuXi, and shows how to train and fast infer the model through MindEarth. More information can be found in [paper](https://www.nature.com/articles/s41612-023-00512-1). The [ERA5_0_25_tiny400](https://download-mindspore.osinfra.cn/mindscience/mindearth/dataset/ERA5_0_25_tiny400/) with a resolution of 0.25° is used in this tutorial, and the results is shown below.

## Running Model

### Base Backbone

You can download dataset from [ERA5_0_25_tiny400](https://download-mindspore.osinfra.cn/mindscience/mindearth/dataset/ERA5_0_25_tiny400/) for model evaluation. Save these dataset at `./dataset`.

#### Quick Start

*Option 1: use command line*

```shell
bash ./scripts/run_standalone_train.sh $device_id $device_target $config_file_path
```

where:

`--device_id` NPU id.

`--device_target` device type, default 'Ascend'.

`--config_file_path` the path of config file, default "./configs/FuXi.yaml".

*Option 2: Run Jupyter Notebook*

You can use [Chinese](https://gitee.com/mindspore/mindscience/blob/f93ea7a7f90d67c983256844a2bcab094a3c7084/MindEarth/applications/medium-range/fuxi/fuxi_CN.ipynb) or [English](https://gitee.com/mindspore/mindscience/raw/f93ea7a7f90d67c983256844a2bcab094a3c7084/MindEarth/applications/medium-range/fuxi/fuxi.ipynb) Jupyter Notebook to run the training and evaluation code line-by-line.

#### Multi-Card Parallel

Running in Multi-Card parallel mode requires setting the `distribute` in the configuration file specified by `config_file_path` to `True`.

```shell
bash ./scripts/run_distributed_train.sh $rank_table_file $device_num $device_start_id $config_file_path
```

where:

`--rank_table_file` [path to the networking information file](https://www.mindspore.cn/tutorials/experts/zh-CN/r2.2/parallel/rank_table.html#%E6%A6%82%E8%BF%B0).

`--device_num` the numbers of networking device.

`--device_start_id` the start ID of networking device.

`--config_file_path` the path of config file.

### Results

#### Visualization of basic meteorological variables

The following figure shows the ground truth, predicion and their errors using the checkpoint of training epoch 100.

![epoch100](images/key_info_comparison.png)

## Total Training Results

By using 13years of ERA5 0.25° resolution data and conducting 6-hour interval data training, we can get the following result which shows the comparison of key meteorological indicators with IFS results over a period of 5 days for training models to predict 6 hours.

![13yr_rmse](./images/RMSE_0.25_multi_years.png)
![13yr_acc](./images/ACC_0.25_multi_years.png)

## Performance

### Base Backbone

|      Parameter        |        NPU              |        NPU             |    GPU       |
|:----------------------:|:--------------------------:|:--------------------------:|:---------------:|
|    Hardware        |     Ascend, memory 32G     |     Ascend, memory 32G      |     V100, memory 32G       |
|     MindSpore   |        2.2.0             |         2.2.0             |      2.2.0       |
|     Dataset      |      ERA5_0_25_13yr             |      [ERA5_0_25_tiny400](https://download-mindspore.osinfra.cn/mindscience/mindearth/dataset/ERA5_0_25_tiny400/)     |     [ERA5_0_25_tiny400](https://download-mindspore.osinfra.cn/mindscience/mindearth/dataset/ERA5_0_25_tiny400/)     |
|    Parameters     |    61 mil.         |          61 mil.         |        61 mil.    |
|    Train parameters  |        batch_size=1<br>steps_per_epoch=1187<br>epochs=200    |    batch_size=1<br>steps_per_epoch=67<br>epochs=100   |     batch_size=1<br>steps_per_epoch=67<br>epochs=100     |
|        Test parameters      |    batch_size=1<br>steps=39 | batch_size=1<br>steps=9 |    batch_size=1<br>steps=9  |
|    Optimizer    |        Adam      |         Adam              |    Adam     |
|        Train loss(RMSE)      |    0.046    |   0.12     |  0.12   |
|        Z500  (6h, 72h, 120h)      |   30, 197.5, 372 |  176, 992, 1187 |   176, 992, 1187    |
|        T850  (6h, 72h, 120h)      |   0.48, 1.39, 2.15 |  1.23, 5.78, 7.29 |1.23, 5.78, 7.29   |
|        U10  (6h, 72h, 120h)      |    0.5, 1.9, 2.9 | 1.48, 5.58, 6.33 | 1.48, 5.58, 6.33   |
|        T2m  (6h, 72h, 120h)      |    0.69, 1.39, 1.94|  2.52, 5.78, 7.29 | 2.52, 5.78, 7.29   |
|    Training resources      | 2Node 16NPUs  | 1Node 1NPU   | 1Node 1GPU    |
|    Running time     | 445 hours  | 7.8 hours   | 15 hours    |
|    Speed(ms/step)          |     4386     |     3821       |   8071 |

## Contributor

gitee id: alancheng511, liulei277

email: alanalacheng@gmail.com, liulei2770919@163.com