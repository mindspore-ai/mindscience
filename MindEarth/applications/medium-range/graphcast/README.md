ENGLISH | [简体中文](README_CN.md)

# Medium-range Global Weather Forecasting Based on GNN

## Overview

GraphCast is a data-driven global weather forecast model developed by researchers from DeepMind and Google. It provides medium-term forecasts of key global weather indicators with a resolution of 0.25°. Equivalent to a spatial resolution of approximately 25 km x 25 km near the equator and a global grid of 721 x 1440 pixels in size. Compared with the previous ML-based weather forecast model, this model improves the accuarcy to 99.2% of the 252 targets.

![winde_quiver](images/wind_quiver_0.25.png)

This tutorial introduces the research background and technical path of GraphCast, and shows how to train and fast infer the model through MindEarth. More information can be found in [paper](https://arxiv.org/abs/2212.12794). The partial dataset with a resolution of 1.4° is used in this tutorial, and the results is shown below.

## QuickStart

You can download dataset from graphcast/dataset for model evaluation. Save these dataset at `./dataset`.

### Run Option 1: Call `main.py` from command line

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

where:
`--device_target` device type, default 'Ascend'.

`--device_id` NPU id, default 0.

`--processing_steps` preocessing steps, default 16.

`--latent_dims` hidden layer dimensions, default 512.

`--mesh_level` mesh node levels, default 4.

`--grid_resolution` grid resolution, default 1.4.

`--output_dir` the path of output file, default "./summary".

### Run Option 2: Run Jupyter Notebook

You can use [Chinese](https://gitee.com/mindspore/mindscience/raw/master/MindEarth/applications/medium-range/graphcast/graphcast_CN.ipynb) or [English](https://gitee.com/mindspore/mindscience/raw/master/MindEarth/applications/medium-range/graphcast/graphcast.ipynb) Jupyter Notebook to run the training and evaluation code line-by-line.

### Analysis

The following figure shows the ground truth, predicion and their errors using the checkpoint of training epoch 100.

![epoch100](images/key_info_comparison.png)

Summary of skill score for 6-hours to 5-days lead time is shown below.

![image_earth](images/Eval_RMSE_epoch100.png)
![image_earth](images/Eval_ACC_epoch100.png)

## Contributor

gitee id: liulei277, email: liulei2770919@163.com

gitee id: Bokai Li, email: 1052173504@qq.com