ENGLISH | [简体中文](README_CN.md)

# Skillful Nowcasting with Deep Generative Model of Radar: DgmrNet

## Overview

DgmrNet(Deep Generative Model of Radar Network) is a deep generative model for the probabilistic nowcasting of precipitation from radar developed by researchers from DeepMind. It produces realistic and spatiotemporally consistent predictions over regions up to 1,536 km × 1,280 km and with lead times from 5–90 min ahead.

![dgmr](images/dgmr_DgmrNet.png)

This tutorial introduces the research background and technical path of DgmrNet, and shows how to train and fast infer the model through MindEarth. More information can be found in [paper](https://arxiv.org/abs/2104.00954).

## QuickStart

You can download dataset from dgmr/dataset for model evaluation. Save these dataset at `./dataset`.

### Run Option 1: Call `main.py` from command line

```shell
python -u ./main.py \
  --device_target Ascend \
  --device_id 0 \
  --output_dir ./summary
```

where:
--device_target device type, default Ascend.
--device_id NPU id, default 0.
--output_dir the path of output file, default "./summary".

### Run Option 2: Run Jupyter Notebook

You can use '[Chinese](DgmrNet_CN.ipynb)' or '[English](DgmrNet.ipynb)' Jupyter Notebook to run the training and evaluation code line-by-line.

## Analysis

The following figure shows the ground truth, predicion and their errors using the checkpoint of training epoch 100.

![epoch 100](images/dgmr_pre_image.png)

Summary of score for CRPS_MAX is shown below.
![image_earth](images/dgmr_crps_max.png)

## Contributor

gitee id: alancheng511

c