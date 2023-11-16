# Raynold-averaged Navier-Stokes equations

## Overview

The Raynold-averaged Navier-Stokes equations are classic numerical simulation case in the fields of fluid mechanics and meteorology. It is used to study the flow behavior of air or fluid over a periodic hilly terrain. This problem aims to explore the influence of hilly terrain on atmospheric or fluid motion, leading to a deeper understanding of meteorological phenomena, terrain effects, and fluid characteristics over complex terrain. This project utilizes the Reynolds-averaged model to simulate turbulent flow over a two-dimensional periodic hill.

### Reynolds-Averaged Model

The Reynolds-Averaged Navier-Stokes equations (RANS) are a commonly used numerical simulation approach in fluid mechanics to study the averaged behavior of fluids under different Reynolds numbers. Named after the British scientist Osborne Reynolds, this model involves time-averaging of flow field variables and provides an engineering-oriented approach to deal with turbulent flows. The Reynolds-averaged model is based on Reynolds decomposition, which separates flow field variables into mean and fluctuating components. By time-averaging the Reynolds equations, the unsteady fluctuating terms are eliminated, resulting in time-averaged equations describing the macroscopic flow. Taking the two-dimensional Reynolds-averaged momentum and continuity equations as examples:

**Reynolds-Averaged Momentum Equation:**

$$
\rho \bar{u}_j \frac{\partial \bar{u}_i}{\partial x_j}=\rho \bar{f}_i+\frac{\partial}{\partial x_j}\left[-\bar{p} \delta_{i j}+\mu\left(\frac{\partial \bar{u}_i}{\partial x_j}+\frac{\partial \bar{u}_j}{\partial x_i}\right)-\rho \overline{u_i^{\prime} u_j^{\prime}}\right]
$$

**Continuity Equation:**

$$\frac{\partial \overline{u}}{\partial x} + \frac{\partial \overline{v}}{\partial y} = 0$$

Here, $\overline{u}$ and $\overline{v}$ represent the time-averaged velocity components in the x and y directions, $\overline{p}$ is the time-averaged pressure, $\rho$ is fluid density, $\nu$ is the kinematic viscosity, and $u$ and $v$ are the velocity components in the x and y directions.

### Model Solution Introduction

The core idea of the RANS-PINNs (Reynolds-Averaged Navier-Stokes - Physics-Informed Neural Networks) method is to combine physical equations with neural networks to achieve simulation results that possess both the accuracy of traditional RANS models and the flexibility of neural networks. In this approach, the Reynolds-averaged equations for mean flow, along with an isotropic eddy viscosity model for turbulence, are combined to form an accurate baseline solution. Then, the remaining turbulent fluctuation part is modeled using Physics-Informed Neural Networks (PINNs), further enhancing the simulation accuracy.

The structure of the RANS-PINNs model is depicted below:

<figure class="half">
    <img src="./images/rans_pinns_structure.png" title="prediction result" width="500"/>
</figure>

### Dataset

Source: Numerical simulation flow field data around a two-dimensional cylinder, provided by Associate Professor Yu Jian's team at the School of Aeronautic Science and Engineering, Beihang University.

Data Description:
The data format is numpy's npy format with dimensions [300, 700, 10]. The first two dimensions represent the length and width of the flow field, and the last dimension includes variables (x, y, u, v, p, uu, uv, vv, rho, nu), totaling 10 variables. Among these, x, y, u, v, p represent the x-coordinate, y-coordinate, x-direction velocity, y-direction velocity, and pressure of the flow field, respectively. uu, uv, vv are Reynolds-averaged statistical quantities, while rho is fluid density and nu is kinematic viscosity.

Dataset Download Link:
[periodic_hill.npy](https://download.mindspore.cn/mindscience/mindflow/dataset/periodic_hill_2d/)

## Quick Start

### Training Method 1: Calling `train.py` Script from Command Line

This model is trained on a single device with a single GPU card. To start training the RANS-PINNs model, simply execute train.py. Before training, make sure to set parameters related to data loading, saving, and training in `rans.yaml`:

```bash
python -u train.py --mode GRAPH --device_target GPU --device_id 0 --config_file_path ./configs/rans.yaml
```

Where,

`--mode` specifies the running mode: 'GRAPH' for static graph mode and 'PYNATIVE' for dynamic graph mode (default is 'GRAPH');

`--device_target` specifies the target computing platform, which can be 'Ascend' or 'GPU' (default is 'GPU');

`--device_id` specifies the device ID to be used (default is 0);

`--config_file_path` specifies the path of the parameter file (default is './configs/rans.yaml').

### Training Method 2: Running Jupyter Notebook

You can use the Jupyter Notebook versions in both Chinese and English to run training and validation code line by line:

Chinese version: [rans_CN.ipynb](./rans_CN.ipynb)

English version: [rans.ipynb](./rans.ipynb)

## Visualization of Prediction Results

After training, execute eval.py to perform model inference based on the weight parameter file generated during training. This operation will predict the periodic hill flow field based on the training results.

```bash
python -u eval.py
```

The default output path for post-processing is './prediction_result', which can be modified in `rans.yaml`.

## Performance

|        Parameter         |        NPU               |    GPU       |
|:----------------------:|:--------------------------:|:---------------:|
|     Hardware         |     Ascend, Memory32G    |      NVIDIA V100, Memory32G       |
|     MindSpore version   |        2.0.0             |      2.0.0       |
| Dataset | [Periodic_hill](https://download.mindspore.cn/mindscience/mindflow/dataset/periodic_hill_2d/) | [Periodic_hill](https://download.mindspore.cn/mindscience/mindflow/dataset/periodic_hill_2d/) |
|  Parameters | 1.3e5 | 1.3e5 |
|  Training hyperparameters |  batch_size=4002, epochs=1600, steps_per_epoch=50 | batch_size=4002, epochs=1600, steps_per_epoch=50 |
|  Testing hyperparameters | batch_size=4002 | batch_size=4002 |
|  Optimizer | Adam | Adam |
|        Train loss      |        6.21e-4          |   5.30e-4       |
|        Validation loss      |        0.103          |   0.113           |
|        Speed          |     180ms/step        |    389ms/step  |

## Results Display

Below is a comparison between the predicted results of the RANS-PINNs model and the ground truth:

<figure class="half">
    <img src="./images/prediction_result.png" title="prediction result" width="500"/>
</figure>

The images display the distribution of lateral velocity and vertical velocity at different positions within the flow field. The lower image shows the ground truth, while the upper image displays the predicted values.

The following is a cross-velocity profile of the RANS-PINNs model:

<figure class="harf">
    <img src="./images/speed_contour.png" title="prediction_result" width="500"/>
</figure>

where the blue line is the true value and the orange dashed line is the predicted value.

## Contributor

Gitee ID of code contributor: [Marc-Antoine-6258](https://gitee.com/Marc-Antoine-6258)

Email of code contributor: 775493010@qq.com