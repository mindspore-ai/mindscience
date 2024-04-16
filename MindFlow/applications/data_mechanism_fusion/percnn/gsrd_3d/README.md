[中文版](README_CN.md) | ENGLISH

# Solving 3D reaction-diffusion Equation by Using PeRCNN

## Overview

Reaction-diffusion equation is an important partial derivative equation that has been broadly used to describe a variety of phenomenon in physics, biology and chemistry.

$$
u_t = \mu_u \Delta u - u{v*2} + F(1-v)
$$

$$
v_t = \mu_v \Delta v + u{v*2} + (F+\kappa)v
$$

where,
$$
\mu_v = 0.1, \mu_u = 0.2, F = 0.025, \kappa = 0.055
$$

In this case, we will simulate the flow dynamics in 100 time steps (dt=0.5s) in a $ \Omega \times \tau = {[-50,50]}^3 \times [0,500] $ physical domain. The initial condition of the problem would go through gaussian noise and periodic BC is adpoted.

## Quick Start

Download the data from [link](https://download.mindspore.cn/mindscience/mindflow/dataset/applications/data_mechanism_fusion/PeRCNN) and save them in `./dataset`.

### call `train.py` to carry out data-driven PeRCNN simulation

```shell
python train.py --mode GRAPH --device_target Ascend --device_id 0 --config_file_path ./configs/percnn_3d_rd.yaml
```

where,

`--mode` is the running mode. 'GRAPH' indicates static graph mode. 'PYNATIVE' indicates dynamic graph mode. Default 'GRAPH'.

`--device_target` represents the type of computing platform used, which can be selected as 'Ascend' or 'GPU', default 'Ascend';

`--device_id` represents the calculation card number used, which can be filled in according to the actual situation, default 0;

`--config_file_path` represents the parameter and path control file, default './configs/percnn_3d_rd.yaml';

## 结果展示

![3d GS RD](images/result.jpg)

## 性能

|        Parameter         |        Ascend              |    GPU       |
|:----------------------:|:--------------------------:|:---------------:|
|     Hardware         |     memory 32G, CPU: 2.6GHz, 192 cores      |      NVIDIA V100 memory 32G       |
|     MindSpore version   |        2.1.0             |      2.1.0       |
|     dataset      |      [3DRD](https://download.mindspore.cn/mindscience/mindflow/dataset/applications/data_mechanism_fusion/PeRCNN)             |     [3DRD](https://download.mindspore.cn/mindscience/mindflow/dataset/applications/data_mechanism_fusion/PeRCNN)        |
|     parameters      |          10078         |         10078    |
|     training config    |        batch_size=1, steps_per_epoch=1, epochs=10000              |     batch_size=1, steps_per_epoch=1, epochs=10000        |
|     test config      |        batch_size=1,steps=1              |     batch_size=1,steps=1        |
|     optimizer      |        Adam              |     Adam       |
|        training loss(MSE)      |        2e-3              |       2e-3      |
|        test loss(RMSE)      |        6e-2               |       6e-2    |
|        speed(ms/step)          |     5000       |    3500 |

## Contributor

gitee id: [chengzrz](https://gitee.com/chengzrz)

email: czrzrichard@gmail.com