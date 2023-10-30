ENGLISH | [简体中文](README_CN.md)

# Kovasznay Flow

## Overview

Kovasznay flow is an exact solution to the Navier-Stokes equations and is widely used in the field of fluid mechanics as a benchmark problem. It finds applications in areas such as aerodynamics and computational fluid dynamics. This example demonstrates how to use MindFlow, a fluid simulation toolkit, and the Physics Informed Neural Networks (PINNs) method to solve the Kovasznay flow problem.

## Quick Start

### Training Option 1: Run the `train.py` script from the command line

```shell
python train.py --mode GRAPH --device_target GPU --device_id 0 --config_file_path ./configs/kovasznay_cfg.yaml
```

where,
`--mode` indicates the running mode, with 'GRAPH' representing static graph mode and 'PYNATIVE' representing dynamic graph mode. Refer to the [MindSpore documentation](https://www.mindspore.cn/docs/zh-CN/r2.0.0-alpha/design/dynamic_graph_and_static_graph.html?highlight=pynative) for more details. The default value is 'GRAPH'.

`--device_target` indicates the target computing platform, with options of 'Ascend' or 'GPU'. The default value is 'Ascend'.

`--device_id` indicates the device ID to use. The default value is 0.

`--config_file_path` indicates the path to the configuration file. The default value is './configs/kovasznay_cfg.yaml'.

### Training Option 2: Run Jupyter Notebook

You can also run the Jupyter Notebook for this example, available in [Chinese version](./kovasznay_CN.ipynb) and [English version](./kovasznay.ipynb), and execute the training and validation code line by line.

## Results

![Kovasznay PINNs](images/result.png)

## Performance

|     Parameter     |             Ascend             |                 GPU                 |
| :---------------: | :----------------------------: | :---------------------------------: |
|     Hardware      | Ascend | NVIDIA RTX 3090, 24G；CPU: 40 cores |
| MindSpore version |            2.0.dev             |                 2.0                 |
|    train loss     |             6.5e-5             |               4.4e-5                |
|    valid loss     |              4e-3              |                3e-3                 |
|       speed       |           5.8s/epoch           |             12.8s/epoch             |

## Contributor

liangjiaming2023
