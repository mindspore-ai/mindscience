ENGLISH | [简体中文](README_CN.md)

# 2D Poisson equation with a point source

## Overview

In this case, MindFlow fluid simulation suite is used to solve the 2D Poisson equation with a point source based on the physical-driven PINNs (Physics Informed Neural Networks) method.
The equation is defined by
$$
\Delta u = - \delta(x-x_{src})\delta(y-y_{src}),
$$
where $(x_{src}, y_{src})$  is the coordinate corresponding to the point source position. The point source can be represented mathematically using the Dirac $\delta$ function
$$
\delta(x) = \begin{cases}
+\infty, & x = 0    \\
0,       & x \neq 0
\end{cases}
\qquad
\int_{-\infty}^{+\infty}\delta(x)dx = 1.
$$

## QuickStart

### Run Option 1: Call `train.py` from command line

```shell
python train.py --mode GRAPH --save_graphs_path ./graphs --device_target Ascend --device_id 0 --n_epochs 250 --config_file_path ./poisson_cfg.yaml
```

where:

`--mode` is the running mode. 'GRAPH' indicates static graph mode. 'PYNATIVE' indicates dynamic graph mode. You can refer to [MindSpore official website](https://www.mindspore.cn/docs/en/r2.0.0-alpha/design/dynamic_graph_and_static_graph.html) for details.Default 'GRAPH'.

`--save_graphs` indicates whether to save the computational graph. Default 'False'.

`--save_graphs_path` indicates the path to save the computational graph. Default './graphs'.

`--device_target` indicates the computing platform. You can choose 'Ascend' or 'GPU'. Default 'Ascend'.

`--device_id` indicates the index of NPU or GPU. Default 0.

`--n_epochs` indicates the number of training epochs. Default 250.

`--config_file_path` indicates the path of the parameter file. Default './poisson_cfg.yaml'；

### Run Option 2: Run Jupyter Notebook

You can use [Chinese](https://gitee.com/mindspore/mindscience/blob/master/MindFlow/applications/physics_driven/poisson/point_source/poisson_CN.ipynb) or [English](https://gitee.com/mindspore/mindscience/blob/master/MindFlow/applications/physics_driven/poisson/point_source/poisson.ipynb) Jupyter Notebook to run the training and evaluation code line-by-line.

## Results

![Poisson point source](images/result.png)

## Contributor

gitee id: huangxiang360729

email: <sahx@mail.ustc.edu.cn>
