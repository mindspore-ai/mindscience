# 泊松方程

## 概述

本案例采用 MindFlow 流体仿真套件，基于物理驱动的 PINNs (Physics Informed Neural Networks)方法，求解一维，二维和三维的泊松方程:

泊松方程是理论物理中广泛应用的偏微分方程，其形式如下:

$$
\Delta u = f
$$

其中，$\Delta$是拉普拉斯算子，$u$ 和$f$ 是定义在流形上的实值或复值函数。通常，$f$ 是已知的，而$\varphi$ 是需要求解的。

在本案例中，对于一维泊松方程，我们有:

$$
\Delta u = -\sin(4\pi x),
$$

对于二维泊松方程，我们有:

$$
\Delta u = -\sin(4\pi x)\sin(4\pi y),
$$

对于三维泊松方程，我们有:

$$
\Delta u = -\sin(4\pi x)\sin(4\pi y)\sin(4\pi z),
$$

并且我们可以通过修改 yaml 中的 BC_type 来选择使用的边界条件，目前可选择的有:迪利克雷边界、罗宾边界以及周期性边界条件。

对于一维问题，本案例使用一维数轴区间作为求解域，对于二维问题，本例演示在矩形，圆形，三角形，L 形和五边形区域求解方程，而对于三维问题，我们将在四面体，圆柱和圆锥区域内求解方程。

## 快速开始

### 训练方式一：在命令行调用 `train.py`脚本

在命令行中输入以下命令，即可开始训练:

```bash
python train.py --geom_name disk --mode GRAPH --save_graphs_path ./graphs --device_target GPU --device_id 0 --config_file_path ./configs/poisson_cfg.yaml
```

其中， `--geom_name`表示几何形状的名称，可以选择 `'interval'`，`'rectangle'`，`'disk'`，`'triangle'`，`'polygon'`，`'pentagon'`，`'tetrahedron'`，`'cylinder'`，`'cone'`，默认值 `'disk'`；

`--mode`表示运行的模式，`'GRAPH'`表示静态图模式, `'PYNATIVE'`表示动态图模式，详见 MindSpore 官网，默认值'GRAPH'；

`--device_target`表示使用的计算平台类型，可以选择'Ascend'或'GPU'，默认值'GPU'；

`--device_id`表示使用的计算卡编号，可按照实际情况填写，默认值 0；

`--ckpt_dir`表示模型保存的路径，默认值'./ckpt'；

`--n_epochs`表示训练 epoch 数量；

`--config_file_path`表示参数文件的路径，默认值'./configs/poisson_cfg.yaml'；

可通过修改 poisson_cfg.yaml 中的 BC_type 来选择所使用的边界条件函数，同时，以下是在各边界条件以及采样区间下，根据案例参数所制表格：

|    边界条件    |                                               Dirichlet                                               |                                                robin                                                 |                                               Periodic                                                |
| :------------: | :---------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------: |
|    硬件资源    |                                       GPU: 1\*T4,<br /> CPU: 4                                        |                                       GPU: 1\*T4,<br /> CPU: 4                                       |                                       GPU: 1\*T4,<br /> CPU: 4                                        |
| MindSpore 版本 |                                                 2.2.0                                                 |                                                2.2.0                                                 |                                                 2.2.0                                                 |
|    训练参数    | n_epochs=50,<br />batch_size=5000,<br />BC_size=1000000, <br />domain_size=1000000，<br />max_lr=5e-4 | n_epochs=50,<br />batch_size=5000,<br />BC_size=2000000,<br />domain_size=2000000，<br />max_lr=5e-4 | n_epochs=100,<br />batch_size=5000,<br />BC_size=2000000,<br />domain_size=2000000，<br />max_lr=5e-4 |
|    测试参数    |                                            batch_size=5000                                            |                                           batch_size=5000                                            |                                            batch_size=5000                                            |
|    采样区间    |                                               rectangle                                               |                                              rectangle                                               |                                               rectangle                                               |
|    训练误差    |                                                0.00818                                                |                                               0.00258                                                |                                                0.01809                                                |
|     优化器     |                                                 Adam                                                  |                                                 Adam                                                 |                                                 Adam                                                  |
|    测试误差    |                                                0.0028                                                 |                                                0.0812                                                |                                                0.8790                                                 |
| 单个 step 时间 |                                                163.4ms                                                |                                               286.1ms                                                |                                                146.0ms                                                |
|     参数量     |                                                1688002                                                |                                               1688002                                                |                                                1688002                                                |

其他采样区间的训练参数以及结果[在此](./evaluation_table.md)

### 训练方式二：运行 Jupyter Notebook

您可以使用[中文版](./poisson_CN.ipynb)和[英文版](./poisson.ipynb) Jupyter Notebook 逐行运行训练和验证代码。

### 测试结果

<p align = "center">
<img src="./images/dirichlet-rectangle.png" width="180"/>
<img src="./images/periodic-rectangle.png" width="180"/>
<img src="./images/robin-rectangle.png" width="180"/>
<img src="./images/dirichlet-tetrahedron.png" width="180"/>
</p>

## 性能

|      参数      |                     Ascend                      |                       GPU                       |
| :------------: | :---------------------------------------------: | :---------------------------------------------: |
|    硬件资源    |                Ascend, 显存 32G                 |              NVIDIA V100, 显存 32G              |
| MindSpore 版本 |                     >=2.0.0                     |                     >=2.0.0                     |
|     参数量     |                       1e5                       |                       1e5                       |
|    训练参数    | batch_size=5000, steps_per_epoch=200, epochs=50 | batch_size=5000, steps_per_epoch=200, epochs=50 |
|    测试参数    |                 batch_size=5000                 |                 batch_size=5000                 |
|     优化器     |                      Adam                       |                      Adam                       |
| 训练损失(MSE)  |                      0.001                      |                      0.001                      |
| 验证损失(RMSE) |                      0.01                       |                      0.01                       |
| 速度(ms/step)  |                       0.3                       |                        1                        |
