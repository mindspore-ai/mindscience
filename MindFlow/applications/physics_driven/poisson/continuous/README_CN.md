# 泊松方程

## 概述

本案例采用MindFlow流体仿真套件，基于物理驱动的PINNs (Physics Informed Neural Networks)方法，求解一维，二维和三维的泊松方程:

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

并且我们设定几何边界条件满足狄利克雷边界条件。

对于一维问题，本案例使用一维数轴区间作为求解域，对于二维问题，本例演示在矩形，圆形，三角形，L形和五边形区域求解方程，而对于三维问题，我们将在四面体，圆柱和圆锥区域内求解方程。

## 快速开始

### 训练方式一：在命令行调用`train.py`脚本

在命令行中输入以下命令，即可开始训练:

```bash
python train.py --geom_name disk --mode GRAPH --save_graphs_path ./graphs --device_target GPU --device_id 0 --config_file_path ./configs/poisson_cfg.yaml
```

其中， `--geom_name`表示几何形状的名称，可以选择`'interval'`，`'rectangle'`，`'disk'`，`'triangle'`，`'polygon'`，`'pentagon'`，`'tetrahedron'`，`'cylinder'`，`'cone'`，默认值`'disk'`；

`--mode`表示运行的模式，`'GRAPH'`表示静态图模式, `'PYNATIVE'`表示动态图模式，详见MindSpore官网，默认值'GRAPH'；

`--device_target`表示使用的计算平台类型，可以选择'Ascend'或'GPU'，默认值'GPU'；

`--device_id`表示使用的计算卡编号，可按照实际情况填写，默认值0；

`--ckpt_dir`表示模型保存的路径，默认值'./ckpt'；

`--n_epochs`表示训练epoch数量；

`--config_file_path`表示参数文件的路径，默认值'./configs/poisson_cfg.yaml'；

### 训练方式二：运行Jupyter Notebook

您可以使用中文版和英文版Jupyter Notebook逐行运行训练和验证代码。
