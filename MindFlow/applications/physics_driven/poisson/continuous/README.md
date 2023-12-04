# Poisson Equation

## Overview

This case study uses the MindFlow fluid simulation suite and the Physics Informed Neural Networks (PINNs) method to solve the one-dimensional, two-dimensional, and three-dimensional Poisson equations:

The Poisson equation is a widely used partial differential equation in theoretical physics, given by:

$$
\Delta u = f
$$

where $\Delta$ is the Laplacian operator, and $u$ and $f$ are real or complex-valued functions defined on a manifold. Usually, $f$ is given, and $\varphi$ is sought.

In this case study, for the one-dimensional Poisson equation, we have:

$$
\Delta u = -\sin(4\pi x),
$$

for the two-dimensional Poisson equation, we have:

$$
\Delta u = -\sin(4\pi x)\sin(4\pi y),
$$

and for the three-dimensional Poisson equation, we have:

$$
\Delta u = -\sin(4\pi x)\sin(4\pi y)\sin(4\pi z),
$$

and we set the geometric boundary conditions to satisfy the Dirichlet boundary conditions.

For the one-dimensional problem, this case study uses the one-dimensional axis interval as the solution domain. For the two-dimensional problem, it demonstrates solving the equation in rectangular, circular, triangular, L-type, and pentagonal regions. For the three-dimensional problem, we solve the equation in tetrahedral, cylindrical, and conical regions.

## Quick Start

### Training Method 1: Call the `train.py` Script in Command Line

In the command line, enter the following command to start the training:

```bash
python train.py --geom_name disk --mode GRAPH --save_graphs_path ./graphs --device_target GPU --device_id 0 --config_file_path ./poisson_cfg.yaml
```

where `--geom_name` specifies the name of the geometric shape, and you can choose from `'interval'`, `'rectangle'`, `'disk'`, `'triangle'`, `'polygon'`, `'pentagon'`, `'tetrahedron'`, `'cylinder'`, `'cone'`, with the default value `'disk'`.

`--mode` specifies the running mode, with `'GRAPH'` indicating static graph mode, and `'PYNATIVE'` indicating dynamic graph mode. Refer to the MindSpore official website for more details, with the default value `'GRAPH'`.

`--device_target` specifies the computing platform type, and you can choose from `'Ascend'` or `'GPU'`, with the default value `'GPU'`.

`--device_id` specifies the device ID, which can be filled in according to the actual situation, with the default value `0`.

`--ckpt_dir` specifies the path to save the model, with the default value `'./ckpt'`.

`--n_epochs` specifies the number of training epochs.

`--config_file_path` specifies the path to the parameter file, with the default value `'./configs/poisson_cfg.yaml'`.

### Training Method 2: Run Jupyter Notebook

You can run the training and validation code line by line using the Chinese or English version of the Jupyter Notebook.

## Performance

| Parameter               | Ascend               | GPU                |
|:----------------------:|:--------------------------:|:---------------:|
| Hardware                | Ascend 32G           | NVIDIA V100 32G    |
| MindSpore version       | >=2.0.0                | >=2.0.0                   |
| Parameters              | 1e5                  | 1e5                   |
| Train Config            | batch_size=5000, steps_per_epoch=200, epochs=50 | batch_size=5000, steps_per_epoch=200, epochs=50 |
| Evaluation Config       | batch_size=5000      | batch_size=5000               |
| Optimizer               | Adam                 | Adam                   |
| Train Loss(MSE)         | 0.001                | 0.001             |
| Evaluation Error(RMSE)  | 0.01                 | 0.01              |
| Speed(ms/step)          | 0.3                  | 1                |