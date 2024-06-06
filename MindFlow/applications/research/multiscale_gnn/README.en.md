# MultiscaleGNN

## Introduction

When solving the incompressible Navier-Stokes equations using projection method (or fractional step method), the projection step involves solving the large-scale Pressure Poisson Equation (PPE), which is typically the most computationally expensive and time-consuming step in the entire calculation process. A machine learning-based approach for solving the PPE problem is proposed, and a novel multi-scale Graph Neural Network (GNN) embedded solver is designed to accelerate the numerical solution of the incompressible Navier-Stokes equations. By replacing the traditional iterative solver for solving the PPE, the multi-scale GNN is seamlessly integrated into the numerical solution framework of the incompressible Navier-Stokes equations. In the multi-scale GNN framework, the original high-resolution graph corresponds to the discretized grid of the solution domain, graphs of the same resolution are connected through graph convolution operations, and graphs of different resolutions are connected through up-sampling and down-sampling operations. The well-trained multi-scale GNN serves as a universal PPE solver for a certain class of flow problems.

### Model framework

The model framework is as shown in the following diagram:

![MultiscaleGNN](images/MultiscaleGNN.png)

where

a. Numerical solution framework for the incompressible Navier-Stokes equations embedded with an ML-block (second-order explicit-implicit temporal discretization scheme)；

b. ML-block (multi-scale GNN), $\mathcal{G}^{1h}$ is the original high-resolution graph, $\mathcal{G}^{2h}$, $\mathcal{G}^{3h}$ and  $\mathcal{G}^{4h}$ are the low-resolution graph of level-2, level-3 and level-4, respectively, and the number represents the number of neurons in the corresponding layer.

## QuickStart

To run the current application, please make sure that MindSpore Version=2.2.X and MindFlow Version>=0.1.0. Dataset download link：[data_driven/multiscale_gnn/dataset](https://download-mindspore.osinfra.cn/mindscience/mindflow/dataset/applications/research/MultiScaleGNN/), Save the dataset to `./dataset`.

### Run Option 1: Call `train.py` from command line

```shell
python train.py --config_path ./config/config.yaml

```

where

`--config_path` indicates the path of the parameter file. Default "./config/config.yaml".

In the "./config/config.yaml" configuration file:

`grid_type` represents the type of grid, where 'structure' indicates structured grid selection, and 'unstructure' indicates unstructured grid selection, with a default value of 'structure'.

`activ_fun` represents the activation function type, with options including 'swish', 'elu', or 'gelu', with a default value of 'swish'.

`device_target` specifies the type of computing platform used, with options of 'CPU', 'GPU', or 'Ascend', with a default value of 'CPU'.

`device_id` represents the index of NPU or GPU. Default 0.

`mode` represents the running mode. 'GRAPH' indicates static graph mode. 'PYNATIVE' indicates dynamic graph mode. Default "GRAPH".

`lambda_p` represents the weight coefficient for the supervised pressure matching loss term, with a default value of 1.

`lambda_eq` represents the weight coefficient for the unsupervised equation residual loss term, with a default value of 1.

#### Note

Two model implementation methods are provided for the 'unstructure' and 'structure' grid types. In the 'unstructure' grid type model, graph convolution involves sparse matrix multiplication operations, while in the 'structure' model, graph convolution is computed according to the definition, avoiding the use of sparse matrix multiplication operations.

For `grid_type=unstructure`, the computing platform devices supported are 'CPU' and 'GPU';

For `grid_type=structure`, the computing platform devices supported are 'CPU', 'GPU', and 'Ascend';

### Run Option 2: Run Jupyter Notebook

You can use [Chinese](./multiscale_gnn_CN.ipynb) or [English](./multiscale_gnn.ipynb) Jupyter Notebook to run the training and evaluation code line-by-line.

## Solving case: Kolmogorov Flow

### Case description

$Re=1000$, driving force $\boldsymbol{f} = \sin(8y)\hat{\boldsymbol{x}} $, initial conditions $u_0=\sin(x)\cos(y), v_0=-\cos(x)\sin(y)$, with bi-periodic boundary conditions.

Training set: 100 samples within $t \in [0, 5]$; Test set: $t \in [0, 10]$.

By default, the swish activation function is used. Multiple models with different grid types and combinations of weight coefficients have been trained, specifically including:

grid_type=unstructure: lambda_p=1,lambda_eq=1; lambda_p=10,lambda_eq=1; lambda_p=20,lambda_eq=1; lambda_p=1,lambda_eq=10;

grid_type=structure: lambda_p=1,lambda_eq=1; lambda_p=10,lambda_eq=1; lambda_p=20,lambda_eq=1; lambda_p=50,lambda_eq=1; lambda_p=100,lambda_eq=1; lambda_p=1,lambda_eq=10;

These well-trained models are all saved in the './Savers/' folder.

### Training loss curves

grid_type=structure:

![structure_losses_curve](images/structure_losses_curve.png)

grid_type=unstructure:

![unstructure_losses_curve](images/unstructure_losses_curve.png)

### Test mode: Call `test.py` from command line

```shell
python test.py --grid_type=structure --activ_fun=swish --device=CPU --lambda_p=1 --lambda_eq=1 --plot_figure=1

```

### Performance comparison: $\lambda_p=20, \lambda_{eq}=1$

|         Parameters         |        NPU         |           GPU           |           CPU           |
| :------------------------: | :----------------: | :---------------------: | :---------------------: |
|          hardware          | Ascend(memory 32G) | NVIDIA V100(memory 32G) |        memory 32G       |
|     MindSpore version      |       2.2.10       |          2.2.10         |          2.2.10         |
|         data_size          |         50         |            50           |            50           |
|           epochs           |        2000        |           2000          |           2000          |
|         optimizer          |  AdamWeightDecay   |     AdamWeightDecay     |     AdamWeightDecay     |
| structure train loss(MSE)  |       8.93e-4      |         9.53e-4         |         5.68e-4         |
| structure test loss(MSE)   |       2.10e-3      |         1.18e-3         |         2.14e-3         |
|  structure speed(s/step)   |         4.3        |           2.2           |           8.5           |
| unstructure train loss(MSE) |         -         |         1.02e-3         |         9.31e-4         |
| unstructure test loss(MSE) |          -         |         2.27e-3         |         2.01e-3         |
|  unstructure speed(s/step) |          -         |           1.7           |           6.4           |

### Test result display：

`grid_type=structure`, `activ_fun=swish`, `lambda_p=20`, `lambda_eq=1`

![animate_pressure_field](images/animate_pressure_field.gif)

## Contributor

gitee id: [chenruilin2024](https://gitee.com/chenruilin2024)

email: chenruilin@isrc.iscas.ac.cn

## Reference

Chen R, Jin X, Li H. A machine learning based solver for pressure Poisson equations[J]. Theoretical and Applied Mechanics Letters, 2022, 12(5): 100362. https://www.sciencedirect.com/science/article/pii/S2095034922000423
