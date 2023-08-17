[简体中文](README_CN.md) | ENGLISH

# CMA-ES combined with multi-objective gradient descent algorithm to train PINNS neural network

## Overview

### Overall background

PINNs training often needs to solve a highly nonlinear, non-convex optimization problem, and the setting of weight value in the loss function has a significant impact on the training effect of the network. In this case, the gradient-free optimization algorithm CMA-ES and multi-objective gradient optimization algorithm (mgda) are combined to overcome the problems of highly non-convex and gradient anomalies in optimization.

### Technical route

The overall framework of the algorithm is CMA-ES algorithm, in which a multi-objective gradient optimization algorithm is embedded.
The CMA-ES algorithm is mainly referred to the article [Neuroevolution Surpasses Stochastic Gradient Descent for Physics-Informed Neural Networks](https://arxiv.org/abs/2212.07624) and [cma code package](https://pypi.org/project/cma/), Mgda code reference [code](https://github.com/isl-org/MultiObjectiveOptimization).

The algorithm flow is as follows:

- step1: Build train_dataset for training in multi-objective gradient descent, and build train_dataset_loss for calculating the fitness of a given solution (loss).
- step2: Build the neural network model and optimizer
- step3: Select a certain number of parameter descendants randomly from the Gaussian distribution by CMA-ES algorithm
- step4: Randomly select a certain proportion of offspring from the screened offspring for multi-objective gradient descent, and replace the original offspring
- step5: Calculate the offspring fitness at this time
- step6: Update parameters in the CMA-ES algorithm according to the fitness of offspring and parameter offspring
- step7: Repeat step3 to step6 until the stop conditions are met

### Verification equation

The proposed neural network training method is verified on two equations, Burgers equation and Navier-Stokes equation, respectively.

#### Introduction to Burgers Equation

Burgers' equation is a nonlinear partial differential equation that simulates the propagation and reflection of shock waves.

The Burgers' equation has the following form:

$$
u_t + uu_x = \epsilon u_{xx}, \quad x \in[-1,1], t \in[0, T],
$$

where $\epsilon=0.01/\pi$, the left of the equal sign is the convection term, and the right is the dissipation term. In this case, the Dirichlet boundary condition and the initial condition of the sine function are used. The format is as follows:

$$
u(t, -1) = u(t, 1) = 0,
$$

$$
u(0, x) = -sin(\pi x).
$$

In this case, the PINNs method is used to learn the mapping $(x, t) \mapsto u$ from position and time to corresponding physical quantities. So that the solution of Burgers' equation is realized.

#### Introduction to Navier-Stokes equation

Flow around a cylinder, that is, low speed unsteady flow around a two-dimensional cylinder, the flow characteristics are related to the Reynolds number `Re`.

The Navier-Stokes equation, referred to as `N-S` equation, is a classical partial differential equation in the field of fluid mechanics. In the case of viscous incompressibility, the dimensionless `N-S` equation has the following form:

$$
\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} = 0
$$

$$
\frac{\partial u} {\partial t} + u \frac{\partial u}{\partial x} + v \frac{\partial u}{\partial y} = - \frac{\partial p}{\partial x} + \frac{1} {Re} (\frac{\partial^2u}{\partial x^2} + \frac{\partial^2u}{\partial y^2})
$$

$$
\frac{\partial v} {\partial t} + u \frac{\partial v}{\partial x} + v \frac{\partial v}{\partial y} = - \frac{\partial p}{\partial y} + \frac{1} {Re} (\frac{\partial^2v}{\partial x^2} + \frac{\partial^2v}{\partial y^2})
$$

where `Re` stands for Reynolds number.

In this case, the PINNs method is used to learn the mapping from the location and time to flow field quantities to solve the `N-S` equation.

$$
(x, y, t) \mapsto (u, v, p)
$$

## Quick start

You can download dataset from [physics_driven/burgers_pinns/dataset](https://download.mindspore.cn/mindscience/mindflow/dataset/applications/physics_driven/burgers_pinns/dataset/)、[physics_driven/cylinder_flow_pinns/dataset](https://download.mindspore.cn/mindscience/mindflow/dataset/applications/physics_driven/flow_past_cylinder/dataset/)  for model evaluation. Save these dataset at `./dataset`.

### Run Option 1: Call `train.py` from command line

```shell
python --case burgers --mode GRAPH --device_target Ascend --device_id 0 --config_file_path ./configs/burgers.yaml
```

where:
`--case` represents the selection of cases, 'burgers' represents the selection of burgers equation, 'cylinder_flow' represents the selection of Navier-Stokes equation and cylinder flow datasets, 'periodic_hill' means selecting Reynolds mean Navier-Stokes equations to train the periodic hill dataset.

`--mode` is the running mode. 'GRAPH' indicates static graph mode. 'PYNATIVE' indicates dynamic graph mode. You can refer to [MindSpore official website](https://www.mindspore.cn/docs/en/r2.0.0-alpha/design/dynamic_graph_and_static_graph.html) for details.Default 'GRAPH'.

`--device_target` indicates the computing platform. You can choose 'Ascend' or 'GPU'. Default 'Ascend'.

`--device_id` indicates the index of NPU or GPU. Default 0.

`--config_file_path` indicates the path of the parameter file. Default './configs/burgers.yaml'.

### Run Option 2: Run Jupyter Notebook

You can use [Chinese](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/research/cma_es_mgda_CN.ipynb) or [English](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/research/cma_es_mgda.ipynb)Jupyter Notebook to run the training and evaluation code line-by-line.

## Result

### Verification effect on Burgers equation

PINNS composed of the same neural network (5-layer neural network with 20 neurons in each layer) are trained with 4000 epochs. The blue curve is the loss curve using this method (cma_es genetic algorithm combined with multi-objective gradient descent), and the orange curve is the loss curve using Adam. The effectiveness of this method is fully demonstrated:

![Burgers](images/burgers_loss_contrast.png)

After training with 4000 epochs, the model predicted values are shown in the figure below:

![Burgers](images/burgers_4000-result.jpg)

### Verification effect on Navier-Stokes equation

PINNS composed of the same neural network (5-layer neural network with 20 neurons in each layer) are trained with 4000 epochs. The blue curve is the loss curve using this method (cma_es genetic algorithm combined with multi-objective gradient descent), and the orange curve is the loss curve using Adam. The effectiveness of this method is fully demonstrated:

![cylinder_flow](images/cylinder_flow_loss_contrast.png)

After training with 4000 epochs, the predicted value of the model is matched with the true value as shown in the following figure:

![cylinder flow](images/cylinder_FlowField_4000.gif)

### Verification effect on Raynold-averaged Navier-Stokes equations

PINNS composed of the same neural network (-layer neural network with  neurons in each layer) are trained with  epochs. The blue curve is the loss curve using this method (cma_es genetic algorithm combined with multi-objective gradient descent), and the orange curve is the loss curve using Adam. The effectiveness of this method is fully demonstrated:

After training with  epochs, the predicted value of the model is matched with the true value as shown in the following figure:

![periodic_hill](images/periodic_hill_contrast.png)

After the training of 160 epochs, the predicted value of the model is matched with the true value as shown in the following figure:

![periodic_hill](images/periodic_hill_160.png)

## Performance

### Burgers equation

|        Parameter         |        Ascend               |    GPU       |
|:----------------------:|:--------------------------:|:---------------:|
|     Hardware         |     Ascend910A, Memory：32G；CPU: 2.6HZ, 192 cores      |      NVIDIA V100, Memory：32G       |
|     MindSpore version   |        2.0.0             |      2.0.0       |
|        Train loss      |        6.44e-4               |       7.28e-4       |
|        Validation loss      |        0.020              |       0.058       |
|        Speed          |     970ms/step        |    1330ms/step  |

### Navier-Stokes equations

|        Parameter         |        Ascend               |    GPU       |
|:----------------------:|:--------------------------:|:---------------:|
|     Hardware         |     Ascend910A, Memory32G；CPU: 2.6HZ, 192 cores      |      NVIDIA V100, Memory32G       |
|     MindSpore version   |        2.0.0             |      2.0.0       |
|        Train loss      |        3.46e-4               |       3.23e-4      |
|        Validation loss      |        0.091              |       0.124       |
|        Speed          |     1220ms/step        |    1150ms/step  |

### Raynold-averaged Navier-Stokes equations

|        Parameter         |        Ascend               |    GPU       |
|:----------------------:|:--------------------------:|:---------------:|
|     Hardware         |     Ascend910A, Memory32G；CPU: 2.6HZ, 192 cores      |      NVIDIA V100, Memory32G       |
|     MindSpore version   |        2.0.0             |      2.0.0       |
|        Train loss      |        8.92e-05               |       1.06e-4      |
|        Validation loss      |        0.115              |       0.125       |
|        Speed          |     1650ms/step        |    2250ms/step  |

## Contributor

gitee id: [lin109](https://gitee.com/lin109)

email: 1597702543@qq.com