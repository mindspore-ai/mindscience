[ENGLISH](README.md) | 简体中文

# Meta-pinns

## Introduction

In different physical application scenarios, selecting an appropriate PINNs loss function still mainly relies on experience and manual design. In order to solve the above problems, Apostolos F Psarosa et al. proposed the Meta-PINNs algorithm in paper [Meta-learning PINN loss functions](https://www.sciencedirect.com/science/article/pii/S0021999122001838). This algorithm updates the hyperparameters acting on the loss function in a gradient descent manner during training, thereby training a set of hyperparameters suitable for a class of homogeneous partial differential equations.

### Background

In traditional neural network training, loss function selection is crucial to the convergence speed and performance of the model. Apostolos F Psarosa et al. proposed the idea of selecting the loss function as a hyperparameter and optimizing the loss function through meta-learning.

### Method

Specifically, the algorithm is mainly divided into internal optimization and external optimization in the meta-training stage. The algorithm process is as follows:

- Extracting parameterized partial differential equation (PDE) tasks from task distributions.
- In the internal optimization stage, use the currently learned loss function to iteratively solve the PINN several times, track the gradient of the learned loss parameters, and update the parameters of the PINN model.
- In the external optimization stage, update the loss function parameters based on the mean square error (MSE) of the final (optimized) PINN parameters.

After completing meta-training, the resulting learned loss function is used for meta-testing, i.e., solving unseen tasks until convergence.

The pseudocode is as follows:
![algorithm](images/algorithm.png)

The algorithm example diagram is as follows:
![algorithm_pic](images/algorithm_pic.png)

### Verification equation

This meta learning method is validated on one-dimensional Burgers equation, one-dimensional linearized Burgers equation, Navier Stokes equation, and  equation.

#### Introduction to the one-dimensional Burgers equation

The Burgers' equation is a nonlinear partial differential equation that simulates the propagation and reflection of shock waves.
The form of the Burgers equation is as follows:

$$
u_t + uu_x = \lambda u_{xx}, \quad x \in[-1,1], t \in[0, 1],
$$

The left side of the equal sign represents the convective term, the right side represents the dissipative term, and $ lambda $represents the dynamic viscosity coefficient, which is a parameter that changes during training. $is selected from the uniform distribution $U [0.001, 0.005].
This case uses the Dirichlet boundary condition and the initial condition of the sine function, as follows:

$$
u(t, -1) = u(t, 1) = 0,
$$

$$
u(0, x) = -sin(\pi x),
$$

$\lambda=0.01/\pi$is the validation status of this case.

#### Introduction to one-dimensional linearized Burgers equation

The one-dimensional linearized Burgers equation is an approximate form of the Burgers equation, used to describe compressible flow in fluids or gases. It simplifies the model by linearizing the nonlinear term of the Burgers equation.

The form of the one-dimensional linearized Burgers equation is as follows:

$$
u_t + u_x = \lambda u_{xx}, \quad x \in[-1.5, 4.5], t \in[0, 2],
$$

The dynamic viscosity coefficient is a parameter that changes during training, selected from a uniform distribution of $U[0.01, 0.03]$. The initial conditions used in this case are as follows:

$$
u(x, 0) = 10 * e^{-(2x)^2},
$$

$\lambda=0.02$ is the validation status of this case.

#### Introduction to Navier-Stokes Equations

The flow around a cylinder, specifically the two-dimensional unsteady flow past a circular cylinder, exhibits characteristics that are dependent on the Reynolds number (Re).

The Navier-Stokes equations, abbreviated as N-S equations, are a set of classic partial differential equations in the field of fluid mechanics. In the case of incompressible viscous flow, the dimensionless form of the N-S equations is given as follows:

$$
\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} = 0
$$

$$
\frac{\partial u} {\partial t} + u \frac{\partial u}{\partial x} + v \frac{\partial u}{\partial y} = - \frac{\partial p}{\partial x} + \frac{1} {Re} (\frac{\partial^2u}{\partial x^2} + \frac{\partial^2u}{\partial y^2})
$$

$$
\frac{\partial v} {\partial t} + u \frac{\partial v}{\partial x} + v \frac{\partial v}{\partial y} = - \frac{\partial p}{\partial y} + \frac{1} {Re} (\frac{\partial^2v}{\partial x^2} + \frac{\partial^2v}{\partial y^2})
$$

Here, Re represents the Reynolds number, which is a parameter varying during training and selected from a uniform distribution $U[90, 110]$. This case study is validated using the periodic_hill dataset with $\lambda=100$.

#### The Reynolds-averaged Navier-Stokes equation

The Reynolds-averaged Navier-Stokes equation is used to simulate the periodic mountain flow problem in the field of fluid mechanics and meteorology, studying the behavior of air or fluid flow over periodic mountain topography. The Reynolds-averaged momentum equation is given by:

$$\rho \bar{u}_j \frac{\partial \bar{u}_i}{\partial x_j}=\rho \bar{f}_i + \frac{\partial}{\partial x_j}\left[-\bar{p} {\delta \_ {i j}+}\mu\left(\frac{\partial \bar{u}_i}{\partial x_j}+\frac{\partial \bar{u}_j}{\partial x_i}\right)-\rho \overline{u_i^{\prime} u_j^{\prime}}\right]$$

Here, $\rho$ represents the fluid density, which is a parameter with varying values during training, selected from a uniform distribution $U[0.8, 1.2]$. For this particular case, $\rho=1.0$ serves as the validation scenario.

## Quick Start

### Training Method 1: Using the `train.py` Script in Command Line

```shell
python train.py --case burgers --mode GRAPH --device_target Ascend --device_id 0 --config_file_path ./configs/burgers.yaml
```

Where:

- `--case` indicates the choice of the case, which includes burgers, l_burgers, periodic_hill, cylinder_flow, with burgers being the default.
- `--mode` indicates the operating mode, where 'GRAPH' represents static graph mode, 'PYNATIVE' represents dynamic graph mode. For more details, refer to the [MindSpore official website](https://www.mindspore.cn/docs/zh-CN/r2.0.0-alpha/design/dynamic_graph_and_static_graph.html?highlight=pynative). The default value is 'GRAPH'.
- `--device_target` indicates the type of computing platform to be used, which can be 'Ascend' or 'GPU', with 'Ascend' being the default.
- `--device_id` indicates the ID of the computing card to be used, which can be filled in according to the actual situation, with the default value being 0.
- `--config_file_path` indicates the path of the parameter file, with the default value being './configs/burgers.yaml'.

### Training Method 2: Running Jupyter Notebook

You can run and validate the training code line by line using the [Chinese version](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/research/) and [English version](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/research/) of the Jupyter Notebook.

## Results Display

### Validation Results for Equations

#### Burgers Equation

![burgersl2](./images/burgers_l2.png)

#### Navier-Stokes Equation

![cylinder_flow](./images/cylinder_flow_l2.png)

#### The Reynolds-averaged Navier-Stokes equation

![rans](./images/periodic_hill_l2.png)

#### One-Dimensional Linearized Burgers Equation

![l_burgersl2](./images/lburgers_l2.png)

## Performance

### Burgers Equation

|         Parameter         |                             NPU                              |                             GPU                              |
| :-----------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|    Hardware Resources     |                     Ascend, 32GB memory                      |                   NVIDIA V100, 32GB memory                   |
|     MindSpore Version     |                            2.0.0                             |                            2.0.0                             |
|          Dataset          | [Burgers Dataset](https://download.mindspore.cn/mindscience/mindflow/dataset/applications/physics_driven/burgers_pinns/) | [Burgers Dataset](https://download.mindspore.cn/mindscience/mindflow/dataset/applications/physics_driven/burgers_pinns/) |
|        Parameters         |                             2751                             |                             2751                             |
|    Training Parameters    |       batch_size=8192, steps_per_epoch=1, epochs=10000       |       batch_size=8192, steps_per_epoch=1, epochs=10000       |
|    Testing Parameters     |                   batch_size=8192,steps=1                   |                  batch_size=8192,steps=1                  |
|         Optimizer         |                             Adam                             |                             Adam                             |
|    Regular PINNs (MSE)    |                           3.97e-2                            |                           4.46e-2                            |
| Meta-learning PINNs (MSE) |                           9.76e-3                            |                           8.69e-3                            |
|      Speed (ms/step)      |                              91                              |                              84                              |

### Navier-Stokes Equation

|         Parameter         |                             NPU                              |                             GPU                              |
| :-----------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|    Hardware Resources     |                     Ascend, 32GB memory                      |                   NVIDIA V100, 32GB memory                   |
|     MindSpore Version     |                            2.0.0                             |                            2.0.0                             |
|          Dataset          | [Cylinder Flow Dataset](https://download.mindspore.cn/mindscience/mindflow/dataset/applications/physics_driven/flow_past_cylinder/dataset/) | [Cylinder Flow Dataset](https://download.mindspore.cn/mindscience/mindflow/dataset/applications/physics_driven/flow_past_cylinder/dataset/) |
|        Parameters         |                            17411                             |                            17411                             |
|    Training Parameters    |       batch_size=8192, steps_per_epoch=2, epochs=10000       |       batch_size=8192, steps_per_epoch=2, epochs=10000       |
|    Testing Parameters     |                   batch_size=8192, steps=2                   |                   batch_size=8192, steps=2                   |
|         Optimizer         |                             Adam                             |                             Adam                             |
|    Regular PINNs (MSE)    |                           2.67e-2                            |                           2.79e-2                            |
| Meta-learning PINNs (MSE) |                           1.82e-2                            |                           1.75e-2                            |
|      Speed (ms/step)      |                             385                              |                             375                              |

### The Reynolds-averaged Navier-Stokes equation

|         Parameter         |                             NPU                              |                             GPU                              |
| :-----------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|    Hardware Resources     |                     Ascend, 32GB memory                      |                   NVIDIA V100, 32GB memory                   |
|     MindSpore Version     |                            2.0.0                             |                            2.0.0                             |
|          Dataset          | [Periodic Hill Dataset](https://download.mindspore.cn/mindscience/mindflow/dataset/periodic_hill_2d/) | [Periodic Hill Dataset](https://download.mindspore.cn/mindscience/mindflow/dataset/periodic_hill_2d/) |
|        Parameters         |                            17383                             |                            17383                             |
|    Training Parameters    |       batch_size=1000, steps_per_epoch=52, epochs=250        |       batch_size=1000, steps_per_epoch=52, epochs=250        |
|    Testing Parameters     |                  batch_size=20000, steps=3                  |                  batch_size=20000, steps=3                  |
|         Optimizer         |                             Adam                             |                             Adam                             |
|    Regular PINNs (MSE)    |                           1.63e-1                            |                           1.58e-1                            |
| Meta-learning PINNs (MSE) |                           1.31e-1                            |                           1.22e-1                            |
|      Speed (ms/step)      |                             8646                             |                             8703                             |

### One-Dimensional Linearized Burgers Equation

|         Parameter         |                        NPU                         |                        GPU                        |
| :-----------------------: | :------------------------------------------------: | :-----------------------------------------------: |
|    Hardware Resources     |                Ascend, 32GB memory                 |             NVIDIA V100, 32GB memory              |
|     MindSpore Version     |                       2.0.0                        |                       2.0.0                       |
|          Dataset          |                 [L_Burgers Dataset](https://download-mindspore.osinfra.cn/mindscience/mindflow/dataset/applications/research/meta-pinns/linear.npz)                  |                 [L_Burgers Dataset](https://download-mindspore.osinfra.cn/mindscience/mindflow/dataset/applications/research/meta-pinns/linear.npz)                 |
|        Parameters         |                        2751                        |                       2751                        |
|    Training Parameters    | batch_size=10000, steps_per_epoch=2, epochs=10000 | batch_size=10000, steps_per_epoch=2, epochs=10000 |
|    Testing Parameters     |             batch_size=10000, steps=2              |            batch_size=10000, steps=2             |
|         Optimizer         |                        Adam                        |                       Adam                        |
|    Regular PINNs (MSE)    |                      1.26e-3                       |                      1.23e-3                      |
| Meta-learning PINNs (MSE) |                      9.83e-4                       |                      8.56e-4                      |
|      Speed (ms/step)      |                        467                         |                        474                        |

## Contributor

gitee id: [lin109](https://gitee.com/lin109)

email: 1597702543@qq.com