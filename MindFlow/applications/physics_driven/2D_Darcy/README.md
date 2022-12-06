# PINNS-based solution for 2D stabilized Darcy equations

## Overview

Darcy equation is a second-order, elliptic PDE (partial differential equation), which describes the flow through a porous medium at low speed. It is widely used in hydraulic engineering and petroleum engineering. The Darcy equation was originally formulated by Henry Darcy on the basis of experimental results of permeability  experiments in sandy soil, and later derived from the Navier-Stokes equation by Stephen Whitaker via the homogenization method.

It is difficult to obtain a generalized analytical solution of the Darcy equation for the permeability field of different fluids and the numerical method is usually used to solve the Darcy governing equation describing a specific scenario, and then the pressure field and velocity field of flow under the scenario are simulated. The numerical simulation results of Darcy flow can be used for further scientific research and engineering practice. Finite element method (FEM) for Darcy equation is designed to work with finite element spaces. In addition, for many problems of practical interest, some physical terms of the Darcy equation will be stabilized. The finite element method is designed in the standard finite element grid space. The higher accuracy required for numerical solution, the more fine the grid needs to be divided, and it costs larger time and storage overhead.

As the research of parallel algorithm for numerical solution tends to slow down, the method based on neural network has been developed and achieved the solution accuracy close to the traditional numerical method. In 2019, the Applied Mathematics team of Brown University proposed a Physics-informed Neural Networks (PINNs) and provided a complete code framework to construct PINNs for solving a wide variety of PDEs. In this case, MindFlow suite of fluid equations is used to solve the two-dimensional stabilized Darcy equation based on PINNs method.

## Problem Description

Considering the two-dimensional cube $\Omega=(0, 1)\times(0, 1)$, The boundary of the cube is $\Gamma$. Ignoring the effects of gravity, in the range of $\Omega$, the two-dimensional stabilized Darcy equation satisfied by the fluid pressure $p$ and velocity $u$ is as follows:

$$
\begin{align}
u + \nabla p &= 0, (x, y)\in\Omega\\
\nabla \cdot u &= f, (x, y)\in\Omega
\end{align}
$$

The Dirichlet boundary conditions are used in this case in the following form:

$$
\begin{align}
u_x &= -2 \pi cos(2 \pi x) cos(2 \pi y) &(x, y)\in\Gamma\\
u_y &= 2 \pi sin(2 \pi x) sin(2 \pi y) &(x, y)\in\Gamma\\
p &= sin(2 \pi x) cos(2 \pi y) &(x, y)\in\Gamma
\end{align}
$$

In which $f$ is **forcing function** in the Darcy equation. In this case, **forcing function** $f$ is used to learn the mapping $(x, y) \mapsto (u, p)$ from position to corresponding physical quantities when **forcing function** $f$ is $8 \pi^2 sin(2 \pi x)cos(2 \pi y)$. So that the solution of Darcy equation is realized.

## Technology Path

MindFlow solves 2D constant Darcy problem is as follows:

1. Random sampling is performed on the solution domain and boundary value conditions to create a training data set.
2. Constructing `Darcy` equations and Dirichlet boundary conditions.
3. Design the structure and parameters of the neural network.
4. Model Training.
5. Model Inference and Visualization.

## Training Example

### Configuration file

The overall configuration file is shown below, which defines key parameters such as problem domain boundary, neural network structure, learning rate, learning rate damping coefficient, training epoch, batch size, etc. File access path of checkpoint and visualization, case naming and other elements can also be configured here.

```yaml
geometry:
  coord_min: [0.0, 0.0]
  coord_max: [1.0, 1.0]
  axis_size: 101
data:
  domain:
    size: [256, 256]
    random_sampling: false
  BC:
    size: 65536
    random_sampling: false
model:
  name: FCNN_with_Residual
  input_size: 2
  output_size: 3
  activation: tanh
  neurons: 64
optimizer:
  lr: 0.001
train_epoch: 2000
train_batch_size: 8192
vision_path: "./images"
save_ckpt: false
save_ckpt_path: "./ckpt"
train_with_eval: false
visual_resolution: 100
```

### Training Dataset Construction

For the training dataset, this case conducts random sampling according to the problem domain and boundary conditions. The sampling configuration information is as follows, and samples are collected according to uniform distribution. The problem domain of cube is constructed, and then the known problem domain and boundary are sampled.

```python
def create_random_dataset(config, name):
    """create training dataset by online sampling"""
    # define geometry
    coord_min = config["geometry"]["coord_min"]
    coord_max = config["geometry"]["coord_max"]
    data_config = config["data"]

    flow_region = Rectangle(
        name,
        coord_min=coord_min,
        coord_max=coord_max,
        sampling_config=generate_sampling_config(data_config),
    )
    geom_dict = {flow_region: ["domain", "BC"]}

    # create dataset for train
    dataset = Dataset(geom_dict)
    return dataset
```

The validation dataset is constructed based on the real solution results.

```python
def get_test_data(config):
    """load labeled data for evaluation"""
    # acquire config
    coord_min = config["geometry"]["coord_min"]
    coord_max = config["geometry"]["coord_max"]
    axis_size = config["geometry"]["axis_size"]

    # set mesh
    axis_x = np.linspace(coord_min[0], coord_max[0], num=axis_size, endpoint=True)
    axis_y = np.linspace(coord_min[1], coord_max[1], num=axis_size, endpoint=True)

    mesh_x, mesh_y = np.meshgrid(axis_x, axis_y)

    input_data = np.hstack(
        (mesh_y.flatten()[:, None], mesh_x.flatten()[:, None])
    ).astype(np.float32)

    label = np.zeros((axis_size, axis_size, 3))
    for i in range(axis_size):
        for j in range(axis_size):
            in_x = axis_x[i]
            in_y = axis_y[j]
            label[i, j, 0] = -2 * PI * np.cos(2 * PI * in_x) * np.cos(2 * PI * in_y)
            label[i, j, 1] = 2 * PI * np.sin(2 * PI * in_x) * np.sin(2 * PI * in_y)
            label[i, j, 2] = np.sin(2 * PI * in_x) * np.cos(2 * PI * in_y)

    label = label.reshape(-1, 3).astype(np.float32)
    return input_data, label
```

### Modeling based on Darcy equation

`Problem` contains the governing equations and boundary conditions for solving 2D stabilized Darcy problem.

```python
class Darcy2D(Problem):
    r"""
    The steady-state 2D Darcy flow's equations with Dirichlet boundary condition

    Args:
      model (Cell): The solving network.
      domain_name (str): The corresponding column name of data which governed by maxwell's equation.
      bc_name (str): The corresponding column name of data which governed by boundary condition.
    """

    def __init__(self, model, domain_name=None, bc_name=None):
        super(Darcy2D, self).__init__()
        self.domain_name = domain_name
        self.bc_name = bc_name
        self.model = model
        self.grad = Grad(self.model)
        self.sin = ops.Sin()
        self.cos = ops.Cos()

        # constants
        self.PI = Tensor(PI, mstype.float32)

    def force_function(self, in_x, in_y):
        """"forcing function in Darcy Equation"""
        return 8 * self.PI**2 * self.sin(2 * self.PI * in_x) * self.cos(2 * self.PI * in_y)

    @ms_function
    def governing_equation(self, *output, **kwargs):
        """darcy equation"""
        u_x, u_y, _ = ops.split(output[0], axis=1, output_num=3)

        data = kwargs[self.domain_name]
        in_x = ops.Reshape()(data[:, 0], (-1, 1))
        in_y = ops.Reshape()(data[:, 1], (-1, 1))

        duxdx = ops.Cast()(self.grad(data, 0, 0, output[0]), mstype.float32)
        duydy = ops.Cast()(self.grad(data, 1, 1, output[0]), mstype.float32)
        dpdx = ops.Cast()(self.grad(data, 0, 2, output[0]), mstype.float32)
        dpdy = ops.Cast()(self.grad(data, 1, 2, output[0]), mstype.float32)

        loss_1 = -1 * (duxdx + duydy - self.force_function(in_x, in_y))
        loss_2 = 1 * (u_x + dpdx)
        loss_3 = 2 * self.PI * (u_y + dpdy)

        return ops.Concat(1)((loss_1, loss_2, loss_3))

    @ms_function
    def boundary_condition(self, *output, **kwargs):
        """Dirichlet boundary condition"""

        out_vars = output[0]
        u_x, u_y, pressure = ops.split(out_vars, axis=1, output_num=3)
        data = kwargs[self.bc_name]
        in_x = ops.Reshape()(data[:, 0], (-1, 1))
        in_y = ops.Reshape()(data[:, 1], (-1, 1))
        ux_boundary = -1 * (
            u_x - (-2 * self.PI * self.cos(2 * self.PI * in_x) * self.cos(2 * self.PI * in_y))
        )

        uy_boundary = 1 * (
            u_y - (2 * self.PI * self.sin(2 * self.PI * in_x) * self.sin(2 * self.PI * in_y))
        )

        p_boundary = (
            2 * self.PI * (pressure - self.sin(2 * self.PI * in_x) * self.cos(2 * self.PI * in_y))
        )
        return ops.Concat(1)((ux_boundary, uy_boundary, p_boundary))
```

### Neural Network Construction

This example uses a simple fully-connected network with a depth of 6 layers and the activation function is the tanh function.

```python
    model = FCSequential(
        in_channels=config["model"]["input_size"],
        out_channels=config["model"]["output_size"],
        neurons=config["model"]["neurons"],
        layers=config["model"]["layers"],
        residual=config["model"]["residual"],
        act=config["model"]["activation"],
        weight_init=config["model"]["weight_init"]
    )
```

### Define Losses

Instantiate `Contraints` as a loss.

```python
    # define problem and Constraints
    darcy_problem = [
        Darcy2D(model=model) for _ in range(flow_train_dataset.num_dataset)
    ]
    train_constraints = Constraints(flow_train_dataset, darcy_problem)
```

### Model Training

Invoke the `Solver` interface for model training and inference. pass optimizer, network model, loss function to instantiated `Solver`.

```python
    # optimizer
    params = model.trainable_params()
    optim = nn.Adam(params, learning_rate=config["optimizer"]["lr"])

    # solver
    solver = Solver(
        model,
        optimizer=optim,
        mode="PINNs",
        train_constraints=train_constraints,
        test_constraints=None,
        metrics={"l2": L2(), "distance": nn.MAE()},
        loss_scale_manager=DynamicLossScaleManager(),
    )

    # training

    # define callbacks
    callbacks = [LossAndTimeMonitor(len(flow_train_dataset))]

    if config["save_ckpt"]:
        ckpt_config = CheckpointConfig(save_checkpoint_steps=10, keep_checkpoint_max=2)
        ckpoint_cb = ModelCheckpoint(
            prefix="ckpt_darcy", directory=config["save_ckpt_path"], config=ckpt_config
        )
        callbacks += [ckpoint_cb]

    solver.train(
        epoch=config["train_epoch"], train_dataset=train_data, callbacks=callbacks
    )

    visual_result(model, config)
```

### Model training result

The model results are as follows:

```log
epoch time: 1137.334 ms, per step time: 142.167 ms
epoch: 1991 step: 8, loss is 0.12258543819189072
epoch time: 1117.534 ms, per step time: 139.692 ms
epoch: 1992 step: 8, loss is 0.10140248388051987
epoch time: 1155.795 ms, per step time: 144.474 ms
epoch: 1993 step: 8, loss is 0.030582554638385773
epoch time: 1146.296 ms, per step time: 143.287 ms
epoch: 1994 step: 8, loss is 0.10011541098356247
epoch time: 2366.454 ms, per step time: 295.807 ms
epoch: 1995 step: 8, loss is 0.24885042011737823
epoch time: 502.493 ms, per step time: 62.812 ms
epoch: 1996 step: 8, loss is 0.2624998688697815
epoch time: 2406.218 ms, per step time: 300.777 ms
epoch: 1997 step: 8, loss is 0.14243541657924652
epoch time: 322.166 ms, per step time: 40.271 ms
epoch: 1998 step: 8, loss is 0.17884144186973572
epoch time: 1493.348 ms, per step time: 186.669 ms
epoch: 1999 step: 8, loss is 0.07444168627262115
epoch time: 623.304 ms, per step time: 77.913 ms
epoch: 2000 step: 8, loss is 0.0650666207075119
================================Start Evaluation================================
Total prediction time: 0.0147705078125 s
l2_error, ux:  0.012288654921565733 , uy:  0.010292700640242451 , p:  0.008429703507824701
=================================End Evaluation=================================
epoch time: 1879.475 ms, per step time: 234.934 ms
End-to-End total time: 2449.483253479004 s
```

### Model Inference and Visualization

After training, all data points in the flow field can be inferred. And related results can be visualized.

![PINNS result](images/result.png)
