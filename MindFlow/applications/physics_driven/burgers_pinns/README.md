# Solve Burgers' Equation based on PINNs

## Overview

Computational fluid dynamics is one of the most important techniques in the field of fluid mechanics in the 21st century. The flow analysis, prediction and control can be realized by solving the governing equations of fluid mechanics by numerical method. Traditional finite element method (FEM) and finite difference method (FDM) are inefficient because of the complex simulation process (physical modeling, meshing, numerical discretization, iterative solution, etc.) and high computing costs. Therefore, it is necessary to improve the efficiency of fluid simulation with AI.

In recent years, while the development of classical theories and numerical methods with computer performance tends to be smooth, machine learning methods combine a large amount of data with neural networks realize the flow field's fast simulation. These methods can obtain the accuracy close to the traditional methods, which provides a new idea for flow field solution.

Burgers' equation is a nonlinear partial differential equation that simulates the propagation and reflection of shock waves. It is widely used in the fields of fluid mechanics, nonlinear acoustics, gas dynamics et al. It is named after Johannes Martins Hamburg (1895-1981). In this case, MindFlow fluid simulation suite is used to solve the Burgers' equation in one-dimensional viscous state based on the physical-driven PINNs (Physics Informed Neural Networks) method.

## Problem Description

The form of Burgers' equation is as follows:

$$
u_t + uu_x = \epsilon u_{xx}, \quad x \in[-1,1], t \in[0, T],
$$

where $\epsilon=0.01/\pi$, the left of the equal sign is the convection term, and the right is the dissipation term. In this case, the Dirichlet boundary condition and the initial condition of the sine function are used. The format is as follows:

$$
u(t, -1) = u(t, 1) = 0,
$$

$$
u(0, x) = -sin(\pi x),
$$

In this case, the PINNs method is used to learn the mapping $(x, t) \mapsto u$ from position and time to corresponding physical quantities. So that the solution of Burgers' equation is realized.

## Technology Path

MindFlow solves the problem as follows:

1. Training Dataset Construction.
2. Neural Network Construction.
3. Problem Modeling.
4. Model Training.
5. Model Evaluation and Visualization.

## Training Dataset Construction

In this case, random sampling is performed according to the solution domain, initial condition and boundary value condition to generate training data sets and test data sets. The specific settings are as follows:

```python
from mindflow.data import Dataset
from mindflow.geometry import Interval, TimeDomain, GeometryWithTime
from mindflow.geometry import generate_sampling_config


def create_training_dataset(config):
    """create training dataset by online sampling"""
    geom_config = config["geometry"]
    data_config = config["data"]

    time_interval = TimeDomain("time", geom_config["time_min"], geom_config["time_max"])
    spatial_region = Interval("domain", geom_config["coord_min"], geom_config["coord_max"])
    region = GeometryWithTime(spatial_region, time_interval)
    region.set_sampling_config(generate_sampling_config(data_config))

    geom_dict = {region: ["domain", "IC", "BC"]}
    dataset = Dataset(geom_dict)

    return dataset
```

## Neural Network Construction

This example uses a simple fully-connected network with a depth of 6 layers and the activation function is the `tanh` function.

```python
from mindflow import MultiScaleFCCell

model = MultiScaleFCCell(in_channels=2,
                         out_channels=1,
                         layers=6,
                         neurons=128,
                         residual=True,
                         act='tan',
                         num_scales=1)

# define optimizer
optimizer = nn.Adam(model.trainable_params(), config["optimizer"]["initial_lr"])
```

## Problem Modeling

The following `Burgers1D` defines the burgers' problem. The `sympy` is used for delineating partial differential equations in symbolic forms and computing all equations' loss. Specifically, it includes 3 parts: governing equation, initial condition and boundary conditions.

```python
from mindspore import nn
from mindspore import dtype as mstype
from mindflow.pde import Burgers, sympy_to_mindspore


class Burgers1D(Burgers):
    def __init__(self, model, loss_fn=nn.MSELoss()):
        super(Burgers1D, self).__init__(model, loss_fn=loss_fn)
        self.ic_nodes = sympy_to_mindspore(self.ic(), self.in_vars, self.out_vars)
        self.bc_nodes = sympy_to_mindspore(self.bc(), self.in_vars, self.out_vars)

    def ic(self):
        ic_eq = self.u + sympy.sin(np.pi * self.x)
        equations = {"ic": ic_eq}
        return equations

    def bc(self):
        bc_eq = self.u
        equations = {"bc": bc_eq}
        return equations

    def get_loss(self, pde_data, ic_data, bc_data):
        pde_res = self.parse_node(self.pde_nodes, inputs=pde_data)
        pde_loss = self.loss_fn(pde_res[0], Tensor(np.array([0.0]), mstype.float32))

        ic_res = self.parse_node(self.ic_nodes, inputs=ic_data)
        ic_loss = self.loss_fn(ic_res[0], Tensor(np.array([0.0]), mstype.float32))

        bc_res = self.parse_node(self.bc_nodes, inputs=bc_data)
        bc_loss = self.loss_fn(bc_res[0], Tensor(np.array([0.0]), mstype.float32))

        return pde_loss + ic_loss + bc_loss
```

## Model Training

With MindSpore version >= 2.0.0, we can use the functional programming for training neural networks.

```python
def train():
    problem = Burgers1D(model)

    from mindspore.amp import DynamicLossScaler, auto_mixed_precision, all_finite
    if is_ascend:
        loss_scaler = DynamicLossScaler(1024, 2, 100)
        auto_mixed_precision(model, 'O1')
    else:
        loss_scaler = None

    # the loss function receives 3 data sources: pde, ic and bc
    def forward_fn(pde_data, ic_data, bc_data):
        loss = problem.get_loss(pde_data, ic_data, bc_data)
        if is_ascend:
            loss = loss_scaler.scale(loss)
        return loss

    grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=False)

    # using jit function to accelerate training process
    @jit
    def train_step(pde_data, ic_data, bc_data):
        loss, grads = grad_fn(pde_data, ic_data, bc_data)
        if is_ascend:
            loss = loss_scaler.unscale(loss)
            if all_finite(grads):
                grads = loss_scaler.unscale(grads)

        loss = ops.depend(loss, optimizer(grads))
        return loss

    steps = config["train_steps"]
    sink_process = mindspore.data_sink(train_step, train_dataset, sink_size=1)
    model.set_train()
    for step in range(steps + 1):
        time_beg = time.time()
        cur_loss = sink_process()
        if step % 100 == 0:
            print(f"loss: {cur_loss.asnumpy():>7f}")
            print("step: {}, time elapsed: {}ms".format(step, (time.time() - time_beg)*1000))
            calculate_l2_error(model, inputs, label, config["train_batch_size"])

time_beg = time.time()
train()
print("End-to-End total time: {} s".format(time.time() - time_beg))
```

The model results are as follows:

```python
loss: 0.000082
step: 14500, time elapsed: 51.5749454498291ms
    predict total time: 8.36324691772461 ms
    l2_error:  0.0047564595594806035
==================================================================================================
loss: 0.000081
step: 14600, time elapsed: 51.32031440734863ms
    predict total time: 9.172677993774414 ms
    l2_error:  0.005077659280011354
==================================================================================================
loss: 0.000099
step: 14700, time elapsed: 50.887107849121094ms
    predict total time: 4.549264907836914 ms
    l2_error:  0.0049527912578844506
==================================================================================================
loss: 0.000224
step: 14800, time elapsed: 51.982879638671875ms
    predict total time: 9.348869323730469 ms
    l2_error:  0.0055557865591330845
==================================================================================================
loss: 0.000080
step: 14900, time elapsed: 51.56064033508301ms
    predict total time: 8.392810821533203 ms
    l2_error:  0.004695746950148064
==================================================================================================
End-to-End total time: 897.3836033344269 s
```

## Model Evaluation and Visualization

After training, all data points in the flow field can be inferred. And related results can be visualized.

```python
from src import visual_result

visual_result(model, resolution=config["visual_resolution"])
```

![PINNs_results](images/result.jpg)
