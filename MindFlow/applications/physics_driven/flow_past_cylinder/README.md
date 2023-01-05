# PINNS-based solution for flow past a cylinder

## Overview

Flow past cylinder problem is a two-dimensional low velocity steady flow around a cylinder which is only related to the `Re` number. When `Re` is less than or equal to 1, the inertial force in the flow field is secondary to the viscous force, the streamlines in the upstream and downstream directions of the cylinder are symmetrical, and the drag coefficient is approximately inversely proportional to `Re` . The flow around this `Re` number range is called the Stokes zone; With the increase of `Re` , the streamlines in the upstream and downstream of the cylinder gradually lose symmetry. This special phenomenon reflects the peculiar nature of the interaction between the fluid and the surface of the body. Solving flow past a cylinder is a classical problem in hydromechanics.

Since it is difficult to obtain the generalized theoretical solution of the Navier-Stokes equation,the numerical method is used to solve the governing equation in the flow past cylinder scenario to predict the flow field, which is also a classical problem in computational fluid mechanics. Traditional solutions often require fine discretization of the fluid to capture the phenomena that need to be modeled. Therefore, traditional finite element method (FEM) and finite difference method (FDM) are often costly.

Physics-informed Neural Networks (PINNs) provides a new method for quickly solving complex fluid problems by using loss functions that approximate governing equations coupled with simple network configurations. In this case, the data-driven characteristic of neural network is used along with `PINNs` to solve the flow past cylinder problem.

## Problem Description

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

## Technology Path

MindFlow solves the problem as follows:

1. Training Dataset Construction.
2. Neural Network Construction.
3. Multi-task learning for adaptive losses.
4. Problem Modeling.
5. Model training.
6. Model Evaluation and Visualization.

### Import Dependency

Import the modules and interfaces on which this tutorial depends:

```python
"""train process"""
import os
import time

import numpy as np
from sympy import diff, Function, symbols

import mindspore
from mindspore import context, nn, ops, Tensor, jit, set_seed, load_checkpoint, load_param_into_net
from mindspore import dtype as mstype

from mindflow import MTLWeightedLossCell, load_yaml_config, NavierStokes, sympy_to_mindspore

from src import create_training_dataset, create_test_dataset, calculate_l2_error

```

## Training Dataset Construction

In this case, the initial condition and boundary condition data of the existing flow around a cylinder with Reynolds number 100 are sampled. For the training dataset, the problem domain and time dimension of planar rectangle are constructed. Then the known initial conditions and boundary conditions are sampled. The validation set is constructed based on the existing points in the flow field.

Download the training and test dataset: [physics_driven/flow_past_cylinder/dataset](https://download.mindspore.cn/mindscience/mindflow/dataset/applications/physics_driven/flow_past_cylinder/dataset/) .

```python
from mindflow.data import Dataset, ExistedDataConfig
from mindflow.geometry import Rectangle, TimeDomain, GeometryWithTime, generate_sampling_config


def create_test_dataset(test_data_path):
    """load labeled data for evaluation"""
    print("get dataset path: {}".format(test_data_path))
    paths = [test_data_path + '/eval_points.npy', test_data_path + '/eval_label.npy']
    inputs = np.load(paths[0])
    label = np.load(paths[1])
    print("check eval dataset length: {}".format(inputs.shape))
    return inputs, label


def create_training_dataset(config):
    """create training dataset by online sampling"""
    coord_min = config["coord_min"]
    coord_max = config["coord_max"]
    rectangle = Rectangle("rect", coord_min, coord_max)

    time_interval = TimeDomain("time", 0.0, config["range_t"])
    domain_region = GeometryWithTime(rectangle, time_interval)
    domain_region.set_name("domain")
    domain_region.set_sampling_config(create_config_from_edict(domain_sampling_config))

    geom_dict = {domain_region: ["domain"]}

    data_path = config["train_data_path"]
    config_bc = ExistedDataConfig(name="bc",
                                  data_dir=[data_path + "/bc_points.npy", data_path + "/bc_label.npy"],
                                  columns_list=["points", "label"],
                                  constraint_type="BC",
                                  data_format="npy")
    config_ic = ExistedDataConfig(name="ic",
                                  data_dir=[data_path + "/ic_points.npy", data_path + "/ic_label.npy"],
                                  columns_list=["points", "label"],
                                  constraint_type="IC",
                                  data_format="npy")
    dataset = Dataset(geom_dict, existed_data_list=[config_bc, config_ic])
    return dataset

```

## Neural Network Construction

This example uses a simple fully-connected network with a depth of 6 layers and the activation function is the `tanh` function.

```python
from mindflow import MultiScaleFCCell
coord_min = np.array(config["geometry"]["coord_min"] + [config["geometry"]["time_min"]]).astype(np.float32)
coord_max = np.array(config["geometry"]["coord_max"] + [config["geometry"]["time_max"]]).astype(np.float32)
input_center = list(0.5 * (coord_max + coord_min))
input_scale = list(2.0 / (coord_max - coord_min))
model = MultiScaleFCCell(in_channels=config["model"]["in_channels"],
                            out_channels=config["model"]["out_channels"],
                            layers=config["model"]["layers"],
                            neurons=config["model"]["neurons"],
                            residual=config["model"]["residual"],
                            act='tanh',
                            num_scales=1,
                            input_scale=input_scale,
                            input_center=input_center)
```

## Multi-task learning for adaptive losses

The PINNs method needs to optimize multiple losses at the same time, and brings challenges to the optimization process. Here, we adopt the uncertainty weighting algorithm proposed in ***Kendall, Alex, Yarin Gal, and Roberto Cipolla. "Multi-task learning using uncertainty to weigh losses for scene geometry and semantics." CVPR, 2018.*** to dynamically adjust the weights.

```python
mtl = MTLWeightedLossCell(num_losses=cylinder_flow_train_dataset.num_dataset)

if config["load_ckpt"]:
    param_dict = load_checkpoint(config["load_ckpt_path"])
    load_param_into_net(model, param_dict)
    load_param_into_net(mtl, param_dict)

# define optimizer
params = model.trainable_params() + mtl.trainable_params()
optimizer = nn.Adam(params, config["optimizer"]["initial_lr"])
```

## Problem Modeling

The following NavierStokes2D defines the navier-stokes' problem. The `sympy` is used for delineating partial differential equations in symbolic forms and computing all equations' loss. Specifically, it includes 3 parts: governing equation, initial condition and boundary conditions.

```python
class NavierStokes2D(NavierStokes):
    def __init__(self, model, re=100, loss_fn=nn.MSELoss()):
        super(NavierStokes2D, self).__init__(model, re=re, loss_fn=loss_fn)
        self.ic_nodes = sympy_to_mindspore(self.ic(), self.in_vars, self.out_vars)
        self.bc_nodes = sympy_to_mindspore(self.bc(), self.in_vars, self.out_vars)

    def ic(self):
        ic_u = self.u
        ic_v = self.v
        equations = {"ic_u": ic_u, "ic_v": ic_v}
        return equations

    def bc(self):
        bc_u = self.u
        bc_v = self.v
        bc_p = self.p
        equations = {"bc_u": bc_u, "bc_v": bc_v, "bc_p": bc_p}
        return equations

    def get_loss(self, pde_data, ic_data, ic_label, bc_data, bc_label):
        pde_res = self.parse_node(self.pde_nodes, inputs=pde_data)
        pde_residual = ops.Concat(1)(pde_res)
        pde_loss = self.loss_fn(pde_residual, Tensor(np.array([0.0]).astype(np.float32), mstype.float32))

        ic_res = self.parse_node(self.ic_nodes, inputs=ic_data)
        ic_residual = ops.Concat(1)(ic_res)
        ic_loss = self.loss_fn(ic_residual, ic_label)

        bc_res = self.parse_node(self.bc_nodes, inputs=bc_data)
        bc_residual = ops.Concat(1)(bc_res)
        bc_loss = self.loss_fn(bc_residual, bc_label)

        return pde_loss + ic_loss + bc_loss
```

## Model Training

With MindSpore version >= 2.0.0, we can use the functional programming for training neural networks.

```python
def train():
    problem = NavierStokes2D(model)

    from mindspore.amp import DynamicLossScaler, auto_mixed_precision, all_finite
    if use_ascend:
        loss_scaler = DynamicLossScaler(1024, 2, 100)
        auto_mixed_precision(model, 'O3')
    else:
        loss_scaler = None

    # the loss function receives 5 data sources: pde, ic, ic_label, bc and bc_label
    def forward_fn(pde_data, ic_data, ic_label, bc_data, bc_label):
        loss = problem.get_loss(pde_data, ic_data, ic_label, bc_data, bc_label)
        if use_ascend:
            loss = loss_scaler.scale(loss)
        return loss

    grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=False)

    # using jit function to accelerate training process
    @jit
    def train_step(pde_data, ic_data, ic_label, bc_data, bc_label):
        loss, grads = grad_fn(pde_data, ic_data, ic_label, bc_data, bc_label)
        if use_ascend:
            loss = loss_scaler.unscale(loss)
            if all_finite(grads):
                grads = loss_scaler.unscale(grads)

        loss = ops.depend(loss, optimizer(grads))
        return loss


    steps = config["train_steps"]
    sink_process = mindspore.data_sink(train_step, cylinder_dataset, sink_size=1)
    model.set_train()

    for step in range(steps + 1):
        local_time_beg = time.time()
        cur_loss = sink_process()
        if step % 100 == 0:
            print(f"loss: {cur_loss.asnumpy():>7f}")
            print("step: {}, time elapsed: {}ms".format(step, (time.time() - local_time_beg)*1000))
            calculate_l2_error(model, inputs, label, config)

time_beg = time.time()
train()
print("End-to-End total time: {} s".format(time.time() - time_beg))
```

The model results are as follows:

```python
step: 4500, time elapsed: 401.7298221588135ms
    predict total time: 34.17372703552246 ms
    l2_error, U:  0.06336409400901151 , V:  0.2589800209573793 , P:  0.34167427991249655 , Total:  0.10642616781913976
==================================================================================================
loss: 0.000452
step: 4600, time elapsed: 402.61220932006836ms
    predict total time: 34.90447998046875 ms
    l2_error, U:  0.062382466103748126 , V:  0.25132992417815014 , P:  0.31638189557928253 , Total:  0.10285521629387122
==================================================================================================
loss: 0.001991
step: 4700, time elapsed: 402.57716178894043ms
    predict total time: 34.70349311828613 ms
    l2_error, U:  0.07896903562757136 , V:  0.2652466317087061 , P:  0.3036429776439537 , Total:  0.1145695518800529
==================================================================================================
loss: 0.000889
step: 4800, time elapsed: 402.6777744293213ms
    predict total time: 34.42740440368652 ms
    l2_error, U:  0.058614692034967684 , V:  0.2414685389277242 , P:  0.3107724054671294 , Total:  0.0985094087524046
==================================================================================================
loss: 0.000381
step: 4900, time elapsed: 401.6759395599365ms
    predict total time: 34.93666648864746 ms
    l2_error, U:  0.05813861797271185 , V:  0.237321794767128 , P:  0.292845942377899 , Total:  0.0963624185597883
==================================================================================================
loss: 0.000343
step: 5000, time elapsed: 401.6103744506836ms
    predict total time: 31.789302825927734 ms
    l2_error, U:  0.056819929136297694 , V:  0.22960231322852553 , P:  0.30507615478534533 , Total:  0.0948311305565182
==================================================================================================
End-to-End total time: 2056.573511123657 s
```

## Model Evaluation and Visualization

The following figure shows the errors versus time training epoch 5000. As the epoch number increases, the errors decreases accordingly.
Loss corresponding to 5000 epochs:

![epoch5000](images/TimeError_epoch5000.png)

During the calculation, the callback records the predicted values of U, V, and P at each step. The difference between the predicted values and the actual values is small.

![image_flow](images/image-flow.png)
