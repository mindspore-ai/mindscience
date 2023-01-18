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
train_epoch: 4000
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
class Darcy2D(PDEWithLoss):
    r"""
    The steady-state 2D Darcy flow's equations with Dirichlet boundary condition

    Args:
      model (Cell): The solving network.
      domain_name (str): The corresponding column name of data which governed by maxwell's equation.
      bc_name (str): The corresponding column name of data which governed by boundary condition.
    """

    def __init__(self, model, loss_fn=nn.MSELoss()):
        self.x, self.y = symbols("x y")
        self.u = Function("u")(self.x, self.y)
        self.v = Function("v")(self.x, self.y)
        self.p = Function("p")(self.x, self.y)
        self.in_vars = [self.x, self.y]
        self.out_vars = [self.u, self.v, self.p]
        self.loss_fn = loss_fn
        self.bc_nodes = sympy_to_mindspore(self.bc(), self.in_vars, self.out_vars)
        super(Darcy2D, self).__init__(model, self.in_vars, self.out_vars)

    def force_function(self, x, y):
        """ "forcing function in Darcy Equation"""
        return 8 * pi**2 * sin(2 * pi * x) * cos(2 * pi * y)

    def pde(self):
        """darcy equation"""
        loss_1 = (
            self.u.diff(self.x)
            + self.v.diff(self.y)
            - self.force_function(self.x, self.y)
        )
        loss_2 = self.u + self.p.diff(self.x)
        loss_3 = self.v + self.p.diff(self.y)
        return {"loss_1": loss_1, "loss_2": loss_2, "loss_3": loss_3}

    def bc(self):
        """Dirichlet boundary condition"""
        u_boundary = self.u - (-2 * pi * cos(2 * pi * self.x) * cos(2 * pi * self.y))

        v_boundary = self.v - (2 * pi * sin(2 * pi * self.x) * sin(2 * pi * self.y))

        p_boundary = self.p - (sin(2 * pi * self.x) * cos(2 * pi * self.y))

        return {
            "u_boundary": u_boundary,
            "v_boundary": v_boundary,
            "p_boundary": p_boundary,
        }

    def get_loss(self, pde_data, bc_data):
        """
        Compute loss of 2 parts: governing equation and boundary conditions.
        """
        pde_res = ops.Concat(1)(self.parse_node(self.pde_nodes, inputs=pde_data))
        pde_loss = self.loss_fn(
            pde_res, Tensor(np.array([0.0]).astype(np.float32), mstype.float32)
        )

        bc_res = ops.Concat(1)(self.parse_node(self.bc_nodes, inputs=bc_data))
        bc_loss = self.loss_fn(
            bc_res, Tensor(np.array([0.0]).astype(np.float32), mstype.float32)
        )

        return pde_loss + bc_loss
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
    # define problem
    problem = Darcy2D(model)
```

### Model Training

Invoke the `Solver` interface for model training and inference. pass optimizer, network model, loss function to instantiated `Solver`.

```python
    # optimizer
    params = model.trainable_params()
    optim = nn.Adam(params, learning_rate=config["optimizer"]["lr"])

    def forward_fn(pde_data, bc_data):
        return problem.get_loss(pde_data, bc_data)

    grad_fn = ops.value_and_grad(forward_fn, None, optim.parameters, has_aux=False)

    @jit
    def train_step(pde_data, bc_data):
        loss, grads = grad_fn(pde_data, bc_data)
        loss = ops.depend(loss, optim(grads))
        return loss

    epochs = config["train_epoch"]
    sink_process = data_sink(train_step, train_data, sink_size=1)
    model.set_train()

    for epoch in range(epochs):
        local_time_beg = time.time()
        cur_loss = sink_process()
        if epoch % 200 == 0:
            print(
                "epoch: {}, loss: {}, time: {}ms.".format(
                    epoch, cur_loss, (time.time() - local_time_beg) * 1000
                )
            )
            calculate_l2_error(
                model, test_input, test_label, config["train_batch_size"]
            )

    visual_result(model, config)
```

### Model training result

The model results are as follows:

```log
u_boundary: u(x, y) + 2*pi*cos(2*pi*x)*cos(2*pi*y)
    Item numbers of current derivative formula nodes: 2
v_boundary: v(x, y) - 2*pi*sin(2*pi*x)*sin(2*pi*y)
    Item numbers of current derivative formula nodes: 2
p_boundary: p(x, y) - sin(2*pi*x)*cos(2*pi*y)
    Item numbers of current derivative formula nodes: 2
loss_1: -8*pi**2*sin(2*pi*x)*cos(2*pi*y) + Derivative(u(x, y), x) + Derivative(v(x, y), y)
    Item numbers of current derivative formula nodes: 3
loss_2: u(x, y) + Derivative(p(x, y), x)
    Item numbers of current derivative formula nodes: 2
loss_3: v(x, y) + Derivative(p(x, y), y)
    Item numbers of current derivative formula nodes: 2
epoch: 0, loss: 540.41064, time: 6506.239652633667ms.
    predict total time: 1118.4625625610352 ms
    l2_error:  1.0001721637046521
==================================================================================================
epoch: 200, loss: 519.95905, time: 224.0011692047119ms.
    predict total time: 107.98430442810059 ms
    l2_error:  1.000227361534047
==================================================================================================
epoch: 400, loss: 530.78564, time: 250.22459030151367ms.
    predict total time: 3.2651424407958984 ms
    l2_error:  1.000569918705146
==================================================================================================
epoch: 600, loss: 205.96858, time: 263.2002830505371ms.
    predict total time: 267.98462867736816 ms
    l2_error:  0.8051833259470584
==================================================================================================
epoch: 800, loss: 6.81559, time: 247.53975868225098ms.
    predict total time: 3.2553672790527344 ms
    l2_error:  0.5721603283228421
==================================================================================================
epoch: 1000, loss: 0.7342617, time: 233.46233367919922ms.
    predict total time: 47.98769950866699 ms
    l2_error:  0.15113541987718068
==================================================================================================
epoch: 1200, loss: 0.32787272, time: 189.26453590393066ms.
    predict total time: 223.968505859375 ms
    l2_error:  0.06990242878007565
==================================================================================================
epoch: 1400, loss: 0.92689145, time: 354.1455268859863ms.
    predict total time: 303.9853572845459 ms
    l2_error:  0.05815758712471397
==================================================================================================
epoch: 1600, loss: 0.5284673, time: 309.88574028015137ms.
    predict total time: 147.98974990844727 ms
    l2_error:  0.047257754766852345
==================================================================================================
epoch: 1800, loss: 0.06530425, time: 274.4631767272949ms.
    predict total time: 8.200645446777344 ms
    l2_error:  0.03284622712221182
==================================================================================================
epoch: 2000, loss: 0.84967047, time: 316.03217124938965ms.
    predict total time: 7.172822952270508 ms
    l2_error:  0.04487714822865241
==================================================================================================
epoch: 2200, loss: 0.70964915, time: 252.62761116027832ms.
    predict total time: 154.88839149475098 ms
    l2_error:  0.039425191815509394
==================================================================================================
epoch: 2400, loss: 0.056157738, time: 220.26371955871582ms.
    predict total time: 6.469488143920898 ms
    l2_error:  0.024135588727258288
==================================================================================================
epoch: 2600, loss: 0.22577585, time: 316.00093841552734ms.
    predict total time: 128.19910049438477 ms
    l2_error:  0.032751112166644045
==================================================================================================
epoch: 2800, loss: 0.04019782, time: 132.07292556762695ms.
    predict total time: 127.98666954040527 ms
    l2_error:  0.021813896887517493
==================================================================================================
epoch: 3000, loss: 0.1239146, time: 283.0181121826172ms.
    predict total time: 57.543277740478516 ms
    l2_error:  0.024571568144330987
==================================================================================================
epoch: 3200, loss: 0.018467793, time: 219.33817863464355ms.
    predict total time: 76.04241371154785 ms
    l2_error:  0.019597676617278563
==================================================================================================
epoch: 3400, loss: 0.098201014, time: 160.3076457977295ms.
    predict total time: 91.98641777038574 ms
    l2_error:  0.021949914163476116
==================================================================================================
epoch: 3600, loss: 0.73904705, time: 336.02404594421387ms.
    predict total time: 211.99345588684082 ms
    l2_error:  0.046544678917597206
==================================================================================================
epoch: 3800, loss: 0.22243327, time: 316.00022315979004ms.
    predict total time: 211.98391914367676 ms
    l2_error:  0.022450780339445277
==================================================================================================
End-to-End total time: 839.0130681991577 s
```

### Model Inference and Visualization

After training, all data points in the flow field can be inferred. And related results can be visualized.

![PINNS result](images/result.png)
