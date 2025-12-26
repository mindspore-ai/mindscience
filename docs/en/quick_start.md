# FNO for Solving the Burgers Equation

## Overview

### Introduction

With the rapid development of neural networks, neural network-based methods have provided a new paradigm for scientific computing. The MindSpore Science suite offers a variety of tools and interfaces for this new computational paradigm, including operator neural networks for numerically solving partial differential equations. Next, we will use the most classic and simple case in fluid mechanics—solving the Burgers equation in one dimension—as an example to introduce some of the interfaces and usage of the MindSpore Science suite.

Classical neural networks operate in finite-dimensional spaces and can only learn solutions tied to specific discretizations. Unlike classical neural networks, the Fourier Neural Operator (FNO) is a novel deep learning architecture capable of learning mappings in infinite-dimensional function spaces. It directly learns the mapping from arbitrary functional parameters to solutions, addressing a class of partial differential equation solving problems with stronger generalization capabilities. For more information, please refer to [Fourier Neural Operator for Parametric Partial Differential Equations](https://arxiv.org/abs/2010.08895).

### Problem Setup

The one-dimensional Burgers' equation is a nonlinear partial differential equation with wide-ranging applications, including the modeling of one-dimensional viscous fluid flow. Its form is as follows:

$$
\partial_t u(x, t)+\partial_x (u^2(x, t)/2)=\nu \partial_{xx} u(x, t), \quad x \in(0,1), t \in(0, T] \\
u(x, 0)=u_0(x), \quad x \in(0,1)
$$

where $u$ represents the velocity field, $u_0$ denotes the initial condition, and $\nu$ is the viscosity coefficient.

In this case we use the Fourier Neural Operator to learn the mapping from the initial state to the state at the next time step, achieving the solution of the one-dimensional Burgers' equation:

$$
u_0 \mapsto u(\cdot, t)
$$

## Solution

The architecture of the Fourier Neural Operator model is illustrated in the figure below. In the diagram, $w_0(x)$ represents the initial vorticity, which undergoes a high-dimensional mapping of the input vector via the Lifting Layer. The mapped result is then used as the input to the Fourier Layer, where nonlinear transformations of frequency-domain information are performed. Finally, the Decoding Layer maps the transformed result to the final prediction outcome $w_1(x)$.

The Lifting Layer, Fourier Layer, and Decoding Layer collectively constitute the Fourier Neural Operator.

![Fourier Neural Operator模型构架](https://raw.gitcode.com/mindspore-lab/mindscience/raw/master/MindFlow/applications/data_driven/burgers/fno1d/images/FNO.png)

The network structure of the Fourier Layer is illustrated in the figure below. In the diagram, $V$ represents the input vector. The upper branch shows the vector undergoing Fourier transform, followed by linear transformation $R$ to filter high-frequency information, and then inverse Fourier transform. The other branch undergoes linear transformation $W$, and finally passes through an activation function to produce the output vector of the Fourier Layer.

![Fourier Layer网络结构](https://raw.gitcode.com/mindspore-lab/mindscience/raw/master/MindFlow/applications/data_driven/burgers/fno1d/images/FNO-2.png)

### Quick Start

Dataset download address: [data_driven/burgers/](https://download.mindspore.cn/mindscience/mindflow/dataset/applications/data_driven/burgers/). Save the dataset under the `./dataset` directory.

#### Key Interface

Set the working directory to `mindscience/MindFlow/applications/data_driven/burgers/fno1d`. Under the working directory, create a `train.py` file and import the following key interfaces:

```python
from mindspore.amp import DynamicLossScaler, auto_mixed_precision, all_finite
from mindspore import context, nn, Tensor, set_seed, ops, data_sink, jit, save_checkpoint
from mindspore import dtype as mstype
from mindscience import FNO1D, RelativeRMSELoss, load_yaml_config, get_warmup_cosine_annealing_lr
from mindscience.pde import UnsteadyFlowWithLoss
```

#### Create Dataset
Generate initial values $u_0$ that satisfy the following distribution, based on periodic boundary conditions.

$$
u_0 \sim \mathcal{N}\left(0,625(-\Delta + 25I)^{-2}\right).
$$

This distribution has already been implemented in `create_training_dataset`. Simply call it in the following format:
```python
from src import create_training_dataset, visual
# create training dataset
train_dataset = create_training_dataset(data_params, model_params, shuffle=True)

# create test dataset
test_input, test_label = np.load(os.path.join(data_params["root_dir"], "test/inputs.npy")), \
                         np.load(os.path.join(data_params["root_dir"], "test/label.npy"))
test_input = Tensor(np.expand_dims(test_input, -2), mstype.float32)
test_label = Tensor(np.expand_dims(test_label, -2), mstype.float32)
```

#### Construction of Neural Network

The network consists of a 1-layer Lifting layer, a 1-layer Decoding layer, and multiple stacked Fourier Layers, defined as follows:

```python
model = FNO1D(in_channels=model_params["in_channels"],
              out_channels=model_params["out_channels"],
              n_modes=model_params["modes"],
              resolutions=model_params["resolutions"],
              hidden_channels=model_params["hidden_channels"],
              n_layers=model_params["depths"],
              projection_channels=4*model_params["hidden_channels"],
              )
model_params_list = []
for k, v in model_params.items():
    model_params_list.append(f"{k}:{v}")
model_name = "_".join(model_params_list)
```

#### Optimizer

Select the Adam algorithm as the optimizer, and take into account the decay of the learning rate with the number of iterations, as well as mixed-precision training.

```python
steps_per_epoch = train_dataset.get_dataset_size()
lr = get_warmup_cosine_annealing_lr(lr_init=optimizer_params["learning_rate"],
last_epoch=optimizer_params["epochs"],
steps_per_epoch=steps_per_epoch,warmup_epochs=1)
optimizer = nn.Adam(model.trainable_params(), learning_rate=Tensor(lr))

if use_ascend:
    from mindspore.amp import DynamicLossScaler, auto_mixed_precision, all_finite
    loss_scaler = DynamicLossScaler(1024, 2, 100)
    auto_mixed_precision(model, 'O3')
else:
    loss_scaler = None
```

#### Training Process

Using **MindSpore with version >= 2.0.0**, we can employ functional programming to train neural networks. Utilize the training interface for unsteady problems, `UnsteadyFlowWithLoss`, which is designed for model training and evaluation. The loss function is selected as the residual of the Burgers equation under the relative root mean square error.

```python
problem = UnsteadyFlowWithLoss(model, loss_fn=RelativeRMSELoss(), data_format="NHWTC")

summary_dir = os.path.join(config["summary"]["summary_dir"], model_name)
print(summary_dir)

def forward_fn(data, label):
    loss = problem.get_loss(data, label)
    if use_ascend:
        loss = loss_scaler.scale(loss)
    return loss

grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=False)

@jit
def train_step(data, label):
    loss, grads = grad_fn(data, label)
    if use_ascend:
        loss = loss_scaler.unscale(loss)
        if all_finite(grads):
            grads = loss_scaler.unscale(grads)
    loss = ops.depend(loss, optimizer(grads))
    return loss

sink_process = data_sink(train_step, train_dataset, 1)
ckpt_dir = os.path.join(summary_dir, "ckpt")
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)
for epoch in range(1, optimizer_params["epochs"] + 1):
    model.set_train()
    local_time_beg = time.time()
    for _ in range(steps_per_epoch):
        cur_loss = sink_process()
        save_checkpoint(model, os.path.join(ckpt_dir, model_params["name"] + '_epoch' + str(epoch)))
```

#### Training Method 1: Call the `train.py` script via the command line.

```shell
export PYTHONPATH=$(cd ../../../../../ && pwd):$PYTHONPATH
python train.py --config_file_path ./configs/fno1d.yaml --device_target Ascend --device_id 0 --mode GRAPH
```

in which

`--config_file_path` represents the path to the configuration file, with the default value being './configs/fno1d.yaml';

`--device_target` represents the type of computing platform used, which can be either 'Ascend' or 'GPU', with the default being 'GPU';

`--device_id` represents the number of the computing device used, which can be filled in according to the actual situation, with a default value of 0;

`--mode` represents the mode of operation, with 'GRAPH' indicating static graph mode and 'PYNATIVE' indicating dynamic graph mode.

#### Training Method 2: Run Jupyter Notebook

You can use the [Chinese Version](https://atomgit.com/mindspore-lab/mindscience/blob/master/MindFlow/applications/data_driven/burgers/fno1d/FNO1D_CN.ipynb) and [English Version](https://atomgit.com/mindspore-lab/mindscience/blob/master/MindFlow/applications/data_driven/burgers/fno1d/FNO1D.ipynb) Jupyter Notebooks to run the training and validation code line by line.

### Numerical Results

The following figures show the numerical solution of the equation on $t\in(0,3]$ and slices at different time steps, respectively:

![FNO1D Solves Burgers](https://raw.gitcode.com/mindspore-lab/mindscience/raw/master/MindFlow/applications/data_driven/burgers/fno1d/images/101-result.jpg)

## Introduction to the Included Interface

| Name        | Definition |
|:----------------------:|:--------------------------:|
| `FNO1D` | In `mindscience.models`. Defines the one-dimensional Fourier Neural Operator neural network. |
| `RelativeRMSELoss` | In `mindscience.common`. The relative MSE loss function. |
| `get_warmup_cosine_annealing_lr` | In `mindscience.common`. Cosine annealing with warm restarts, a widely used learning rate adjustment strategy. |
| `UnsteadyFlowWithLoss` | In `mindscience.pde`. Defines the loss function for unsteady flow problems. |

## Core Contributors

gitee id：[liulei277](https://gitee.com/liulei277), [yezhenghao2023](https://gitee.com/yezhenghao2023), [huangwangwen2025](https://gitee.com/huangwangwen2025)

email: liulei2770919@163.com, yezhenghao@isrc.iscas.ac.cn, wangwen@isrc.iscas.ac.cn
