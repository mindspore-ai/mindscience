# 基于PINNs的2D定常达西方程（Darcy equation）求解

## 概述

达西方程（Darcy equation）是一个描述了流体在多孔介质中低速流动时渗流规律的二阶椭圆型偏微分方程，被广泛应用于水利工程，石油工程等领域中。达西方程最初由亨利·达西根据沙土渗流实验的实验结果制定，后来由斯蒂芬·惠特克通过均质化方法从纳维-斯托克斯方程推导出来。

由于对于不同流体的渗流情况，达西方程难以得到泛化的解析解。通常采用数值方法对描述特定场景的达西控制方程进行求解，进而对该场景下渗流的压力场和速度场进行仿真。利用达西渗流的数值仿真结果，可以进一步施行科学研究和工程实践。传统达西方程的数值求解通常采用有限元法（finite element method，FEM），此外，在实践中，达西方程的一些物理项会被定常化。有限元法被设计在标准的有限元网格空间，数值求解要求的精度越高，网格需要被划分得越精细，时间开销和存储开销会变得越大。

随着数值求解的并行算法研究趋于放缓，利用基于神经网络的方法开始得到发展并取得了接近传统数值方法的求解精度。在2019年，布朗大学应用数学团队提出了一种基于物理信息的神经网络（Physics-informed Neural Networks，PINNs）并给出了完整构建PINNs的代码体系用于求解多种多样的偏微分方程。本案例利用MindFlow流体方程套件，使用基于PINNs的方法，求解二维定常达西方程。

## 问题描述

考虑二维正方体$\Omega=(0,\;1)\times(0,\;1)$，该正方体的边界为$\Gamma$。忽略重力的影响，在$\Omega$的范围内，流体压力$p$和速度$u$满足的定常2D Darcy方程如下：

$$
\begin{align}
u + \nabla p &= 0,\;\;(x,\;y)\in\Omega\\
\nabla \cdot u &= f,\;\;(x,\;y)\in\Omega
\end{align}
$$

本案例使用迪利克雷边界条件，形式如下：

$$
\begin{align}
u_x &= -2 \pi cos(2 \pi x) cos(2 \pi y)\;&(x,\;y)\in\Gamma\\
u_y &= 2 \pi sin(2 \pi x) sin(2 \pi y)\;&(x,\;y)\in\Gamma\\
p &= sin(2 \pi x) cos(2 \pi y)\;&(x,\;y)\in\Gamma
\end{align}
$$

其中$f$为Darcy微分方程中的**forcing function**。本案例利用PINNs学习**forcing function** $f$为$8 \pi^2 sin(2 \pi x)cos(2 \pi y)$时位置到相应物理量的映射$(x,\;y) \mapsto (u,\;p)$，实现对Darcy方程的求解。

## 技术路线

MindFlow求解2D定常达西方程的具体流程如下：

1. 对问题域以及边界进行随机采样，创建训练数据集。
2. 构建`Darcy`方程和迪利克雷边界条件。
3. 构建神经网络模型，并设置网络模型参数。
4. 模型训练。
5. 模型推理和可视化。

## 训练示例

### 配置文件

总体配置文件如下所示，定义了问题域的边界，神经网络的结构，学习率，训练epoch，batch size等关键参数以及checkpoint文件、可视化文件的存取路径，案例命名等配置。

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

### 创建数据集

对于训练数据集，本案例根据问题域及边值条件进行随机采样，采样配置信息如下，根据均匀分布采样。构建平面正方形的问题域，再对已知的问题域和边界进行采样。

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

基于真实求解结果构建验证数据集。

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

### 达西方程建模

`Problem`包含求解2D定常Darcy问题的控制方程、边界条件。

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

### 构建神经网络

本案例使用层数为6层，每层128个神经元的神经网络结构，其中包含6层全连接层和5层残差层。残差层可以有效使得梯度在每层传递中不消失，使得更深的网络结构成为可能。

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

### 定义损失

实例化`Contraints`作为损失。

```python
    # define problem and Constraints
    darcy_problem = [
        Darcy2D(model=model) for _ in range(flow_train_dataset.num_dataset)
    ]
    train_constraints = Constraints(flow_train_dataset, darcy_problem)
```

### 模型训练

调用 `Solver`接口用于模型的训练和推理。向实例化的`solver`输入优化器、网络模型、损失函数。

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

### 网络训练结果

模型训练结果如下：

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

### 模型推理和可视化

训练后可对流场内所有数据点进行推理，并可视化相关结果。

![PINNS结果](images/result.png)
