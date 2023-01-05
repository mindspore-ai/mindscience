# 基于PINNs的伯格斯方程（Burgers' equation）求解

## 概述

计算流体力学是21世纪流体力学领域的重要技术之一，其通过使用数值方法在计算机中对流体力学的控制方程进行求解，从而实现流动的分析、预测和控制。传统的有限元法（finite element method，FEM）和有限差分法（finite difference method，FDM）常囿于复杂的仿真流程（物理建模，网格划分，数值离散，迭代求解等）和较高的计算成本，往往效率低下。因此，借助AI提升流体仿真效率是十分必要的。

在经典理论与结合计算机性能的数值求解方法的发展趋于平缓的时候，近年来机器学习方法通过神经网络结合大量数据，实现流场的快速仿真，获得了接近传统方法的求解精度，为流场求解提供了新思路。

伯格斯方程（Burgers' equation）是一个模拟冲击波的传播和反射的非线性偏微分方程，被广泛应用于流体力学，非线性声学，气体动力学等领域，它以约翰内斯·马丁斯汉堡（1895-1981）的名字命名。本案例采用MindFlow流体仿真套件，基于物理驱动的PINNs (Physics Informed Neural Networks)方法，求解一维有粘性情况下的Burgers'方程。

## 问题描述

Burgers'方程的形式如下：

$$
u_t + uu_x = \epsilon u_{xx}, \quad x \in[-1,1], t \in[0, T],
$$

其中$\epsilon=0.01/\pi$，等号左边为对流项，右边为耗散项，本案例使用迪利克雷边界条件和正弦函数的初始条件，形式如下：

$$
u(t, -1) = u(t, 1) = 0,
$$

$$
u(0, x) = -sin(\pi x),
$$

本案例利用PINNs方法学习位置和时间到相应物理量的映射$(x, t) \mapsto u$，实现Burgers'方程的求解。

## 技术路径

MindFlow求解该问题的具体流程如下：

1. 创建训练数据集。
2. 构建神经网络。
3. 问题建模。
4. 模型训练。
5. 模型推理及可视化。

## 创建数据集

本案例根据求解域、初始条件及边值条件进行随机采样，生成训练数据集与测试数据集，具体设置如下：

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

## 构建神经网络

本例使用简单的全连接网络，深度为6层，激发函数为`tanh`函数。

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

## 问题建模

`Burgers1D`包含求解问题的控制方程、边界条件、初始条件等。使用`sympy`以符号形式定义偏微分方程并求解所有方程的损失值。

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

## 模型训练

使用2.0.0及以后版本的MindSpore，采用函数式编程的方式训练网络。

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

模型结果如下：

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

## 模型推理及可视化

训练后可对流场内所有数据点进行推理，并可视化相关结果。

```python
from src import visual_result

visual_result(model, resolution=config["visual_resolution"])
```

![PINNS结果](images/result.jpg)
