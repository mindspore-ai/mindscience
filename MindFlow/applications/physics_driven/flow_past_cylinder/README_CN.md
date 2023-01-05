
# 基于PINNs的圆柱绕流

## 概述

圆柱绕流，是指二维圆柱低速定常绕流的流型只与`Re`数有关。在`Re`≤1时，流场中的惯性力与粘性力相比居次要地位，圆柱上下游的流线前后对称，阻力系数近似与`Re`成反比，此`Re`数范围的绕流称为斯托克斯区；随着Re的增大，圆柱上下游的流线逐渐失去对称性。这种特殊的现象反映了流体与物体表面相互作用的奇特本质，求解圆柱绕流则是流体力学中的经典问题。

由于控制方程纳维-斯托克斯方程（Navier-Stokes equation）难以得到泛化的理论解，使用数值方法对圆柱绕流场景下控制方程进行求解，从而预测流场的流动，成为计算流体力学中的样板问题。传统求解方法通常需要对流体进行精细离散化，以捕获需要建模的现象。因此，传统有限元法（finite element method，FEM）和有限差分法（finite difference method，FDM）往往成本比较大。

物理启发的神经网络方法（Physics-informed Neural Networks），以下简称`PINNs`，通过使用逼近控制方程的损失函数以及简单的网络构型，为快速求解复杂流体问题提供了新的方法。本案例利用神经网络数据驱动特性，结合`PINNs`求解圆柱绕流问题。

## 纳维-斯托克斯方程（Navier-Stokes equation）

纳维-斯托克斯方程（Navier-Stokes equation），简称`N-S`方程，是流体力学领域的经典偏微分方程，在粘性不可压缩情况下，无量纲`N-S`方程的形式如下：

$$
\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} = 0
$$

$$
\frac{\partial u} {\partial t} + u \frac{\partial u}{\partial x} + v \frac{\partial u}{\partial y} = - \frac{\partial p}{\partial x} + \frac{1} {Re} (\frac{\partial^2u}{\partial x^2} + \frac{\partial^2u}{\partial y^2})
$$

$$
\frac{\partial v} {\partial t} + u \frac{\partial v}{\partial x} + v \frac{\partial v}{\partial y} = - \frac{\partial p}{\partial y} + \frac{1} {Re} (\frac{\partial^2v}{\partial x^2} + \frac{\partial^2v}{\partial y^2})
$$

其中，`Re`表示雷诺数。

本案例利用PINNs方法学习位置和时间到相应流场物理量的映射，实现`N-S`方程的求解：

$$
(x, y, t) \mapsto (u, v, p)
$$

## 技术路径

MindFlow求解该问题的具体流程如下：

1. 创建训练数据集。
2. 构建神经网络。
3. 多任务学习loss权重自适应。
4. 问题建模。
5. 模型训练。
6. 模型推理及可视化。

### 导入依赖

导入本教程所依赖模块与接口:

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

## 创建数据集

本案例对已有的雷诺数为100的标准圆柱绕流进行初始条件和边界条件数据的采样。对于训练数据集，构建平面矩形的问题域以及时间维度，再对已知的初始条件，边界条件采样；基于已有的流场中的点构造验证集。

[下载](https://download.mindspore.cn/mindscience/mindflow/dataset/applications/physics_driven/flow_past_cylinder/dataset/) 本案例使用的训练和测试数据。

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

## 构建神经网络

本例使用简单的全连接网络，深度为6层，激发函数为`tanh`函数。

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

## 多任务学习loss权重自适应

基于PINNs的学习方法在同一时间需要优化多个loss，这给优化过程带来了挑战。我们采用 ***Kendall, Alex, Yarin Gal, and Roberto Cipolla. "Multi-task learning using uncertainty to weigh losses for scene geometry and semantics." CVPR, 2018.*** 中提出的不确定性算法，动态调整loss权重。

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

### 问题建模

`NavierStokes2D`包含求解问题的控制方程、边界条件、初始条件等。使用`sympy`以符号形式定义偏微分方程并求解所有方程的损失值。

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

## 模型训练

使用2.0.0及以后版本的MindSpore，采用函数式编程的方式训练网络。

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

运行结果如下：

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

## 模型推理及可视化

训练过程中的error如图所示，随着epoch增长，error逐渐下降。

5000 epochs 对应的loss：

![epoch5000](images/TimeError_epoch5000.png)

计算过程中callback记录了每个时刻U，V，P的预测情况，与真实值偏差比较小。

![image_flow](images/image-flow.png)
