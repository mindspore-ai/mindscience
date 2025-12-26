# FNO算子求解Burgers方程

## 概述

### 背景

随着神经网络的迅猛发展，基于神经网络的方法为科学计算提供了新的范式。MindSpore Science 套件为这种新的计算范式提供了多种工具与接口，其中就包括了针对数值求解偏微分方程的算子神经网络。接下来，我们选取流体力学中最经典与简单的案例——求解一维情形下的 Burgers 方程作为示例，来介绍 MindSpore Science 套件的部分接口与使用方式。

经典的神经网络是在有限维度的空间进行映射，只能学习与特定离散化相关的解。与经典神经网络不同，傅里叶神经算子（Fourier Neural Operator，FNO）是一种能够学习无限维函数空间映射的新型深度学习架构。该架构可直接学习从任意函数参数到解的映射，用于解决一类偏微分方程的求解问题，具有更强的泛化能力。更多信息可参考[Fourier Neural Operator for Parametric Partial Differential Equations](https://arxiv.org/abs/2010.08895)。

### 问题描述

一维伯格斯方程（1-d Burgers' equation）是一个非线性偏微分方程，具有广泛应用，包括一维粘性流体流动建模。它的形式如下：

$$
\partial_t u(x, t)+\partial_x (u^2(x, t)/2)=\nu \partial_{xx} u(x, t), \quad x \in(0,1), t \in(0, T] \\
u(x, 0)=u_0(x), \quad x \in(0,1)
$$

其中$u$表示速度场，$u_0$表示初始条件，$\nu$表示粘度系数。

本案例利用 Fourier Neural Operator 学习初始状态到下一时刻状态的映射，实现一维Burgers'方程的求解：

$$
u_0 \mapsto u(\cdot, t)
$$

## 技术路径

Fourier Neural Operator模型构架如下图所示。图中$w_0(x)$表示初始涡度，通过Lifting Layer实现输入向量的高维映射，然后将映射结果作为Fourier Layer的输入，进行频域信息的非线性变换，最后由Decoding Layer将变换结果映射至最终的预测结果$w_1(x)$。

Lifting Layer、Fourier Layer以及Decoding Layer共同组成了Fourier Neural Operator。

![Fourier Neural Operator模型构架](https://raw.gitcode.com/mindspore-lab/mindscience/raw/master/MindFlow/applications/data_driven/burgers/fno1d/images/FNO.png)

Fourier Layer网络结构如下图所示。图中V表示输入向量，上框表示向量经过傅里叶变换后，经过线性变换R，过滤高频信息，然后进行傅里叶逆变换；另一分支经过线性变换W，最后通过激活函数，得到Fourier Layer输出向量。

![Fourier Layer网络结构](https://raw.gitcode.com/mindspore-lab/mindscience/raw/master/MindFlow/applications/data_driven/burgers/fno1d/images/FNO-2.png)

### 快速开始

数据集下载地址：[data_driven/burgers/](https://download.mindspore.cn/mindscience/mindflow/dataset/applications/data_driven/burgers/)。将数据集保存在`./dataset` 路径下。

#### 关键接口导入

工作目录选择为 `mindscience/MindFlow/applications/data_driven/burgers/fno1d`，在工作目录下创建 `train.py`，导入如下关键接口：

```python
from mindspore.amp import DynamicLossScaler, auto_mixed_precision, all_finite
from mindspore import context, nn, Tensor, set_seed, ops, data_sink, jit, save_checkpoint
from mindspore import dtype as mstype
from mindscience import FNO1D, RelativeRMSELoss, load_yaml_config, get_warmup_cosine_annealing_lr
from mindscience.pde import UnsteadyFlowWithLoss
```

#### 创建数据集
基于周期边界条件，生成满足如下分布的初始条件 $u_0$：

$$
u_0 \sim \mathcal{N}\left(0,625(-\Delta + 25I)^{-2}\right).
$$

该分布已在 `create_training_dataset` 中实现，只需按如下格式调用：
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

#### 构建模型

网络由 1 层 Lifting layer、1 层 Decoding layer 以及多层 Fourier Layer 叠加组成，定义如下：

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

#### 优化器

选择 Adam 算法作为优化器，并考虑 learning rate 随迭代次数的衰减，以及混合精度训练：

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

#### 模型训练

使用 **MindSpore version >= 2.0.0**, 我们可以使用函数式编程来训练神经网络。 使用非稳态问题的训练接口 `UnsteadyFlowWithLoss` ，该接口用于模型训练和评估，损失函数选择为相对均方根误差下的 Burgers 方程残差。

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

#### 训练方式一：在命令行中调用 `train.py` 脚本

```shell
export PYTHONPATH=$(cd ../../../../../ && pwd):$PYTHONPATH
python train.py --config_file_path ./configs/fno1d.yaml --device_target Ascend --device_id 0 --mode GRAPH
```

其中，

`--config_file_path` 表示配置文件的路径，默认值 './configs/fno1d.yaml'；

`--device_target` 表示使用的计算平台类型，可以选择 'Ascend' 或 'GPU'，默认值 'GPU'；

`--device_id` 表示使用的计算卡编号，可按照实际情况填写，默认值 0；

`--mode` 表示运行的模式，'GRAPH' 表示静态图模式, 'PYNATIVE' 表示动态图模式。

#### 训练方式二：运行 Jupyter Notebook

您可以使用[中文版](https://atomgit.com/mindspore-lab/mindscience/blob/master/MindFlow/applications/data_driven/burgers/fno1d/FNO1D_CN.ipynb)和[英文版](https://atomgit.com/mindspore-lab/mindscience/blob/master/MindFlow/applications/data_driven/burgers/fno1d/FNO1D.ipynb) Jupyter Notebook 逐行运行训练和验证代码。

### 结果展示

下图分别表示在 $t \in (0,3]$ 上的方程数值解以及不同时间下的切片：

![FNO1D Solves Burgers](https://raw.gitcode.com/mindspore-lab/mindscience/raw/master/MindFlow/applications/data_driven/burgers/fno1d/images/101-result.jpg)

## 使用接口简介

| 接口名称           | 简介             |
|:----------------------:|:--------------------------:|
| `FNO1D` | 来自 `mindscience.models`，定义一维 FNO 算子神经网络。 |
| `RelativeRMSELoss` | 来自 `mindscience.common`，相对 MSE 误差函数。 |
| `get_warmup_cosine_annealing_lr` | 来自 `mindscience.common`，带热重启的余弦退火，一种应用广泛的学习率的调整策略。 |
| `UnsteadyFlowWithLoss` | 来自 `mindscience.pde`，定义非稳态流问题的损失函数。 |

## 核心贡献者

gitee id：[liulei277](https://gitee.com/liulei277), [yezhenghao2023](https://gitee.com/yezhenghao2023), [huangwangwen2025](https://gitee.com/huangwangwen2025)

email: liulei2770919@163.com, yezhenghao@isrc.iscas.ac.cn, wangwen@isrc.iscas.ac.cn
