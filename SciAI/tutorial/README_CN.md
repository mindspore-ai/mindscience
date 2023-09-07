[ENGLISH](README.md) | 简体中文

## 示例网络

- [网络搭建](#网络搭建)
- [损失函数定义](#损失函数定义)
- [网络训练](#网络训练)

SciAI基础框架由若干基础模块构成，涵盖有神经网络搭建、训练、验证以及其他辅助函数等。下面的一个示例简要展示了SciAI框架的功能。

### 网络搭建

使用SciAI基础框架创建神经网络的原理与[使用MindSpore构建网络](https://www.mindspore.cn/tutorials/zh-CN/r2.0/beginner/model.html)一致，但过程将会十分简便。
如下示例代码创建了一个输入维度为2，输出维度为1，包含两层维度为5的中间层的神经网络。

```python
from sciai.architecture import MLP

example_net = MLP(layers=[2, 5, 5, 1], weight_init=XavierTruncNormal(), bias_init='zeros', activation="tanh")
```

`MLP`将默认使用正态分布随机生成网络权重，偏差`bias`默认为0，激活函数默认为`tanh`。`MLP`同时接受多样化的初始化方式和MindSpore提供的所有激活函数，您可在模型库中自行探索。

### 损失函数定义

损失函数定义为[Cell](https://www.mindspore.cn/tutorials/zh-CN/r2.0/advanced/modules/layer.html?highlight=cell)的子类，
并将损失的计算方法写在方法`construct`中。

```python
from mindspore import nn
from sciai.architecture import MSE

class ExampleLoss(nn.Cell):
    def __init__(self, network):
        super().__init__()
        self.network = network
        self.mse = MSE()

    def construct(self, x, y_true):
        y_predict = self.network(x)
        return self.mse(y_predict - y_true)

example_loss = ExampleLoss(example_net)
```

此时，通过直接调用`example_loss`，并将输入`x`与真实值`y_true`作为参数，便可计算得到当前`example_net`预测的损失。

### 网络训练

得到损失函数后，我们即可使用SciAI框架中已封装好的训练类，使用数据集进行训练，代码如下。

```python
from mindspore import nn
from sciai.common import TrainCellWithCallBack
from sciai.context import init_project

# Get the correct platform automatically and set to GRAPH_MODE by default.
init_project()

example_optimizer = nn.Adam(example_loss.trainable_params())
example_trainer = TrainCellWithCallBack(example_loss, example_optimizer, loss_interval=100, time_interval=100)
for _ in range(num_iters):
    example_trainer(x_train, y_true)
```

在训练结束并且损失收敛时，通过调用`y = example_net(x)`即可得到`x`处的预测值`y`。
`./example_net.py`文件有该示例网络的完整训练流程。此示例网络仅仅展示了SciAI框架的基本能力和网络训练的基本流程，
通过此基础示例，您可以更好地理解在下面的高频模型库中，各个模型是如何使用SciAI和MindSpore实现训练与推理。
