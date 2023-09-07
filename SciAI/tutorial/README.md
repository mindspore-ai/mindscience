ENGLISH | [简体中文](README_CN.md)

## Example

- [Setup Neural Networks](#setup-neural-networks)
- [Loss Definition](#loss-definition)
- [Networks Training](#networks-training)

SciAI base framework consists of several modules covering network setup, network training, validation and auxiliary functions.
The following simple example, indicating the fundamental processes in using SciAI.

### Setup Neural Networks

The principle of setting up a neural networks in ScAI is the same as in [MindSpore](https://www.mindspore.cn/tutorials/en/r2.0/beginner/model.html),
but in SciAI it is much easier. The following code segment creates a neural networks with 2-D input, 1-D output and two 5-D hidden layers.

```python
from sciai.architecture import MLP

example_net = MLP(layers=[2, 5, 5, 1], weight_init=XavierTruncNormal(), bias_init='zeros', activation="tanh")
```

`MLP` accepts various initialization method and all [activation functions](https://www.mindspore.cn/docs/en/r2.0/api_python/mindspore.nn.html) provided by MindSpore.

### Loss Definition

We define the loss function as a sub-class of [Cell](https://www.mindspore.cn/tutorials/en/r2.0/advanced/modules/layer.html), and calculate the loss in method `construct`.

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

At this moment, we can calculate the prediction loss by calling `example_loss` with input `x` and ground truth `y_true`.

### Networks Training

Then, by creating instance of trainer class provided by SciAI, we can start training with datasets.

```python
from mindspore import nn
from sciai.common import TrainCellWithCallBack
from sciai.context.context import init_project

# Get the correct platform automatically and set to GRAPH_MODE by default.
init_project()

example_optimizer = nn.Adam(example_loss.trainable_params())
example_trainer = TrainCellWithCallBack(example_loss, example_optimizer, loss_interval=100, time_interval=100)
for _ in range(num_iters):
    example_trainer(x_train, y_true)
```

When the training of `example_net` is finished and loss converges, we can use the net to predict the value at `x` by calling `y = example_net(x)`.
The complete example code can be found in `./example_net.py`.
Ths example network only indicates the basic ability of SciAI framework and fundamental training process for a neural network.
With this exmpale, you can understand the models in the following model library better, how each model are trained and validated with SciAI and MindSpore.