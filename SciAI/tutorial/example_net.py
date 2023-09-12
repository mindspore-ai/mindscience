# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
""" An example neural network

This is an example of preparing and training a simple neural networks with 2 hidden layers.
We aim to give you a quick insight of how to use SciAI to train your NNs,
and make it easier for you to understand the models provided in SciAI.
"""

import numpy as np

from mindspore import nn
from sciai.architecture import MSE, MLP
from sciai.common import TrainCellWithCallBack
from sciai.common.initializer import XavierTruncNormal
from sciai.utils import to_tensor, print_log, log_config
from sciai.context.context import init_project

init_project()  # Get the correct platform automatically and set to GRAPH_MODE by default.
log_config("./logs")  # Configure the logging path


class ExampleLoss(nn.Cell):
    """ Loss definition class"""
    def __init__(self, network):
        """
        Everything besides input data should be initialized in '__init__', such as networks, gradient operators etc.
        Args:
            network: the neural networks to be trained.
        """
        super().__init__()
        self.network = network
        self.mse = MSE()

    def construct(self, x, y_real):
        """
        In this method, we define how the loss is calculated.
        Args:
            x: input of the neural networks
            y_real: ground truth

        Returns:
            Mean-Squared-Error loss
        """
        y_predict = self.network(x)
        return self.mse(y_predict - y_real)


def func(x):
    """The function to be learned to"""
    return x[:, 0:1] ** 2 + np.sin(x[:, 1:2])


layers = [2, 5, 5, 1]
example_net = MLP(layers=layers, weight_init=XavierTruncNormal(), bias_init='zeros', activation="tanh")
example_loss = ExampleLoss(example_net)
example_optimizer = nn.Adam(example_loss.trainable_params(), learning_rate=1e-3)
example_trainer = TrainCellWithCallBack(example_loss, example_optimizer,
                                        loss_interval=100, time_interval=100,
                                        ckpt_interval=10000)

x_train = np.random.rand(1000, 2)
y_true = func(x_train)
x_train, y_true = to_tensor((x_train, y_true))  # In MindSpore, we should feed data with `ms.Tensor` type.

for _ in range(10001):
    example_trainer(x_train, y_true)

x_val = np.random.rand(5, 2)
y_true = func(x_val)
y_pred = example_net(to_tensor(x_val)).asnumpy()
print_log("y_true: ", y_true)
print_log("y_pred: ", y_pred)
