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
We aim to give you a quick insight of how to use SciAI to train your NNs to predict function,
which is defined by differential equation, and make it easier for you to understand the models provided in SciAI.
"""

import numpy as np

from mindspore import nn, ops
from sciai.architecture import MSE, MLP
from sciai.common import TrainCellWithCallBack
from sciai.common.initializer import XavierTruncNormal
from sciai.context import init_project
from sciai.operators import grad
from sciai.utils import to_tensor, print_log, log_config

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
        self.dy_dx = grad(net=self.network, output_index=0, input_index=0)
        self.mse = MSE()

    def construct(self, x, x_bc, y_bc_true):
        """
        In this method, we define how the loss is calculated.
        No ground truth is given, what we know is PDEs and its boundary conditions
        Args:
            x: input of the neural networks
            x_bc: boundary positions
            y_bc_true: true value at boundary positions

        Returns:
            Mean-Squared of PDE residual error and BC error
        """
        y = self.network(x)
        dy_dx = self.dy_dx(x)
        domain_res = dy_dx - 2 * ops.div(y, x) + ops.mul(ops.pow(x, 2), ops.pow(y, 2))  # PDE residual error

        y_bc = self.network(x_bc)
        bc_res = y_bc_true - y_bc
        return self.mse(domain_res) + 10 * self.mse(bc_res)


def func(x):
    """
    The function to be learned to, which is the explicit solution of following Bernoulli equation (PDE):
    y' - 2y / x + x^2 * y^2 = 0
    with boundary condition: y(1) = 1, y(0) = 0
    """
    return x ** 2 / (0.2 * x ** 5 + 0.8)


layers = [1, 5, 5, 1]
example_net = MLP(layers=layers, weight_init=XavierTruncNormal(), bias_init='zeros', activation="tanh")
example_loss = ExampleLoss(example_net)
example_optimizer = nn.Adam(example_loss.trainable_params(), learning_rate=1e-3)
example_trainer = TrainCellWithCallBack(example_loss, example_optimizer, loss_interval=100, time_interval=100)

x_train = np.random.rand(100, 1)
x_bc_train = np.array([[0], [1]])
y_bc_train = np.array([[0], [1]])
x_train, x_bc_train, y_bc_train = to_tensor((x_train, x_bc_train, y_bc_train))  # In MindSpore, we should feed data with `ms.Tensor` type.

for _ in range(10000):
    example_trainer(x_train, x_bc_train, y_bc_train)

x_val = np.random.rand(5, 1)
y_true = func(x_val)
y_pred = example_net(to_tensor(x_val)).asnumpy()
print_log("y_true: ", y_true)
print_log("y_pred: ", y_pred)
