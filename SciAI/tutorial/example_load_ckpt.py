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

This is an example of preparing and loading checkpoints for a simple neural networks with 2 hidden layers.
We aim to give you a quick insight of how to load pre-trained checkpoints into your NNs,
and make it easier for you to understand the models provided in SciAI.
"""

import numpy as np

import mindspore as ms
from sciai.architecture import MLP
from sciai.common.initializer import XavierTruncNormal
from sciai.context import init_project
from sciai.utils import to_tensor, print_log, log_config

init_project()  # Get the correct platform automatically and set to GRAPH_MODE by default.
log_config("./logs")  # Configure the logging path


def func(x):
    """The function to be learned to"""
    return x[:, 0:1] ** 2 + np.sin(x[:, 1:2])


layers = [2, 5, 5, 1]
example_net = MLP(layers=layers, weight_init=XavierTruncNormal(), bias_init='zeros', activation="tanh")
ms.load_checkpoint("./checkpoints/example_net_10000.ckpt", example_net)

x_val = np.random.rand(5, 2)
y_true = func(x_val)
y_pred = example_net(to_tensor(x_val)).asnumpy()
print_log("y_true: ", y_true)
print_log("y_pred: ", y_pred)
