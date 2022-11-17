# Copyright 2021 Huawei Technologies Co., Ltd
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
"""
feedforward neural network
"""

import numpy as np
from mindspore import Tensor
import mindspore.nn as nn
from mindspore.common.initializer import TruncatedNormal
from mindflow.cell import get_activation, LinearBlock


class FlowNetwork(nn.Cell):
    """Full-connect networks."""

    def __init__(self, in_channels, out_channels, coord_min, coord_max,
                 num_layers=10, neurons=64, activation="tanh", residual=False):
        super(FlowNetwork, self).__init__()
        self.activation = get_activation(activation)
        self.lower_x = Tensor(np.array(coord_min).astype(np.float32))
        self.upper_x = Tensor(np.array(coord_max).astype(np.float32))
        self.residual = residual

        self.fc1 = LinearBlock(in_channels, neurons,
                               weight_init=TruncatedNormal(sigma=np.sqrt(2.0 / (in_channels + neurons))))
        self.cell_list = nn.CellList()
        if num_layers < 2:
            raise ValueError("Total layers number should be at least 2, but got: {}".format(num_layers))
        self.num_hidden_layers = num_layers - 2
        for _ in range(self.num_hidden_layers):
            linear = LinearBlock(neurons, neurons, weight_init=TruncatedNormal(sigma=np.sqrt(1.0 / neurons)))
            self.cell_list.append(linear)
        self.fc2 = LinearBlock(neurons, out_channels,
                               weight_init=TruncatedNormal(sigma=np.sqrt(2.0 / (neurons + out_channels))))

    def construct(self, *inputs):
        """fc network"""
        x = inputs[0]
        x = 2.0 * (x - self.lower_x) / (self.upper_x - self.lower_x) - 1.0
        out = self.fc1(x)
        out = self.activation(out)
        for i in range(self.num_hidden_layers):
            if self.residual:
                out = self.activation(out + self.cell_list[i](out))
            else:
                out = self.activation(self.cell_list[i](out))
        out = self.fc2(out)
        return out
