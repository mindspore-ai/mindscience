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
# ============================================================================
"""Networks definition"""
import numpy as np

import mindspore as ms
from mindspore import Tensor, nn, ops
from mindspore.common.initializer import XavierNormal

from sciai.architecture import MLP
from sciai.operators import grad


class ODENN(nn.Cell):
    """ODE-network"""
    def __init__(self, fnn):
        super(ODENN, self).__init__()
        self.fnn = fnn
        self.grad = grad(self.fnn, output_index=0, input_index=0)

    def construct(self, t):
        """Network forward pass"""
        u_t = self.grad(t)
        t_new = ops.mul(t, 0.5 * np.pi)
        f = ops.sub(u_t, ops.add(ops.mul(ops.cos(t_new), 0.5 * np.pi), 1))

        return f


class FNN(nn.Cell):
    """F-network"""
    def __init__(self, layers, x_min, x_max):
        super(FNN, self).__init__()
        self.size = layers
        self.x_min = Tensor(x_min, dtype=ms.float32)
        self.x_max = Tensor(x_max, dtype=ms.float32)
        self.mlp = MLP(layers, weight_init=XavierNormal())

    def construct(self, x):
        """Network forward pass"""
        a = ops.sub(x, self.x_min) / ops.sub(self.x_max, self.x_min)
        a = ops.mul(a, 2.0)
        a = ops.sub(a, 1.0)
        y = self.mlp(a)

        return y
