# Copyright 2021-2023 @ Shenzhen Bay Laboratory &
#                       Peking University &
#                       Huawei Technologies Co., Ltd
#
# This code is a part of Cybertron package.
#
# The Cybertron is open-source software based on the AI-framework:
# MindSpore (https://www.mindspore.cn/)
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
"""
Filter networks
"""

from mindspore import Tensor
from mindspore.nn import Cell

from mindsponge.function import get_integer

from .block import MLP, Dense, Residual

class DenseFilter(Cell):
    r"""Dense type filter network.

    Args:

        num_basis (int):    Number of basis functions.

        dim_filter (int):   Dimension of output filter Tensor.

        activation (Cell):  Activation function. Default: None

        n_hidden (int):     Number of hidden layers. Default: 1

    """
    def __init__(self,
                 num_basis: int,
                 dim_filter: int,
                 activation: Cell,
                 n_hidden: int = 1,
                 ):

        super().__init__()

        self.num_basis = get_integer(num_basis)
        self.dim_filter = get_integer(dim_filter)

        if n_hidden > 0:
            hidden_layers = [self.dim_filter for _ in range(n_hidden)]
            self.dense_layers = MLP(
                self.num_basis, self.dim_filter, hidden_layers, activation=activation)
        else:
            self.dense_layers = Dense(
                self.num_basis, self.dim_filter, activation=activation)

    def construct(self, x: Tensor):
        return self.dense_layers(x)


class ResFilter(Cell):
    r"""Residual type filter network.

    Args:

        num_basis (int):    Number of basis functions.

        dim_filter (int):   Dimension of output filter Tensor.

        activation (Cell):  Activation function. Default: None

        n_hidden (int):     Number of hidden layers. Default: 1

    """
    def __init__(self,
                 num_basis: int,
                 dim_filter: int,
                 activation: Cell,
                 n_hidden: int = 1,
                 ):

        super().__init__()

        self.num_basis = get_integer(num_basis)
        self.dim_filter = get_integer(dim_filter)

        self.linear = Dense(self.num_basis, self.dim_filter, activation=None)
        self.residual = Residual(
            self.dim_filter, activation=activation, n_hidden=n_hidden)

    def construct(self, x: Tensor):
        lx = self.linear(x)
        return self.residual(lx)
