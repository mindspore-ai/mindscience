# Copyright 2021-2023 @ Shenzhen Bay Laboratory &
#                       Peking University &
#                       Huawei Technologies Co., Ltd
#
# This code is a part of AIchemist package.
#
# The AIchemist is open-source software based on the AI-framework:
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
from mindspore import Tensor, nn
from ..configs import Config

from .common import MLP
from .residuals import Residual
from ..configs import Registry as R


class Filter(nn.Cell):
    r"""Base class for filter network.

    Args:
        dim_in (int):    Number of basis functions.

        dim_out (int):   Dimension of output filter Tensor.

        activation (Cell):  Activation function. Default: ``None``.

        n_hidden (int):     Number of hidden layers. Default: 1

    """

    def __init__(self,
                 dim_in: int,
                 dim_out: int,
                 activation: nn.Cell = None,
                 **kwargs,
                 ):
        super().__init__()
        self._kwargs = kwargs

        self.dim_in = int(dim_in)
        self.dim_out = int(dim_out)
        self.activation = activation

    def construct(self, x: Tensor):
        return x


@R.register('filter.dense')
class DenseFilter(Filter):
    r"""Dense type filter network.

    Args:
        dim_in (int):    Number of basis functions.

        dim_out (int):   Dimension of output filter Tensor.

        activation (Cell):  Activation function. Default: ``None``.

        n_hidden (int):     Number of hidden layers. Default: 1

    """

    def __init__(self,
                 dim_in: int,
                 dim_out: int,
                 activation: nn.Cell,
                 n_hidden: int = 1,
                 **kwargs,
                 ):

        super().__init__(
            dim_in=dim_in,
            dim_out=dim_out,
            activation=activation,
        )
        self._kwargs = Config.get_arguments(locals(), kwargs)
        hidden_layers = [self.dim_out for _ in range(n_hidden)]
        self.dense_layers = MLP(
            self.dim_in, hidden_layers, activation=self.activation)

    def construct(self, x: Tensor):
        return self.dense_layers(x)


@R.register('filter.residual')
class ResFilter(Filter):
    r"""Residual type filter network.

    Args:
        dim_in (int):    Number of basis functions.

        dim_out (int):   Dimension of output filter Tensor.

        activation (Cell):  Activation function. Default: ``None``.

        n_hidden (int):     Number of hidden layers. Default: 1

    """

    def __init__(self,
                 dim_in: int,
                 dim_out: int,
                 activation: nn.Cell,
                 n_hidden: int = 1,
                 **kwargs,
                 ):
        super().__init__(
            dim_in=dim_in,
            dim_out=dim_out,
            activation=activation,
        )
        self._kwargs = Config.get_arguments(locals(), kwargs)

        self.linear = nn.Dense(self.dim_in, self.dim_out, weight_init='xavier_uniform', activation=None)
        self.residual = Residual(
            self.dim_out, activation=self.activation, n_hidden=n_hidden)

    def construct(self, x: Tensor):
        lx = self.linear(x)
        return self.residual(lx)
