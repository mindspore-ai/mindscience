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
Basic Neural network layer
"""

from typing import Union

import mindspore as ms
from mindspore import nn
from mindspore.nn import Cell

from ..configs import Config
from .common import MLP
from ..configs import Registry as R


__all__ = [
    "FeedForward",
    "Residual",
    "PreActDense",
    "PreActResidual",
    "SeqPreActResidual",
]


class FeedForward(Cell):
    r"""Feed forward network for transformer.

    Args:
        dim (int):          Last dimension of Tensor

        activation (Cell):  Activation function.

        n_hidden (int):     Number of hidden layers. Default: 1

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """

    def __init__(self,
                 dim: int,
                 activation: Cell,
                 n_hidden: int = 1,
                 **kwargs
                 ):
        super().__init__()
        self._kwargs = Config.get_arguments(locals(), kwargs)

        dim = int(dim)
        self.norm = nn.LayerNorm((dim,), -1, -1)
        self.residual = Residual(dim, activation=activation, n_hidden=int(n_hidden))

    def construct(self, x: ms.Tensor):
        """Compute feed forward network.

        Args:
            x (Tensor): Tensor with shape (B, A, F). Data type is float.

        Returns:
            y (Tensor): Tensor with shape (B, A, F). Data type is float.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """

        nx = self.norm(x)
        return self.residual(nx)


class Residual(Cell):
    r"""Residual block

    Args:
        dim (int): The number of channels in the input space.

        activation (Union[Cell, str]): Activation function.

        n_hidden (int): Number of hidden layers. Default: 1

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """

    def __init__(self,
                 dim: int,
                 activation: Union[Cell, str],
                 n_hidden: int = 1,
                 **kwargs
                 ):
        super().__init__()
        self._kwargs = Config.get_arguments(locals(), kwargs)

        dim = int(dim)
        n_hidden = int(n_hidden)
        hidden_dims = [dim] * n_hidden
        self.nonlinear = MLP(dim, hidden_dims, activation=activation)

    def construct(self, x):
        r"""Compute neural network.

        Args:
            x (Tensor): Tensor of shape (..., X). Data type is float.

        Returns:
            y (Tensor): Tensor of shape (..., Y). Data type is float.

        Note:
            X:  Input dimension
            Y:  output dimension

        """
        return x + self.nonlinear(x)


class PreActDense(Cell):
    r"""Pre-activated dense layer

    Args:
        dim_in (int):  Input dimension.

        dim_out (int): Output dimension.

        activation (Union[Cell, str]): Activation function.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """

    def __init__(self,
                 dim_in: int,
                 dim_out: int,
                 activation: Union[Cell, str],
                 **kwargs
                 ):
        super().__init__()
        self._kwargs = Config.get_arguments(locals(), kwargs)

        self.activation = R.build('activation', activation)
        self.dense = nn.Dense(int(dim_in), int(dim_out), weight_init='xavier_uniform', activation=None)

    def construct(self, x):
        r"""Compute neural network.

        Args:
            x (Tensor): Tensor of shape (..., X). Data type is float.

        Returns:
            y (Tensor): Tensor of shape (..., Y). Data type is float.

        Note:
            X:  Input dimension
            Y:  output dimension

        """
        x = self.activation(x)
        return self.dense(x)


class PreActResidual(Cell):
    r"""Pre-activated residual block

    Args:
        dim (int): Dimension.

        activation (Union[Cell, str]): Activation function.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """

    def __init__(self, dim: int, activation: Union[Cell, str], **kwargs):
        super().__init__()
        self._kwargs = Config.get_arguments(locals(), kwargs)

        dim = int(dim)
        self.preact_dense1 = PreActDense(dim, dim, activation)
        self.preact_dense2 = PreActDense(dim, dim, activation)

    def construct(self, x):
        r"""Compute neural network.

        Args:
            x (Tensor): Tensor of shape (..., X). Data type is float.

        Returns:
            y (Tensor): Tensor of shape (..., Y). Data type is float.

        Note:
            X:  Input dimension
            Y:  output dimension

        """
        x1 = self.preact_dense1(x)
        x2 = self.preact_dense1(x1)
        return x + x2


class SeqPreActResidual(Cell):
    r"""Sequential of pre-activated residual block

    Args:
        dim (int): The number of channels in the input space.

        activation (Union[Cell, str]): Activation function.

        n_res (int): Number of residual blocks.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """

    def __init__(self,
                 dim: int,
                 activation: Union[Cell, str],
                 n_res: int,
                 **kwargs
                 ):
        super().__init__()
        self._kwargs = Config.get_arguments(locals(), kwargs)

        self.sequential = nn.SequentialCell(
            [PreActResidual(int(dim), activation) for _ in range(int(n_res))]
        )

    def construct(self, x):
        r"""Compute neural network.

        Args:
            x (Tensor): Tensor of shape (..., X). Data type is float.

        Returns:
            y (Tensor): Tensor of shape (..., Y). Data type is float.

        Note:
            X:  Input dimension
            Y:  output dimension

        """
        return self.sequential(x)
