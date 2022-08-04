# Copyright 2021-2022 @ Shenzhen Bay Laboratory &
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
Basic Neural network module
"""

from mindspore import nn
from mindspore import Tensor
from mindspore.nn import Cell
from mindspore.common.initializer import Initializer

from mindsponge.function import get_integer

from .activation import get_activation

__all__ = [
    "Dense",
    "MLP",
    "Residual",
    "PreActDense",
    "PreActResidual",
    "SeqPreActResidual",
]


class Dense(nn.Dense):
    r"""Full connected neural network layer.

    Args:
        in_channels (int):          The number of channels in the input space.

        out_channels (int):         The number of channels in the output space.

        weight_init (Initializer):  The trainable weight_init parameter.
                                    Default: 'xavier_uniform'

        bias_init (Initializer):    The trainable bias_init parameter.
                                    Default: 'zeros'

        has_bias (bool):            Specifies whether the layer uses a bias vector.
                                    Default: True

        activation (Cell):          Activate function.
                                    Default: None.

    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 weight_init: Initializer = 'xavier_uniform',
                 bias_init: Initializer = 'zero',
                 has_bias: bool = True,
                 activation: Cell = None,
                 ):

        super().__init__(
            in_channels=get_integer(in_channels),
            out_channels=get_integer(out_channels),
            weight_init=weight_init,
            bias_init=bias_init,
            has_bias=has_bias,
            activation=get_activation(activation),
        )


class MLP(Cell):
    r"""Multi-layer perceptron

    Args:
        n_in (int):                 The number of channels in the input space.

        n_out (int):                The number of channels in the output space.

        activation (Cell):          Activation function.
                                    Default: None.

        layer_dims (list):          Dimension of hidden layers.
                                    Default: None

        weight_init (Initializer):  The trainable weight_init parameter.
                                    Default: 'xavier_uniform'

        bias_init (Initializer):    The trainable bias_init parameter.
                                    Default: 'zeros'

        use_last_activation (bool): Whether to use activation function at the last layer.
                                    Default: False

    """

    def __init__(self,
                 n_in: int,
                 n_out: int,
                 layer_dims: list = None,
                 activation: Cell = None,
                 weight_init: Initializer = 'xavier_uniform',
                 bias_init: Initializer = 'zero',
                 use_last_activation: bool = False,
                 ):

        super().__init__()

        self.n_in = get_integer(n_in)
        self.n_out = get_integer(n_out)

        # get list of number of dimensions in input, hidden & output layers
        if (layer_dims is None) or (not layer_dims):
            self.mlp = nn.Dense(self.n_in, self.n_out, activation=activation)
        else:
            # assign a Dense layer (with activation function) to each hidden layer
            nets = []
            indim = self.n_in
            for ldim in layer_dims:
                # nets.append(Dense(indim, ldim,activation=activation))
                nets.append(
                    nn.Dense(
                        in_channels=indim,
                        out_channels=ldim,
                        weight_init=weight_init,
                        bias_init=bias_init,
                        has_bias=True,
                        activation=get_activation(activation),
                    )
                )
                indim = ldim

            # assign a Dense layer to the output layer
            if use_last_activation and activation is not None:
                nets.append(
                    nn.Dense(
                        in_channels=indim,
                        out_channels=self.n_out,
                        weight_init=weight_init,
                        bias_init=bias_init,
                        has_bias=True,
                        activation=get_activation(activation),
                    )
                )
            else:
                nets.append(
                    nn.Dense(
                        in_channels=indim,
                        out_channels=self.n_out,
                        weight_init=weight_init,
                        bias_init=bias_init,
                        has_bias=True,
                        activation=None)
                )
            # put all layers together to make the network
            self.mlp = nn.SequentialCell(nets)

    def construct(self, x: Tensor):
        r"""Compute neural network.

        Args:
            x (Tensor): Tensor of shape (..., X). Data type is float.

        Returns:
            y (Tensor): Tensor of shape (..., Y). Data type is float.

        Symbols:
            X:  Input dimension
            Y:  output dimension

        """
        y = self.mlp(x)

        return y


class Residual(Cell):
    r"""Residual block

    Args:
        dim (int):          The number of channels in the input space.

        activation (Cell):  Activation function.

        n_hidden (int):     Number of hidden layers. Default: 1

    """
    def __init__(self,
                 dim: int,
                 activation: Cell,
                 n_hidden: int = 1
                 ):
        super().__init__()

        if n_hidden > 0:
            hidden_layers = [dim] * n_hidden
            self.nonlinear = MLP(dim, dim, hidden_layers,
                                 activation=activation)
        else:
            self.nonlinear = Dense(dim, dim, activation=activation)

    def construct(self, x):
        r"""Compute neural network.

        Args:
            x (Tensor): Tensor of shape (..., X). Data type is float.

        Returns:
            y (Tensor): Tensor of shape (..., Y). Data type is float.

        Symbols:
            X:  Input dimension
            Y:  output dimension

        """
        return x + self.nonlinear(x)


class PreActDense(Cell):
    r"""Pre-activated dense layer

    Args:
        dim_in (int):       Input dimension.

        dim_out (int):      output dimension.

        activation (Cell):  Activation function.

    """
    def __init__(self,
                 dim_in: int,
                 dim_out: int,
                 activation: Cell
                 ):
        super().__init__()

        self.activation = get_activation(activation)
        self.dense = Dense(dim_in, dim_out, activation=None)

    def construct(self, x):
        r"""Compute neural network.

        Args:
            x (Tensor): Tensor of shape (..., X). Data type is float.

        Returns:
            y (Tensor): Tensor of shape (..., Y). Data type is float.

        Symbols:
            X:  Input dimension
            Y:  output dimension

        """
        x = self.activation(x)
        return self.dense(x)


class PreActResidual(Cell):
    r"""Pre-activated residual block

    Args:
        dim (int):          Dimension.

        activation (Cell):  Activation function.

    """
    def __init__(self, dim: int, activation: Cell):
        super().__init__()

        self.preact_dense1 = PreActDense(dim, dim, activation)
        self.preact_dense2 = PreActDense(dim, dim, activation)

    def construct(self, x):
        r"""Compute neural network.

        Args:
            x (Tensor): Tensor of shape (..., X). Data type is float.

        Returns:
            y (Tensor): Tensor of shape (..., Y). Data type is float.

        Symbols:
            X:  Input dimension
            Y:  output dimension

        """
        x1 = self.preact_dense1(x)
        x2 = self.preact_dense1(x1)
        return x + x2


class SeqPreActResidual(Cell):
    r"""Sequential of pre-activated residual block

    Args:
        dim (int):          The number of channels in the input space.

        activation (Cell):  Activation function.

        n_res (int):        Number of residual blocks.

    """
    def __init__(self,
                 dim: int,
                 activation: Cell,
                 n_res: int
                 ):
        super().__init__()

        self.sequential = nn.SequentialCell(
            [PreActResidual(dim, activation) for _ in range(n_res)]
        )

    def construct(self, x):
        r"""Compute neural network.

        Args:
            x (Tensor): Tensor of shape (..., X). Data type is float.

        Returns:
            y (Tensor): Tensor of shape (..., Y). Data type is float.

        Symbols:
            X:  Input dimension
            Y:  output dimension

        """
        return self.sequential(x)
