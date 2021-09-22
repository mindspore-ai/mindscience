# ============================================================================
# Copyright 2021 The AIMM team at Shenzhen Bay Laboratory & Peking University
#
# People: Yi Isaac Yang, Jun Zhang, Diqing Chen, Yaqiang Zhou, Huiyang Zhang,
#         Yupeng Huang, Yijie Xia, Yao-Kun Lei, Lijiang Yang, Yi Qin Gao
#
# This code is a part of Cybertron-Code package.
#
# The Cybertron-Code is open-source software based on the AI-framework:
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
"""cybertron blocks"""

from mindspore import nn
from .activations import get_activation

__all__ = [
    "Dense",
    "MLP",
    "Residual",
    "PreActDense",
    "PreActResidual",
]


class Dense(nn.Dense):
    def __init__(self,
                 in_channels,
                 out_channels,
                 weight_init='xavier_uniform',
                 bias_init='zero',
                 has_bias=True,
                 activation=None,
                 ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            weight_init=weight_init,
            bias_init=bias_init,
            has_bias=has_bias,
            activation=get_activation(activation),
        )


class MLP(nn.Cell):
    """Multiple layer fully connected perceptron neural network.

    Args:
        n_in (int): number of input dimensions.
        n_out (int): number of output dimensions.
        layer_dims (list of int or int): number hidden layer dimensions.
            If an integer, same number of node is used for all hidden layers resulting
            in a rectangular network.
            If None, the number of neurons is divided by two after each layer starting
            n_in resulting in a pyramidal network.
        n_layers (int, optional): number of layers.
        activation (callable, optional): activation function. All hidden layers would
            the same activation function except the output layer that does not apply
            any activation function.

    """

    def __init__(self, n_in, n_out, layer_dims=None, activation=None, weight_init='xavier_uniform', bias_init='zero',
                 use_last_activation=False,):
        super().__init__()

        # get list of number of dimensions in input, hidden & output layers
        if layer_dims is None or not layer_dims:
            self.mlp = nn.Dense(n_in, n_out, activation=activation)
        else:
            # assign a Dense layer (with activation function) to each hidden
            # layer
            nets = []
            indim = n_in
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
                        out_channels=n_out,
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
                        out_channels=n_out,
                        weight_init=weight_init,
                        bias_init=bias_init,
                        has_bias=True,
                        activation=None)
                )
            # put all layers together to make the network
            self.mlp = nn.SequentialCell(nets)

    def construct(self, x):
        """Compute neural network output.

        Args:
            inputs (torch.Tensor): network input.

        Returns:
            torch.Tensor: network output.

        """

        y = self.mlp(x)

        return y


class Residual(nn.Cell):
    """Residual"""
    def __init__(self, dim, activation, n_hidden=1):
        super().__init__()

        if n_hidden > 0:
            hidden_layers = [dim for _ in range(n_hidden)]
            self.nonlinear = MLP(
                dim, dim, hidden_layers, activation=activation)
        else:
            self.nonlinear = Dense(dim, dim, activation=activation)

    def construct(self, x):
        return x + self.nonlinear(x)


class PreActDense(nn.Cell):
    def __init__(self, dim_in, dim_out, activation):
        super().__init__()

        self.activation = get_activation(activation)
        self.dense = Dense(dim_in, dim_out, activation=None)

    def construct(self, x):
        x = self.activation(x)
        return self.dense(x)


class PreActResidual(nn.Cell):
    """PreActResidual"""
    def __init__(self, dim, activation):
        super().__init__()

        self.preact_dense1 = PreActDense(dim, dim, activation)
        self.preact_dense2 = PreActDense(dim, dim, activation)

    def construct(self, x):
        x1 = self.preact_dense1(x)
        x2 = self.preact_dense1(x1)
        return x + x2


class SeqPreActResidual(nn.Cell):
    def __init__(self, dim, activation, n_res):
        super().__init__()

        self.sequential = nn.SequentialCell(
            [PreActResidual(dim, activation) for i in range(n_res)]
        )

    def construct(self, x):
        return self.sequential(x)
