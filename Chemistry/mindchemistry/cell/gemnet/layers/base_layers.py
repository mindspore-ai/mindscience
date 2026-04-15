# Copyright 2024 Huawei Technologies Co., Ltd
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
"""base layers"""

import math
import mindspore as ms
import mindspore.mint as mint
from mindspore.common.initializer import initializer
from .he_orthogonal import he_orthogonal_init


class DenseWithActivation(ms.nn.Cell):
    r"""
    Combines dense layer with scaling for swish activation.

    Args:
        in_features (int): The number of input features.
        out_features (int): The number of output features.
        bias (bool): Whether to include a bias term in the linear transformation. Default is False.
        activation (str): The activation function to use. Supported values are "silu" and "siqu". Default is None.
        scale_factor (float): The scaling factor for the activation function. Default is 0.6.

    Inputs:
        - **x** (Tensor) - The shape of tensor is :math:`(*, in\_features)`.

    Outputs:
        - **x** (Tensor) - The shape of tensor is :math:`(*, out\_features)`.
    """

    def __init__(self, in_features, out_features, bias=False, activation=None, scale_factor=0.6):
        super().__init__()
        self.linear = mint.nn.Linear(in_features, out_features, bias=bias)
        self.activation = activation
        self.scale_factor = 1 / scale_factor
        self.activation_fun = mint.nn.functional.silu
        self.reset_parameters()

    def reset_parameters(self, init=he_orthogonal_init):
        self.linear.weight = init(self.linear.weight)
        if self.linear.bias is not None:
            self.linear.bias.set_data(initializer(
                "zero", self.linear.bias.shape, self.linear.bias.dtype))

    def construct(self, x):
        """DenseWithActivation construct"""
        x = self.linear(x)
        if isinstance(self.activation, str):
            self.activation = self.activation.lower()
        if self.activation == "silu":
            x = mint.mul(self.activation_fun(x), self.scale_factor)
        elif self.activation == "siqu":
            x = mint.mul(self.activation_fun(x), x)
        elif self.activation is None:
            pass
        else:
            raise NotImplementedError(
                f"Activation function '{self.activation}' not implemented."
            )
        return x


class ResidualLayer(ms.nn.Cell):
    """
    Residual block with output scaled by 1/sqrt(2).

    Args:
        units (int): Output embedding size.
        n_layers (int): Number of dense layers. Default: 2
        layer (class): The class of the dense layer to use. Default: DenseWithActivation

    Inputs:
        - **res_input** (Tensor) - The shape of tensor is :math:`(*, units)`.
    Outputs:
        - **x** (Tensor) - The shape of tensor is :math:`(*, units)`.
    """

    def __init__(self, units: int, n_layers=2, layer=DenseWithActivation, **layer_kwargs):
        super().__init__()
        self.dense_mlp = ms.nn.SequentialCell(
            *[
                layer(
                    in_features=units,
                    out_features=units,
                    bias=False,
                    **layer_kwargs
                )
                for _ in range(n_layers)
            ]
        )
        self.inv_sqrt_2 = ms.Tensor(1 / math.sqrt(2))

    def construct(self, res_input):
        x = self.dense_mlp(res_input)
        x = mint.add(res_input, x)
        x = mint.mul(x, self.inv_sqrt_2)
        return x


class MLP(ms.nn.Cell):
    r"""
    Multi-layer perceptron.

    Args:
        in_dim (int): Input dimension.
        hidden_dim (int): Hidden layer dimension.
        fc_num_layers (int): Number of fully connected layers.
        out_dim (int): Output dimension.
        activation (str): Name of the activation function to use.
        last_activation (str): Name of the activation function to use for the last layer.

    Inputs:
        - **x** (Tensor) - The shape of tensor is :math:`(*, in\_dim)`.

    Outputs:
        - **x** (Tensor) - The shape of tensor is :math:`(*, out\_dim)`.
    """

    def __init__(self, in_dim, hidden_dim, fc_num_layers, out_dim,
                 activation=None, last_activation=None):
        super().__init__()
        self.activation = activation
        self.last_activation = last_activation
        self.in_layer = mint.nn.Linear(in_dim, hidden_dim, bias=True)
        self.dense_mlp = ms.nn.SequentialCell(
            *[
                mint.nn.Linear(
                    in_features=hidden_dim,
                    out_features=hidden_dim,
                    bias=False,
                )
                for _ in range(fc_num_layers)
            ]
        )
        self.out_layer = mint.nn.Linear(hidden_dim, out_dim, bias=True)

    def construct(self, x):
        """MLP construct"""
        if self.activation == 'ReLU':
            x = mint.relu(self.in_layer(x))
            for layer in self.dense_mlp:
                x = mint.relu(layer(x))
        elif self.activation is None:
            x = self.in_layer(x)
            for layer in self.dense_mlp:
                x = layer(x)
        else:
            raise NotImplementedError(
                f"Activation function '{self.activation}' not implemented."
            )

        if self.last_activation == 'ReLU':
            x = mint.relu(self.out_layer(x))
        elif self.last_activation is None:
            x = self.out_layer(x)
        else:
            raise NotImplementedError(
                f"Activation function '{self.last_activation}' not implemented."
            )
        return x
