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
"""basic"""
from __future__ import absolute_import

from collections.abc import Sequence
from typing import Union

import mindspore.nn as nn
import mindspore.nn.layer.activation as activation
from mindspore import ops, float16, float32, Tensor
from mindspore.common.initializer import Initializer

from .activation import _activation


def _get_dropout(dropout_rate):
    """
    Gets the dropout functions.

    Inputs:
        dropout_rate (Union[int, float]): The dropout rate of the dropout function.
            If dropout_rate was int or not in range (0,1], it would be rectify to closest float value.

    Returns:
        Function, the dropout function.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> from mindchemistry.cell import _get_dropout
        >>> dropout = get_dropout(0.5)
        >>> dropout.set_train
        Dropout<keep_prob=0.5>
    """
    dropout_rate = float(max(min(dropout_rate, 1.), 1e-7))
    return nn.Dropout(keep_prob=dropout_rate)


def _get_layernorm(channel, epsilon):
    """
    Gets the layer normalization functions.

    Inputs:
        channel (Union[int, list]): The normalized shape of the layer normalization function.
            If channel was int, it would be wrap into a list.
        epsilon (float): The epsilon of the layer normalization function.

    Returns:
        Function, the layer normalization function.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> from mindchemistry.cell import _get_layernorm
        >>> from mindspore import Tensor
        >>> input_x = Tensor(np.array([[1.2, 0.1], [0.2, 3.2]], dtype=np.float32))
        >>> layernorm = get_layernorm([2], 1e-7)
        >>> output = layernorm(input_x)
        >>> print(output)
        [[ 9.99999881e-01, -9.99999881e-01],
        [-1.00000000e+00,  1.00000000e+00]]
    """
    if isinstance(channel, int):
        channel = [channel]
    return nn.LayerNorm(channel, epsilon=epsilon)


def _get_activation(name):
    """
    Gets the activation function.

    Inputs:
        name (Union[str, None]): The name of the activation function. If name was None, it would return [].

    Returns:
        Function, the activation function.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> from mindchemistry.cell import _get_activation
        >>> from mindspore import Tensor
        >>> input_x = Tensor(np.array([[1.2, 0.1], [0.2, 3.2]], dtype=np.float32))
        >>> sigmoid = _get_activation('sigmoid')
        >>> output = sigmoid(input_x)
        >>> print(output)
        [[0.7685248  0.5249792 ]
         [0.54983395 0.96083426]]
    """
    if name is None:
        return []
    if isinstance(name, str):
        name = name.lower()
        if name not in _activation:
            return activation.get_activation(name)
        return _activation.get(name)()
    return name


def _get_layer_arg(arguments, index):
    """
    Gets the argument of each network layers.

    Inputs:
        arguments (Union[str, int, float, List, None]): The arguments of each layers.
            If arguments was List return the argument at the index of the List.
        index (int): The index of layer in the network

    Returns:
        Argument of the indexed layer.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> from mindchemistry.cell import _get_layer_arg
        >>> from mindspore import Tensor
        >>> dropout_rate = _get_layer_arg([0.1, 0.2, 0.3], index=2)
        >>> print(dropout_rate)
        0.2
        >>> dropout_rate = _get_layer_arg(0.2, index=2)
        >>> print(dropout_rate)
        0.2
    """
    if isinstance(arguments, list):
        if len(arguments) <= index:
            if len(arguments) == 1:
                return [] if arguments[0] is None else arguments[0]
            return []
        return [] if arguments[index] is None else arguments[index]
    return [] if arguments is None else arguments


def get_linear_block(
        in_channels,
        out_channels,
        weight_init='normal',
        has_bias=True,
        bias_init='zeros',
        has_dropout=False,
        dropout_rate=0.5,
        has_layernorm=False,
        layernorm_epsilon=1e-7,
        has_activation=True,
        act='relu'
):
    """
    Gets the linear block list.

    Inputs:
        in_channels (int): The number of input channel.
        out_channels (int): The number of output channel.
        weight_init (Union[str, float, mindspore.common.initializer]): The initializer of the weights of dense layer
        has_bias (bool): The switch for whether dense layer has bias.
        bias_init (Union[str, float, mindspore.common.initializer]): The initializer of the bias of dense layer
        has_dropout (bool): The switch for whether linear block has a dropout layer.
        dropout_rate (float): The dropout rate for dropout layer, the dropout rate must be a float in range (0, 1]
        has_layernorm (bool): The switch for whether linear block has a layer normalization layer.
        layernorm_epsilon (float): The hyper parameter epsilon for layer normalization layer.
        has_activation (bool): The switch for whether linear block has an activation layer.
        act (Union[str, None]): The activation function in linear block

    Returns:
        List of mindspore.nn.Cell, linear block list .

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> from mindchemistry.cell import get_layer_arg
        >>> from mindspore import Tensor
        >>> dropout_rate = get_layer_arg([0.1, 0.2, 0.3], index=2)
        >>> print(dropout_rate)
        0.2
        >>> dropout_rate = get_layer_arg(0.2, index=2)
        >>> print(dropout_rate)
        0.2
    """
    dense = nn.Dense(
        in_channels, out_channels, weight_init=weight_init, bias_init=bias_init, has_bias=has_bias, activation=None
    )
    dropout = _get_dropout(dropout_rate) if (has_dropout is True) else []
    layernorm = _get_layernorm(out_channels, layernorm_epsilon) if (has_layernorm is True) else []
    act = _get_activation(act) if (has_activation is True) else []
    block_list = [dense, dropout, layernorm, act]
    while [] in block_list:
        block_list.remove([])
    return block_list


class FCNet(nn.Cell):
    r"""
    The Fully Connected Network. Applies a series of fully connected layers to the incoming data.

    Args:
        channels (List): the list of numbers of channel of each fully connected layers.
        weight_init (Union[str, float, mindspore.common.initializer, List]): initialize layer weights.
            if weight_init was List, each element corresponds to each layer.
        has_bias (Union[bool, List]): The switch for whether the dense layers has bias.
            if has_bias was List, each element corresponds to each dense layer.
        bias_init (Union[str, float, mindspore.common.initializer, List]): The initializer of the bias of dense layer
            if bias_init was List, each element corresponds to each dense layer.
        has_dropout (Union[bool, List]): The switch for whether linear block has a dropout layer.
            if has_dropout was List, each element corresponds to each layer.
        dropout_rate (float): The dropout rate for dropout layer, the dropout rate must be a float in range (0, 1]
            if dropout_rate was List, each element corresponds to each dropout layer.
        has_layernorm (Union[bool, List]): The switch for whether linear block has a layer normalization layer.
            if has_layernorm was List, each element corresponds to each layer.
        layernorm_epsilon (float): The hyper parameter epsilon for layer normalization layer.
            if layernorm_epsilon was List, each element corresponds to each layer normalization layer.
        has_activation (Union[bool, List]): The switch for whether linear block has an activation layer.
            if has_activation was List, each element corresponds to each layer.
        act (Union[str, None, List]): The activation function in linear block.
            if act was List, each element corresponds to each activation layer.

    Inputs:
        - **input** (Tensor) - Tensor of shape :math:`(*, channels[0])

    Returns:
        - **output** (Tensor) - Tensor of shape :math:`(*, channels[-1])

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> from mindchemistry.cell import FCNet
        >>> from mindspore import Tensor
        >>> inputs = Tensor(np.array([[180, 234, 154], [244, 48, 247]], np.float32))
        >>> net = FCNet([3, 16, 32, 16, 8])
        >>> output = net(inputs)
        >>> print(output.shape)
        (2, 8)

    """

    def __init__(
            self,
            channels,
            weight_init='normal',
            has_bias=True,
            bias_init='zeros',
            has_dropout=False,
            dropout_rate=0.5,
            has_layernorm=False,
            layernorm_epsilon=1e-7,
            has_activation=True,
            act='relu'
    ):
        super(FCNet, self).__init__()
        self.channels = channels
        self.weight_init = weight_init
        self.has_bias = has_bias
        self.bias_init = bias_init
        self.has_dropout = has_dropout
        self.dropout_rate = dropout_rate
        self.has_layernorm = has_layernorm
        self.layernorm_epsilon = layernorm_epsilon
        self.has_activation = has_activation
        self.activation = act
        self.network = nn.SequentialCell(self._create_network())

    def _create_network(self):
        """ create the network """
        cell_list = []
        for i in range(len(self.channels) - 1):
            cell_list += get_linear_block(
                self.channels[i],
                self.channels[i + 1],
                weight_init=_get_layer_arg(self.weight_init, i),
                has_bias=_get_layer_arg(self.has_bias, i),
                bias_init=_get_layer_arg(self.bias_init, i),
                has_dropout=_get_layer_arg(self.has_dropout, i),
                dropout_rate=_get_layer_arg(self.dropout_rate, i),
                has_layernorm=_get_layer_arg(self.has_layernorm, i),
                layernorm_epsilon=_get_layer_arg(self.layernorm_epsilon, i),
                has_activation=_get_layer_arg(self.has_activation, i),
                act=_get_layer_arg(self.activation, i)
            )
        return cell_list

    def construct(self, x):
        return self.network(x)


class MLPNet(nn.Cell):
    r"""
    The MLPNet Network. Applies a series of fully connected layers to the incoming data among which hidden layers have
        same number of channels.

    Args:
        in_channels (int): the number of input layer channel.
        out_channels (int): the number of output layer channel.
        layers (int): the number of layers.
        neurons (int): the number of channels of hidden layers.
        weight_init (Union[str, float, mindspore.common.initializer, List]): initialize layer weights
            if weight_init was List, each element corresponds to each layer.
        has_bias (Union[bool, List]): The switch for whether the dense layers has bias.
            if has_bias was List, each element corresponds to each dense layer.
        bias_init (Union[str, float, mindspore.common.initializer, List]): The initializer of the bias of dense layer
            if bias_init was List, each element corresponds to each dense layer.
        has_dropout (Union[bool, List]): The switch for whether linear block has a dropout layer.
            if has_dropout was List, each element corresponds to each layer.
        dropout_rate (float): The dropout rate for dropout layer, the dropout rate must be a float in range (0, 1]
            if dropout_rate was List, each element corresponds to each dropout layer.
        has_layernorm (Union[bool, List]): The switch for whether linear block has a layer normalization layer.
            if has_layernorm was List, each element corresponds to each layer.
        layernorm_epsilon (float): The hyper parameter epsilon for layer normalization layer.
            if layernorm_epsilon was List, each element corresponds to each layer normalization layer.
        has_activation (Union[bool, List]): The switch for whether linear block has an activation layer.
            if has_activation was List, each element corresponds to each layer.
        act (Union[str, None, List]): The activation function in linear block.
            if act was List, each element corresponds to each activation layer.

    Inputs:
        - **input** (Tensor) - Tensor of shape :math:`(*, channels[0])

    Outputs:
        - **output** (Tensor) - Tensor of shape :math:`(*, channels[-1])

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> from mindchemistry.cell import FCNet
        >>> from mindspore import Tensor
        >>> inputs = Tensor(np.array([[180, 234, 154], [244, 48, 247]], np.float32))
        >>> net = MLPNet(in_channels=3, out_channels=8, layers=5, neurons=32)
        >>> output = net(inputs)
        >>> print(output.shape)
        (2, 8)

    """

    def __init__(
            self,
            in_channels,
            out_channels,
            layers,
            neurons,
            weight_init='normal',
            has_bias=True,
            bias_init='zeros',
            has_dropout=False,
            dropout_rate=0.5,
            has_layernorm=False,
            layernorm_epsilon=1e-7,
            has_activation=True,
            act='relu'
    ):
        super(MLPNet, self).__init__()
        self.channels = (in_channels,) + (layers - 2) * \
                        (neurons,) + (out_channels,)
        self.network = FCNet(
            channels=self.channels,
            weight_init=weight_init,
            has_bias=has_bias,
            bias_init=bias_init,
            has_dropout=has_dropout,
            dropout_rate=dropout_rate,
            has_layernorm=has_layernorm,
            layernorm_epsilon=layernorm_epsilon,
            has_activation=has_activation,
            act=act
        )

    def construct(self, x):
        return self.network(x)


class MLPMixPrecision(nn.Cell):
    """MLPMixPrecision
    """

    def __init__(
            self,
            input_dim: int,
            hidden_dims: Sequence,
            short_cut=False,
            batch_norm=False,
            activation_fn='relu',
            has_bias=False,
            weight_init: Union[Initializer, str] = 'xavier_uniform',
            bias_init: Union[Initializer, str] = 'zeros',
            dropout=0,
            dtype=float16
    ):
        super().__init__()
        self.dtype = dtype
        self.div = ops.Div()

        self.dims = [input_dim] + hidden_dims
        self.short_cut = short_cut
        self.nonlinear_const = 1.0
        if isinstance(activation_fn, str):
            self.activation = _activation.get(activation_fn)()
            if activation_fn is not None and activation_fn == 'silu':
                self.nonlinear_const = 1.679177
        else:
            self.activation = activation_fn
        self.dropout = None
        if dropout:
            self.dropout = nn.Dropout(dropout)
        fcs = [
            nn.Dense(dim, self.dims[i + 1], weight_init=weight_init, bias_init=bias_init,
                     has_bias=has_bias).to_float(self.dtype) for i, dim in enumerate(self.dims[:-1])
        ]
        self.layers = nn.CellList(fcs)
        self.batch_norms = None
        if batch_norm:
            bns = [nn.BatchNorm1d(dim) for dim in self.dims[1:-1]]
            self.batch_norms = nn.CellList(bns)

    def construct(self, inputs):
        """construct

        Args:
            inputs: inputs

        Returns:
            inputs
        """
        hidden = inputs
        norm_from_last = 1.0
        for i, layer in enumerate(self.layers):
            sqrt_dim = ops.sqrt(Tensor(float(self.dims[i])))
            hidden = self.div(layer(hidden).astype(float32) * norm_from_last, sqrt_dim)
            norm_from_last = self.nonlinear_const
            if i < len(self.layers) - 1:
                if self.batch_norms is not None:
                    x = hidden.flatten(0, -2)
                    hidden = self.batch_norms[i](x).view_as(hidden)
                if self.activation is not None:
                    hidden = self.activation(hidden)
                if self.dropout is not None:
                    hidden = self.dropout(hidden)
                if self.short_cut and hidden.shape == hidden.shape:
                    hidden += inputs
        return hidden


class AutoEncoder(nn.Cell):
    r"""
    The AutoEncoder. Applies an encoder to get the latent code and applies a decoder to get the reconstruct data

    Args:
        channels (list): The number of channels of each encoder and decoder layer.
        weight_init (Union[str, float, mindspore.common.initializer, List]): initialize layer parameters
        if weight_init was List, each element corresponds to each layer.
            has_bias (Union[bool, List]): The switch for whether the dense layers has bias.
        if has_bias was List, each element corresponds to each dense layer.
            bias_init (Union[str, float, mindspore.common.initializer, List]): initialize layer parameters
        if bias_init was List, each element corresponds to each dense layer.
            has_dropout (Union[bool, List]): The switch for whether linear block has a dropout layer.
        if has_dropout was List, each element corresponds to each layer.
            dropout_rate (float): The dropout rate for dropout layer, the dropout rate must be a float in range (0, 1]
        if dropout_rate was List, each element corresponds to each dropout layer.
            has_layernorm (Union[bool, List]): The switch for whether linear block has a layer normalization layer.
        if has_layernorm was List, each element corresponds to each layer.
            layernorm_epsilon (float): The hyper parameter epsilon for layer normalization layer.
        if layernorm_epsilon was List, each element corresponds to each layer normalization layer.
            has_activation (Union[bool, List]): The switch for whether linear block has an activation layer.
        if has_activation was List, each element corresponds to each layer.
        act (Union[str, None, List]): The activation function in linear block.
            if act was List, each element corresponds to each activation layer.
        out_act (Union[None, str, mindspore.nn.Cell]): The activation function to output layer.

    Inputs:
        - x (Tensor) - Tensor of shape :math:`(*, channels[0])`.

    Outputs:
        - latents (Tensor) - Tensor of shape :math:`(*, channels[-1])`.
        - x_recon (Tensor) - Tensor of shape :math:`(*, channels[0])`.

    Supported Platforms:
        ``GPU``

    Examples:
        >>> import numpy as np
        >>> from mindchemistry import AutoEncoder
        >>> from mindspore import Tensor
        >>> inputs = Tensor(np.array([[180, 234, 154], [244, 48, 247]], np.float32))
        >>> net = AutoEncoder([3, 6, 2])
        >>> output = net(inputs)
        >>> print(output[0].shape, output[1].shape)
        (2, 2) (2, 3)

        """

    def __init__(
            self,
            channels,
            weight_init='normal',
            has_bias=True,
            bias_init='zeros',
            has_dropout=False,
            dropout_rate=0.5,
            has_layernorm=False,
            layernorm_epsilon=1e-7,
            has_activation=True,
            act='relu',
            out_act=None
    ):
        super(AutoEncoder, self).__init__()
        self.channels = channels
        self.weight_init = weight_init
        self.bias_init = bias_init
        self.has_bias = has_bias
        self.has_dropout = has_dropout
        self.dropout_rate = dropout_rate
        self.has_layernorm = has_layernorm
        self.has_activation = has_activation
        self.layernorm_epsilon = layernorm_epsilon
        self.activation = act
        self.output_activation = out_act
        self.encoder = nn.SequentialCell(self._create_encoder())
        self.decoder = nn.SequentialCell(self._create_decoder())

    def _create_encoder(self):
        """ create the network encoder """
        encoder_cell_list = []
        for i in range(len(self.channels) - 1):
            encoder_cell_list += get_linear_block(
                self.channels[i],
                self.channels[i + 1],
                weight_init=_get_layer_arg(self.weight_init, i),
                has_bias=_get_layer_arg(self.has_bias, i),
                bias_init=_get_layer_arg(self.bias_init, i),
                has_dropout=_get_layer_arg(self.has_dropout, i),
                dropout_rate=_get_layer_arg(self.dropout_rate, i),
                has_layernorm=_get_layer_arg(self.has_layernorm, i),
                layernorm_epsilon=_get_layer_arg(self.layernorm_epsilon, i),
                has_activation=_get_layer_arg(self.has_activation, i),
                act=_get_layer_arg(self.activation, i)
            )
        return encoder_cell_list

    def _create_decoder(self):
        """ create the network decoder """
        decoder_channels = self.channels[::-1]
        decoder_weight_init = self.weight_init[::-1] if isinstance(self.weight_init, list) else self.weight_init
        decoder_bias_init = self.bias_init[::-1] if isinstance(self.bias_init, list) else self.bias_init
        decoder_cell_list = []
        for i in range(len(decoder_channels) - 1):
            decoder_cell_list += get_linear_block(
                decoder_channels[i],
                decoder_channels[i + 1],
                weight_init=_get_layer_arg(decoder_weight_init, i),
                has_bias=_get_layer_arg(self.has_bias, i),
                bias_init=_get_layer_arg(decoder_bias_init, i),
                has_dropout=_get_layer_arg(self.has_dropout, i),
                dropout_rate=_get_layer_arg(self.dropout_rate, i),
                has_layernorm=_get_layer_arg(self.has_layernorm, i),
                layernorm_epsilon=_get_layer_arg(self.layernorm_epsilon, i),
                has_activation=_get_layer_arg(self.has_activation, i),
                act=_get_layer_arg(self.activation, i)
            )
        if self.output_activation is not None:
            decoder_cell_list.append(_get_activation(self.output_activation))
        return decoder_cell_list

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def construct(self, x):
        latents = self.encode(x)
        x_recon = self.decode(latents)
        return x_recon, latents
