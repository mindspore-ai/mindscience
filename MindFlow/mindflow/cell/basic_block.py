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
"""basic"""
from __future__ import absolute_import

import math
import numpy as np

import mindspore.ops as ops
import mindspore.nn as nn
import mindspore.common.dtype as mstype
from mindspore import Tensor, Parameter
from mindspore.ops.primitive import constexpr

from .activation import get_activation
from ..utils.check_func import check_param_type

__all__ = ['LinearBlock', 'ResBlock', 'InputScale',
           'FCSequential', 'MultiScaleFCSequential']


@constexpr
def _check_dense_input_shape(x, prim_name=None):
    msg_prefix = f"For '{prim_name}', the" if prim_name else "The"
    if len(x) < 2:
        raise ValueError(
            f"{msg_prefix} dimension of 'x' should not be less than 2, but got {len(x)}.")


class LinearBlock(nn.Cell):
    r"""
    The LinearBlock. Applies a linear transformation to the incoming data.

    Args:
        in_channels (int): The number of channels in the input space.
        out_channels (int): The number of channels in the output space.
        weight_init (Union[Tensor, str, Initializer, numbers.Number]): The trainable weight_init parameter. The dtype
            is same as input `input` . For the values of str, refer to the function `mindspore.common.initializer`.
            Default: ``"normal"``.
        bias_init (Union[Tensor, str, Initializer, numbers.Number]): The trainable bias_init parameter. The dtype is
            same as input `input` . The values of str refer to the function `mindspore.common.initializer`.
            Default: ``"zeros"``.
        has_bias (bool): Specifies whether the layer uses a bias vector. Default: ``True``.
        activation (Union[str, Cell, Primitive, None]): activate function applied to the output of the fully connected
            layer. Default: ``None``.

    Inputs:
        - **input** (Tensor) - Tensor of shape :math:`(*, in\_channels)`.

    Outputs:
        Tensor of shape :math:`(*, out\_channels)`.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> from mindflow.cell import LinearBlock
        >>> from mindspore import Tensor
        >>> input = Tensor(np.array([[180, 234, 154], [244, 48, 247]], np.float32))
        >>> net = LinearBlock(3, 4)
        >>> output = net(input)
        >>> print(output.shape)
        (2, 4)

    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 weight_init='normal',
                 bias_init='zeros',
                 has_bias=True,
                 activation=None):
        super(LinearBlock, self).__init__()
        self.activation = get_activation(activation) if isinstance(
            activation, str) else activation
        self.dense = nn.Dense(in_channels,
                              out_channels,
                              weight_init=weight_init,
                              bias_init=bias_init,
                              has_bias=has_bias,
                              activation=self.activation)

    def construct(self, x):
        out = self.dense(x)
        return out


class ResBlock(nn.Cell):
    r"""
    The ResBlock of dense layer.

    Args:
        in_channels (int): The number of channels in the input space.
        out_channels (int): The number of channels in the output space.
        weight_init (Union[Tensor, str, Initializer, numbers.Number]): The trainable weight_init parameter. The dtype
            is same as input x. The values of str refer to the function `initializer`. Default: ``'normal'``.
        bias_init (Union[Tensor, str, Initializer, numbers.Number]): The trainable bias_init parameter. The dtype is
            same as input x. The values of str refer to the function `initializer`. Default: ``'zeros'``.
        has_bias (bool): Specifies whether the layer uses a bias vector. Default: ``True``.
        activation (Union[str, Cell, Primitive, None]): activate function applied to the output of the dense layer.
            Default: ``None``.
        weight_norm (bool): Whether to compute the sum of squares of weight. Default: ``False``.

    Inputs:
        - **input** (Tensor) - Tensor of shape :math:`(*, in\_channels)`.

    Outputs:
        Tensor of shape :math:`(*, out\_channels)`.

    Raises:
        ValueError: If `in_channels` not equal out_channels.
        TypeError: If `activation` is not in str or Cell or Primitive.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> from mindflow.cell import ResBlock
        >>> from mindspore import Tensor
        >>> input = Tensor(np.array([[180, 234, 154], [244, 48, 247]], np.float32))
        >>> net = ResBlock(3, 3)
        >>> output = net(input)
        >>> print(output.shape)
        (2, 3)

    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 weight_init='normal',
                 bias_init='zeros',
                 has_bias=True,
                 activation=None,
                 weight_norm=False):
        super(ResBlock, self).__init__()
        check_param_type(in_channels, "in_channels",
                         data_type=int, exclude_type=bool)
        check_param_type(out_channels, "out_channels",
                         data_type=int, exclude_type=bool)
        if in_channels != out_channels:
            raise ValueError("in_channels of ResBlock should be equal of out_channels, but got in_channels: {}, "
                             "out_channels: {}".format(in_channels, out_channels))
        self.dense = LinearBlock(in_channels,
                                 out_channels,
                                 weight_init=weight_init,
                                 bias_init=bias_init,
                                 has_bias=has_bias,
                                 activation=None)
        self.activation = get_activation(activation) if isinstance(
            activation, str) else activation
        if activation is not None and not isinstance(self.activation, (nn.Cell, ops.Primitive)):
            raise TypeError(
                "The activation must be str or Cell or Primitive,"" but got {}.".format(type(activation)))
        if not activation:
            self.activation = ops.Identity()

    def construct(self, x):
        out = self.activation(self.dense(x) + x)
        return out


def _bias_init(fan_in, fan_out):
    """initializer function for bias"""
    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
    out = np.random.uniform(-bound, bound, fan_out)
    return Tensor(out, mstype.float32)


class InputScale(nn.Cell):
    r"""
    Scale the input value to specified region based on :math:`(x_i - input_center)*input_scale`

    Args:
        input_scale (list): The scale factor of input.
        input_center (Union[list, None]): Position offset of coordinate translation. Default: ``None``.

    Inputs:
        - **input** (Tensor) - Tensor of shape :math:`(*, channels)`.

    Outputs:
        Tensor of shape :math:`(*, channels)`.

    Raises:
        TypeError: If `input_scale` is not a list.
        TypeError: If `input_center` is not a list or ``None``.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> from mindflow.cell import InputScale
        >>> from mindspore import Tensor
        >>> inputs = np.random.uniform(size=(16, 3)) + 3.0
        >>> inputs = Tensor(inputs.astype(np.float32))
        >>> input_scale = [1.0, 2.0, 4.0]
        >>> input_center = [3.5, 3.5, 3.5]
        >>> net = InputScale(input_scale, input_center)
        >>> output = net(inputs).asnumpy()
        >>> assert np.all(output[:, 0] <= 0.5) and np.all(output[:, 0] >= -0.5)
        >>> assert np.all(output[:, 1] <= 1.0) and np.all(output[:, 0] >= -1.0)
        >>> assert np.all(output[:, 2] <= 2.0) and np.all(output[:, 0] >= -2.0)
    """

    def __init__(self, input_scale, input_center=None):
        super(InputScale, self).__init__()
        check_param_type(input_scale, "input_scale", data_type=list)
        check_param_type(input_center, "input_center",
                         data_type=(type(None), list))
        input_scale = np.array(input_scale)
        self.input_scale = Tensor(input_scale, mstype.float32)
        if input_center is None:
            self.input_center = Tensor(
                np.zeros(input_scale.shape), mstype.float32)
        else:
            self.input_center = Tensor(np.array(input_center), mstype.float32)
        self.mul = ops.Mul()

    def construct(self, x):
        out = self.mul(x - self.input_center, self.input_scale)
        return out


def _get_out_net_activation(is_out, act):
    if is_out:
        return None
    return act


class FCSequential(nn.Cell):
    r"""
    A sequential container of the dense layers, dense layers are added to the container sequentially.

    Args:
        in_channels (int): The number of channels in the input space.
        out_channels (int): The number of channels in the output space.
        layers (int): The total number of layers, include input/hidden/output layers.
        neurons (int): The number of neurons of hidden layers.
        residual (bool): full-connected of residual block for the hidden layers. Default: ``True``.
        act (Union[str, Cell, Primitive, None]): activate function applied to the output of the fully connected layer,
            eg. ``'ReLU'``.Default: ``"sin"``.
        weight_init (Union[Tensor, str, Initializer, numbers.Number]): The trainable weight_init parameter. The dtype
            is same as input x. The values of str refer to the function `initializer`. Default: ``'normal'``.
        has_bias (bool): Specifies whether the layer uses a bias vector. Default: ``True``.
        bias_init (Union[Tensor, str, Initializer, numbers.Number]): The trainable bias_init parameter. The dtype
            is same as input x. The values of str refer to the function `initializer`. Default: ``'default'``.
        weight_norm (bool): Whether to compute the sum of squares of weight. Default: ``False``.

    Inputs:
        - **input** (Tensor) - Tensor of shape :math:`(*, in\_channels)`.

    Outputs:
        Tensor of shape :math:`(*, out\_channels)`.

    Raises:
        TypeError: If `layers` is not an int.
        TypeError: If `neurons` is not an int.
        TypeError: If `residual` is not a bool.
        ValueError: If `layers` is less than 3.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> from mindflow.cell import FCSequential
        >>> from mindspore import Tensor
        >>> inputs = np.ones((16, 3))
        >>> inputs = Tensor(inputs.astype(np.float32))
        >>> net = FCSequential(3, 3, 5, 32, weight_init="ones", bias_init="zeros")
        >>> output = net(inputs).asnumpy()
        >>> print(output.shape)
        (16, 3)
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 layers,
                 neurons,
                 residual=True,
                 act="sin",
                 weight_init='normal',
                 has_bias=True,
                 bias_init='default',
                 weight_norm=False):
        super(FCSequential, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.layers = layers
        self.neurons = neurons
        self.residual = residual
        self.act = act
        self.weight_init = weight_init
        self.has_bias = has_bias
        self.bias_init = bias_init
        self.weight_norm = weight_norm

        self._check_params()
        self.network = nn.SequentialCell()
        self._create_networks()

    def construct(self, x):
        return self.network(x)

    def _create_networks(self):
        self._add_linear_block(
            self.in_channels, self.neurons, weight_init=self.weight_init)
        self._add_hidden_blocks(
            self.neurons, self.neurons, weight_init=self.weight_init)
        self._add_linear_block(
            self.neurons, self.out_channels, weight_init=self.weight_init, is_out_net=True)

    def _check_params(self):
        check_param_type(self.layers, "layers",
                         data_type=int, exclude_type=bool)
        check_param_type(self.neurons, "neurons",
                         data_type=int, exclude_type=bool)
        check_param_type(self.residual, "residual", data_type=bool)
        if self.layers < 3:
            raise ValueError(
                "FCSequential have at least 3 layers, but got layers: {}".format(self.layers))

    def _add_linear_block(self, in_channels, out_channels, weight_init, is_out_net=False):
        act = _get_out_net_activation(is_out_net, self.act)
        self.network.append(LinearBlock(in_channels,
                                        out_channels,
                                        activation=act,
                                        weight_init=weight_init,
                                        has_bias=self.has_bias,
                                        bias_init=_bias_init(
                                            in_channels, out_channels)
                                        if self.bias_init == "default" else self.bias_init,
                                        ))

    def _add_res_block(self, in_channels, out_channels, weight_init, is_out_net=False):
        act = _get_out_net_activation(is_out_net, self.act)
        self.network.append(ResBlock(in_channels,
                                     out_channels,
                                     activation=act,
                                     weight_init=weight_init,
                                     has_bias=self.has_bias,
                                     bias_init=_bias_init(
                                         in_channels, out_channels)
                                     if self.bias_init == "default" else self.bias_init,
                                     ))

    def _add_hidden_blocks(self, in_channels, out_channels, weight_init):
        for _ in range(self.layers - 2):
            if self.residual:
                self._add_res_block(in_channels, out_channels, weight_init)
            else:
                self._add_linear_block(in_channels, out_channels, weight_init)


class MultiScaleFCSequential(nn.Cell):
    r"""
    The multi-scale fully conneted network.

    Args:
        in_channels (int): The number of channels in the input space.
        out_channels (int): The number of channels in the output space.
        layers (int): The total number of layers, include input/hidden/output layers.
        neurons (int): The number of neurons of hidden layers.
        residual (bool): full-connected of residual block for the hidden layers. Default: ``True``.
        act (Union[str, Cell, Primitive, None]): activate function applied to the output of the fully connected layer,
            eg. ``'ReLU'``.Default: ``"sin"``.
        weight_init (Union[Tensor, str, Initializer, numbers.Number]): The trainable weight_init parameter. The dtype
            is same as `input`. The values of str refer to the function `initializer`. Default: ``'normal'``.
        weight_norm (bool): Whether to compute the sum of squares of weight. Default: ``False``.
        has_bias (bool): Specifies whether the layer uses a bias vector. Default: ``True``.
        bias_init (Union[Tensor, str, Initializer, numbers.Number]): The trainable bias_init parameter. The dtype
            is same as `input`. The values of str refer to the function `initializer`. Default: ``'default'``.
        num_scales (int): The subnet number of multi-scale network. Default: ``4``.
        amp_factor (Union[int, float]): The amplification factor of input. Default: ``1.0``.
        scale_factor (Union[int, float]): The base scale factor. Default: ``2.0``.
        input_scale (Union[list, None]): The scale factor of input x/y/t. If not ``None``, the inputs will be
            scaled before set in the network. Default: ``None``.
        input_center (Union[list, None]): Center position of coordinate translation. If not ``None``, the inputs will be
            translated before set in the network. Default: ``None``.
        latent_vector (Union[Parameter, None]): Trainable papameter which will be concated will the sampling inputs
            and updated during training. Default: ``None``.

    Inputs:
        - **input** (Tensor) - Tensor of shape :math:`(*, in\_channels)`.

    Outputs:
        Tensor of shape :math:`(*, out\_channels)`.

    Raises:
        TypeError: If `num_scales` is not an int.
        TypeError: If `amp_factor` is neither int nor float.
        TypeError: If `scale_factor` is neither int nor float.
        TypeError: If `latent_vector` is neither a Parameter nor ``None``.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> from mindflow.cell import MultiScaleFCSequential
        >>> from mindspore import Tensor, Parameter
        >>> inputs = np.ones((64,3)) + 3.0
        >>> inputs = Tensor(inputs.astype(np.float32))
        >>> num_scenarios = 4
        >>> latent_size = 16
        >>> latent_init = np.ones((num_scenarios, latent_size)).astype(np.float32)
        >>> latent_vector = Parameter(Tensor(latent_init), requires_grad=True)
        >>> input_scale = [1.0, 2.0, 4.0]
        >>> input_center = [3.5, 3.5, 3.5]
        >>> net = MultiScaleFCSequential(3, 3, 5, 32,
        ...                        weight_init="ones", bias_init="zeros",
        ...                        input_scale=input_scale, input_center=input_center, latent_vector=latent_vector)
        >>> output = net(inputs).asnumpy()
        >>> print(output.shape)
        (64, 3)
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 layers,
                 neurons,
                 residual=True,
                 act="sin",
                 weight_init='normal',
                 weight_norm=False,
                 has_bias=True,
                 bias_init="default",
                 num_scales=4,
                 amp_factor=1.0,
                 scale_factor=2.0,
                 input_scale=None,
                 input_center=None,
                 latent_vector=None
                 ):
        super(MultiScaleFCSequential, self).__init__()
        check_param_type(num_scales, "num_scales",
                         data_type=int, exclude_type=bool)
        check_param_type(amp_factor, "amp_factor",
                         data_type=(int, float), exclude_type=bool)
        check_param_type(scale_factor, "scale_factor",
                         data_type=(int, float), exclude_type=bool)

        self.cell_list = nn.CellList()
        self.num_scales = num_scales
        self.scale_coef = [amp_factor * (scale_factor**i)
                           for i in range(self.num_scales)]

        self.latent_vector = latent_vector
        if self.latent_vector is not None:
            check_param_type(latent_vector, "latent_vector",
                             data_type=Parameter)
            self.num_scenarios = latent_vector.shape[0]
            self.latent_size = latent_vector.shape[1]
            in_channels += self.latent_size
        else:
            self.num_scenarios = 1
            self.latent_size = 0

        for _ in range(self.num_scales):
            self.cell_list.append(FCSequential(in_channels=in_channels,
                                               out_channels=out_channels,
                                               layers=layers,
                                               neurons=neurons,
                                               residual=residual,
                                               act=act,
                                               weight_init=weight_init,
                                               has_bias=has_bias,
                                               bias_init=bias_init,
                                               ))
        if input_scale is not None:
            self.input_scale = InputScale(input_scale, input_center)
        else:
            self.input_scale = ops.Identity()

        self.cast = ops.Cast()
        self.concat = ops.Concat(axis=1)

    def construct(self, x):
        x = self.input_scale(x)
        if self.latent_vector is not None:
            batch_size = x.shape[0]
            latent_vectors = self.latent_vector.view(self.num_scenarios, 1,
                                                     self.latent_size).repeat(
                batch_size // self.num_scenarios,
                axis=1).view((-1, self.latent_size))
            x = self.concat((x, latent_vectors))
        out = 0
        for i in range(self.num_scales):
            x_s = x * self.scale_coef[i]
            out = out + self.cast(self.cell_list[i](x_s), mstype.float32)
        return out
