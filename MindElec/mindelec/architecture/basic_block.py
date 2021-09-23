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

from .activation import get_activation
from .util import check_mode

__all__ = ['LinearBlock', 'ResBlock', 'InputScaleNet', 'FCSequential', 'MultiScaleFCCell']


class LinearBlock(nn.Cell):
    r"""
    The LinearBlock.

    Args:
        in_channels (int): The number of channels in the input space.
        out_channels (int): The number of channels in the output space.
        weight_init (Union[Tensor, str, Initializer, numbers.Number]): The trainable weight_init parameter. The dtype
            is same as input x. The values of str refer to the function `initializer`. Default: 'normal'.
        bias_init (Union[Tensor, str, Initializer, numbers.Number]): The trainable bias_init parameter. The dtype is
            same as input x. The values of str refer to the function `initializer`. Default: 'zeros'.
        has_bias (bool): Specifies whether the layer uses a bias vector. Default: True.
        activation (Union[str, Cell, Primitive, None]): activate function applied to the output of the fully connected
            layer. Default: None.

    Inputs:
        - **input** (Tensor) - Tensor of shape :math:`(*, in\_channels)`.

    Outputs:
        Tensor of shape :math:`(*, out\_channels)`.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import numpy as np
        >>> from mindelec.architecture import LinearBlock
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
        check_mode("LinearBlock")
        self.activation = get_activation(activation) if isinstance(activation, str) else activation
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
            is same as input x. The values of str refer to the function `initializer`. Default: 'normal'.
        bias_init (Union[Tensor, str, Initializer, numbers.Number]): The trainable bias_init parameter. The dtype is
            same as input x. The values of str refer to the function `initializer`. Default: 'zeros'.
        has_bias (bool): Specifies whether the layer uses a bias vector. Default: True.
        activation (Union[str, Cell, Primitive, None]): activate function applied to the output of the dense layer.
            Default: None.

    Inputs:
        - **input** (Tensor) - Tensor of shape :math:`(*, in\_channels)`.

    Outputs:
        Tensor of shape :math:`(*, out\_channels)`.

    Supported Platforms:
        ``Ascend``

    Raises:
        ValueError: If `in_channels` not equal out_channels.
        TypeError: If `activation` is not in str or Cell or Primitive.

    Examples:
        >>> import numpy as np
        >>> from mindelec.architecture import ResBlock
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
                 activation=None):
        super(ResBlock, self).__init__()
        check_mode("ResBlock")
        _check_type(in_channels, "in_channels", int)
        _check_type(out_channels, "out_channels", int)
        if in_channels != out_channels:
            raise ValueError("in_channels of ResBlock should be equal of out_channels, but got in_channels: {}, "
                             "out_channels: {}".format(in_channels, out_channels))
        self.dense = nn.Dense(in_channels,
                              out_channels,
                              weight_init=weight_init,
                              bias_init=bias_init,
                              has_bias=has_bias,
                              activation=None)
        self.activation = get_activation(activation) if isinstance(activation, str) else activation
        if activation is not None and not isinstance(self.activation, (nn.Cell, ops.Primitive)):
            raise TypeError("The activation must be str or Cell or Primitive,"" but got {}.".format(type(activation)))
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


class InputScaleNet(nn.Cell):
    r"""
    Scale the input value to specified region.

    Args:
        input_scale (list): The scale factor of input x/y/t.
        input_center (Union[list, None]): Center position of coordinate translation. Default: None.

    Inputs:
        - **input** (Tensor) - Tensor of shape :math:`(*, channels)`.

    Outputs:
        Tensor of shape :math:`(*, channels)`.

    Supported Platforms:
        ``Ascend``

    Raises:
        TypeError: If `input_scale` is not a list.
        TypeError: If `input_center` is not a list or None.

    Examples:
        >>> import numpy as np
        >>> from mindelec.architecture import InputScaleNet
        >>> from mindspore import Tensor
        >>> inputs = np.random.uniform(size=(16, 3)) + 3.0
        >>> inputs = Tensor(inputs.astype(np.float32))
        >>> input_scale = [1.0, 2.0, 4.0]
        >>> input_center = [3.5, 3.5, 3.5]
        >>> net = InputScaleNet(input_scale, input_center)
        >>> output = net(inputs).asnumpy()
        >>> assert np.all(output[:, 0] <= 0.5) and np.all(output[:, 0] >= -0.5)
        >>> assert np.all(output[:, 1] <= 1.0) and np.all(output[:, 0] >= -1.0)
        >>> assert np.all(output[:, 2] <= 2.0) and np.all(output[:, 0] >= -2.0)
    """
    def __init__(self, input_scale, input_center=None):
        super(InputScaleNet, self).__init__()
        check_mode("InputScaleNet")
        _check_type(input_scale, "input_scale", list)
        _check_type(input_center, "input_center", (type(None), list))
        input_scale = np.array(input_scale)
        self.input_scale = Tensor(input_scale, mstype.float32)
        if input_center is None:
            self.input_center = Tensor(np.zeros(input_scale.shape), mstype.float32)
        else:
            self.input_center = Tensor(np.array(input_center), mstype.float32)
        self.mul = ops.Mul()

    def construct(self, x):
        out = self.mul(x - self.input_center, self.input_scale)
        return out


class FCSequential(nn.Cell):
    r"""
    The dense layer sequential.

    Args:
        in_channel (int): The number of channels in the input space.
        out_channel (int): The number of channels in the output space.
        layers (int): The total number of layers, include input/hidden/output layers.
        neurons (int): The number of neurons of hidden layers.
        residual (bool): full-connected of residual block for the hidden layers. Default: True.
        act (Union[str, Cell, Primitive, None]): activate function applied to the output of the fully connected layer,
            eg. 'ReLU'.Default: "sin".
        weight_init (Union[Tensor, str, Initializer, numbers.Number]): The trainable weight_init parameter. The dtype
            is same as input x. The values of str refer to the function `initializer`. Default: 'normal'.
        has_bias (bool): Specifies whether the layer uses a bias vector. Default: True.
        bias_init (Union[Tensor, str, Initializer, numbers.Number]): The trainable bias_init parameter. The dtype
            is same as input x. The values of str refer to the function `initializer`. Default: 'default'.

    Inputs:
        - **input** (Tensor) - Tensor of shape :math:`(*, in\_channels)`.

    Outputs:
        Tensor of shape :math:`(*, out\_channels)`.

    Supported Platforms:
        ``Ascend``

    Raises:
        TypeError: If `layers` is not an int.
        TypeError: If `neurons` is not an int.
        TypeError: If `residual` is not a bool.
        ValueError: If `layers` is less than 3.

    Examples:
        >>> import numpy as np
        >>> from mindelec.architecture import FCSequential
        >>> from mindspore import Tensor
        >>> inputs = np.ones((16, 3))
        >>> inputs = Tensor(inputs.astype(np.float32))
        >>> net = FCSequential(3, 3, 5, 32, weight_init="ones", bias_init="zeros")
        >>> output = net(inputs).asnumpy()
        >>> print(output.shape)
        (16, 3)
    """
    def __init__(self,
                 in_channel,
                 out_channel,
                 layers,
                 neurons,
                 residual=True,
                 act="sin",
                 weight_init='normal',
                 has_bias=True,
                 bias_init='default'):
        super(FCSequential, self).__init__()
        check_mode("FCSequential")
        _check_type(layers, "layers", int)
        _check_type(neurons, "neurons", int)
        _check_type(residual, "residual", bool)
        if layers < 3:
            raise ValueError("FCSequential have at least 3 layers, but got layers: {}".format(layers))
        self.network = nn.SequentialCell([LinearBlock(in_channel,
                                                      neurons,
                                                      activation=act,
                                                      weight_init=weight_init,
                                                      has_bias=has_bias,
                                                      bias_init=_bias_init(in_channel, neurons) \
                                                                if bias_init == "default" else bias_init
                                                      )])
        for _ in range(layers - 2):
            if residual:
                self.network.append(ResBlock(neurons,
                                             neurons,
                                             activation=act,
                                             weight_init=weight_init,
                                             has_bias=has_bias,
                                             bias_init=_bias_init(neurons, neurons) \
                                                       if bias_init == "default" else bias_init
                                             ))
            else:
                self.network.append(LinearBlock(neurons,
                                                neurons,
                                                activation=act,
                                                weight_init=weight_init,
                                                has_bias=has_bias,
                                                bias_init=_bias_init(neurons, neurons) \
                                                          if bias_init == "default" else bias_init
                                                ))
        self.network.append(LinearBlock(neurons,
                                        out_channel,
                                        weight_init=weight_init,
                                        has_bias=has_bias,
                                        bias_init=_bias_init(neurons, out_channel)  \
                                                  if bias_init == "default" else bias_init
                                        ))

    def construct(self, x):
        return self.network(x)


class MultiScaleFCCell(nn.Cell):
    r"""
    The multi-scale network.

    Args:
        in_channel (int): The number of channels in the input space.
        out_channel (int): The number of channels in the output space.
        layers (int): The total number of layers, include input/hidden/output layers.
        neurons (int): The number of neurons of hidden layers.
        residual (bool): full-connected of residual block for the hidden layers. Default: True.
        act (Union[str, Cell, Primitive, None]): activate function applied to the output of the fully connected layer,
            eg. 'ReLU'.Default: "sin".
        weight_init (Union[Tensor, str, Initializer, numbers.Number]): The trainable weight_init parameter. The dtype
            is same as input x. The values of str refer to the function `initializer`. Default: 'normal'.
        has_bias (bool): Specifies whether the layer uses a bias vector. Default: True.
        bias_init (Union[Tensor, str, Initializer, numbers.Number]): The trainable bias_init parameter. The dtype
            is same as input x. The values of str refer to the function `initializer`. Default: 'default'.
        num_scales (int): The subnet number of multi-scale network. Default: 4
        amp_factor (Union[int, float]): The amplification factor of input. Default: 1.0
        scale_factor (Union[int, float]): The base scale factor. Default: 2.0
        input_scale (Union[list, None]): The scale factor of input x/y/t. If not None, the inputs will be scaled before
            set in the network. Default: None.
        input_center (Union[list, None]): Center position of coordinate translation. If not None, the inputs will be
            translated before set in the network. Default: None.
        latent_vector (Union[Parameter, None]): Trainable papameter which will be concated will the sampling inputs
            and updated during training. Default: None.

    Inputs:
        - **input** (Tensor) - Tensor of shape :math:`(*, in\_channels)`.

    Outputs:
        Tensor of shape :math:`(*, out\_channels)`.

    Supported Platforms:
        ``Ascend``

    Raises:
        TypeError: If `num_scales` is not an int.
        TypeError: If `amp_factor` is neither int nor float.
        TypeError: If `scale_factor` is neither int nor float.
        TypeError: If `latent_vector` is neither a Parameter nor None.

    Examples:
        >>> import numpy as np
        >>> from mindelec.architecture import MultiScaleFCCell
        >>> from mindspore import Tensor, Parameter
        >>> inputs = np.ones((64,3)) + 3.0
        >>> inputs = Tensor(inputs.astype(np.float32))
        >>> num_scenarios = 4
        >>> latent_size = 16
        >>> latent_init = np.ones((num_scenarios, latent_size)).astype(np.float32)
        >>> latent_vector = Parameter(Tensor(latent_init), requires_grad=True)
        >>> input_scale = [1.0, 2.0, 4.0]
        >>> input_center = [3.5, 3.5, 3.5]
        >>> net = MultiScaleFCCell(3, 3, 5, 32,
        ...                        weight_init="ones", bias_init="zeros",
        ...                        input_scale=input_scale, input_center=input_center, latent_vector=latent_vector)
        >>> output = net(inputs).asnumpy()
        >>> print(output.shape)
        (64, 3)
    """
    def __init__(self,
                 in_channel,
                 out_channel,
                 layers,
                 neurons,
                 residual=True,
                 act="sin",
                 weight_init='normal',
                 has_bias=True,
                 bias_init="default",
                 num_scales=4,
                 amp_factor=1.0,
                 scale_factor=2.0,
                 input_scale=None,
                 input_center=None,
                 latent_vector=None
                 ):
        super(MultiScaleFCCell, self).__init__()
        check_mode("MultiScaleFCCell")
        _check_type(num_scales, "num_scales", int)
        _check_type(amp_factor, "amp_factor", (int, float))
        _check_type(scale_factor, "scale_factor", (int, float))

        self.cell_list = nn.CellList()
        self.num_scales = num_scales
        self.scale_coef = [amp_factor * (scale_factor**i) for i in range(self.num_scales)]

        self.latent_vector = latent_vector
        if self.latent_vector is not None:
            _check_type(latent_vector, "latent_vector", Parameter)
            self.num_scenarios = latent_vector.shape[0]
            self.latent_size = latent_vector.shape[1]
            in_channel += self.latent_size
        else:
            self.num_scenarios = 1
            self.latent_size = 0

        for _ in range(self.num_scales):
            self.cell_list.append(FCSequential(in_channel=in_channel,
                                               out_channel=out_channel,
                                               layers=layers,
                                               neurons=neurons,
                                               residual=residual,
                                               act=act,
                                               weight_init=weight_init,
                                               has_bias=has_bias,
                                               bias_init=bias_init))
        if input_scale:
            self.input_scale = InputScaleNet(input_scale, input_center)
        else:
            self.input_scale = ops.Identity()

        self.cast = ops.Cast()
        self.concat = ops.Concat(axis=1)

    def construct(self, x):
        """running multi-scale net"""
        x = self.input_scale(x)
        if self.latent_vector is not None:
            batch_size = x.shape[0]
            latent_vectors = self.latent_vector.view(self.num_scenarios, 1, \
                self.latent_size).repeat(batch_size // self.num_scenarios, axis=1).view((-1, self.latent_size))
            x = self.concat((x, latent_vectors))
        out = 0
        for i in range(self.num_scales):
            x_s = x * self.scale_coef[i]
            out = out + self.cast(self.cell_list[i](x_s), mstype.float32)
        return out


def _check_type(param, param_name, param_type):
    """check type"""
    if not isinstance(param, param_type):
        raise TypeError("The type of {} should be instance of {}, but got {}".format(
            param_name, param_type, type(param)))
