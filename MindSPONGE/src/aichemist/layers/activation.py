# Copyright 2021-2022 @ Shenzhen Bay Laboratory &
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
Extra activation function
"""

import mindspore as ms
from mindspore import nn
from mindspore.nn import Cell
from mindspore.nn.layer import activation
from mindspore import ops
from mindspore.ops.primitive import Primitive, PrimitiveWithInfer, PrimitiveWithCheck

__all__ = [
    "ShiftedSoftplus",
    "Swish",
    "get_activation",
]


class ShiftedSoftplus(Cell):
    r"""Compute shifted soft-plus activation function.

    .. math::
       y = \ln\left(1 + e^{-x}\right) - \ln(2)

    Args:
        x (mindspore.Tensor): input tensor.

    Returns:
        mindspore.Tensor: shifted soft-plus of input.

    """

    def __init__(self):
        super().__init__()
        self.shift = ops.log(ms.Tensor(2.0))
        self.softplus = ops.Softplus()

    def __str__(self):
        return 'ShiftedSoftplus<>'

    def construct(self, x):
        # if softpuls has no bug, then use " return self.softplus(x) - self.shift"
        return ops.log1p(ops.exp(x)) - self.log2


class Swish(Cell):
    r"""Compute swish\SILU\SiL function.

    .. math::
       y_i = x_i / (1 + e^{-beta * x_i})

    Args:
        x (mindspore.Tensor): input tensor.

    Returns:
        mindspore.Tensor: shifted soft-plus of input.

    """

    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()

    def __str__(self):
        return 'Swish<>'

    def construct(self, x):
        return x * self.sigmoid(x)


_ACTIVATIONS_BY_KEY = {
    'ssp': ShiftedSoftplus,
    'swish': Swish,
}

# pylint: disable=protected-access
_ACTIVATIONS_BY_KEY.update(activation._activation)

_ACTIVATIONS_BY_NAME = {a.__name__: a for a in _ACTIVATIONS_BY_KEY.values()}


def get_activation(activation_) -> Cell:
    """
    Gets the activation function.

    Args:
        obj (str): The obj of the activation function.

    Returns:
        Function, the activation function.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> sigmoid = nn.get_activation('sigmoid')
        >>> print(sigmoid)
        Sigmoid<>
    """

    if activation_ is None:
        return None

    if isinstance(activation_, (Cell, Primitive, PrimitiveWithCheck,
                                PrimitiveWithInfer, PrimitiveWithCheck)):
        return activation_
    if isinstance(activation_, str):
        if activation_.lower() == 'none':
            return None
        if activation_.lower() in _ACTIVATIONS_BY_KEY:
            return _ACTIVATIONS_BY_KEY[activation_.lower()]()
        if activation_ in _ACTIVATIONS_BY_NAME.keys():
            return _ACTIVATIONS_BY_NAME[activation_]()
        raise ValueError(
            f"The activation corresponding to '{activation_}' was not found.")
    raise TypeError("Unsupported activation type: "+str(type(activation_)))
