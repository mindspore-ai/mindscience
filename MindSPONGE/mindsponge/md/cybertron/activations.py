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
"""activations"""

from mindspore import nn
from mindspore.nn.layer.activation import _activation
from mindspore.ops import operations as P

__all__ = [
    "ShiftedSoftplus",
    "ScaledShiftedSoftplus",
    "Swish",
    "get_activation",
    ]


class ShiftedSoftplus(nn.Cell):
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
        self.log1p = P.Log1p()
        self.exp = P.Exp()
        self.ln2 = 0.6931471805599453

    def __str__(self):
        return "shifted_softplus"

    def construct(self, x):
        return self.log1p(self.exp(x)) - self.ln2


class ScaledShiftedSoftplus(nn.Cell):
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
        self.softplus = P.Softplus()
        self.ln2 = 0.6931471805599453

    def __str__(self):
        return "scaled_shifted_softplus"

    def construct(self, x):
        return 2 * (self.softplus(x) - self.ln2)


class Swish(nn.Cell):
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
        return "swish"

    def construct(self, x):
        return x * self.sigmoid(x)


_EXTENDED_ACTIVATIONS = {
    'shifted': ShiftedSoftplus,
    'scaledshifted': ScaledShiftedSoftplus,
    'swish': Swish,
}


def get_activation(name):
    """get activation"""
    if name is None or isinstance(name, nn.Cell):
        return name
    if isinstance(name, str):
        if name.lower() in _activation.keys():
            return name
        if name.lower() not in _EXTENDED_ACTIVATIONS.keys():
            raise ValueError("The class corresponding to '{}' was not found.".format(name))
        return _EXTENDED_ACTIVATIONS[name.lower()]()
    raise TypeError("Unsupported activation type '{}'.".format(type(name)))
