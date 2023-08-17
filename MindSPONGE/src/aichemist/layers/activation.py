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
Extra activation function
"""

import mindspore as ms
from mindspore import nn
from mindspore.nn import Cell
from mindspore import ops
from ..configs import Registry as R

__all__ = [
    "ShiftedSoftplus",
    "Swish",
]


@R.register('activation.ssp')
class ShiftedSoftplus(Cell):
    r"""Compute shifted soft-plus activation function.

    .. math::
       y = \ln\left(1 + e^{-x}\right) - \ln(2)

    Args:
        x (mindspore.Tensor): input tensor.

    Returns:
        mindspore.Tensor: shifted soft-plus of input.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """

    def __init__(self):
        super().__init__()
        self.shift = ops.log(ms.Tensor(2.0))
        self.softplus = ops.Softplus()

    def __str__(self):
        return 'ShiftedSoftplus<>'

    def construct(self, x):
        # if softpuls has no bug, then use " return self.softplus(x) - self.shift"
        return ops.log1p(ops.exp(x)) - self.shift


@R.register('activation.swish')
class Swish(Cell):
    r"""Compute swish\SILU\SiL function.

    .. math::
       y_i = x_i / (1 + e^{-beta * x_i})

    Args:
        x (mindspore.Tensor): input tensor.

    Returns:
        mindspore.Tensor: shifted soft-plus of input.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """

    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()

    def __str__(self):
        return 'Swish<>'

    def construct(self, x):
        return x * self.sigmoid(x)
