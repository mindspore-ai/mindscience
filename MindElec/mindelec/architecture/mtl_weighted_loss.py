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
"""mtl_weighted_loss"""
from __future__ import absolute_import

import numpy as np

import mindspore.ops as ops
import mindspore.nn as nn
import mindspore.common.dtype as mstype
from mindspore.common.tensor import Tensor
from mindspore import Parameter
from .util import check_mode

__all__ = ['MTLWeightedLossCell']


class MTLWeightedLossCell(nn.Cell):
    r"""
    The MTL strategy weighted multi-task losses automatically.

    Args:
        num_losses (int): The number of multi-task losses, should be positive integer.

    Inputs:
        - **input** - tuple of Tensors.

    Outputs:
        Tensor.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import numpy as np
        >>> from mindelec.architecture import MTLWeightedLossCell
        >>> import mindspore
        >>> from mindspore import Tensor
        >>> net = MTLWeightedLossCell(num_losses=2)
        >>> input1 = Tensor(1.0, mindspore.float32)
        >>> input2 = Tensor(0.8, mindspore.float32)
        >>> output = net((input1, input2))
        >>> print(output)
        2.2862945
    """
    def __init__(self, num_losses):
        super(MTLWeightedLossCell, self).__init__(auto_prefix=False)
        check_mode("MTLWeightedLossCell")
        if not isinstance(num_losses, int):
            raise TypeError("the type of num_losses should be int, but got {}".format(type(num_losses)))
        if num_losses <= 0:
            raise ValueError("the value of num_losses should be positive, but got {}".format(num_losses))
        self.num_losses = num_losses
        self.params = Parameter(Tensor(np.ones(num_losses), mstype.float32), requires_grad=True)
        self.concat = ops.Concat(axis=0)
        self.pow = ops.Pow()
        self.log = ops.Log()
        self.div = ops.RealDiv()

    def construct(self, losses):
        loss_sum = 0
        params = self.pow(self.params, 2)
        for i in range(self.num_losses):
            weighted_loss = 0.5 * self.div(losses[i], params[i]) + self.log(params[i] + 1.0)
            loss_sum = loss_sum + weighted_loss
        return loss_sum
