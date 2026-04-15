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
Basic functions
"""
from mindspore import Tensor
from mindspore.nn import Cell
from mindspore.ops import functional as F
from ...configs import Config


class Aggregate(Cell):
    r"""A network to aggregate Tensor

    Args:
        axis (int):     Axis to aggregate.

        mean (bool):    Whether to average the Tensor. Default: ``False``.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """

    def __init__(self,
                 axis: int,
                 mean: bool = False,
                 **kwargs,
                 ):
        super().__init__()
        self._kwargs = Config.get_arguments(locals(), kwargs)

        self.average = mean
        self.axis = int(axis)

    def construct(self, inputs: Tensor, mask: Tensor = None):
        """To aggregate the representation of each nodes

        Args:
            inputs (Tensor):    Tensor with shape (B, A, N, F). Data type is float.
            mask (Tensor):      Tensor with shape (B, A, N). Data type is bool.

        Returns:
            output (Tensor):    Tensor with shape (B, A, F). Data type is float.

        """
        # mask input
        if mask is not None:
            inputs = inputs * F.expand_dims(mask, -1)
        # compute sum of input along axis

        y = F.reduce_sum(inputs, self.axis)
        # compute average of input along axis
        if self.average:
            # get the number of items along axis
            if mask is not None:
                num = F.reduce_sum(mask, self.axis)
                num = F.maximum(num, other=F.ones_like(num))
            else:
                num = inputs.shape[self.axis]

            y = y / num
        return y
