# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
'''Module providing loss functions'''

from mindspore import nn, ops
from mindspore import dtype as mstype
from mindspore.ops import operations as P
from mindspore.ops import functional as F


class RelativeRMSELoss(nn.LossBase):
    r"""
    Relative Root Mean Square Error (RRMSE) is the root mean squared error normalized by the root mean square value
    where each residual is scaled against the actual value. Relative RMSELoss creates a criterion to measure the root
    mean square error between :math:`x` and :math:`y` element-wise,
    where :math:`x` is the input and :math:`y` is the labels.

    For simplicity, let :math:`x` and :math:`y` be 1-dimensional Tensor with length :math:`N`,
    the loss of :math:`x` and :math:`y` is given as:

    .. math::
        loss = \sqrt{\frac{\frac{1}{N}\sum_{i=1}^{N}{(x_i-y_i)^2}}{sum_{i=1}^{N}{(y_i)^2}}}

    Args:
        reduction (str): Type of reduction to be applied to loss. The optional values are ``"mean"``,
            ``"sum"``, and ``"none"``. Default: ``"sum"``.

    Inputs:
        - **prediction** (Tensor) - Tensor of shape :math:`(N, *)` where :math:`*` means, any number of
          additional dimensions.
        - **labels** (Tensor) - Tensor of shape :math:`(N, *)`, same shape as the `prediction` in common cases.
          However, it supports the shape of `prediction` is different from the shape of `label`
          and they should be broadcasted to each other.

    Outputs:
        Tensor, weighted loss float tensor.

        - **output** (Tensor) - Tensor of shape :math:`()`

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore
        >>> from mindspore import Tensor
        >>> from mindearth.core import RelativeRMSELoss
        >>> # Case: prediction.shape = labels.shape = (3, 3)
        >>> prediction = Tensor(np.array([[1, 2, 3],[1, 2, 3],[1, 2, 3]]), mindspore.float32)
        >>> labels = Tensor(np.array([[1, 2, 2],[1, 2, 3],[1, 2, 3]]), mindspore.float32)
        >>> loss_fn = RelativeRMSELoss()
        >>> loss = loss_fn(prediction, labels)
        >>> print(loss)
        0.11111112
    """

    def construct(self, prediction, labels):
        prediction = P.Cast()(prediction, mstype.float32)
        batch_size = prediction.shape[0]
        diff_norms = F.square(prediction.reshape(batch_size, -1) - labels.reshape(batch_size, -1)).sum(axis=1)
        label_norms = F.square(labels.reshape(batch_size, -1)).sum(axis=1)
        rel_error = ops.div(F.sqrt(diff_norms), F.sqrt(label_norms))
        rel_error = F.reduce_mean(rel_error)
        return rel_error
