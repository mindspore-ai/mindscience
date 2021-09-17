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
# ============================================================================
"""
metric
"""

import numpy as np
import mindspore.nn as nn


class L2(nn.Metric):
    r"""
    Calculates l2 metric.

    Creates a criterion that measures the l2 metric between each element
    in the input: :math:`x` and the target: :math:`y`.

    .. math::
        \text{l2} = \sqrt {\sum_{i=1}^n \frac {(y_i - x_i)^2}{y_i^2}}

    Here :math:`y_i` is the true value and :math:`x_i` is the prediction.

    Note:
        The method `update` must be called with the form `update(y_pred, y)`.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import numpy as np
        >>> from mindelec.common import L2
        >>> from mindspore import nn, Tensor
        >>> import mindspore
        ...
        >>> x = Tensor(np.array([0.1, 0.2, 0.6, 0.9]), mindspore.float32)
        >>> y = Tensor(np.array([0.1, 0.25, 0.7, 0.9]), mindspore.float32)
        >>> metric = L2()
        >>> metric.clear()
        >>> metric.update(x, y)
        >>> result = metric.eval()
        >>> print(result)
        0.09543302997807275
    """

    def __init__(self):
        super(L2, self).__init__()
        self.clear()

    def clear(self):
        """clear the internal evaluation result."""
        self.square_error_sum = 0
        self.square_label_sum = 0

    def update(self, *inputs):
        """
        Updates the internal evaluation result :math:`y_{pred}` and :math:`y`.

        Args:
            inputs: Input `y_pred` and `y` for calculating L2 where the shape of
                `y_pred` and `y` are same. The input data type must be tensor, list or numpy.array.

        Raises:
            ValueError: if the length of inputs is not 2.
            ValueError: if the shape of y_pred and y are not same.
        """
        if len(inputs) != 2:
            raise ValueError("The L2 needs 2 inputs (y_pred, y), but got {}".format(inputs))
        y_pred = self._convert_data(inputs[0])
        y = self._convert_data(inputs[1])
        if y_pred.shape != y.shape:
            raise ValueError("The shape of y_pred and y should be same but got y_pred: {} and y: {}"
                             .format(y_pred.shape, y.shape))

        square_error_sum = np.square(y.reshape(y_pred.shape) - y_pred)
        self.square_error_sum += square_error_sum.sum()
        square_label_sum = np.square(y)
        self.square_label_sum += square_label_sum.sum()

    def eval(self):
        """
        Computes l2 metric.

        Returns:
            Float, the computed result.

        """
        return np.sqrt(self.square_error_sum / self.square_label_sum)
