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
"""initializer"""
import numpy as np
import scipy.stats
from mindspore.common.initializer import Initializer, _assignment, _register


@_register('lecun_normal')
class LeCunNormal(Initializer):
    r"""
    Yann LeCun Normal Initialization
    :math:`{N}(0, \text{sigma}^2)` in order to initialize a tensor, where

    .. math::
        sigma = \sqrt{\frac{1}{fan\_in}}

    'fan_in' is the number of input units of the weight tensor.

    For details of LeCun Normal Initialization, please check:
    `Neural Tangent Kernel: Convergence and Generalization in Neural Networks
    <https://proceedings.neurips.cc/paper/2018/hash/5a4be1fa34e62bb8a6ec6b91d2462f5a-Abstract.html>`_.

    Supported Platforms:
        ``GPU`` ``CPU`` ``Ascend``

    Examples:
        >>> import mindspore
        >>> from mindspore.common.initializer import initializer
        >>> from sciai.common.initializer import LeCunNormal
        >>> tensor = initializer(LeCunNormal(), [1, 2, 3], mindspore.float32)
    """

    def __init__(self):  # pylint: disable=W0235
        super().__init__()

    def _initialize(self, arr):
        shape = arr.shape
        in_dim, out_dim = (shape[0], shape[1]) if len(shape) == 2 else (1, shape[0])
        std = np.sqrt(1. / in_dim)
        res = np.random.normal(loc=0, scale=std, size=(in_dim, out_dim))
        _assignment(arr, res)


@_register('lecun_uniform')
class LeCunUniform(Initializer):
    r"""
    Yann LeCun Normal Initialization
    :math:`{U}(-\text{boundary}, \text{boundary})` in order to initialize a tensor, where

    .. math::
        boundary = \sqrt{\frac{3}{fan\_in}}

    'fan_in' is the number of input units of the weight tensor.

    For details of LeCun Uniform Initialization, please check:
    `Neural Tangent Kernel: Convergence and Generalization in Neural Networks
    <https://proceedings.neurips.cc/paper/2018/hash/5a4be1fa34e62bb8a6ec6b91d2462f5a-Abstract.html>`_.

    Supported Platforms:
        ``GPU`` ``CPU`` ``Ascend``

    Examples:
        >>> import mindspore
        >>> from mindspore.common.initializer import initializer
        >>> from sciai.common.initializer import LeCunUniform
        >>> tensor = initializer(LeCunUniform(), [1, 2, 3], mindspore.float32)
    """

    def __init__(self):  # pylint: disable=W0235
        super().__init__()

    def _initialize(self, arr):
        shape = arr.shape
        in_dim, out_dim = (shape[0], shape[1]) if len(shape) == 2 else (1, shape[0])
        bound = np.sqrt(3. / in_dim)
        res = np.random.uniform(low=-bound, high=bound, size=(in_dim, out_dim))
        _assignment(arr, res)


@_register('standard_uniform')
class StandardUniform(Initializer):
    r"""
    Generates an array with values sampled from Standard Uniform distribution
    :math:`{U}(-\text{boundary}, \text{boundary})` in order to initialize a tensor, where

    .. math::
        boundary = \sqrt{\frac{1}{fan\_in}}

    'fan_in' is the number of input units of the weight tensor.

    Supported Platforms:
        ``GPU`` ``CPU`` ``Ascend``

    Examples:
        >>> import mindspore
        >>> from mindspore.common.initializer import initializer
        >>> from sciai.common.initializer import StandardUniform
        >>> tensor = initializer(StandardUniform(), [1, 2, 3], mindspore.float32)
    """

    def __init__(self):  # pylint: disable=W0235
        super().__init__()

    def _initialize(self, arr):
        shape = arr.shape
        in_dim, out_dim = (shape[0], shape[1]) if len(shape) == 2 else (1, shape[0])
        bound = np.sqrt(1. / in_dim)
        res = np.random.uniform(low=-bound, high=bound, size=(in_dim, out_dim))
        _assignment(arr, res)


@_register('xavier_trunc_normal')
class XavierTruncNormal(Initializer):
    """
    Xavier Truncated Normal Initialization with clip of 2 times of stddev from mean of Xavier Normal Initialization.

    Args:
         trunc_interval (Union[None, tuple[Number]]): Truncated normal interval. If (-2, 2), discarding and re-drawing
             any samples that are more than two standard deviations from mean 0. Default: (-2, 2).

    Supported Platforms:
        ``GPU`` ``CPU`` ``Ascend``

    Examples:
        >>> import mindspore
        >>> from mindspore.common.initializer import initializer
        >>> from sciai.common.initializer import XavierTruncNormal
        >>> tensor = initializer(XavierTruncNormal(trunc_interval=(-2, 2)), [1, 2, 3], mindspore.float32)
    """

    def __init__(self, trunc_interval=(-2, 2)):
        super(XavierTruncNormal, self).__init__()
        self.trunc_interval = trunc_interval

    def _initialize(self, arr):
        shape = arr.shape
        in_dim, out_dim = (shape[0], shape[1]) if len(shape) == 2 else (1, shape[0])
        xavier_stddev = np.sqrt(2. / (in_dim + out_dim))
        res = scipy.stats.truncnorm.rvs(*self.trunc_interval, loc=0, scale=xavier_stddev, size=(in_dim, out_dim))
        _assignment(arr, res)
