# Copyright 2022 Huawei Technologies Co., Ltd
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
"""Moment(sum rules) and Kernel(convolution kernel) converter"""
import numpy as np
from scipy.special import factorial

import mindspore.common.dtype as mstype
from mindspore import nn, ops, Tensor
from mindspore.numpy import tensordot


class _M2K(nn.Cell):
    '''M2K module'''
    def __init__(self, shape):
        super(_M2K, self).__init__()
        self._shape = shape
        self._ndim = len(shape)
        self._m = []
        self._invm = []
        self.cast = ops.Cast()
        for l in shape:
            zero_to_l = np.arange(l)
            mat = np.power(zero_to_l - l // 2, zero_to_l[:, None]) / factorial(zero_to_l[:, None])
            self._m.append(Tensor.from_numpy(mat))
            self._invm.append(Tensor.from_numpy(np.linalg.inv(mat)))

    @property
    def m(self):
        return self._m

    @property
    def invm(self):
        return self._invm

    @property
    def shape(self):
        return self._shape

    @property
    def ndim(self):
        return self._ndim

    def _packdim(self, x):
        if x.ndim == self.ndim:
            x = x[None, :]
        x = x.view((-1, x.shape[1], x.shape[2]))
        return x

    def _apply_axis_left_dot(self, x, mats):
        x_shape = x.shape
        k = x.ndim - 1
        for i in range(k):
            x = tensordot(mats[k - i - 1].astype(mstype.float32), x.astype(mstype.float32), axes=[1, k])
        x = ops.transpose(x, (2, 0, 1))
        x = x.view(x_shape)
        return x

    def construct(self, m):
        m_size = m.shape
        m = self.cast(m, mstype.float32)
        m = self._packdim(m)
        k = self._apply_axis_left_dot(m, self.invm)
        k = k.view(m_size)
        return k
