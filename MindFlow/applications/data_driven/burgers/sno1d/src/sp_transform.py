# Copyright 2024 Huawei Technologies Co., Ltd
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
"""spectral transform"""
from enum import Enum

import mindspore.common.dtype as mstype
from mindspore import ops, nn


class Dim(Enum):
    x = 'x'
    y = 'y'
    z = 'z'


class ConvCell(nn.Cell):
    """Convolution Cell"""
    def __init__(self, dim: int, in_channels: int, out_channels: int,
                 kernel_size: int = 1, compute_dtype=mstype.float32):
        super().__init__()
        self.dim = dim
        self.conv = None
        args = {
            'in_channels': in_channels,
            'out_channels': out_channels,
            'kernel_size': kernel_size
            }
        if kernel_size > 1:
            pad_args = {
                'pad_mode': 'pad',
                'padding': kernel_size // 2
            }
            args.update(pad_args)
        self.conv = nn.Conv1d(**args).to_float(compute_dtype)

    def construct(self, x):
        return self.conv(x)


class TransformCell1D(nn.Cell):
    """ Linear layer, performs polynomial transform in 1D"""
    def __init__(self, transform):
        super().__init__()
        self.transform = ops.permute(transform, (1, 0))
    def construct(self, x):
        return ops.matmul(x, self.transform)


class TransformCell(nn.Cell):
    """ Linear layer, performs polynomial transform on the specified axis"""
    def __init__(self, transform):
        super().__init__()
        self.cell = ops.Identity()
        self.transform = transform
        self.dtype = self.transform.dtype
        self.cell = TransformCell1D(transform)

    def construct(self, x):
        xdtype = x.dtype
        if xdtype != self.dtype:
            cast = ops.Cast()
            x = cast(x, self.dtype)
            x = self.cell(x)
            return cast(x, xdtype)
        return self.cell(x)
