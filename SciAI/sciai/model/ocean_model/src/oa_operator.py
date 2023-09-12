
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
# ============================================================================

"""stencil operations kernel"""
import mindspore.nn as nn
from mindspore.ops import operations as P


class AXBKernel(nn.Cell):
    """create axb_kernel"""
    def __init__(self):
        super(AXBKernel, self).__init__()
        self.pad = P.Pad(((1, 0), (0, 0), (0, 0)))
        self.slice = P.Slice()
        self.shape = P.Shape()

    def construct(self, x):
        x1 = self.pad(x)
        x_shape = self.shape(x)
        x1 = self.slice(x1, (0, 0, 0), x_shape)
        out = 0.5 * (x + x1)
        return out


class AYBKernel(nn.Cell):
    """create ayb_kernel"""
    def __init__(self):
        super(AYBKernel, self).__init__()
        self.pad = P.Pad(((0, 0), (1, 0), (0, 0)))
        self.slice = P.Slice()
        self.shape = P.Shape()

    def construct(self, x):
        x1 = self.pad(x)
        x_shape = self.shape(x)
        x1 = self.slice(x1, (0, 0, 0), x_shape)
        out = 0.5 * (x + x1)
        return out


class AZBKernel(nn.Cell):
    """create azb_kernel"""
    def __init__(self):
        super(AZBKernel, self).__init__()
        self.pad = P.Pad(((0, 0), (0, 0), (1, 0)))
        self.slice = P.Slice()
        self.shape = P.Shape()

    def construct(self, x):
        x1 = self.pad(x)
        x_shape = self.shape(x)
        x1 = self.slice(x1, (0, 0, 0), x_shape)
        out = 0.5 * (x + x1)
        return out


class AXFKernel(nn.Cell):
    """create axf_kernel"""
    def __init__(self):
        super(AXFKernel, self).__init__()
        self.pad = P.Pad(((0, 1), (0, 0), (0, 0)))
        self.slice = P.Slice()
        self.shape = P.Shape()

    def construct(self, x):
        x1 = self.pad(x)
        x_shape = self.shape(x)
        x1 = self.slice(x1, (1, 0, 0), x_shape)
        out = 0.5 * (x + x1)
        return out


class AYFKernel(nn.Cell):
    """create ayf_kernel"""
    def __init__(self):
        super(AYFKernel, self).__init__()
        self.pad = P.Pad(((0, 0), (0, 1), (0, 0)))
        self.slice = P.Slice()
        self.shape = P.Shape()

    def construct(self, x):
        x1 = self.pad(x)
        x_shape = self.shape(x)
        x1 = self.slice(x1, (0, 1, 0), x_shape)
        out = 0.5 * (x + x1)
        return out


class AZFKernel(nn.Cell):
    """create azf_kernel"""
    def __init__(self):
        super(AZFKernel, self).__init__()
        self.pad = P.Pad(((0, 0), (0, 0), (0, 1)))
        self.slice = P.Slice()
        self.shape = P.Shape()

    def construct(self, x):
        x1 = self.pad(x)
        x_shape = self.shape(x)
        x1 = self.slice(x1, (0, 0, 1), x_shape)
        out = 0.5 * (x + x1)
        return out


class DXBKernel(nn.Cell):
    """create dxb_kernel"""
    def __init__(self):
        super(DXBKernel, self).__init__()
        self.pad = P.Pad(((1, 0), (0, 0), (0, 0)))
        self.slice = P.Slice()
        self.shape = P.Shape()

    def construct(self, x):
        x1 = self.pad(x)
        x_shape = self.shape(x)
        x1 = self.slice(x1, (0, 0, 0), x_shape)
        x = x - x1
        return x


class DXFKernel(nn.Cell):
    """create dxf_kernel"""
    def __init__(self):
        super(DXFKernel, self).__init__()
        self.pad = P.Pad(((0, 1), (0, 0), (0, 0)))
        self.slice = P.Slice()
        self.shape = P.Shape()

    def construct(self, x):
        x1 = self.pad(x)
        x_shape = self.shape(x)
        x1 = self.slice(x1, (1, 0, 0), x_shape)
        x = x1 - x
        return x


class DYBKernel(nn.Cell):
    """create dyb_kernel"""
    def __init__(self):
        super(DYBKernel, self).__init__()
        self.pad = P.Pad(((0, 0), (1, 0), (0, 0)))
        self.slice = P.Slice()
        self.shape = P.Shape()

    def construct(self, x):
        x1 = self.pad(x)
        x_shape = self.shape(x)
        x1 = self.slice(x1, (0, 0, 0), x_shape)
        x = x - x1
        return x


class DYFKernel(nn.Cell):
    """create dyf_kernel"""
    def __init__(self):
        super(DYFKernel, self).__init__()
        self.pad = P.Pad(((0, 0), (0, 1), (0, 0)))
        self.slice = P.Slice()
        self.shape = P.Shape()

    def construct(self, x):
        x1 = self.pad(x)
        x_shape = self.shape(x)
        x1 = self.slice(x1, (0, 1, 0), x_shape)
        x = x1 - x
        return x


class DZBKernel(nn.Cell):
    """create dzb_kernel"""
    def __init__(self):
        super(DZBKernel, self).__init__()
        self.pad = P.Pad(((0, 0), (0, 0), (1, 0)))
        self.slice = P.Slice()
        self.shape = P.Shape()

    def construct(self, x):
        x1 = self.pad(x)
        x_shape = self.shape(x)
        x1 = self.slice(x1, (0, 0, 0), x_shape)
        x = x - x1
        return x


class DZFKernel(nn.Cell):
    """create dzf_kernel"""
    def __init__(self):
        super(DZFKernel, self).__init__()
        self.pad = P.Pad(((0, 0), (0, 0), (0, 1)))
        self.slice = P.Slice()
        self.shape = P.Shape()

    def construct(self, x):
        x1 = self.pad(x)
        x_shape = self.shape(x)
        x1 = self.slice(x1, (0, 0, 1), x_shape)
        x = x1 - x
        return x
