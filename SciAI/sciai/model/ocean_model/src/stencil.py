
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

"""stencil operations"""
import mindspore.nn as nn
from mindspore.ops import operations as P
from .oa_operator import AXBKernel, AXFKernel, AYBKernel, AYFKernel, AZBKernel, AZFKernel
from .oa_operator import DXBKernel, DXFKernel, DYBKernel, DYFKernel, DZBKernel, DZFKernel


class AXB(nn.Cell):
    """
    backward averaging operation along x direction
    output = (input[i, j, k] + input[i-1, j, k]) / 2

    Inputs:
        - **input** (Tensor) - The input should be a 3-dimension tensor.

    Outputs:
        Tensor, the shape is the same as the input.

    Supported Platforms:
        ``Ascend`` ``GPU``
    """
    def __init__(self, stencil_width=1):
        super(AXB, self).__init__()
        self.stencil_width = stencil_width
        self.pad = P.Pad(((self.stencil_width, self.stencil_width), (self.stencil_width, self.stencil_width),
                          (self.stencil_width, self.stencil_width)))
        self.slice = P.Slice()
        self.shape = P.Shape()
        self.axb_kernel = AXBKernel()

    def construct(self, x):
        x1 = self.pad(x)
        x_shape = self.shape(x)
        x1 = self.axb_kernel(x1)
        x1 = self.slice(x1, (self.stencil_width, self.stencil_width, self.stencil_width), x_shape)
        return x1


class AXF(nn.Cell):
    """
    forward averaging operation along x direction
    output = (input[i, j, k] + input[i+1, j, k]) / 2

    Inputs:
        - **input** (Tensor) - The input should be a 3-dimension tensor.

    Outputs:
        Tensor, the shape is the same as the input.

    Supported Platforms:
        ``Ascend`` ``GPU``
    """
    def __init__(self, stencil_width=1):
        super(AXF, self).__init__()
        self.stencil_width = stencil_width
        self.pad = P.Pad(((self.stencil_width, self.stencil_width), (self.stencil_width, self.stencil_width),
                          (self.stencil_width, self.stencil_width)))
        self.slice = P.Slice()
        self.shape = P.Shape()
        self.axf_kernel = AXFKernel()

    def construct(self, x):
        x1 = self.pad(x)
        x_shape = self.shape(x)
        x1 = self.axf_kernel(x1)
        x1 = self.slice(x1, (self.stencil_width, self.stencil_width, self.stencil_width), x_shape)
        return x1


class AYB(nn.Cell):
    """
    backward averaging operation along y direction
    output = (input[i, j, k] + input[i, j-1, k]) / 2

    Inputs:
        - **input** (Tensor) - The input should be a 3-dimension tensor.

    Outputs:
        Tensor, the shape is the same as the input.

    Supported Platforms:
        ``Ascend`` ``GPU``
    """
    def __init__(self, stencil_width=1):
        super(AYB, self).__init__()
        self.stencil_width = stencil_width
        self.pad = P.Pad(((self.stencil_width, self.stencil_width), (self.stencil_width, self.stencil_width),
                          (self.stencil_width, self.stencil_width)))
        self.slice = P.Slice()
        self.shape = P.Shape()
        self.ayb_kernel = AYBKernel()

    def construct(self, x):
        x1 = self.pad(x)
        x_shape = self.shape(x)
        x1 = self.ayb_kernel(x1)
        x1 = self.slice(x1, (self.stencil_width, self.stencil_width, self.stencil_width), x_shape)
        return x1


class AYF(nn.Cell):
    """
    forward averaging operation along y direction
    output = (input[i, j, k] + input[i, j+1, k]) / 2

    Inputs:
        - **input** (Tensor) - The input should be a 3-dimension tensor.

    Outputs:
        Tensor, the shape is the same as the input.

    Supported Platforms:
        ``Ascend`` ``GPU``
    """
    def __init__(self, stencil_width=1):
        super(AYF, self).__init__()
        self.stencil_width = stencil_width
        self.pad = P.Pad(((self.stencil_width, self.stencil_width), (self.stencil_width, self.stencil_width),
                          (self.stencil_width, self.stencil_width)))
        self.slice = P.Slice()
        self.shape = P.Shape()
        self.ayf_kernel = AYFKernel()

    def construct(self, x):
        x1 = self.pad(x)
        x_shape = self.shape(x)
        x1 = self.ayf_kernel(x1)
        x1 = self.slice(x1, (self.stencil_width, self.stencil_width, self.stencil_width), x_shape)
        return x1


class AZB(nn.Cell):
    """
    backward averaging operation along z direction
    output = (input[i, j, k] + input[i, j, k-1]) / 2

    Inputs:
        - **input** (Tensor) - The input should be a 3-dimension tensor.

    Outputs:
        Tensor, the shape is the same as the input.

    Supported Platforms:
        ``Ascend`` ``GPU``
    """
    def __init__(self, stencil_width=1):
        super(AZB, self).__init__()
        self.stencil_width = stencil_width
        self.pad = P.Pad(((self.stencil_width, self.stencil_width), (self.stencil_width, self.stencil_width),
                          (self.stencil_width, self.stencil_width)))
        self.slice = P.Slice()
        self.shape = P.Shape()
        self.azb_kernel = AZBKernel()

    def construct(self, x):
        x1 = self.pad(x)
        x_shape = self.shape(x)
        x1 = self.azb_kernel(x1)
        x1 = self.slice(x1, (self.stencil_width, self.stencil_width, self.stencil_width), x_shape)
        return x1


class AZF(nn.Cell):
    """
    forward averaging operation along z direction
    output = (input[i, j, k] + input[i, j, k+1]) / 2

    Inputs:
        - **input** (Tensor) - The input should be a 3-dimension tensor.

    Outputs:
        Tensor, the shape is the same as the input.

    Supported Platforms:
        ``Ascend`` ``GPU``
    """
    def __init__(self, stencil_width=1):
        super(AZF, self).__init__()
        self.stencil_width = stencil_width
        self.pad = P.Pad(((self.stencil_width, self.stencil_width), (self.stencil_width, self.stencil_width),
                          (self.stencil_width, self.stencil_width)))
        self.slice = P.Slice()
        self.shape = P.Shape()
        self.azf_kernel = AZFKernel()

    def construct(self, x):
        x1 = self.pad(x)
        x_shape = self.shape(x)
        x1 = self.azf_kernel(x1)
        x1 = self.slice(x1, (self.stencil_width, self.stencil_width, self.stencil_width), x_shape)
        return x1


class DXB(nn.Cell):
    """
    backward differential operation along x direction
    output = input[i, j, k] - input[i-1, j, k]

    Inputs:
        - **input** (Tensor) - The input should be a 3-dimension tensor.

    Outputs:
        Tensor, the shape is the same as the input.

    Supported Platforms:
        ``Ascend`` ``GPU``
    """
    def __init__(self, stencil_width=1):
        super(DXB, self).__init__()
        self.stencil_width = stencil_width
        self.pad = P.Pad(((self.stencil_width, self.stencil_width), (self.stencil_width, self.stencil_width),
                          (self.stencil_width, self.stencil_width)))
        self.slice = P.Slice()
        self.shape = P.Shape()
        self.dxb_kernel = DXBKernel()

    def construct(self, x):
        x1 = self.pad(x)
        x_shape = self.shape(x)
        x1 = self.dxb_kernel(x1)
        x1 = self.slice(x1, (self.stencil_width, self.stencil_width, self.stencil_width), x_shape)
        return x1


class DXF(nn.Cell):
    """
    forward differential operation along x direction
    output = input[i+1, j, k] - input[i, j, k]

    Inputs:
        - **input** (Tensor) - The input should be a 3-dimension tensor.

    Outputs:
        Tensor, the shape is the same as the input.

    Supported Platforms:
        ``Ascend`` ``GPU``
    """
    def __init__(self, stencil_width=1):
        super(DXF, self).__init__()
        self.stencil_width = stencil_width
        self.pad = P.Pad(((self.stencil_width, self.stencil_width), (self.stencil_width, self.stencil_width),
                          (self.stencil_width, self.stencil_width)))
        self.slice = P.Slice()
        self.shape = P.Shape()
        self.dxf_kernel = DXFKernel()

    def construct(self, x):
        x1 = self.pad(x)
        x_shape = self.shape(x)
        x1 = self.dxf_kernel(x1)
        x1 = self.slice(x1, (self.stencil_width, self.stencil_width, self.stencil_width), x_shape)
        return x1


class DYB(nn.Cell):
    """
    backward differential operation along y direction
    output = input[i, j, k] - input[i, j-1, k]

    Inputs:
        - **input** (Tensor) - The input should be a 3-dimension tensor.

    Outputs:
        Tensor, the shape is the same as the input.

    Supported Platforms:
        ``Ascend`` ``GPU``
    """
    def __init__(self, stencil_width=1):
        super(DYB, self).__init__()
        self.stencil_width = stencil_width
        self.pad = P.Pad(((self.stencil_width, self.stencil_width), (self.stencil_width, self.stencil_width),
                          (self.stencil_width, self.stencil_width)))
        self.slice = P.Slice()
        self.shape = P.Shape()
        self.dyb_kernel = DYBKernel()

    def construct(self, x):
        x1 = self.pad(x)
        x_shape = self.shape(x)
        x1 = self.dyb_kernel(x1)
        x1 = self.slice(x1, (self.stencil_width, self.stencil_width, self.stencil_width), x_shape)
        return x1


class DYF(nn.Cell):
    """
    forward differential operation along y direction
    output = input[i, j+1, k] - input[i, j, k]

    Inputs:
        - **input** (Tensor) - The input should be a 3-dimension tensor.

    Outputs:
        Tensor, the shape is the same as the input.

    Supported Platforms:
        ``Ascend`` ``GPU``
    """
    def __init__(self, stencil_width=1):
        super(DYF, self).__init__()
        self.stencil_width = stencil_width
        self.pad = P.Pad(((self.stencil_width, self.stencil_width), (self.stencil_width, self.stencil_width),
                          (self.stencil_width, self.stencil_width)))
        self.slice = P.Slice()
        self.shape = P.Shape()
        self.dyf_kernel = DYFKernel()

    def construct(self, x):
        x1 = self.pad(x)
        x_shape = self.shape(x)
        x1 = self.dyf_kernel(x1)
        x1 = self.slice(x1, (self.stencil_width, self.stencil_width, self.stencil_width), x_shape)
        return x1


class DZB(nn.Cell):
    """
    backward differential operation along z direction
    output = input[i, j, k] - input[i, j, k-1]

    Inputs:
        - **input** (Tensor) - The input should be a 3-dimension tensor.

    Outputs:
        Tensor, the shape is the same as the input.

    Supported Platforms:
        ``Ascend`` ``GPU``
    """
    def __init__(self, stencil_width=1):
        super(DZB, self).__init__()
        self.stencil_width = stencil_width
        self.pad = P.Pad(((self.stencil_width, self.stencil_width), (self.stencil_width, self.stencil_width),
                          (self.stencil_width, self.stencil_width)))
        self.slice = P.Slice()
        self.shape = P.Shape()
        self.dzb_kernel = DZBKernel()

    def construct(self, x):
        x1 = self.pad(x)
        x_shape = self.shape(x)
        x1 = self.dzb_kernel(x1)
        x1 = self.slice(x1, (self.stencil_width, self.stencil_width, self.stencil_width), x_shape)
        return x1


class DZF(nn.Cell):
    """
    forward differential operation along z direction
    output = input[i, j, k+1] - input[i, j, k]

    Inputs:
        - **input** (Tensor) - The input should be a 3-dimension tensor.

    Outputs:
        Tensor, the shape is the same as the input.

    Supported Platforms:
        ``Ascend`` ``GPU``
    """

    def __init__(self, stencil_width=1):
        super(DZF, self).__init__()
        self.stencil_width = stencil_width
        self.pad = P.Pad(((self.stencil_width, self.stencil_width), (self.stencil_width, self.stencil_width),
                          (self.stencil_width, self.stencil_width)))
        self.slice = P.Slice()
        self.shape = P.Shape()
        self.dzf_kernel = DZFKernel()

    def construct(self, x):
        x1 = self.pad(x)
        x_shape = self.shape(x)
        x1 = self.dzf_kernel(x1)
        x1 = self.slice(x1, (self.stencil_width, self.stencil_width, self.stencil_width), x_shape)
        return x1
