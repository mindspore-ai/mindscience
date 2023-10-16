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
"""basic"""
from __future__ import absolute_import

import numpy as np

import mindspore as ms
import mindspore.nn as nn
from mindspore import set_seed

set_seed(0)
np.random.seed(0)


class ResBlock(nn.Cell):
    r"""
    Digital Elevation Model is based on deep residual network and transfer learning.
    The details can be found in `Super-resolution reconstruction of a 3 arc-second global DEM dataset
    <https://pubmed.ncbi.nlm.nih.gov/36604030/>`_.

    Args:
        channels (int): The number of output channels.
        kernel_size (int): Kernel size.
        scale (float): Scale factor of the network.

    Inputs:
         - **input** (Tensor) - Tensor of shape :math:`(batch\_size, channels, height\_size, width\_size)`.

    Outputs:
        Tensor, the output of the DEMNet.

         - **output** (Tensor) - Tensor of shape :math:`(batch\_size, channels, new_height\_size, new_width\_size)`.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore as ms
        >>> from mindspore import ops, Tensor
        >>> from mindspore.nn import Cell
        >>> from mindearth.cell.dem_srnet.dem_srnet import ResBlock
        >>> input_images = np.random.rand(64, 256, 32, 32).astype(np.float32)
        >>> net = ResBlock(channels=256, kernel_size=3)
        >>> print(input_images.shape)
        >>> out = net(Tensor(input_images, ms.float32))
        >>> print(out.shape)
        (64, 256, 32, 32)
    """
    def __init__(self,
                 channels=256,
                 kernel_size=3,
                 scale=0.1):
        super(ResBlock, self).__init__()
        self.scale = scale
        self.res_conv1 = nn.Conv2d(channels, channels, kernel_size, pad_mode='same')
        self.relu = nn.ReLU()
        self.res_conv2 = nn.Conv2d(channels, channels, kernel_size, pad_mode='same')

    def construct(self, x):
        y = self.relu(self.res_conv1(x))
        y = self.res_conv2(y)
        y *= self.scale
        y += x
        return y


class DEMNet(nn.Cell):
    r"""
    Digital Elevation Model is based on deep residual network and transfer learning.
    The details can be found in `Super-resolution reconstruction of a 3 arc-second global DEM dataset
    <https://pubmed.ncbi.nlm.nih.gov/36604030/>`_.

    Args:
        in_channels(int): The channels of input image.
        out_channels (int): The number of output channels.
        kernel_size (int): Kernel size.
        scale (int): The scale factor of new size of the tensor.
        num_blocks (int): The number of blocks in the DEMNet.

    Inputs:
         - **x** (Tensor) - Tensor of shape :math:`(batch\_size, out_channels, height\_size, width\_size)`.

    Outputs:
        Tensor, the output of the DEMNet.

         - **output** (Tensor) - Tensor of shape :math:`(batch\_size, out_channels, new_height\_size, new_width\_size)`.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore as ms
        >>> from mindspore import ops, Tensor
        >>> from mindspore.nn import Cell
        >>> from mindearth.cell import DEMNet
        >>> input_images = np.random.rand(64, 1, 32, 32).astype(np.float32)
        >>> net = DEMNet(in_channels=1, out_channels=256, kernel_size=3, scale=5, num_blocks=42)
        >>> out = net(Tensor(input_images, ms.float32))
        >>> print(out.shape)
        (64, 1, 160, 160)
    """
    def __init__(self,
                 in_channels=1,
                 channels=256,
                 kernel_size=3,
                 scale=5,
                 num_blocks=42):
        super(DEMNet, self).__init__()
        self.scale = scale
        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size, pad_mode='same')
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, pad_mode='same')
        self.conv_up = nn.Conv2d(channels, channels, kernel_size, pad_mode='same')
        self.conv_out = nn.Conv2d(channels, in_channels, kernel_size, pad_mode='same')
        self.body = self.make_layer(ResBlock, num_blocks)

    def make_layer(self, block, layers):
        res_block = []
        for _ in range(layers):
            res_block.append(block())
        return nn.SequentialCell(*res_block)

    def construct(self, x):
        out = self.conv1(x)
        out = self.body(out)
        out = self.conv2(out)
        out += x
        out = self.conv_up(out)
        out = ms.nn.ResizeBilinear()(out, scale_factor=self.scale)
        out = self.conv_out(out)
        return out
