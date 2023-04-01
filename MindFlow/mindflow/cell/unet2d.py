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
"""UNet2D"""
from mindspore import ops, nn, Tensor


class DoubleConv(nn.Cell):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.SequentialCell(
            [nn.Conv2d(
                in_channels, out_channels, 3, pad_mode="same", has_bias=True,
                weight_init="HeUniform"),
             nn.BatchNorm2d(out_channels),
             nn.ReLU(),
             nn.Conv2d(
                 out_channels, out_channels, 3, pad_mode="same", has_bias=True,
                 weight_init="HeUniform"),
             nn.BatchNorm2d(out_channels),
             nn.ReLU()])

    def construct(self, x: Tensor):
        '''
        Args:
            - x (Tensor): Input tensor.
        '''
        x = self.conv(x)
        return x


class InConv(nn.Cell):
    '''input convolutional'''

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)

    def construct(self, x: Tensor):
        x = self.conv(x)
        return x


class Down(nn.Cell):
    '''downsample convolutional'''

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.mpconv = nn.SequentialCell([
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels)
        ])

    def construct(self, x: Tensor):
        x = self.mpconv(x)
        return x


class Up(nn.Cell):
    '''upsample convolutional'''

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)

    def construct(self, x1: Tensor, x2: Tensor):
        """
        Args:
            - x1 (Tensor): Low resolution input tensor.
            - x2 (Tensor): High resolution input tensor.
        """
        x1 = ops.interpolate(x1, scales=(1., 1., 2., 2.), mode="bilinear")

        diff_y = x1.shape[2] - x2.shape[2]
        diff_x = x1.shape[3] - x2.shape[3]

        x2 = ops.pad(x2, (diff_x // 2, diff_x - diff_x // 2,
                          diff_y // 2, diff_y - diff_y // 2))

        x = ops.concat([x2, x1], axis=1)
        x = self.conv(x)
        return x


class OutConv(nn.Cell):
    '''output convolutional'''

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                              has_bias=True, weight_init="HeUniform")

    def construct(self, x: Tensor):
        x = self.conv(x)
        return x


class UNet2D(nn.Cell):
    r"""
    The 2-dimensional U-Net model.
    U-Net is a U-shaped convolutional neural network for biomedical image segmentation.
    It has a contracting path that captures context and an expansive path that enables
    precise localization. The details can be found in `U-Net: Convolutional Networks for
    Biomedical Image Segmentation <https://arxiv.org/abs/1505.04597>`.

    Args:
        in_channels (int): number of input channels.
        out_channels (int): number of output channels.

    Inputs:
        - **x** (Tensor) - input tensor, shape is :math:`(batch\_size, C_{in}, H, W)`.

    Outputs:
        Tensor, the output of this network.

        - **output** (Tensor) - output tensor, shape is :math:`(batch\_size, C_{out}, H, W)`.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore
        >>> from mindflow.cell import UNet2D
        >>> unet = UNet2D(2, 2)
        >>> x = Tensor(np.random.rand(1, 2, 64, 64), mindspore.float32) # [B, C, H, W]
        >>> y = unet(x) # [B, C, H, W]
        >>> print(y.shape)
        (1, 2, 64, 64)

    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.inc = InConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, out_channels)

    def construct(self, x: Tensor):
        '''
        Args:
            - x (Tensor): Input tensor.
        '''
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x
