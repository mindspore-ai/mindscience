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
"""
unet2d
"""
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.ops import operations as P


class Unet2D(nn.Cell):
    r"""
    UNet2D model

    Args:
        in_channels (int): The input feature size of input.
        out_channels (int): The output feature size of output.
        resolution (int): The spatial resolution of the input.
        kernel_size (int): Specifies the height and width of the 2D convolution kernel. Default: 2.
        stride (Union[int, tuple[int]]): The distance of kernel moving,
            an int number that represents the height and width of movement are both stride,
            or a tuple of two int numbers that represent height and width of movement respectively. Default: 2.

    Inputs:
            - **input** (Tensor) - Tensor of shape :math:`(batch\_size, resolution, resolution, channels)`.

    Outputs:
            - **output** (Tensor) - Tensor of shape :math:`((batch\_size, resolution, resolution, channels)`.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import mindspore as ms
        >>> from mindspore import Tensor
        >>> import mindspore.common.dtype as mstype
        >>> ms.set_context(mode=ms.GRAPH_MODE, save_graphs=False, device_target="GPU")
        >>> x=Tensor(np.ones([2, 128, 128, 3]), mstype.float32)
        >>> unet = Unet2D(in_channels=3, out_channels=3)
        >>> output=unet(x)
        >>> print(res_x.shape)
        (2, 128, 128, 3)
    """

    def __init__(self, in_channels, out_channels, base_channels, kernel_size=2, stride=2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.base_channels = base_channels

        self.inc = DoubleConv(self.in_channels, self.base_channels, mid_channels=None)
        self.down1 = Down(self.base_channels, self.base_channels * 2, self.kernel_size, self.stride)
        self.down2 = Down(self.base_channels * 2, self.base_channels * 4, self.kernel_size, self.stride)
        self.down3 = Down(self.base_channels * 4, self.base_channels * 8, self.kernel_size, self.stride)
        self.down4 = Down(self.base_channels * 8, self.base_channels * 16, self.kernel_size, self.stride)
        self.up1 = Up(self.base_channels * 16, self.base_channels * 8, self.kernel_size, self.stride)
        self.up2 = Up(self.base_channels * 8, self.base_channels * 4, self.kernel_size, self.stride)
        self.up3 = Up(self.base_channels * 4, self.base_channels * 2, self.kernel_size, self.stride)
        self.up4 = Up(self.base_channels * 2, self.base_channels, self.kernel_size, self.stride)
        self.outc = nn.Conv2d(self.base_channels + self.in_channels, self.out_channels, kernel_size=3, stride=1)
        self.transpose = P.Transpose()
        self.cat = P.Concat(axis=1)

    def construct(self, x):
        """forward"""
        x0 = self.transpose(x, (0, 3, 1, 2))
        x1 = self.inc(x0)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.cat((x, x0))
        x = self.outc(x)
        out = self.transpose(x, (0, 2, 3, 1))

        return out


class DoubleConv(nn.Cell):
    """double conv"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.SequentialCell(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def construct(self, x):
        """forward"""
        return self.double_conv(x)


class Down(nn.Cell):
    """down"""

    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)

    def construct(self, x):
        """forward"""
        x = self.maxpool(x)
        return self.conv(x)


class Up(nn.Cell):
    """up"""

    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2):
        super().__init__()
        self.up = nn.Conv2dTranspose(in_channels, in_channels // 2, kernel_size=kernel_size, stride=stride)
        self.conv = DoubleConv(in_channels, out_channels)
        self.cat = ops.Concat(axis=1)

    def construct(self, x1, x2):
        """forward"""
        x1 = self.up(x1)

        _, _, h1, w1 = ops.shape(x1)
        _, _, h2, w2 = ops.shape(x2)

        diff_y = w2 - w1
        diff_x = h2 - h1

        x1 = ops.Pad(((0, 0), (0, 0), (diff_x // 2, diff_x - diff_x // 2), (diff_y // 2, diff_y - diff_y // 2)))(x1)
        x = self.cat((x2, x1))
        return self.conv(x)
