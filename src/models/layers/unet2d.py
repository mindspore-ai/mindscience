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
"""
unet2d
"""
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.ops import operations as P

from .activation import get_activation
from ..utils.check_func import check_param_type


class DoubleConv(nn.Cell):
    """double conv"""

    def __init__(self, in_channels, out_channels, mid_channels=None, activation="relu", enable_bn=True):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.activation = get_activation(activation) if isinstance(activation, str) else activation

        self.double_conv = nn.SequentialCell(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3))
        if enable_bn:
            self.double_conv.append(nn.BatchNorm2d(mid_channels))
        self.double_conv.append(self.activation)
        self.double_conv.append(nn.Conv2d(mid_channels, out_channels, kernel_size=3))
        if enable_bn:
            self.double_conv.append(nn.BatchNorm2d(out_channels))
        self.double_conv.append(self.activation)

    def construct(self, x):
        """forward"""
        return self.double_conv(x)


class Down(nn.Cell):
    """down"""

    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2, activation="relu", enable_bn=True):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels, activation=activation, enable_bn=enable_bn)
        self.maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)

    def construct(self, x):
        """forward"""
        x = self.maxpool(x)
        return self.conv(x)


class Up(nn.Cell):
    """up"""

    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2, activation="relu", enable_bn=True):
        super().__init__()
        self.up = nn.Conv2dTranspose(in_channels, in_channels // 2, kernel_size=kernel_size, stride=stride)
        self.conv = DoubleConv(in_channels, out_channels, activation=activation, enable_bn=enable_bn)
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


class UNet2D(nn.Cell):
    r"""
    The 2-dimensional U-Net model.
    U-Net is a U-shaped convolutional neural network for biomedical image segmentation.
    It has a contracting path that captures context and an expansive path that enables
    precise localization. The details can be found in `U-Net: Convolutional Networks for
    Biomedical Image Segmentation <https://arxiv.org/abs/1505.04597>`_ .

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        base_channels (int): The number of base channels of UNet2D.
        n_layers (int): The number of downsample and upsample convolutions. Default: 4.
        data_format (str): The format of input data. Default: 'NHWC'
        kernel_size (int): Specifies the height and width of the 2D convolution kernel. Default: 2.
        stride (Union[int, tuple[int]]): The distance of kernel moving,
            an int number that represents the height and width of movement are both stride,
            or a tuple of two int numbers that represent height and width of movement respectively. Default: 2.
        activation (Union[str, class]): The activation function, could be either str or class. Default: ``relu``.
        enable_bn (bool): Specifies whether to use batch norm in convolutions.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(batch\_size, resolution, resolution, channels)`.

    Outputs:
        - **output** (Tensor) - Tensor of shape :math:`(batch\_size, resolution, resolution, channels)`.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import mindspore as ms
        >>> from mindspore import Tensor
        >>> import mindspore.common.dtype as mstype
        >>> import mindflow
        >>> from mindflow.cell import UNet2D
        >>> ms.set_context(mode=ms.GRAPH_MODE, save_graphs=False, device_target="GPU")
        >>> x=Tensor(np.ones([2, 128, 128, 3]), mstype.float32)
        >>> unet = UNet2D(in_channels=3, out_channels=3, base_channels=3)
        >>> output = unet(x)
        >>> print(output.shape)
        (2, 128, 128, 3)
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 base_channels,
                 n_layers=4,
                 data_format="NHWC",
                 kernel_size=2,
                 stride=2,
                 activation="relu",
                 enable_bn=True):
        super().__init__()
        check_param_type(in_channels, "in_channels",
                         data_type=int, exclude_type=bool)
        check_param_type(out_channels, "out_channels",
                         data_type=int, exclude_type=bool)
        check_param_type(base_channels, "base_channels",
                         data_type=int, exclude_type=bool)
        check_param_type(n_layers, "n_layers",
                         data_type=int, exclude_type=bool)
        check_param_type(data_format, "data_format",
                         data_type=str, exclude_type=bool)
        check_param_type(enable_bn, "enable_bn", data_type=bool)

        if data_format not in ("NHWC", "NCHW"):
            raise ValueError(
                "data_format must be 'NHWC' or 'NCHW', but got data_format: {}".format(data_format))
        if n_layers == 0:
            raise ValueError("UNet block should contain at least one downsample convolution")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.stride = stride
        self.base_channels = base_channels
        self.data_format = data_format
        self.activation = get_activation(activation) if isinstance(activation, str) else activation
        self.enable_bn = enable_bn

        self.inc = DoubleConv(self.in_channels, self.base_channels, None, self.activation, self.enable_bn)

        self.layers_down = nn.CellList()
        self.layers_up = nn.CellList()
        for i in range(n_layers):
            self.layers_down.append(Down(self.base_channels * 2**i, self.base_channels * 2 ** (i+1),
                                         self.kernel_size, self.stride, self.activation, self.enable_bn))
            self.layers_up.append(Up(self.base_channels * 2 ** (i+1), self.base_channels * 2 ** i,
                                     self.kernel_size, self.stride, self.activation, self.enable_bn))
        self.layers_up = self.layers_up[::-1]

        self.outc = nn.Conv2d(self.base_channels + self.in_channels, self.out_channels,
                              kernel_size=3, stride=1)
        self.transpose = P.Transpose()
        self.cat = P.Concat(axis=1)

    def construct(self, x):
        """forward"""
        if self.data_format == "NHWC":
            x0 = self.transpose(x, (0, 3, 1, 2))
        else:
            x0 = x
        x = self.inc(x0)
        tensor_list = [x.copy()]
        for layer in self.layers_down:
            x = layer(x)
            tensor_list.append(x.copy())
        for i, layer in enumerate(self.layers_up):
            x = layer(x, tensor_list[-i-2])

        x = self.outc(self.cat((x, x0)))
        if self.data_format == "NHWC":
            out = self.transpose(x, (0, 2, 3, 1))
        else:
            out = x

        return out
