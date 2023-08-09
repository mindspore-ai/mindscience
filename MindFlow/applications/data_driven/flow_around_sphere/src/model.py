# ============================================================================
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
"""ResUnet3D structure"""
from mindspore import nn
from mindspore.ops import operations as P


class ResidualUnit(nn.Cell):
    """
    The basic Residual Unit to construct Down part and Up part.

    Args:
        in_channel(int): The number of channels in the input space.
        out_channel (int): The number of channels in the output space.
        stride(int): The movement stride of the 3D convolution kernel. Default: ``2``.
        kernel_size (Union[int, tuple[int]]): The size of convolutional kernel. Default: ``(3, 3, 3)``.
        down(bool): The flag to distinguish between Down and Up. Default: ``True``.
        is_output(bool): The flag for the last upsampled Residual block. Default: ``False``.
        init(Union[Tensor, str, Initializer, numbers.Number]): The trainable weight_init parameter. The dtype is same
            as input `input` . For the values of str, refer to the function `initializer`.Default:``'XavierUniform'``.

    Inputs:
        - **input** (Tensor) - Tensor of shape :math:`(*, in_channel)`.

    Outputs:
        Tensor of shape :math:`(*, out_channel)`.

    Supported Platforms:
        ``Ascend`` ``GPU````CPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore
        >>> from mindspore import Tensor
        >>> inputs = Tensor(np.ones([10, 4, 64, 128, 64]), mindspore.float32)
        >>> net = ResidualUnit(4, 64)
        >>> output = net(inputs)
        >>> print(output.shape)
        (10, 64, 32, 64, 32)
    """
    def __init__(self, in_channel, out_channel, stride=2, kernel_size=(3, 3, 3), down=True, is_output=False,
                 init='XavierUniform'):
        super().__init__()
        self.stride = stride
        self.down = down
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.down_conv_1 = nn.Conv3d(in_channel, out_channel, kernel_size=(3, 3, 3), pad_mode="pad", stride=self.stride,
                                     padding=1, has_bias=True, weight_init=init, bias_init='zeros')
        self.is_output = is_output
        if not is_output:
            self.batch_normal_1 = nn.BatchNorm3d(num_features=self.out_channel)
            self.relu1 = nn.PReLU()
        if self.down:
            self.down_conv_2 = nn.Conv3d(out_channel, out_channel, kernel_size=(3, 3, 3), pad_mode="pad", stride=1,
                                         padding=1, has_bias=True, weight_init=init, bias_init='zeros')
            self.relu2 = nn.PReLU()
            if kernel_size[0] == 1:
                self.residual = nn.Conv3d(in_channel, out_channel, kernel_size=(1, 1, 1), pad_mode="valid",
                                          stride=self.stride, has_bias=True, weight_init=init, bias_init='zeros')
            else:
                self.residual = nn.Conv3d(in_channel, out_channel, kernel_size=(3, 3, 3), pad_mode="pad",
                                          stride=self.stride, padding=1,
                                          has_bias=True, weight_init=init, bias_init='zeros')
            self.batch_normal_2 = nn.BatchNorm3d(num_features=self.out_channel)

    def construct(self, x):
        """ResidualUnit construct"""
        out = self.down_conv_1(x)
        if self.is_output:
            return out
        out = self.batch_normal_1(out)
        out = self.relu1(out)
        if self.down:
            out = self.down_conv_2(out)
            out = self.batch_normal_2(out)
            out = self.relu2(out)
            res = self.residual(x)
        else:
            res = x
        return out + res


class Down(nn.Cell):
    """
    The basic Downsampled residual block.

    Args:
        in_channel(int): The number of channels in the input space.
        out_channel (int): The number of channels in the output space.
        stride(int): The movement stride of the 3D convolution kernel. Default: ``2``.
        kernel_size (Union[int, tuple[int]]): The size of convolutional kernel. Default: ``(3, 3, 3)``.
        init(Union[Tensor, str, Initializer, numbers.Number]): The trainable weight_init parameter.
            The dtype is same as input `input` . For the values of str, refer to the function `initializer`.
            Default:``'XavierUniform'``.


    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(*, in_channel)`.

    Outputs:
        Tensor of shape :math:`(*, out_channel)`.

    Supported Platforms:
        ``Ascend`` ``GPU````CPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore
        >>> from mindspore import Tensor
        >>> inputs = Tensor(np.ones([10, 4, 64, 128, 64]), mindspore.float32)
        >>> net = Down(4, 64)
        >>> output = net(inputs)
        >>> print(output.shape)
        (10, 64, 32, 64, 32)
    """
    def __init__(self, in_channel, out_channel, stride=2, kernel_size=(3, 3, 3), init='XavierUniform'):
        super().__init__()
        self.stride = stride
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.down_conv = ResidualUnit(self.in_channel, self.out_channel, stride, kernel_size, init=init)

    def construct(self, x):
        """Down construct"""
        x = self.down_conv(x)
        return x


class Up(nn.Cell):
    """
    The basic Upsampled residual block.

    Args:
        in_channel(int): The number of channels in the input_data.
        down_in_channel(int): The number of channels in the down_input.
        out_channel(int): The channel number of the output tensor of the Conv3dTranspose layer.
        stride(int): The movement stride of the 3D convolution kernel. Default: ``2``.
        is_output(bool): The flag for the last upsampled Residual block. Default: ``False``.
        output_padding(Union(int, tuple[int])): The number of padding on the depth, height and width directions of
                the output. Default: ``(1, 1, 1)``.
        init(Union[Tensor, str, Initializer, numbers.Number]): The trainable weight_init parameter.
            Default:``'XavierUniform'``.

    Inputs:
        - **input_data** (Tensor) - Tensor of shape :math:`(*, in_channel)`.
        - **down_input** (Tensor) - Tensor of shape :math:`(*, down_in_channel)`.

    Outputs:
        Tensor of shape :math:`(*, out_channel)`.

    Supported Platforms:
        ``Ascend`` ``GPU````CPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore
        >>> from mindspore import Tensor
        >>> input_data = Tensor(np.ones([10, 128, 32, 64, 32]), mindspore.float32)
        >>> down_input = Tensor(np.ones([10, 128, 32, 64, 32]), mindspore.float32)
        >>> net = Up(128, 128, 64)
        >>> output = net(input_data, down_input)
        >>> print(output.shape)
        (10, 64, 64, 128, 64)
    """
    def __init__(self, in_channel, down_in_channel, out_channel, stride=2, is_output=False, output_padding=(1, 1, 1),
                 init='XavierUniform'):
        super().__init__()
        self.in_channel = in_channel
        self.down_in_channel = down_in_channel
        self.out_channel = out_channel
        self.stride = stride
        self.conv3d_transpose = nn.Conv3dTranspose(in_channels=self.in_channel + self.down_in_channel,
                                                   out_channels=self.out_channel, kernel_size=(3, 3, 3),
                                                   pad_mode="pad", stride=self.stride,
                                                   output_padding=output_padding, padding=1,
                                                   weight_init=init, has_bias=False)

        self.concat = P.Concat(axis=1)
        self.conv = ResidualUnit(self.out_channel, self.out_channel, stride=1, down=False, is_output=is_output,
                                 init=init)
        self.batch_normal_1 = nn.BatchNorm3d(num_features=self.out_channel)
        self.relu = nn.PReLU()

    def construct(self, input_data, down_input):
        """Up construct"""
        x = self.concat((input_data, down_input))
        x = self.conv3d_transpose(x)
        x = self.batch_normal_1(x)
        x = self.relu(x)
        x = self.conv(x)
        return x


class ResUnet3D(nn.Cell):
    """
    ResUnet3D, which contains 5 Down parts and 4 Up parts.

    Args:
        in_channels(int): The number of channels in the input space. Default: 4.
        base(int): The number of channels in the first hidden space. Default: 64.
        out_channels (int): The number of channels in the output space. Default: 4.
        init(Union[Tensor, str, Initializer, numbers.Number]): The trainable weight_init parameter.
            Default:'XavierUniform'.

    Inputs:
        - **input** (Tensor) - Tensor of shape :math:`(*, in_channels)`.

    Outputs:
        Tensor of shape :math:`(*, out_channels)`.

    Supported Platforms:
        ``Ascend`` ``GPU````CPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> inputs = Tensor(np.ones([10, 4, 64, 128, 64]), mstype.float32)
        >>> net = ResUnet3D(in_channels=4, base=64, out_channels=4, init="XavierUniform")
        >>> output = net(inputs)
        >>> print(output.shape)
        (10, 4, 64, 128, 64)
    """
    def __init__(self, in_channels=4, base_channels=64, out_channels=4, init="XavierUniform"):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.init = init

        self.down1 = Down(in_channel=self.in_channels, out_channel=self.base_channels, init=self.init)
        self.down2 = Down(in_channel=self.base_channels, out_channel=2 * self.base_channels, init=self.init)
        self.down3 = Down(in_channel=2 * self.base_channels, out_channel=4 * self.base_channels, init=self.init)
        self.down4 = Down(in_channel=4 * self.base_channels, out_channel=8 * self.base_channels, init=self.init)
        self.down5 = Down(in_channel=8 * self.base_channels, out_channel=16 * self.base_channels, stride=1,
                          kernel_size=(1, 1, 1), init=self.init)

        # up
        self.up1 = Up(in_channel=16 * self.base_channels, down_in_channel=8 * self.base_channels,
                      out_channel=4 * self.base_channels, init=self.init)
        self.up2 = Up(in_channel=4 * self.base_channels, down_in_channel=4 * self.base_channels,
                      out_channel=2 * self.base_channels, init=self.init)
        self.up3 = Up(in_channel=2 * self.base_channels, down_in_channel=2 * self.base_channels,
                      out_channel=self.base_channels, init=self.init)
        self.up4 = Up(in_channel=self.base_channels, down_in_channel=self.base_channels, out_channel=self.out_channels,
                      is_output=True, init=self.init)

    def construct(self, input_data):
        """Build a complete model"""
        x1 = self.down1(input_data)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return x
