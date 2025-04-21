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
"""mindflow st testcase"""
import pytest
import numpy as np
import mindspore as ms
from mindspore import Tensor
from mindspore import dtype as mstype
import torch
from torch import nn
from mindflow.cell import UNet2D as UNet2D_mindspore


RTOL = 0.003

class DoubleConv(nn.Module):
    """double conv"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, bias=False, padding=1, padding_mode='zeros'),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, bias=False, padding=1, padding_mode='zeros'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """forward"""
        return self.double_conv(x)


class Down(nn.Module):
    """down"""

    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        """forward"""
        x = self.maxpool(x)
        return self.conv(x)


class Up(nn.Module):
    """up"""

    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=kernel_size, stride=stride, bias=False)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        """forward"""
        x1 = self.up(x1)

        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        x1 = nn.functional.pad(x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat((x2, x1), 1)
        return self.conv(x)


class UNet2DTorch(nn.Module):
    r"""
    The 2-dimensional U-Net model.
    U-Net is a U-shaped convolutional neural network for biomedical image segmentation.
    It has a contracting path that captures context and an expansive path that enables
    precise localization. The details can be found in `U-Net: Convolutional Networks for
    Biomedical Image Segmentation <https://arxiv.org/abs/1505.04597>`.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        base_channels (int): The number of base channels of UNet2D.
        data_format (str): The format of input data. Default: 'NHWC'
        kernel_size (int): Specifies the height and width of the 2D convolution kernel. Default: 2.
        stride (Union[int, tuple[int]]): The distance of kernel moving,
            an int number that represents the height and width of movement are both stride,
            or a tuple of two int numbers that represent height and width of movement respectively. Default: 2.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(batch\_size, resolution, resolution, channels)`.

    Outputs:
        - **output** (Tensor) - Tensor of shape :math:`(batch\_size, resolution, resolution, channels)`.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import torch
        >>> import numpy as np
        >>> x = torch.Tensor(np.ones([2, 128, 128, 3]))
        >>> unet = Unet2D(in_channels=3, out_channels=3, base_channels=64)
        >>> output = unet(x)
        >>> print(output.shape)
        (2, 128, 128, 3)
    """

    def __init__(self, in_channels, out_channels, base_channels, data_format="NHWC", kernel_size=2, stride=2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.base_channels = base_channels
        self.data_format = data_format

        self.inc = DoubleConv(self.in_channels, self.base_channels, mid_channels=None)
        self.down1 = Down(self.base_channels, self.base_channels * 2, self.kernel_size, self.stride)
        self.down2 = Down(self.base_channels * 2, self.base_channels * 4, self.kernel_size, self.stride)
        self.down3 = Down(self.base_channels * 4, self.base_channels * 8, self.kernel_size, self.stride)
        self.down4 = Down(self.base_channels * 8, self.base_channels * 16, self.kernel_size, self.stride)
        self.up1 = Up(self.base_channels * 16, self.base_channels * 8, self.kernel_size, self.stride)
        self.up2 = Up(self.base_channels * 8, self.base_channels * 4, self.kernel_size, self.stride)
        self.up3 = Up(self.base_channels * 4, self.base_channels * 2, self.kernel_size, self.stride)
        self.up4 = Up(self.base_channels * 2, self.base_channels, self.kernel_size, self.stride)
        self.outc = nn.Conv2d(self.base_channels + self.in_channels, self.out_channels, kernel_size=3, stride=1,
                              bias=False, padding=1, padding_mode='zeros')

    def forward(self, x):
        """forward"""
        if self.data_format == "NHWC":
            x0 = x.permute(0, 3, 1, 2)
        else:
            x0 = x
        x1 = self.inc(x0)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = torch.cat([x, x0], 1)
        x = self.outc(x)
        if self.data_format == "NHWC":
            out = x.permute(0, 2, 3, 1)
        else:
            out = x

        return out


def inference_alignment(base_channels):
    """inference alignment"""
    pt_model = UNet2DTorch(3, 3, base_channels)
    device = torch.device("cpu")
    pt_model.to(device)

    ms_model = UNet2D_mindspore(3, 3, base_channels)
    ms_model.set_train()

    data = np.ones((2, 128, 128, 3))
    data_ms = ms.Tensor(data, dtype=ms.float32)
    data_pt = torch.Tensor(data)
    data_pt = data_pt.to(device)

    ms_key_list = []
    pt_key_list = []
    for k, _ in ms_model.parameters_dict().items():
        if '.0.weight' in k or '.3.weight' in k or 'up.weight' in k or 'outc.weight' in k:
            ms_key_list.append(k)
    for k in pt_model.state_dict().keys():
        if '.0.weight' in k or '.3.weight' in k or 'up.weight' in k or 'outc.weight' in k:
            pt_key_list.append(k)
    ms_to_torch_dict = {c2: c1 for (c1, c2) in zip(pt_key_list, ms_key_list)}
    ms_ckpt_dict = ms_model.parameters_dict()

    for k, v in ms_to_torch_dict.items():
        init = ms.Parameter(pt_model.state_dict()[v].cpu().detach().numpy())
        ms_ckpt_dict[k] = init

    ms.load_param_into_net(ms_model, ms_ckpt_dict)

    output_torch = pt_model(data_pt)
    output_torch = output_torch.cpu().detach().numpy()
    output_mindspore = ms_model(data_ms)
    output_mindspore = output_mindspore.asnumpy()
    return output_torch, output_mindspore


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_unet2d_precision():
    """
    Feature: Test UNet2D network precision on platform cpu.
    Description: None.
    Expectation: Success or throw AssertionError.
    """
    ms.set_context(mode=ms.GRAPH_MODE)
    for base_channels in [4, 8, 16, 32, 64]:
        output_torch, output_mindspore = inference_alignment(base_channels)
        assert np.average(abs(abs(output_mindspore) - abs(output_torch))) / np.average(abs(output_torch)) < RTOL


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_unet2d_shape():
    """
    Feature: Test UNet2D network output shape on platform cpu.
    Description: None.
    Expectation: Success or throw AssertionError.
    """
    ms.set_context(mode=ms.GRAPH_MODE)
    input_tensor = Tensor(np.ones((2, 128, 128, 3)), mstype.float32)
    model = UNet2D_mindspore(in_channels=3, out_channels=3, base_channels=64, data_format="NHWC")
    output_tensor = model(input_tensor)
    assert output_tensor.shape == (2, 128, 128, 3)
    assert output_tensor.dtype == mstype.float32
