# Copyright 2023 @ Shenzhen Bay Laboratory &
#                  Peking University &
#                  Huawei Technologies Co., Ltd
#
# This code is a part of MindSPONGE:
# MindSpore Simulation Package tOwards Next Generation molecular modelling.
#
# MindSPONGE is open-source software based on the AI-framework:
# MindSpore (https://www.mindspore.cn/)
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
"""UFold Network Definition"""
import mindspore.nn as nn
from mindspore import ops
CH_FOLD2 = 1


class ConvBlock(nn.Cell):
    """Convolution block"""
    def __init__(self, ch_in, ch_out):
        super(ConvBlock, self).__init__()
        conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, pad_mode="pad", padding=1, has_bias=True)
        conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, pad_mode="pad", padding=1, has_bias=True)
        bn1 = nn.BatchNorm2d(ch_out, use_batch_statistics=True)
        relu1 = nn.ReLU()
        bn2 = nn.BatchNorm2d(ch_out, use_batch_statistics=True)
        relu2 = nn.ReLU()
        self.conv = nn.SequentialCell([conv1, bn1, relu1, conv2, bn2, relu2])

    def construct(self, x):
        output = self.conv(x)
        return output


class UpConv(nn.Cell):
    """Up-Convolution block"""
    def __init__(self, ch_in, ch_out):
        super(UpConv, self).__init__()

        self.up = nn.ResizeBilinear()
        self.conv = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, pad_mode="pad", padding=1, has_bias=True)
        self.bn = nn.BatchNorm2d(ch_out, use_batch_statistics=True)
        self.relu = nn.ReLU()


    def construct(self, x):
        x = self.up(x, scale_factor=2, align_corners=True)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


# pylint: disable=invalid-name
class Unet(nn.Cell):
    """Unet Definition"""
    def __init__(self, img_ch=3, output_ch=1):
        super(Unet, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.op = ops.Concat(1)
        self.transpose = ops.Transpose()

        self.Conv1 = ConvBlock(ch_in=img_ch, ch_out=int(32*CH_FOLD2))
        self.Conv2 = ConvBlock(ch_in=int(32*CH_FOLD2), ch_out=int(64*CH_FOLD2))
        self.Conv3 = ConvBlock(ch_in=int(64*CH_FOLD2), ch_out=int(128*CH_FOLD2))
        self.Conv4 = ConvBlock(ch_in=int(128*CH_FOLD2), ch_out=int(256*CH_FOLD2))
        self.Conv5 = ConvBlock(ch_in=int(256*CH_FOLD2), ch_out=int(512*CH_FOLD2))

        self.Up5 = UpConv(ch_in=int(512*CH_FOLD2), ch_out=int(256*CH_FOLD2))
        self.Up_conv5 = ConvBlock(ch_in=int(512*CH_FOLD2), ch_out=int(256*CH_FOLD2))

        self.Up4 = UpConv(ch_in=int(256*CH_FOLD2), ch_out=int(128*CH_FOLD2))
        self.Up_conv4 = ConvBlock(ch_in=int(256*CH_FOLD2), ch_out=int(128*CH_FOLD2))

        self.Up3 = UpConv(ch_in=int(128*CH_FOLD2), ch_out=int(64*CH_FOLD2))
        self.Up_conv3 = ConvBlock(ch_in=int(128*CH_FOLD2), ch_out=int(64*CH_FOLD2))

        self.Up2 = UpConv(ch_in=int(64*CH_FOLD2), ch_out=int(32*CH_FOLD2))
        self.Up_conv2 = ConvBlock(ch_in=int(64*CH_FOLD2), ch_out=int(32*CH_FOLD2))

        self.Conv_1x1 = nn.Conv2d(int(32*CH_FOLD2), output_ch, kernel_size=1,
                                  stride=1, pad_mode="valid", padding=0, has_bias=True)


    def construct(self, x):
        """encoding path"""
        x1 = self.Conv1(x)
        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        d5 = self.Up5(x5)
        d5 = self.op((x4, d5))
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = self.op((x3, d4))
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = self.op((x2, d3))
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = self.op((x1, d2))
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)
        d1 = d1.squeeze(1)

        return self.transpose(d1, (0, 2, 1)) * d1
