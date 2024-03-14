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
"""Temporal discriminator"""
from mindspore import nn, ops

from .evolution import SpectralNormal


class TemporalDiscriminator(nn.Cell):
    """ Temporal Discriminator definition """
    def __init__(self, in_channels, hidden1=64, hidden2=84, hidden3=40):
        super(TemporalDiscriminator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden1, kernel_size=9, stride=2, pad_mode='pad', has_bias=True)
        self.conv2 = nn.Conv2d(in_channels, hidden2, kernel_size=9, stride=2, pad_mode='pad', has_bias=True)
        self.conv3 = nn.Conv2d(in_channels, hidden3, kernel_size=9, stride=2, pad_mode='pad', has_bias=True)
        self.block1 = DiscriminatorBlock(hidden1 + hidden2 + hidden3, 128, down_scale=True)
        self.block2 = DiscriminatorBlock(128, 256, down_scale=True)
        self.block3 = DiscriminatorBlock(256, 512, down_scale=True)
        self.block4 = DiscriminatorBlock(512, 512)
        self.bn = nn.BatchNorm2d(512)
        self.leaky_relu = nn.LeakyReLU(2e-1)
        self.conv4 = SpectralNormal(nn.Conv2d(512, 1, kernel_size=3, pad_mode='pad', padding=1, has_bias=True))

    def construct(self, x):
        """temporal discriminator construct"""
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        out = ops.concat([x1, x2, x3], axis=1)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.leaky_relu(self.bn(out))
        out = self.conv4(out)
        return out


class DiscriminatorBlock(nn.Cell):
    """ Discriminator Block """
    def __init__(self, in_channels, out_channels, down_scale=False):
        super(DiscriminatorBlock, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        if down_scale:
            self.one_conv = SpectralNormal(nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=2,
                                                     pad_mode='pad',
                                                     padding=1,
                                                     has_bias=True
                                                     )
                                           )
            self.double_conv = nn.SequentialCell(
                SpectralNormal(nn.Conv2d(in_channels,
                                         out_channels,
                                         kernel_size=3,
                                         pad_mode='pad',
                                         padding=1,
                                         has_bias=True
                                         )
                               ),
                nn.ReLU(),
                SpectralNormal(nn.Conv2d(out_channels,
                                         out_channels,
                                         kernel_size=3,
                                         stride=2,
                                         pad_mode='pad',
                                         padding=1,
                                         has_bias=True
                                         )
                               )
            )
        else:
            self.one_conv = SpectralNormal(nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     pad_mode='pad',
                                                     padding=1,
                                                     has_bias=True
                                                     )
                                           )
            self.double_conv = nn.SequentialCell(
                SpectralNormal(nn.Conv2d(in_channels,
                                         out_channels,
                                         kernel_size=3,
                                         pad_mode='pad',
                                         padding=1,
                                         has_bias=True
                                         )
                               ),
                nn.ReLU(),
                SpectralNormal(nn.Conv2d(out_channels,
                                         out_channels,
                                         kernel_size=3,
                                         pad_mode='pad',
                                         padding=1,
                                         has_bias=True
                                         )
                               )
            )

    def construct(self, x):
        x1 = self.one_conv(self.bn(x))
        x2 = self.double_conv(x)
        out = x1 + x2
        return out
