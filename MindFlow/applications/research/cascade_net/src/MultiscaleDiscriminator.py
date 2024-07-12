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
"""Multiscale Discriminator"""
import mindspore.nn as nn
from mindspore import ops, Tensor


class AveragePool2d(nn.Cell):
    """AveragePool2d"""
    def __init__(self, kernel_size=(20, 20), stride=1, pad_mode='same'):
        super(AveragePool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad_mode = pad_mode
        weight = ops.zeros((2, 2, self.kernel_size[0], self.kernel_size[1]))
        weight[0, 0, :, :] = 1
        weight[1, 1, :, :] = 1
        self.weight = weight / self.kernel_size[0] / self.kernel_size[1]

    def construct(self, x):
        out = ops.conv2d(x, weight=self.weight, stride=self.stride, pad_mode=self.pad_mode)
        return out


class DefineDiscriminatorBlock(nn.Cell):
    """Discriminator Block"""
    def __init__(self, in_channel, filter_num):
        super(DefineDiscriminatorBlock, self).__init__()
        self.discriminator_sequential = nn.SequentialCell(
            # C8
            nn.Conv2d(in_channel, filter_num, (3, 3), stride=(2, 2), has_bias=True),  # (None, 64, 64, 8)
            nn.ReLU(),
            nn.Dropout(p=0.25),
            # C16
            nn.Conv2d(filter_num, 2 * filter_num, (3, 3), stride=(2, 2), has_bias=True),  # (None, 32, 32, 16)
            nn.ReLU(),
            nn.Dropout(p=0.25),
            # C32
            nn.Conv2d(2 * filter_num, 4 * filter_num, (3, 3), stride=(2, 2), has_bias=True),  # (None, 16, 16, 32)
            nn.ReLU(),
            nn.Dropout(p=0.25),
            # C64
            nn.Conv2d(4 * filter_num, 8 * filter_num, (3, 3), stride=(2, 2), has_bias=True),  # (None, 8, 8, 64)
            nn.ReLU(),
            nn.Dropout(p=0.25),
            # second last output layer
            nn.Conv2d(8 * filter_num, 16 * filter_num, (3, 3), has_bias=True),  # (None, 8, 8, 128)
            nn.ReLU(),
        )
        self.conv = nn.Conv2d(16 * filter_num + 1, 1, (3, 3), has_bias=True)
        self.concat = ops.Concat(axis=1)

    def construct(self, in_d, in_merge):
        # source image input
        d = self.discriminator_sequential(in_d)
        d = self.concat((d, in_merge))
        # patch output
        patch_out = self.conv(d)
        return patch_out


class MultiscaleDiscriminator(nn.Cell):
    """Multiscale Discriminator"""
    def __init__(self, input_channel, filter_num, multi_scale=False, single_scale=False):
        super(MultiscaleDiscriminator, self).__init__()
        self.d_weight = Tensor([1, 5, 10])
        self.single_scale = single_scale
        self.multi_scale = multi_scale
        self.discriminator1 = DefineDiscriminatorBlock(input_channel, filter_num)
        if self.multi_scale:
            self.ave_pool20 = AveragePool2d(kernel_size=(20, 20), stride=1, pad_mode='same')
            self.ave_pool10 = AveragePool2d(kernel_size=(10, 10), stride=1, pad_mode='same')
            self.discriminator2 = DefineDiscriminatorBlock(input_channel, filter_num)
            self.discriminator3 = DefineDiscriminatorBlock(input_channel, filter_num)
        self.concat = ops.Concat(axis=1)

    def construct(self, in_d, in_merge):
        """Multiscale Discriminator Model"""
        if self.single_scale:
            out_final = self.discriminator1(in_d, in_merge)  # (None, 8,8,1)
        if self.multi_scale:
            ur20 = self.ave_pool20(in_d)
            out_map_20 = self.discriminator1(ur20, in_merge)  # (None, 8,8,1)
            ur10 = self.ave_pool10(in_d)
            out_map_10 = self.discriminator2(ur10, in_merge)  # (None, 8,8,1)
            out_map_raw = self.discriminator3(in_d, in_merge)  # (None, 8,8,1)
            out_map_concat = self.concat((self.d_weight[0] * out_map_20,
                                          self.d_weight[1] * out_map_10,
                                          self.d_weight[2] * out_map_raw))
            out_final = ops.mean(out_map_concat, axis=1, keep_dims=True)
        return out_final
