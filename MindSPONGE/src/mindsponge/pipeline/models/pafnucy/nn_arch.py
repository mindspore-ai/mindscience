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
"""pafnucy Model"""
from math import ceil

import numpy as np
from mindspore import ops, Tensor, nn
from mindspore.common import dtype as mstype
from mindspore.common.initializer import TruncatedNormal, initializer
from mindspore.ops import operations as P


class Conv3DUsingConv2D(nn.Cell):
    """Construct Conv3D using Conv2D"""

    def __init__(self, in_channel, out_channel, conv_kernel, stride=1, pad_mode='same', has_bias=True, lmbda=0.001):
        super(Conv3DUsingConv2D, self).__init__()

        self.in_channel = in_channel
        self.out_channel = out_channel

        # Ensure kernel_size is a 3-tuple
        if isinstance(conv_kernel, int):
            self.kernel_size = (conv_kernel,) * 3
        else:
            self.kernel_size = conv_kernel

        self.stride = stride
        self.pad_mode = pad_mode

        # Calculate 2D convolution parameters
        # Use depth direction of 3D kernel as input channels
        self.conv2d = nn.Conv2d(
            in_channels=in_channel * self.kernel_size[0],  # input channels * depth kernel size
            out_channels=out_channel,                      # output channels
            kernel_size=(self.kernel_size[1], self.kernel_size[2]),  # height and width kernel size
            stride=stride,
            weight_init=TruncatedNormal(sigma=lmbda),
            pad_mode=pad_mode,
            bias_init=initializer(0.1, [out_channel]),
            has_bias=has_bias
        )

    def construct(self, x):
        """construct"""

        # x shape: (batch, channels, depth, height, width)
        batch_size, channels, depth, height, width = x.shape

        # Padding for depth direction if 'same' padding is used
        if self.pad_mode == 'same':
            pad_size = (self.kernel_size[0] - 1) // 2
            x = ops.pad(x, ((0, 0, 0, 0, pad_size, pad_size)))
            _, _, new_depth, _, _ = x.shape

        # Prepare output list for sliding window
        output_list = []

        # Sliding window operation in depth direction
        for d in range(depth):
            # Extract 3D patch at current position
            d_start = d
            d_end = d + self.kernel_size[0]
            if d_end > new_depth:
                break

            # Extract data for current depth window
            current_input = x[:, :, d_start:d_end, :, :]

            # Reshape for 2D convolution
            # (batch, channels, kernel_depth, height, width) -> (batch, channels*kernel_depth, height, width)
            current_input = current_input.reshape(
                batch_size,
                channels * self.kernel_size[0],
                height,
                width
            )

            # Perform 2D convolution
            current_output = self.conv2d(current_input)
            output_list.append(current_output)

        # Stack outputs along depth dimension
        output = ops.stack(output_list, axis=2)

        return output


class HiddenConv3D(nn.Cell):
    """HiddenConv3D Cell"""

    def __init__(self, in_channel, out_channel, conv_kernel=5, pool_patch=2, lmbda=0.001):
        super(HiddenConv3D, self).__init__()
        self.bias_inits = Tensor(np.array([0.1 * out_channel]).astype(np.float32))
        self.conv = Conv3DUsingConv2D(
            in_channel,
            out_channel,
            conv_kernel=conv_kernel,
            stride=1,
            pad_mode='same',
            has_bias=True,
            lmbda=lmbda
        )

        self.relu = nn.ReLU()
        self.maxpool3d = P.MaxPool3D(kernel_size=pool_patch, strides=pool_patch, pad_mode='SAME')

    def construct(self, x):
        x = self.conv(x)
        h = self.relu(x)
        h_pool = self.maxpool3d(h)
        return h_pool


class Conv3DBlock(nn.Cell):
    """Conv3DBlock"""

    def __init__(self, in_channel, out_channel,
                 conv_kernel=5, pool_patch=2, lmbda=0.001):
        super(Conv3DBlock, self).__init__()
        self.layer1 = HiddenConv3D(in_channel=in_channel[0],
                                   out_channel=out_channel[0],
                                   conv_kernel=conv_kernel,
                                   pool_patch=pool_patch,
                                   lmbda=lmbda)
        self.layer2 = HiddenConv3D(in_channel=in_channel[1],
                                   out_channel=out_channel[1],
                                   conv_kernel=conv_kernel,
                                   pool_patch=pool_patch,
                                   lmbda=lmbda)
        self.layer3 = HiddenConv3D(in_channel=in_channel[2],
                                   out_channel=out_channel[2],
                                   conv_kernel=conv_kernel,
                                   pool_patch=pool_patch,
                                   lmbda=lmbda)

    def construct(self, x):
        """construct"""

        x = self.layer1(x)
        x = self.layer2(x)
        out_c = self.layer3(x)
        return out_c


class FeedForward(nn.Cell):
    """Feed Forward"""

    def __init__(self, fc_size, in_channel, keep_prob=0.5):
        super(FeedForward, self).__init__()
        self.dense1 = nn.Dense(in_channels=in_channel,
                               out_channels=fc_size[0],
                               weight_init=TruncatedNormal(sigma=1 / (in_channel ** 0.5)),
                               bias_init='one',
                               has_bias=True,
                               activation='relu').to_float(mstype.float16)
        self.dense2 = nn.Dense(in_channels=fc_size[0],
                               out_channels=fc_size[1],
                               weight_init=TruncatedNormal(sigma=1 / (fc_size[0] ** 0.5)),
                               bias_init='one',
                               has_bias=True,
                               activation='relu').to_float(mstype.float16)
        self.dense3 = nn.Dense(in_channels=fc_size[1],
                               out_channels=fc_size[2],
                               weight_init=TruncatedNormal(sigma=1 / (fc_size[1] ** 0.5)),
                               bias_init='one',
                               has_bias=True,
                               activation='relu').to_float(mstype.float16)
        self.dropout = nn.Dropout(p=1.0 - keep_prob)
        self.dropout1 = nn.Dropout(p=0.0)

    def construct(self, x, prob=False):
        """construct"""

        x = self.dense1(x)
        if prob:
            x = self.dropout1(x)
        else:
            x = self.dropout(x)
        x = self.dense2(x)
        if prob:
            x = self.dropout1(x)
        else:
            x = self.dropout(x)
        out = self.dense3(x)
        if prob:
            out_f = self.dropout1(out)
        else:
            out_f = self.dropout(out)
        return out_f


class SBNetWork(nn.Cell):
    """SB network"""

    def __init__(self, in_channel=None,
                 out_channel=None,
                 dense_size=None,
                 osize=1, lmbda=0.01, isize=20, conv_kernel=5,
                 pool_patch=2, keep_prob=0.5,
                 is_training=True):
        super(SBNetWork, self).__init__()
        self.conv3dblock = Conv3DBlock(in_channel, out_channel,
                                       conv_kernel, pool_patch, lmbda)
        self.hfsize = isize
        self.out_channel = out_channel
        self.is_training = is_training
        for _ in range(len(self.out_channel)):
            self.hfsize = ceil(self.hfsize / pool_patch)
        self.hfsize = self.out_channel[-1] * self.hfsize ** 3
        self.reshape = P.Reshape()
        self.feedforward = FeedForward(dense_size, self.hfsize, keep_prob=keep_prob).to_float(mstype.float16)
        self.out_dense = nn.Dense(in_channels=dense_size[2],
                                  out_channels=osize,
                                  weight_init=TruncatedNormal(sigma=(1 / (dense_size[2] ** 0.5))),
                                  bias_init='one',
                                  has_bias=True,
                                  activation='relu')
        self.reduce_mean = P.ReduceMean()
        self.pow = P.Pow()
        self.mse = nn.MSELoss()
        self.rmse = nn.RMSELoss()
        self.cast = P.Cast()

    def construct(self, x, target=None, prob=False):
        """construct"""

        x = self.conv3dblock(x)
        h_flat = self.reshape(x, (-1, self.hfsize))
        h_flat = self.cast(h_flat, mstype.float16)
        h_fc = self.feedforward(h_flat, prob=prob)
        h_fc = self.cast(h_fc, mstype.float32)
        y = self.out_dense(h_fc)
        if self.is_training:
            mse_out = self.mse(y, target)
        else:
            mse_out = y

        return mse_out
