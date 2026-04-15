# Copyright 2022 Huawei Technologies Co., Ltd
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
"""model"""
import mindspore as ms
import numpy as np
import mindspore.ops.operations as P
from mindspore import nn, ops, Parameter, Tensor
from mindspore.common.initializer import initializer, Uniform


class CustomConv3d(nn.Cell):
    """
    Applies a 3D convolution over an input tensor which is typically of shape (N, C, D, H, W)
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1,1,1), pad_mode='same',
                 has_bias=False, dtype=ms.float32):
        super(CustomConv3d, self).__init__()
        self.out_channels = out_channels
        self.conv2d_blocks = nn.CellList(
            [nn.Conv2d(in_channels,
                       out_channels,
                       kernel_size=(kernel_size[1], kernel_size[2]),
                       stride=(stride[1], stride[2]),
                       pad_mode=pad_mode,
                       dtype=dtype,
                       ) for _ in range(kernel_size[0])]
        )
        w = Tensor(np.identity(kernel_size[0]), dtype=dtype)
        self.conv2d_weight = ops.expand_dims(ops.expand_dims(w, axis=0), axis=0)
        self.k = kernel_size[0]
        self.stride = stride
        self.pad_mode = pad_mode
        self.conv2d_dtype = dtype
        self.has_bias = has_bias
        if self.has_bias:
            self.bias = Parameter(initializer(Uniform(), [1, out_channels, 1, 1, 1], dtype=dtype))

    def construct(self, x_):
        """
          Process the input tensor through a series of convolutional layers and perform shaping operations.

          Args:
              x: The input tensor with shape (B, C, D, H, W) representing batch size, channels, depth,
              height, and width.

          Returns:
              The output tensor after convolution, reshaping, and optional bias addition, with the final shape
              depending on the input and layer parameters.
          """
        b, c, d, h, w = x_.shape
        x_ = x_.transpose(0, 2, 1, 3, 4).reshape(b * d, c, h, w)
        out = []
        for i in range(self.k):
            out.append(self.conv2d_blocks[i](x_))
        out = ops.stack(out, axis=-1)
        _, cnew, hnew, wnew, _ = out.shape
        out = out.reshape(b, d, cnew, hnew, wnew, self.k).transpose(0, 2, 3, 4, 1, 5).reshape(-1, 1, d, self.k)
        out = ops.conv2d(out, self.conv2d_weight, stride=(self.stride[0], 1), pad_mode='valid')
        out = out.reshape(b, cnew, hnew, wnew, -1).transpose(0, 1, 4, 2, 3)
        if self.has_bias:
            out += self.bias
        return out

class CNN_GGA_1(nn.Cell):
    def __init__(self):
        super(CNN_GGA_1, self).__init__()
        self.rho_type = "GGA"
        self.conv1 = CustomConv3d(4,  8, (4,4,4), has_bias=True, pad_mode='valid') # 4@9x9x9 ->  8@6x6x6, 4x4x4 kernel
        self.conv2 = CustomConv3d(8, 16, (3,3,3), has_bias=True, pad_mode='valid') # 8@6x6x6 -> 16@4x4x4, 3x3x3 kernel
        self.fc1 = nn.Dense(128, 64)
        self.fc2 = nn.Dense(64, 32)
        self.fc3 = nn.Dense(32, 16)
        self.fc4 = nn.Dense(16, 1)

    def construct(self, x):
        # x shape: 4 x 9 x 9 x 9
        # for GGA-like NN, use electron density and its gradients

        x = ops.elu(self.conv1(x))
        x = ops.elu(self.conv2(x))
        x = ops.max_pool3d(x, 2)
        x = x.view(-1, self.num_flat_features(x))
        x = ops.elu(self.fc1(x))
        x = ops.elu(self.fc2(x))
        x = ops.elu(self.fc3(x))
        x = self.fc4(x)
        return x

    def num_flat_features(self, x):
        size = x.shape[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class CNN_GGA_1_zsym(nn.Cell):
    def __init__(self):
        super(CNN_GGA_1_zsym, self).__init__()
        self.rho_type = "GGA"
        self.conv1 = CustomConv3d(4,  8, (4,4,4), has_bias=True, pad_mode='valid') # 4@9x9x9 ->  8@6x6x6, 4x4x4 kernel
        self.conv2 = CustomConv3d(8, 16, (3,3,3), has_bias=True, pad_mode='valid') # 8@6x6x6 -> 16@4x4x4, 3x3x3 kernel
        self.fc1 = nn.Dense(128, 64)
        self.fc2 = nn.Dense(64, 32)
        self.fc3 = nn.Dense(32, 16)
        self.fc4 = nn.Dense(16, 1)

    def construct(self, x):
        # x shape: batch_size x 2 x 4 x 9 x 9 x 9

        xp = x[:, 0]
        xp = ops.elu(self.conv1(xp))
        xp = ops.elu(self.conv2(xp))
        xp = ops.max_pool3d(xp, 2)
        xp = xp.view(-1, self.num_flat_features(xp))
        xp = ops.elu(self.fc1(xp))
        xp = ops.elu(self.fc2(xp))
        xp = ops.elu(self.fc3(xp))
        xp = self.fc4(xp)

        xm = x[:, 1]
        xm = ops.elu(self.conv1(xm))
        xm = ops.elu(self.conv2(xm))
        xm = ops.max_pool3d(xm, 2)
        xm = xm.view(-1, self.num_flat_features(xm))
        xm = ops.elu(self.fc1(xm))
        xm = ops.elu(self.fc2(xm))
        xm = ops.elu(self.fc3(xm))
        xm = self.fc4(xm)

        return (xm, xp)

    def num_flat_features(self, x):
        size = x.shape[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class CNN_LDA_0(nn.Cell):
    def __init__(self):
        super(CNN_LDA_0, self).__init__()
        self.rho_type = "LDA"
        self.conv1 = nn.Conv3d(1, 6, 4, has_bias=True, pad_mode='valid') # 9x9x9 -> 6x6x6
        self.fc1 = nn.Dense(162, 81)
        self.fc2 = nn.Dense(81, 40)
        self.fc3 = nn.Dense(40, 1)

    def construct(self, x):
        # x shape: 4 x 9 x 9 x 9
        # for LDA-like NN, use only electron density, 
        # i.e. [[0][:, :, :]]

        # extract first channel and add back one dimension
        # 4 x 9 x 9 x 9 -> 9 x 9 x 9 -> 1 x 9 x 9 x 9
        x.data = x.data[:, 0].unsqueeze_(1) 
        x = ops.max_pool3d(ops.elu(self.conv1(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = ops.elu(self.fc1(x))
        x = ops.elu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.shape[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class CNN_GGA_0(nn.Cell):
    def __init__(self):
        super(CNN_GGA_0, self).__init__()
        self.rho_type = "GGA"
        self.conv1 = nn.Conv3d(4, 8, 4, has_bias=True, pad_mode='valid') # 9x9x9 -> 6x6x6
        self.fc1 = nn.Dense(216, 108)
        self.fc2 = nn.Dense(108, 50)
        self.fc3 = nn.Dense(50, 25)
        self.fc4 = nn.Dense(25, 1)

    def construct(self, x):
        # x shape: 4 x 9 x 9 x 9
        # for GGA-like NN, use electron density and its gradients

        x = ops.max_pool3d(ops.elu(self.conv1(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = ops.elu(self.fc1(x))
        x = ops.elu(self.fc2(x))
        x = ops.elu(self.fc3(x))
        x = self.fc4(x)
        return x

    def num_flat_features(self, x):
        size = x.shape[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
