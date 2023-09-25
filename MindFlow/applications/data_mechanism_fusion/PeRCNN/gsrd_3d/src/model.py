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
"""3d model"""
import numpy as np
from mindspore import Parameter, Tensor, nn, ops

from .constant import lap_3d_op


class UpScaler(nn.Cell):
    ''' Upscaler (ISG) to convert low-res to high-res initial state '''

    def __init__(self, in_channels, out_channels, hidden_channels, kernel_size, stride, has_bias):
        super(UpScaler, self).__init__()
        self.up0 = nn.Conv3dTranspose(in_channels, hidden_channels, kernel_size=kernel_size, pad_mode='pad',
                                      padding=kernel_size // 2, stride=stride, output_padding=1,
                                      has_bias=has_bias)
        self.conv = nn.Conv3dTranspose(in_channels=hidden_channels, out_channels=hidden_channels,
                                       kernel_size=kernel_size, padding=kernel_size // 2,
                                       stride=1, output_padding=0, pad_mode="pad", has_bias=has_bias)
        # 1x1 layer
        self.out = nn.Conv3d(hidden_channels, out_channels,
                             kernel_size=1, pad_mode="valid", has_bias=has_bias)

    def construct(self, x):
        x = self.up0(x)
        x = ops.sigmoid(x)
        x = self.conv(x)
        x = self.out(x)
        return x


class RecurrentCnn(nn.Cell):
    ''' Recurrent convolutional neural network Cell '''

    def __init__(self, input_channels, hidden_channels, kernel_size, stride, compute_dtype):
        super(RecurrentCnn, self).__init__()

        # the initial parameters, output channel is always 1
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.input_stride = stride
        self.compute_dtype = compute_dtype

        self.dx = 100/48
        self.dt = 0.5
        self.mu_up = 0.274
        # nu from 0 to upper bound (two times the estimate)
        self.nu_up = 0.0108

        # Design the laplace_u term
        self.ca = Parameter(
            Tensor(np.random.rand(), dtype=self.compute_dtype), requires_grad=True)
        self.cb = Parameter(
            Tensor(np.random.rand(), dtype=self.compute_dtype), requires_grad=True)

        # padding_mode='replicate' not working for the test
        laplace = np.array(lap_3d_op)
        self.w_laplace = Tensor(1/self.dx**2*laplace, dtype=self.compute_dtype)

        # Parallel conv layer for u
        self.wh1_u = nn.Conv3d(in_channels=input_channels, out_channels=hidden_channels, kernel_size=kernel_size,
                               stride=self.input_stride, pad_mode="valid", has_bias=True,).to_float(self.compute_dtype)
        self.wh2_u = nn.Conv3d(in_channels=input_channels, out_channels=hidden_channels, kernel_size=kernel_size,
                               stride=self.input_stride, pad_mode="valid", has_bias=True,).to_float(self.compute_dtype)
        self.wh3_u = nn.Conv3d(in_channels=input_channels, out_channels=hidden_channels, kernel_size=kernel_size,
                               stride=self.input_stride, pad_mode="valid", has_bias=True,).to_float(self.compute_dtype)
        # 1x1 layer conv for u
        self.wh4_u = nn.Conv3d(in_channels=hidden_channels,
                               out_channels=1, kernel_size=1, stride=1, has_bias=True).to_float(self.compute_dtype)
        # Parallel conv layer for v
        self.wh1_v = nn.Conv3d(in_channels=input_channels, out_channels=hidden_channels, kernel_size=kernel_size,
                               stride=self.input_stride, pad_mode="valid", has_bias=True,).to_float(self.compute_dtype)
        self.wh2_v = nn.Conv3d(in_channels=input_channels, out_channels=hidden_channels, kernel_size=kernel_size,
                               stride=self.input_stride, pad_mode="valid", has_bias=True,).to_float(self.compute_dtype)
        self.wh3_v = nn.Conv3d(in_channels=input_channels, out_channels=hidden_channels, kernel_size=kernel_size,
                               stride=self.input_stride, pad_mode="valid", has_bias=True,).to_float(self.compute_dtype)
        # 1x1 conv layer for v
        self.wh4_v = nn.Conv3d(in_channels=hidden_channels,
                               out_channels=1, kernel_size=1, stride=1, has_bias=True).to_float(self.compute_dtype)

    def construct(self, h):
        """construct function of RecurrentCnn"""
        # manual periodic padding for diffusion conv layers (5x5 filters)
        h_pad = ops.concat(
            (h[:, :, :, :, -2:], h, h[:, :, :, :, 0:2]), axis=4)
        h_pad = ops.concat(
            (h_pad[:, :, :, -2:, :], h_pad, h_pad[:, :, :, 0:2, :]), axis=3)
        h_pad = ops.concat(
            (h_pad[:, :, -2:, :, :], h_pad, h_pad[:, :, 0:2, :, :]), axis=2)

        u_pad = h_pad[:, 0:1, ...]
        v_pad = h_pad[:, 1:2, ...]
        # previous state
        u_prev = h[:, 0:1, ...]
        v_prev = h[:, 1:2, ...]

        u_res = self.mu_up*ops.sigmoid(self.ca)*ops.conv3d(u_pad, self.w_laplace) + self.wh4_u(
            self.wh1_u(h)*self.wh2_u(h)*self.wh3_u(h))
        v_res = self.mu_up*ops.sigmoid(self.cb)*ops.conv3d(v_pad, self.w_laplace) + self.wh4_v(
            self.wh1_v(h)*self.wh2_v(h)*self.wh3_v(h))

        u_next = u_prev + u_res * self.dt
        v_next = v_prev + v_res * self.dt
        ch = ops.concat((u_next, v_next), axis=1)

        return ch
