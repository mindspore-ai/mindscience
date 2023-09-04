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
"""model"""
import numpy as np
from mindspore import Parameter, Tensor, nn, ops
from prettytable import PrettyTable

from .constant import dx_2d_op, dy_2d_op, lap_2d_op


class UpScaler(nn.Cell):
    ''' Upscaler (ISG) to convert low-res to high-res initial state '''

    def __init__(self, in_channels, out_channels, hidden_channels, kernel_size, stride, has_bais):
        super(UpScaler, self).__init__()
        self.up0 = nn.Conv2dTranspose(in_channels, hidden_channels, kernel_size=kernel_size, pad_mode='pad',
                                      padding=kernel_size // 2, stride=stride,
                                      has_bias=has_bais)
        self.pad = nn.Pad(
            paddings=((0, 0), (0, 0), (0, 1), (0, 1)), mode="CONSTANT")
        self.conv = nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=kernel_size,
                              pad_mode="same", has_bias=has_bais)
        # 1x1 layer
        self.out = nn.Conv2d(hidden_channels, out_channels,
                             kernel_size=1, pad_mode="valid", has_bias=has_bais)

    def construct(self, x):
        x = self.up0(x)
        x = self.pad(x)
        x = self.conv(x)
        x = ops.tanh(x)
        x = self.out(x)
        return x


class RecurrentCNNCell(nn.Cell):
    ''' Recurrent convolutional neural network Cell '''

    def __init__(self, input_channels, hidden_channels, kernel_size, compute_dtype):
        super(RecurrentCNNCell, self).__init__()

        # the initial parameters, output channel is always 1
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.input_stride = 1
        self.compute_dtype = compute_dtype

        self.dx = 1.0/100.0
        self.dt = 0.00025
        # nu from 0 to upper bound (two times the estimate)
        self.nu_up = 0.0108

        # Design the laplace_u term
        self.ca = Parameter(
            Tensor(np.random.rand(), dtype=self.compute_dtype), requires_grad=True)
        self.cb = Parameter(
            Tensor(np.random.rand(), dtype=self.compute_dtype), requires_grad=True)

        # padding_mode='replicate' not working for the test
        laplace = np.array(lap_2d_op)
        self.w_laplace = Tensor(1/self.dx**2*laplace, dtype=self.compute_dtype)

        # Parallel layer for u
        self.wh0_u = nn.Conv2d(in_channels=input_channels, out_channels=hidden_channels, kernel_size=kernel_size,
                               stride=self.input_stride, pad_mode="valid", has_bias=True,).to_float(self.compute_dtype)
        self.wh1_u = nn.Conv2d(in_channels=input_channels, out_channels=hidden_channels, kernel_size=kernel_size,
                               stride=self.input_stride, pad_mode="valid", has_bias=True,).to_float(self.compute_dtype)
        self.wh2_u = nn.Conv2d(in_channels=input_channels, out_channels=hidden_channels, kernel_size=kernel_size,
                               stride=self.input_stride, pad_mode="valid", has_bias=True,).to_float(self.compute_dtype)
        self.wh3_u = nn.Conv2d(in_channels=input_channels, out_channels=hidden_channels, kernel_size=kernel_size,
                               stride=self.input_stride, pad_mode="valid", has_bias=True,).to_float(self.compute_dtype)
        # 1x1 layer for u
        self.wh4_u = nn.Conv2d(in_channels=hidden_channels,
                               out_channels=1, kernel_size=1, stride=1, has_bias=True).to_float(self.compute_dtype)
        # Parallel layer for v
        self.wh0_v = nn.Conv2d(in_channels=input_channels, out_channels=hidden_channels, kernel_size=kernel_size,
                               stride=self.input_stride, pad_mode="valid", has_bias=True,).to_float(self.compute_dtype)
        self.wh1_v = nn.Conv2d(in_channels=input_channels, out_channels=hidden_channels, kernel_size=kernel_size,
                               stride=self.input_stride, pad_mode="valid", has_bias=True,).to_float(self.compute_dtype)
        self.wh2_v = nn.Conv2d(in_channels=input_channels, out_channels=hidden_channels, kernel_size=kernel_size,
                               stride=self.input_stride, pad_mode="valid", has_bias=True,).to_float(self.compute_dtype)
        self.wh3_v = nn.Conv2d(in_channels=input_channels, out_channels=hidden_channels, kernel_size=kernel_size,
                               stride=self.input_stride, pad_mode="valid", has_bias=True,).to_float(self.compute_dtype)
        # 1x1 layer for v
        self.wh4_v = nn.Conv2d(in_channels=hidden_channels,
                               out_channels=1, kernel_size=1, stride=1, has_bias=True).to_float(self.compute_dtype)

    def construct(self, h):
        """construct function of RecurrentCNNCell"""
        # manual periodic padding for diffusion conv layers (5x5 filters)
        h_pad_2 = ops.concat(
            (h[:, :, :, -2:], h, h[:, :, :, 0:2]), axis=3)
        h_pad_2 = ops.concat(
            (h_pad_2[:, :, -2:, :], h_pad_2, h_pad_2[:, :, 0:2, :]), axis=2)
        u_pad_2 = h_pad_2[:, 0:1, ...]
        v_pad_2 = h_pad_2[:, 1:2, ...]
        # previous state
        u_prev = h[:, 0:1, ...]
        v_prev = h[:, 1:2, ...]

        u_res = self.nu_up*ops.sigmoid(self.ca)*ops.conv2d(u_pad_2, self.w_laplace) + self.wh4_u(
            self.wh0_u(h_pad_2)*self.wh1_u(h_pad_2)*self.wh2_u(h_pad_2)*self.wh3_u(h_pad_2))
        v_res = self.nu_up*ops.sigmoid(self.cb)*ops.conv2d(v_pad_2, self.w_laplace) + self.wh4_v(
            self.wh0_v(h_pad_2)*self.wh1_v(h_pad_2)*self.wh2_v(h_pad_2)*self.wh3_v(h_pad_2))

        u_next = u_prev + u_res * self.dt
        v_next = v_prev + v_res * self.dt
        ch = ops.concat((u_next, v_next), axis=1)

        return ch


class RecurrentCNNCellBurgers(nn.Cell):
    ''' Recurrent convolutional neural network Cell '''

    def __init__(self, kernel_size, init_coef, compute_dtype):
        '''
        Args:
        -----------
        kernel_size: int
            Size of the convolutional kernel for input tensor

        compute_dtype: mindspore.dtype
            compute data type of the net
        '''
        super(RecurrentCNNCellBurgers, self).__init__()

        # the initial parameters
        self.kernel_size = kernel_size
        self.input_padding = self.kernel_size//2
        self.compute_dtype = compute_dtype

        self.nu_u = Parameter(
            Tensor(init_coef['nu_u'], dtype=self.compute_dtype), requires_grad=True)
        self.nu_v = Parameter(
            Tensor(init_coef['nu_v'], dtype=self.compute_dtype), requires_grad=True)

        self.c1_u = Parameter(
            Tensor(init_coef['c1_u'], dtype=self.compute_dtype), requires_grad=True)
        self.c2_u = Parameter(
            Tensor(init_coef['c2_u'], dtype=self.compute_dtype), requires_grad=True)

        self.c1_v = Parameter(
            Tensor(init_coef['c1_v'], dtype=self.compute_dtype), requires_grad=True)
        self.c2_v = Parameter(
            Tensor(init_coef['c2_v'], dtype=self.compute_dtype), requires_grad=True)

        self.dx = 1/100
        self.dy = 1/100
        self.dt = 0.00025

        # laplace operator
        laplace = np.array(lap_2d_op)
        self.w_laplace = Tensor(1/self.dx**2*laplace, dtype=self.compute_dtype)

        # dx operator
        dx_op = np.array(dx_2d_op)
        self.w_dx = Tensor(1/self.dx*dx_op, dtype=self.compute_dtype)

        # dy operator
        dy_op = np.array(dy_2d_op)
        self.w_dy = Tensor(1/self.dy*dy_op, dtype=self.compute_dtype)

    def f_rhs(self, h):
        """right hand side"""
        h_pad_2 = ops.concat(
            (h[:, :, :, -self.input_padding:], h, h[:, :, :, 0:self.input_padding]), axis=3)
        h_pad_2 = ops.concat((h_pad_2[:, :, -self.input_padding:, :],
                              h_pad_2, h_pad_2[:, :, 0:self.input_padding, :]), axis=2)
        u_pad_2 = h_pad_2[:, 0:1, ...]
        v_pad_2 = h_pad_2[:, 1:2, ...]
        # previous state
        u_prev = h[:, 0:1, ...]
        v_prev = h[:, 1:2, ...]

        f_u = self.nu_u*ops.conv2d(u_pad_2, self.w_laplace) + self.c1_u*u_prev*ops.conv2d(
            u_pad_2, self.w_dx) + self.c2_u*v_prev*ops.conv2d(u_pad_2, self.w_dy)
        f_v = self.nu_v*ops.conv2d(v_pad_2, self.w_laplace) + self.c1_v*u_prev*ops.conv2d(
            v_pad_2, self.w_dx) + self.c2_v*v_prev*ops.conv2d(v_pad_2, self.w_dy)
        f = ops.concat([f_u, f_v], axis=1)

        return f

    def construct(self, h):
        f = self.f_rhs(h)
        h_next = h + self.dt * f
        return h_next

    def show_coef(self):
        table = PrettyTable()
        table.field_names = ['\\', r"$\nu_u$", r"$\nu_v$",
                             r"$Cu_1$", r"$Cu_2$", r"$Cv_1$", r"$Cv_2$",]
        nu_u, nu_v, cu_1, cu_2, cv_1, cv_2 = self.nu_u.asnumpy(), self.nu_v.asnumpy(), self.c1_u.asnumpy(),\
            self.c2_u.asnumpy(), self.c1_v.asnumpy(), self.c2_v.asnumpy()
        table.add_row(["True", 0.005, 0.005, -1, -1, -1, -1])
        table.add_row(["Identified", nu_u, nu_v, cu_1, cu_2, cv_1, cv_2,])
        print(table)
