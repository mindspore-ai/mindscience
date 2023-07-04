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
from mindspore import nn, ops, Tensor, Parameter, float32

from .constant import lap_2d_op


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
                               stride=self.input_stride, pad_mode="valid", has_bias=True,)
        self.wh1_u = nn.Conv2d(in_channels=input_channels, out_channels=hidden_channels, kernel_size=kernel_size,
                               stride=self.input_stride, pad_mode="valid", has_bias=True,)
        self.wh2_u = nn.Conv2d(in_channels=input_channels, out_channels=hidden_channels, kernel_size=kernel_size,
                               stride=self.input_stride, pad_mode="valid", has_bias=True,)
        self.wh3_u = nn.Conv2d(in_channels=input_channels, out_channels=hidden_channels, kernel_size=kernel_size,
                               stride=self.input_stride, pad_mode="valid", has_bias=True,)
        # 1x1 layer for u
        self.wh4_u = nn.Conv2d(in_channels=hidden_channels, out_channels=1, kernel_size=1,
                               stride=1, has_bias=True)
        # Parallel layer for v
        self.wh0_v = nn.Conv2d(in_channels=input_channels, out_channels=hidden_channels, kernel_size=kernel_size,
                               stride=self.input_stride, pad_mode="valid", has_bias=True,)
        self.wh1_v = nn.Conv2d(in_channels=input_channels, out_channels=hidden_channels, kernel_size=kernel_size,
                               stride=self.input_stride, pad_mode="valid", has_bias=True,)
        self.wh2_v = nn.Conv2d(in_channels=input_channels, out_channels=hidden_channels, kernel_size=kernel_size,
                               stride=self.input_stride, pad_mode="valid", has_bias=True,)
        self.wh3_v = nn.Conv2d(in_channels=input_channels, out_channels=hidden_channels, kernel_size=kernel_size,
                               stride=self.input_stride, pad_mode="valid", has_bias=True,)
        # 1x1 layer for v
        self.wh4_v = nn.Conv2d(in_channels=hidden_channels, out_channels=1, kernel_size=1,
                               stride=1, has_bias=True)

        # initialize filter's wweight and bias
        self.filter_list = [self.wh0_u, self.wh1_u, self.wh2_u, self.wh3_u, self.wh4_u,
                            self.wh0_v, self.wh1_v, self.wh2_v, self.wh3_v, self.wh4_v,]

    def init_filter(self, filter_list, c):
        '''
        :param filter_list: list of filter for initialization
        :param c: constant multiplied on Xavier initialization
        '''
        for f in filter_list:
            f.weight.data.uniform_(-c * np.sqrt(1 / np.prod(
                f.weight.shape[:-1])), c * np.sqrt(1 / np.prod(f.weight.shape[:-1])))
            if f.bias is not None:
                f.bias.data.fill_(0.0)

    def construct(self, h):
        """construct function of RecurrentCNNCell"""
        # manual periodic padding for diffusion conv layers (5x5 filters)
        h_pad_2 = ops.concat(
            (h[:, :, :, -2:], h, h[:, :, :, 0:2]), axis=3)
        h_pad_2 = ops.concat(
            (h_pad_2[:, :, -2:, :], h_pad_2, h_pad_2[:, :, 0:2, :]), axis=2)
        u_pad_2 = h_pad_2[:, 0:1, ...]      # 104x104
        v_pad_2 = h_pad_2[:, 1:2, ...]
        # previous state
        u_prev = h[:, 0:1, ...]             # 100x100
        v_prev = h[:, 1:2, ...]

        u_res = self.nu_up*ops.sigmoid(self.ca)*ops.conv2d(u_pad_2, self.w_laplace) + self.wh4_u(
            self.wh0_u(h_pad_2)*self.wh1_u(h_pad_2)*self.wh2_u(h_pad_2)*self.wh3_u(h_pad_2))
        v_res = self.nu_up*ops.sigmoid(self.cb)*ops.conv2d(v_pad_2, self.w_laplace) + self.wh4_v(
            self.wh0_v(h_pad_2)*self.wh1_v(h_pad_2)*self.wh2_v(h_pad_2)*self.wh3_v(h_pad_2))

        u_next = u_prev + u_res * self.dt
        v_next = v_prev + v_res * self.dt
        ch = ops.concat((u_next, v_next), axis=1)

        return ch, ch


class RCNN(nn.Cell):
    ''' Recurrent convolutional neural network layer '''

    def __init__(self, input_channels, hidden_channels, input_kernel_size,
                 infer_step=1, effective_step=None, compute_dtype=float32):
        super(RCNN, self).__init__()

        # input channels of layer includes input_channels and hidden_channels of cells
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.input_kernel_size = input_kernel_size
        self.step = infer_step + 1
        self.effective_step = effective_step
        self.compute_dtype = compute_dtype

        # Upconv as initial state generator
        self.upconv_block = UpScaler(in_channels=input_channels,
                                     out_channels=2,
                                     hidden_channels=8,
                                     kernel_size=5,
                                     stride=2,
                                     has_bais=True)

        self.cell = RecurrentCNNCell(input_channels=self.input_channels,
                                     hidden_channels=self.hidden_channels,
                                     kernel_size=self.input_kernel_size,
                                     compute_dtype=self.compute_dtype)

    def construct(self, init_state_low):
        """construct function of RCNN"""
        # We can freeze the IC or use UpconvBlock. UpconvBlock works slightly better but needs pretraining.
        init_state = self.upconv_block(init_state_low)
        internal_state = []
        outputs = [init_state]
        second_last_state = []

        for step in range(self.step):
            # all cells are initialized in the first step
            if step == 0:
                h = init_state
                internal_state = h

            # forward
            h = internal_state
            # hidden state + output
            h, o = self.cell(h)
            internal_state = h

            if step == (self.step - 2):
                #  last output is a dummy for central FD
                second_last_state = internal_state.copy()

            # after many layers output the result save at time step t
            if step in self.effective_step:
                outputs.append(o)

        return outputs, second_last_state
