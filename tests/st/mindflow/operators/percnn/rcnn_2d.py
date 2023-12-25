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
# ==============================================================================
"""percnn 2d"""

import torch
import torch.nn as nn
import numpy as np

from constant import lap_2d_op


class PeRCNN2D(nn.Module):
    ''' Recurrent convolutional neural network Cell '''

    def __init__(self, input_channels, hidden_channels, input_kernel_size,
                 input_stride, input_padding):
        super().__init__()
        # the initial parameters
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.input_kernel_size = input_kernel_size
        self.input_stride = input_stride
        self.input_padding = input_padding
        self.dx = 1/100
        self.dt = 0.00025
        # nu from 0 to upper bound (two times the estimate)
        self.nu_up = 0.01
        # Design the laplace_u term
        np.random.seed(1234)
        self.ca = torch.nn.Parameter(torch.tensor(
            np.random.rand(), dtype=torch.float32), requires_grad=True)
        self.cb = torch.nn.Parameter(torch.tensor(
            np.random.rand(), dtype=torch.float32), requires_grad=True)

        # padding_mode='replicate' not working for the test
        self.w_laplace = nn.Conv2d(
            1, 1, self.input_kernel_size, self.input_stride, padding=0, bias=False)
        self.w_laplace.weight.data = 1/self.dx**2 * \
            torch.tensor(lap_2d_op, dtype=torch.float64)
        self.w_laplace.weight.requires_grad = False

        # Nonlinear term for u (up to 2nd order)
        self.wh1_u = nn.Conv2d(in_channels=2, out_channels=16, kernel_size=5,
                               stride=self.input_stride, padding=0, bias=True,)
        self.wh2_u = nn.Conv2d(in_channels=2, out_channels=16, kernel_size=5,
                               stride=self.input_stride, padding=0, bias=True,)
        self.wh3_u = nn.Conv2d(in_channels=2, out_channels=16, kernel_size=5,
                               stride=self.input_stride, padding=0, bias=True,)
        self.wh4_u = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1,
                               stride=1, padding=0, bias=True)
        # Nonlinear term for v (up to 3rd order)
        self.wh1_v = nn.Conv2d(in_channels=2, out_channels=16, kernel_size=5,
                               stride=self.input_stride, padding=0, bias=True,)
        self.wh2_v = nn.Conv2d(in_channels=2, out_channels=16, kernel_size=5,
                               stride=self.input_stride, padding=0, bias=True,)
        self.wh3_v = nn.Conv2d(in_channels=2, out_channels=16, kernel_size=5,
                               stride=self.input_stride, padding=0, bias=True,)
        self.wh4_v = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1,
                               stride=1, padding=0, bias=True)

    def forward(self, h):
        '''
        Calculate the updated gates forward.
        '''
        # padding for diffusion conv layers (5x5 filters)
        h_pad_2 = torch.cat(
            (h[:, :, :, -2:], h, h[:, :, :, 0:2]), dim=3)
        h_pad_2 = torch.cat(
            (h_pad_2[:, :, -2:, :], h_pad_2, h_pad_2[:, :, 0:2, :]), dim=2)
        u_pad_2 = h_pad_2[:, 0:1, ...]      # 104x104
        v_pad_2 = h_pad_2[:, 1:2, ...]
        # previous state
        u_prev = h[:, 0:1, ...]             # 100x100
        v_prev = h[:, 1:2, ...]

        u_res = self.nu_up*torch.sigmoid(self.ca)*self.w_laplace(u_pad_2) + self.wh4_u(
            self.wh1_u(h_pad_2)*self.wh2_u(h_pad_2)*self.wh3_u(h_pad_2))
        v_res = self.nu_up*torch.sigmoid(self.cb)*self.w_laplace(v_pad_2) + self.wh4_v(
            self.wh1_v(h_pad_2)*self.wh2_v(h_pad_2)*self.wh3_v(h_pad_2))
        u_next = u_prev + u_res * self.dt
        v_next = v_prev + v_res * self.dt
        ch = torch.cat((u_next, v_next), dim=1)

        return ch
