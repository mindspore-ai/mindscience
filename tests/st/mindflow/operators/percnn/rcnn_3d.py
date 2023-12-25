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
"""percnn 3d"""

import torch
from torch import nn
import numpy as np

from constant import laplace_3d


class PeRCNN3D(nn.Module):
    """Recurrent convolutional neural network Cell"""

    def __init__(self, input_channels, hidden_channels):
        super().__init__()
        # the initial parameters
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.input_stride = 1

        self.dx = 100 / 48
        self.dt = 0.5
        self.mu_up = 0.274  # upper bound for the diffusion coefficient
        # Design the laplace_u term
        np.random.seed(1234)  # [-1, 1]
        self.ca = torch.nn.Parameter(
            torch.tensor((np.random.rand() - 0.5) * 2, dtype=torch.float32),
            requires_grad=True,
        )
        self.cb = torch.nn.Parameter(
            torch.tensor((np.random.rand() - 0.5) * 2, dtype=torch.float32),
            requires_grad=True,
        )

        # padding_mode='replicate' not working for the test
        self.w_laplace = nn.Conv3d(1, 1, 5, 1, padding=0, bias=False)
        self.w_laplace.weight.data = (
            1 / self.dx**2 * torch.tensor(laplace_3d, dtype=torch.float64)
        )
        self.w_laplace.weight.requires_grad = False

        # Nonlinear term for u (up to 3rd order)
        self.wh1_u = nn.Conv3d(in_channels=2, out_channels=hidden_channels, kernel_size=1,
                               stride=self.input_stride, padding=0, bias=True)
        self.wh2_u = nn.Conv3d(in_channels=2, out_channels=hidden_channels, kernel_size=1,
                               stride=self.input_stride, padding=0, bias=True)
        self.wh3_u = nn.Conv3d(in_channels=2, out_channels=hidden_channels, kernel_size=1,
                               stride=self.input_stride, padding=0, bias=True)
        self.wh4_u = nn.Conv3d(in_channels=hidden_channels, out_channels=1,
                               kernel_size=1, stride=1, padding=0, bias=True)
        # Nonlinear term for v ((up to 3rd order)
        self.wh1_v = nn.Conv3d(in_channels=2, out_channels=hidden_channels, kernel_size=1,
                               stride=self.input_stride, padding=0, bias=True)
        self.wh2_v = nn.Conv3d(in_channels=2, out_channels=hidden_channels,
                               kernel_size=1, stride=self.input_stride, padding=0, bias=True)
        self.wh3_v = nn.Conv3d(in_channels=2, out_channels=hidden_channels, kernel_size=1,
                               stride=self.input_stride, padding=0, bias=True)
        self.wh4_v = nn.Conv3d(in_channels=hidden_channels, out_channels=1, kernel_size=1,
                               stride=1, padding=0, bias=True)

    def forward(self, h):
        '''
        Calculate the updated gates forward.
        '''
        h_pad = torch.cat((h[:, :, :, :, -2:], h, h[:, :, :, :, 0:2]), dim=4)
        h_pad = torch.cat(
            (h_pad[:, :, :, -2:, :], h_pad, h_pad[:, :, :, 0:2, :]), dim=3
        )
        h_pad = torch.cat(
            (h_pad[:, :, -2:, :, :], h_pad, h_pad[:, :, 0:2, :, :]), dim=2
        )
        u_pad = h_pad[:, 0:1, ...]  # (N+4)x(N+4)
        v_pad = h_pad[:, 1:2, ...]
        u_prev = h[:, 0:1, ...]  # NxN
        v_prev = h[:, 1:2, ...]

        u_res = self.mu_up * torch.sigmoid(self.ca) * self.w_laplace(
            u_pad
        ) + self.wh4_u(self.wh1_u(h) * self.wh2_u(h) * self.wh3_u(h))
        v_res = self.mu_up * torch.sigmoid(self.cb) * self.w_laplace(
            v_pad
        ) + self.wh4_v(self.wh1_v(h) * self.wh2_v(h) * self.wh3_v(h))
        u_next = u_prev + u_res * self.dt
        v_next = v_prev + v_res * self.dt
        ch = torch.cat((u_next, v_next), dim=1)
        return ch
