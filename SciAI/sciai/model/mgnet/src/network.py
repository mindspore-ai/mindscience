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

"""Mgnet network"""
from mindspore import nn, ops


class MgIte(nn.Cell):
    """MgIteration"""
    def __init__(self, a, b):
        super().__init__()
        self.a = a
        self.b = b
        self.bn1 = nn.BatchNorm2d(a.weight.shape[0])
        self.bn2 = nn.BatchNorm2d(b.weight.shape[0])

    def construct(self, x):
        """Network forward pass"""
        u, f = x
        x = f - self.a(u)
        x = self.bn1(x)
        x = ops.relu(x)
        x = self.b(x)
        x = self.bn2(x)
        u = u + ops.relu(x)
        x = u, f
        return x


class MgRestriction(nn.Cell):
    """MgRestriction"""
    def __init__(self, a_old, a_conv, pi_conv, r_conv):
        super().__init__()
        self.a_old = a_old
        self.a_conv = a_conv
        self.pi_conv = pi_conv
        self.r_conv = r_conv

        self.bn1 = nn.BatchNorm2d(pi_conv.weight.shape[0])
        self.bn2 = nn.BatchNorm2d(a_old.weight.shape[0])

    def construct(self, out):
        """Network forward pass"""
        u_old, f_old = out
        u = ops.relu(self.bn1(self.pi_conv(u_old)))
        f = ops.relu(self.bn2(self.r_conv(f_old - self.a_old(u_old)))) + self.a_conv(u)
        out = (u, f)
        return out


class MgNet(nn.Cell):
    """MgNet"""
    def __init__(self, args, dtype, num_classes):
        super().__init__()
        self.num_iteration = args.num_ite

        # initialization layer
        if args.dataset == 'mnist':
            self.num_channel_input = 1
        else:
            self.num_channel_input = 3
        self.conv1 = nn.Conv2d(self.num_channel_input, args.num_channel_f, kernel_size=3, stride=1, pad_mode='pad',
                               padding=1, has_bias=False)
        self.bn1 = nn.BatchNorm2d(args.num_channel_f)

        a_conv = nn.Conv2d(args.num_channel_u, args.num_channel_f, kernel_size=3, stride=1, pad_mode='pad',
                           padding=1, has_bias=False)
        if not args.wise_b:
            b_conv = nn.Conv2d(args.num_channel_f, args.num_channel_u, kernel_size=3, stride=1, pad_mode='pad',
                               padding=1, has_bias=False)
        self.cell_list = nn.CellList()
        layers = []
        for l, num_iteration_l in enumerate(self.num_iteration):
            for _ in range(num_iteration_l):
                if args.wise_b:
                    b_conv = nn.Conv2d(args.num_channel_f, args.num_channel_u, kernel_size=3, stride=1, pad_mode='pad',
                                       padding=1, has_bias=False)

                layers.append(MgIte(a_conv, b_conv))
            self.cell_list.append(nn.SequentialCell(*layers))

            if l < len(self.num_iteration) - 1:
                a_old = a_conv
                a_conv = nn.Conv2d(args.num_channel_u, args.num_channel_f, kernel_size=3, stride=1, pad_mode='pad',
                                   padding=1, has_bias=False)
                if not args.wise_b:
                    b_conv = nn.Conv2d(args.num_channel_f, args.num_channel_u, kernel_size=3, stride=1, pad_mode='pad',
                                       padding=1, has_bias=False)
                pi_conv = nn.Conv2d(args.num_channel_u, args.num_channel_u, kernel_size=3, stride=2, pad_mode='pad',
                                    padding=1, has_bias=False)
                r_conv = nn.Conv2d(args.num_channel_f, args.num_channel_u, kernel_size=3, stride=2, pad_mode='pad',
                                   padding=1, has_bias=False)
                layers = [MgRestriction(a_old, a_conv, pi_conv, r_conv)]

        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Dense(args.num_channel_u, num_classes)
        self.dtype = dtype

    def construct(self, u):
        """Network forward pass"""
        f = ops.relu(self.bn1(self.conv1(u)))
        u = ops.zeros(f.shape, self.dtype)
        out = (u, f)
        for l in range(len(self.num_iteration)):
            out = self.cell_list[l](out)
        u, f = out
        u = self.pooling(u)
        u = u.view(u.shape[0], -1)
        u = self.fc(u)
        return u
