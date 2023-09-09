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
"""Network architectures for phy-geonet"""
import math

import mindspore as ms
from mindspore import nn, ops
from mindspore.common.initializer import HeUniform, Orthogonal
from sciai.architecture import MSE
from sciai.utils import print_log


class USCNN(nn.Cell):
    """USCNN"""

    def __init__(self, h, nx, ny, n_var_in=1, n_var_out=1, init_way=None, k=5, s=1, p=2):
        super(USCNN, self).__init__()
        self.init_way = init_way
        self.n_var_in = n_var_in
        self.n_var_out = n_var_out
        self.k = k
        self.s = s
        self.p = p
        self.delta_x = h
        self.nx = nx
        self.ny = ny

        self.relu = nn.ReLU()
        self.us = ops.ResizeBilinearV2()
        self.conv1 = nn.Conv2d(self.n_var_in, 16, kernel_size=self.k, stride=self.s, pad_mode="pad", padding=p,
                               has_bias=True, weight_init=HeUniform(negative_slope=math.sqrt(5)))
        self.conv2 = nn.Conv2d(16, 32, kernel_size=self.k, stride=self.s, pad_mode="pad", padding=p,
                               has_bias=True, weight_init=HeUniform(negative_slope=math.sqrt(5)))
        self.conv3 = nn.Conv2d(32, 16, kernel_size=self.k, stride=self.s, pad_mode="pad", padding=p,
                               has_bias=True, weight_init=HeUniform(negative_slope=math.sqrt(5)))
        self.conv4 = nn.Conv2d(16, self.n_var_out, kernel_size=self.k, stride=self.s, pad_mode="pad", padding=p,
                               has_bias=True, weight_init=HeUniform(negative_slope=math.sqrt(5)))
        self.pixel_shuffle = nn.PixelShuffle(1)
        if self.init_way is not None:
            self._initialize_weights()
        # Specify filter
        dx_filter = ms.Parameter(ms.Tensor([[[[0., 0., 0., 0., 0.],
                                              [0., 0., 0., 0., 0.],
                                              [1., -8., 0., 8., -1.],
                                              [0., 0., 0., 0., 0.],
                                              [0., 0., 0., 0., 0.]]]]) / 12. / self.delta_x, requires_grad=False)
        self.convdx = nn.Conv2d(1, 1, (5, 5), stride=1, pad_mode="pad", padding=0, has_bias=False,
                                weight_init=dx_filter)

        dy_filter = ms.Parameter(ms.Tensor([[[[0., 0., 1., 0., 0.],
                                              [0., 0., -8., 0., 0.],
                                              [0., 0., 0., 0., 0.],
                                              [0., 0., 8., 0., 0.],
                                              [0., 0., -1., 0., 0.]]]]) / 12. / self.delta_x, requires_grad=False)
        self.convdy = nn.Conv2d(1, 1, (5, 5), stride=1, pad_mode="pad", padding=0, has_bias=False,
                                weight_init=dy_filter)

        lap_filter = ms.Parameter(ms.Tensor([[[[0., 0., -1., 0., 0.],
                                               [0., 0., 16., 0., 0.],
                                               [-1., 16., -60., 16., -1.],
                                               [0., 0., 16., 0., 0.],
                                               [0., 0., -1., 0., 0.]]]]) / 12. / self.delta_x / self.delta_x,
                                  requires_grad=False)
        self.convlap = nn.Conv2d(1, 1, (5, 5), stride=1, pad_mode="pad", padding=0, has_bias=False,
                                 weight_init=lap_filter)

    def construct(self, x):
        """Network forward pass"""
        x = self.us(x, (self.ny - 2, self.nx - 2))
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pixel_shuffle(self.conv4(x))
        return x

    def _initialize_weights(self):
        """Initialize weights"""
        if self.init_way == 'kaiming':
            weight_init = HeUniform(mode='fan_out', nonlinearity='relu')
            weight_init_4 = HeUniform()
        elif self.init_way == 'ortho':
            weight_init = Orthogonal(gain=math.sqrt(2.0))  # recommended gain for relu is sqrt(2.0)
            weight_init_4 = Orthogonal()
        else:
            print_log('Only Kaiming or Orthogonal initializer can be used!')
            raise ValueError('Only Kaiming or Orthogonal initializer can be used!')
        self.conv1 = nn.Conv2d(self.n_var_in, 16, kernel_size=self.k, stride=self.s, padding=self.p,
                               weight_init=weight_init, bias_init="HeUniform")
        self.conv2 = nn.Conv2d(16, 32, kernel_size=self.k, stride=self.s, padding=self.p, weight_init=weight_init,
                               bias_init="HeUniform")
        self.conv3 = nn.Conv2d(32, 16, kernel_size=self.k, stride=self.s, padding=self.p, weight_init=weight_init,
                               bias_init="HeUniform")
        self.conv4 = nn.Conv2d(16, self.n_var_out, kernel_size=self.k, stride=self.s, padding=self.p,
                               weight_init=weight_init_4, bias_init="HeUniform")


class Net(nn.Cell):
    """Net"""

    def __init__(self, model, batch_size, h):
        super().__init__()
        self.model = model
        self.batch_size = batch_size
        self.h = h
        self.pad_side = 1
        self.udfpad = nn.ConstantPad2d((self.pad_side, self.pad_side, self.pad_side, self.pad_side), 0)
        self._loss_fn = MSE()

    def construct(self, *args):
        """Network forward pass"""
        coord, jinv, dxdxi, dydxi, dxdeta, dydeta = args
        output = self.model(coord)
        output_pad = self.udfpad(output)
        output_v = output_pad[:, 0, :, :].reshape(output_pad.shape[0], 1, output_pad.shape[2], output_pad.shape[3])
        output_v[:, 0, -self.pad_side:, self.pad_side:-self.pad_side] = 0
        output_v[:, 0, :self.pad_side, self.pad_side:-self.pad_side] = 1
        output_v[:, 0, self.pad_side:-self.pad_side, -self.pad_side:] = 1
        output_v[:, 0, self.pad_side:-self.pad_side, 0:self.pad_side] = 1
        output_v[:, 0, 0, 0] = 0.5 * (output_v[:, 0, 0, 1] + output_v[:, 0, 1, 0])
        output_v[:, 0, 0, -1] = 0.5 * (output_v[:, 0, 0, -2] + output_v[:, 0, 1, -1])
        dvdx = dfdx(output_v, dydeta, dydxi, jinv, self.h)
        d2vdx2 = dfdx(dvdx, dydeta, dydxi, jinv, self.h)
        dvdy = dfdy(output_v, dxdxi, dxdeta, jinv, self.h)
        d2vdy2 = dfdy(dvdy, dxdxi, dxdeta, jinv, self.h)
        continuity = d2vdy2 + d2vdx2
        return self._loss_fn(continuity), output_v


def dfdx(f, dydeta, dydxi, jinv, h):
    """Compute df/dx"""
    dfdeta, dfdxi = compute_df(f, h)
    return jinv * (dfdxi * dydeta - dfdeta * dydxi)


def dfdy(f, dxdxi, dxdeta, jinv, h):
    """Compute df/dy"""
    dfdeta, dfdxi = compute_df(f, h)
    return jinv * (dfdeta * dxdxi - dfdxi * dxdeta)


def compute_df(f, h):
    dfdxi_internal = (-f[:, :, :, 4:] + 8 * f[:, :, :, 3:-1] - 8 * f[:, :, :, 1:-3] + f[:, :, :, 0:-4]) / 12 / h
    dfdxi_left = (-11 * f[:, :, :, 0:-3] + 18 * f[:, :, :, 1:-2] - 9 * f[:, :, :, 2:-1] + 2 * f[:, :, :, 3:]) / 6 / h
    dfdxi_right = (11 * f[:, :, :, 3:] - 18 * f[:, :, :, 2:-1] + 9 * f[:, :, :, 1:-2] - 2 * f[:, :, :, 0:-3]) / 6 / h
    dfdxi = ops.concat((dfdxi_left[:, :, :, 0:2], dfdxi_internal, dfdxi_right[:, :, :, -2:]), 3)
    dfdeta_internal = (-f[:, :, 4:, :] + 8 * f[:, :, 3:-1, :] - 8 * f[:, :, 1:-3, :] + f[:, :, 0:-4, :]) / 12 / h
    dfdeta_down = (-11 * f[:, :, 0:-3, :] + 18 * f[:, :, 1:-2, :] - 9 * f[:, :, 2:-1, :] + 2 * f[:, :, 3:, :]) / 6 / h
    dfdeta_up = (11 * f[:, :, 3:, :] - 18 * f[:, :, 2:-1, :] + 9 * f[:, :, 1:-2, :] - 2 * f[:, :, 0:-3, :]) / 6 / h
    dfdeta = ops.concat((dfdeta_down[:, :, 0:2, :], dfdeta_internal, dfdeta_up[:, :, -2:, :]), 2)
    return dfdeta, dfdxi
