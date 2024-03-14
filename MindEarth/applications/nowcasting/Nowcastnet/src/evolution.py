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
"""Evolution module"""
import numpy as np

import mindspore as ms
import mindspore.numpy as mnp
from mindspore.common.initializer import initializer, Normal
from mindspore import nn, ops, Parameter, Tensor


class SpectralNormal(nn.Cell):
    """Applies spectral normalization to a parameter in the given module.

    Spectral normalization stabilizes the training of discriminators (critics)
    in Generative Adversarial Networks (GANs) by rescaling the weight tensor
    with spectral norm.

    Args:
        module (nn.Cell): containing module.
        n_power_iterations (int): number of power iterations to calculate spectral norm.
        dim (int): dimension corresponding to number of outputs.
        eps (float): epsilon for numerical stability in calculating norms.

    Inputs:
        - **input** - The positional parameter of containing module.
        - **kwargs** - The keyword parameter of containing module.

    Outputs:
        The forward propagation of containing module.
    """
    def __init__(self, module, n_power_iterations=1, dim=0, eps=1e-12):
        super(SpectralNormal, self).__init__()
        self.parametrizations = module
        self.weight = module.weight
        ndim = self.weight.ndim
        if dim >= ndim or dim < -ndim:
            raise IndexError("Dimension out of range (expected to be in range of "
                             f"[-{ndim}, {ndim - 1}] but got {dim})")

        if n_power_iterations <= 0:
            raise ValueError('Expected n_power_iterations to be positive, but '
                             'got n_power_iterations={}'.format(n_power_iterations))
        self.dim = dim if dim >= 0 else dim + ndim
        self.eps = eps
        self.l2_normalize = ops.L2Normalize(epsilon=self.eps)
        self.expand_dims = ops.ExpandDims()
        self.assign = ops.Assign()
        if ndim > 1:
            self.n_power_iterations = n_power_iterations
            weight_mat = self._reshape_weight_to_matrix()
            h, w = weight_mat.shape
            u = initializer(Normal(1.0, 0), [h]).init_data()
            v = initializer(Normal(1.0, 0), [w]).init_data()
            self._u = Parameter(self.l2_normalize(u), requires_grad=False)
            self._v = Parameter(self.l2_normalize(v), requires_grad=False)

    def construct(self, *inputs, **kwargs):
        """SpectralNorm forward function"""
        if self.weight.ndim == 1:
            # Faster and more exact path, no need to approximate anything
            self.l2_normalize(self.weight)
            self.assign(self.parametrizations.weight, self.weight)
        else:
            weight_mat = self._reshape_weight_to_matrix()
            if self.training:
                self._u, self._v = self._power_method(weight_mat, self.n_power_iterations)
            # See above on why we need to clone
            u = self._u.copy()
            v = self._v.copy()

            sigma = ops.tensor_dot(u, mnp.multi_dot([weight_mat, self.expand_dims(v, -1)]), 1)

            self.assign(self.parametrizations.weight, ops.div(self.weight, sigma))

        return self.parametrizations(*inputs, **kwargs)

    def _power_method(self, weight_mat, n_power_iterations):
        for _ in range(n_power_iterations):
            self._u = self.l2_normalize(mnp.multi_dot([weight_mat, self.expand_dims(self._v, -1)]).flatten())
            self._v = self.l2_normalize(mnp.multi_dot([weight_mat.T, self.expand_dims(self._u, -1)]).flatten())
        return self._u, self._v

    def _reshape_weight_to_matrix(self):
        # Precondition
        if self.dim != 0:
            # permute dim to front
            input_perm = [d for d in range(self.weight.dim()) if d != self.dim]
            input_perm.insert(0, self.dim)
            self.weight = ops.transpose(self.weight, input_perm)
        return self.weight.reshape(self.weight.shape[0], -1)


class DoubleConv(nn.Cell):
    """Double Conv"""
    def __init__(self, in_channels, out_channels, kernel=3, mid_channels=None):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.single_conv = nn.SequentialCell(
            nn.BatchNorm2d(in_channels),
            SpectralNormal(nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=kernel,
                                     padding=kernel // 2,
                                     pad_mode='pad',
                                     has_bias=True
                                     )
                           )
        )
        self.double_conv = nn.SequentialCell(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            SpectralNormal(nn.Conv2d(in_channels,
                                     mid_channels,
                                     kernel_size=kernel,
                                     padding=kernel // 2,
                                     pad_mode='pad',
                                     has_bias=True
                                     )
                           ),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            SpectralNormal(nn.Conv2d(mid_channels,
                                     out_channels,
                                     kernel_size=kernel,
                                     padding=kernel // 2,
                                     pad_mode='pad',
                                     has_bias=True
                                     )
                           ),
        )

    def construct(self, x):
        shortcut = self.single_conv(x)
        x = self.double_conv(x)
        out = x + shortcut
        return out


class Up(nn.Cell):
    """Up sample"""
    def __init__(self, in_channels, out_channels, bilinear=True, kernel=3):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2.0, recompute_scale_factor=True, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, kernel=kernel, mid_channels=in_channels // 2)
        else:
            self.up = nn.Conv2dTranspose(in_channels,
                                         in_channels // 2,
                                         kernel_size=2,
                                         stride=2,
                                         has_bias=True,
                                         pad_mode="pad"
                                         )
            self.conv = DoubleConv(in_channels, out_channels, kernel)

    def construct(self, x1, x2):
        x1 = self.up(x1)
        x = ops.cat([x2, x1], axis=1)
        return self.conv(x)


class Down(nn.Cell):
    """Down sample"""
    def __init__(self, in_channels, out_channels, kernel=3):
        super().__init__()
        self.maxpool_conv = nn.SequentialCell(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels, kernel)
        )

    def construct(self, x):
        out = self.maxpool_conv(x)
        return out


class OutConv(nn.Cell):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, pad_mode='pad', has_bias=True)

    def construct(self, x):
        return self.conv(x)


class EvolutionNet(nn.Cell):
    """Evolution network"""
    def __init__(self, t_in, t_out, in_channels=32, bilinear=True):
        super(EvolutionNet, self).__init__()
        self.t_in = t_in
        factor = 2 if bilinear else 1
        self.inc = DoubleConv(t_in, in_channels)
        self.down1 = Down(in_channels * 1, in_channels * 2)
        self.down2 = Down(in_channels * 2, in_channels * 4)
        self.down3 = Down(in_channels * 4, in_channels * 8)
        self.down4 = Down(in_channels * 8, in_channels * 16 // factor)

        self.up1 = Up(in_channels * 16, in_channels * 8 // factor, bilinear)
        self.up2 = Up(in_channels * 8, in_channels * 4 // factor, bilinear)
        self.up3 = Up(in_channels * 4, in_channels * 2 // factor, bilinear)
        self.up4 = Up(in_channels * 2, in_channels, bilinear)
        self.outc = OutConv(in_channels, t_out)
        self.gamma = Parameter(Tensor(np.zeros((1, t_out, 1, 1), dtype=np.float32), ms.float32), requires_grad=True)

        self.up1_v = Up(in_channels * 16, in_channels * 8 // factor, bilinear)
        self.up2_v = Up(in_channels * 8, in_channels * 4 // factor, bilinear)
        self.up3_v = Up(in_channels * 4, in_channels * 2 // factor, bilinear)
        self.up4_v = Up(in_channels * 2, in_channels, bilinear)
        self.outc_v = OutConv(in_channels, t_out * 2)

    def construct(self, all_frames):
        """evolution construct"""
        x = all_frames[:, :self.t_in]
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        intensity = self.outc(x) * self.gamma

        v = self.up1_v(x5, x4)
        v = self.up2_v(v, x3)
        v = self.up3_v(v, x2)
        v = self.up4_v(v, x1)
        motion = self.outc_v(v)
        return intensity, motion
