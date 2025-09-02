''''
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
'''
import numpy as np

import mindspore.common.dtype as mstype
from mindspore import nn, ops, Tensor, Parameter, mint
from mindspore.common.initializer import Zero
from mindspore.ops import operations as P

from ...sciops.fourier import RDFTn, IRDFTn


class SpectralConvDft(nn.Cell):
    """Base Class for Fourier Layer, including DFT, linear transform, and Inverse DFT"""

    def __init__(self, in_channels, out_channels, n_modes, resolutions, compute_dtype=mstype.float32):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if isinstance(n_modes, int):
            n_modes = [n_modes]
        self.n_modes = n_modes
        if isinstance(resolutions, int):
            resolutions = [resolutions]
        self.resolutions = resolutions
        if len(self.n_modes) != len(self.resolutions):
            raise ValueError(
                "The dimension of n_modes should be equal to that of resolutions, \
                but got dimension of n_modes {} and dimension of resolutions {}".format(len(self.n_modes),
                                                                                        len(self.resolutions)))
        self.compute_dtype = compute_dtype

    def construct(self, x: Tensor):
        raise NotImplementedError()

    def _einsum(self, inputs, weights):
        weights = weights.expand_dims(0)
        inputs = inputs.expand_dims(2)
        out = inputs * weights
        return out.sum(1)


class SpectralConv1dDft(SpectralConvDft):
    """1D Fourier Layer. It does DFT, linear transform, and Inverse DFT."""

    def __init__(self, in_channels, out_channels, n_modes, resolutions, compute_dtype=mstype.float32):
        super().__init__(in_channels, out_channels, n_modes, resolutions)
        self._scale = (1. / (self.in_channels * self.out_channels))
        w_re = Tensor(self._scale * np.random.rand(self.in_channels, self.out_channels, self.n_modes[0]),
                      dtype=mstype.float32)
        w_im = Tensor(self._scale * np.random.rand(self.in_channels, self.out_channels, self.n_modes[0]),
                      dtype=mstype.float32)
        self._w_re = Parameter(w_re, requires_grad=True)
        self._w_im = Parameter(w_im, requires_grad=True)
        self._dft1_cell = RDFTn(
            shape=(self.resolutions[0],), norm='ortho', modes=self.n_modes[0], compute_dtype=self.compute_dtype)
        self._idft1_cell = IRDFTn(
            shape=(self.resolutions[0],), norm='ortho', modes=self.n_modes[0], compute_dtype=self.compute_dtype)

    def construct(self, x: Tensor):
        x_re = x
        x_ft_re, x_ft_im = self._dft1_cell(x_re)
        w_re = P.Cast()(self._w_re, self.compute_dtype)
        w_im = P.Cast()(self._w_im, self.compute_dtype)
        out_ft_re = self._einsum(x_ft_re[:, :, :self.n_modes[0]], w_re) - self._einsum(x_ft_im[:, :, :self.n_modes[0]],
                                                                                       w_im)
        out_ft_im = self._einsum(x_ft_re[:, :, :self.n_modes[0]], w_im) + self._einsum(x_ft_im[:, :, :self.n_modes[0]],
                                                                                       w_re)

        x = self._idft1_cell(out_ft_re, out_ft_im)

        return x


class SpectralConv2dDft(SpectralConvDft):
    """2D Fourier Layer. It does DFT, linear transform, and Inverse DFT."""

    def __init__(self, in_channels, out_channels, n_modes, resolutions, compute_dtype=mstype.float32):
        super().__init__(in_channels, out_channels, n_modes, resolutions)
        self._scale = (1. / (self.in_channels * self.out_channels))
        w_re1 = Tensor(
            self._scale * np.random.rand(self.in_channels, self.out_channels, self.n_modes[0], self.n_modes[1]),
            dtype=self.compute_dtype)
        w_im1 = Tensor(
            self._scale * np.random.rand(self.in_channels, self.out_channels, self.n_modes[0], self.n_modes[1]),
            dtype=self.compute_dtype)
        w_re2 = Tensor(
            self._scale * np.random.rand(self.in_channels, self.out_channels, self.n_modes[0], self.n_modes[1]),
            dtype=self.compute_dtype)
        w_im2 = Tensor(
            self._scale * np.random.rand(self.in_channels, self.out_channels, self.n_modes[0], self.n_modes[1]),
            dtype=self.compute_dtype)

        self._w_re1 = Parameter(w_re1, requires_grad=True)
        self._w_im1 = Parameter(w_im1, requires_grad=True)
        self._w_re2 = Parameter(w_re2, requires_grad=True)
        self._w_im2 = Parameter(w_im2, requires_grad=True)

        self._dft2_cell = RDFTn(shape=(self.resolutions[0], self.resolutions[1]), norm='ortho',
                                modes=(self.n_modes[0], self.n_modes[1]), compute_dtype=self.compute_dtype)
        self._idft2_cell = IRDFTn(shape=(self.resolutions[0], self.resolutions[1]), norm='ortho',
                                  modes=(self.n_modes[0], self.n_modes[1]), compute_dtype=self.compute_dtype)
        self._mat = Tensor(shape=(1, self.out_channels, self.resolutions[1] - 2 * self.n_modes[0], self.n_modes[1]),
                           dtype=self.compute_dtype, init=Zero())
        self._concat = ops.Concat(-2)

    def construct(self, x: Tensor):
        x_re = x
        x_ft_re, x_ft_im = self._dft2_cell(x_re)

        out_ft_re1 = self._einsum(x_ft_re[:, :, :self.n_modes[0], :self.n_modes[1]], self._w_re1) - self._einsum(
            x_ft_im[:, :, :self.n_modes[0], :self.n_modes[1]], self._w_im1)
        out_ft_im1 = self._einsum(x_ft_re[:, :, :self.n_modes[0], :self.n_modes[1]], self._w_im1) + self._einsum(
            x_ft_im[:, :, :self.n_modes[0], :self.n_modes[1]], self._w_re1)

        out_ft_re2 = self._einsum(x_ft_re[:, :, -self.n_modes[0]:, :self.n_modes[1]], self._w_re2) - self._einsum(
            x_ft_im[:, :, -self.n_modes[0]:, :self.n_modes[1]], self._w_im2)
        out_ft_im2 = self._einsum(x_ft_re[:, :, -self.n_modes[0]:, :self.n_modes[1]], self._w_im2) + self._einsum(
            x_ft_im[:, :, -self.n_modes[0]:, :self.n_modes[1]], self._w_re2)

        batch_size = x.shape[0]
        mat = mint.repeat_interleave(self._mat, batch_size, 0)
        out_re = self._concat((out_ft_re1, mat, out_ft_re2))
        out_im = self._concat((out_ft_im1, mat, out_ft_im2))

        x = self._idft2_cell(out_re, out_im)

        return x


class SpectralConv3dDft(SpectralConvDft):
    """3D Fourier layer. It does DFT, linear transform, and Inverse DFT."""

    def __init__(self, in_channels, out_channels, n_modes, resolutions, compute_dtype=mstype.float32):
        super().__init__(in_channels, out_channels, n_modes, resolutions)
        self._scale = (1 / (self.in_channels * self.out_channels))

        w_re1 = Tensor(
            self._scale * np.random.rand(self.in_channels, self.out_channels, self.n_modes[0], self.n_modes[1],
                                         self.n_modes[2]), dtype=self.compute_dtype)
        w_im1 = Tensor(
            self._scale * np.random.rand(self.in_channels, self.out_channels, self.n_modes[0], self.n_modes[1],
                                         self.n_modes[2]), dtype=self.compute_dtype)
        w_re2 = Tensor(
            self._scale * np.random.rand(self.in_channels, self.out_channels, self.n_modes[0], self.n_modes[1],
                                         self.n_modes[2]), dtype=self.compute_dtype)
        w_im2 = Tensor(
            self._scale * np.random.rand(self.in_channels, self.out_channels, self.n_modes[0], self.n_modes[1],
                                         self.n_modes[2]), dtype=self.compute_dtype)
        w_re3 = Tensor(
            self._scale * np.random.rand(self.in_channels, self.out_channels, self.n_modes[0], self.n_modes[1],
                                         self.n_modes[2]), dtype=self.compute_dtype)
        w_im3 = Tensor(
            self._scale * np.random.rand(self.in_channels, self.out_channels, self.n_modes[0], self.n_modes[1],
                                         self.n_modes[2]), dtype=self.compute_dtype)
        w_re4 = Tensor(
            self._scale * np.random.rand(self.in_channels, self.out_channels, self.n_modes[0], self.n_modes[1],
                                         self.n_modes[2]), dtype=self.compute_dtype)
        w_im4 = Tensor(
            self._scale * np.random.rand(self.in_channels, self.out_channels, self.n_modes[0], self.n_modes[1],
                                         self.n_modes[2]), dtype=self.compute_dtype)

        self._w_re1 = Parameter(w_re1, requires_grad=True)
        self._w_im1 = Parameter(w_im1, requires_grad=True)
        self._w_re2 = Parameter(w_re2, requires_grad=True)
        self._w_im2 = Parameter(w_im2, requires_grad=True)
        self._w_re3 = Parameter(w_re3, requires_grad=True)
        self._w_im3 = Parameter(w_im3, requires_grad=True)
        self._w_re4 = Parameter(w_re4, requires_grad=True)
        self._w_im4 = Parameter(w_im4, requires_grad=True)

        self._dft3_cell = RDFTn(shape=(self.resolutions[0], self.resolutions[1], self.resolutions[2]), norm='ortho',
                                modes=(self.n_modes[0], self.n_modes[1], self.n_modes[2]),
                                compute_dtype=self.compute_dtype)
        self._idft3_cell = IRDFTn(shape=(self.resolutions[0], self.resolutions[1], self.resolutions[2]), norm='ortho',
                                  modes=(self.n_modes[0], self.n_modes[1], self.n_modes[2]),
                                  compute_dtype=self.compute_dtype)
        self._mat_x = Tensor(
            shape=(1, self.out_channels, self.resolutions[0] - 2 * self.n_modes[0], self.n_modes[1], self.n_modes[2]),
            dtype=self.compute_dtype, init=Zero())
        self._mat_y = Tensor(
            shape=(1, self.out_channels, self.resolutions[0], self.resolutions[1] - 2 * self.n_modes[1],
                   self.n_modes[2]),
            dtype=self.compute_dtype, init=Zero())
        self._concat = ops.Concat(-2)

    def construct(self, x: Tensor):
        x_re = x
        x_ft_re, x_ft_im = self._dft3_cell(x_re)

        out_ft_re1 = self._einsum(x_ft_re[:, :, :self.n_modes[0], :self.n_modes[1], :self.n_modes[2]],
                                  self._w_re1) - self._einsum(x_ft_im[:, :, :self.n_modes[0], :self.n_modes[1],
                                                                      :self.n_modes[2]], self._w_im1)
        out_ft_im1 = self._einsum(x_ft_re[:, :, :self.n_modes[0], :self.n_modes[1], :self.n_modes[2]],
                                  self._w_im1) + self._einsum(x_ft_im[:, :, :self.n_modes[0], :self.n_modes[1],
                                                                      :self.n_modes[2]], self._w_re1)
        out_ft_re2 = self._einsum(x_ft_re[:, :, -self.n_modes[0]:, :self.n_modes[1], :self.n_modes[2]],
                                  self._w_re2) - self._einsum(x_ft_im[:, :, -self.n_modes[0]:, :self.n_modes[1],
                                                                      :self.n_modes[2]], self._w_im2)
        out_ft_im2 = self._einsum(x_ft_re[:, :, -self.n_modes[0]:, :self.n_modes[1], :self.n_modes[2]],
                                  self._w_im2) + self._einsum(x_ft_im[:, :, -self.n_modes[0]:, :self.n_modes[1],
                                                                      :self.n_modes[2]], self._w_re2)
        out_ft_re3 = self._einsum(x_ft_re[:, :, :self.n_modes[0], -self.n_modes[1]:, :self.n_modes[2]],
                                  self._w_re3) - self._einsum(x_ft_im[:, :, :self.n_modes[0], -self.n_modes[1]:,
                                                                      :self.n_modes[2]], self._w_im3)
        out_ft_im3 = self._einsum(x_ft_re[:, :, :self.n_modes[0], -self.n_modes[1]:, :self.n_modes[2]],
                                  self._w_im3) + self._einsum(x_ft_im[:, :, :self.n_modes[0], -self.n_modes[1]:,
                                                                      :self.n_modes[2]], self._w_re3)
        out_ft_re4 = self._einsum(x_ft_re[:, :, -self.n_modes[0]:, -self.n_modes[1]:, :self.n_modes[2]],
                                  self._w_re4) - self._einsum(x_ft_im[:, :, -self.n_modes[0]:, -self.n_modes[1]:,
                                                                      :self.n_modes[2]], self._w_im4)
        out_ft_im4 = self._einsum(x_ft_re[:, :, -self.n_modes[0]:, -self.n_modes[1]:, :self.n_modes[2]],
                                  self._w_im4) + self._einsum(x_ft_im[:, :, -self.n_modes[0]:, -self.n_modes[1]:,
                                                                      :self.n_modes[2]], self._w_re4)

        batch_size = x.shape[0]
        mat_x = mint.repeat_interleave(self._mat_x, batch_size, 0)
        mat_y = mint.repeat_interleave(self._mat_y, batch_size, 0)

        out_re1 = ops.concat((out_ft_re1, mat_x, out_ft_re2), -3)
        out_im1 = ops.concat((out_ft_im1, mat_x, out_ft_im2), -3)

        out_re2 = ops.concat((out_ft_re3, mat_x, out_ft_re4), -3)
        out_im2 = ops.concat((out_ft_im3, mat_x, out_ft_im4), -3)
        out_re = ops.concat((out_re1, mat_y, out_re2), -2)
        out_im = ops.concat((out_im1, mat_y, out_im2), -2)
        x = self._idft3_cell(out_re, out_im)

        return x
