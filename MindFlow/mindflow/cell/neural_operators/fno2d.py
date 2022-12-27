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

import numpy as np

import mindspore.common.dtype as mstype
from mindspore import ops, nn, Tensor, Parameter
from mindspore.ops import operations as P
from mindspore.common.initializer import Zero

from .dft import dft2, idft2
from ...common.math import get_grid_2d
from ...utils.check_func import check_param_type


class SpectralConv2dDft(nn.Cell):
    def __init__(self, in_channels, out_channels, modes1, modes2, column_resolution, raw_resolution,
                 compute_dtype=mstype.float16):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.column_resolution = column_resolution
        self.raw_resolution = raw_resolution
        self.compute_dtype = compute_dtype

        self.scale = (1. / (in_channels * out_channels))

        w_re1 = Tensor(
            self.scale * np.random.rand(in_channels,
                                        out_channels, modes1, modes2),
            dtype=compute_dtype)
        w_im1 = Tensor(
            self.scale * np.random.rand(in_channels,
                                        out_channels, modes1, modes2),
            dtype=compute_dtype)
        w_re2 = Tensor(
            self.scale * np.random.rand(in_channels,
                                        out_channels, modes1, modes2),
            dtype=compute_dtype)
        w_im2 = Tensor(
            self.scale * np.random.rand(in_channels,
                                        out_channels, modes1, modes2),
            dtype=compute_dtype)

        self.w_re1 = Parameter(w_re1, requires_grad=True)
        self.w_im1 = Parameter(w_im1, requires_grad=True)
        self.w_re2 = Parameter(w_re2, requires_grad=True)
        self.w_im2 = Parameter(w_im2, requires_grad=True)

        self.dft2_cell = dft2(shape=(column_resolution, raw_resolution), modes=(modes1, modes2),
                              compute_dtype=compute_dtype)
        self.idft2_cell = idft2(shape=(column_resolution, raw_resolution), modes=(modes1, modes2),
                                compute_dtype=compute_dtype)
        self.mat = Tensor(shape=(1, out_channels, column_resolution - 2 * modes1, modes2), dtype=compute_dtype,
                          init=Zero())
        self.concat = ops.Concat(-2)

    @staticmethod
    def mul2d(inputs, weights):
        weight = weights.expand_dims(0)
        data = inputs.expand_dims(2)
        out = weight * data
        return out.sum(1)

    def construct(self, x: Tensor):
        x_re = x
        x_im = ops.zeros_like(x_re)
        x_ft_re, x_ft_im = self.dft2_cell((x_re, x_im))

        out_ft_re1 = \
            self.mul2d(x_ft_re[:, :, :self.modes1, :self.modes2], self.w_re1) \
            - self.mul2d(x_ft_im[:, :, :self.modes1, :self.modes2], self.w_im1)
        out_ft_im1 = \
            self.mul2d(x_ft_re[:, :, :self.modes1, :self.modes2], self.w_im1) \
            + self.mul2d(x_ft_im[:, :, :self.modes1, :self.modes2], self.w_re1)

        out_ft_re2 = \
            self.mul2d(x_ft_re[:, :, -self.modes1:, :self.modes2], self.w_re2) \
            - self.mul2d(x_ft_im[:, :, -self.modes1:, :self.modes2], self.w_im2)
        out_ft_im2 = \
            self.mul2d(x_ft_re[:, :, -self.modes1:, :self.modes2], self.w_im2) \
            + self.mul2d(x_ft_im[:, :, -self.modes1:, :self.modes2], self.w_re2)

        batch_size = x.shape[0]
        mat = self.mat.repeat(batch_size, 0)
        out_re = self.concat((out_ft_re1, mat, out_ft_re2))
        out_im = self.concat((out_ft_im1, mat, out_ft_im2))

        x, _ = self.idft2_cell((out_re, out_im))
        return x


class FNOBlock(nn.Cell):
    def __init__(self, in_channels, out_channels, modes1, resolution=211, gelu=True, compute_dtype=mstype.float16):
        super().__init__()
        self.conv = SpectralConv2dDft(in_channels, out_channels, modes1, modes1, resolution, resolution,
                                      compute_dtype=compute_dtype)
        self.w = nn.Conv2d(in_channels, out_channels, 1,
                           weight_init='HeUniform').to_float(compute_dtype)

        if gelu:
            self.act = ops.GeLU()
        else:
            self.act = ops.Identity()

    def construct(self, x):
        return self.act(self.conv(x) + self.w(x))


class FNO2D(nn.Cell):
    r"""
    The 2-dimensional Fourier Neural Operator (FNO2D) contains a lifting layer,
    multiple Fourier layers and a decoder layer.
    The details can be found in `Fourier neural operator for parametric
    partial differential equations <https://arxiv.org/pdf/2010.08895.pdf>`_.

    Args:
        in_channels (int): The number of channels in the input space.
        out_channels (int): The number of channels in the output space.
        resolution (int): The spatial resolution of the input.
        modes (int): The number of low-frequency components to keep.
        channels (int): The number of channels after dimension lifting of the input. Default: 20.
        depths (int): The number of FNO layers. Default: 4.
        mlp_ratio (int): The number of channels lifting ratio of the decoder layer. Default: 4.
        compute_dtype (dtype.Number): The computation type of dense. Default mstype.float16.
                Should be mstype.float16 or mstype.float32. mstype.float32 is recommended for the GPU backend,
                mstype.float16 is recommended for the Ascend backend.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(batch\_size, resolution, resolution, in\_channels)`.

    Outputs:
        Tensor, the output of this FNO network.

        - **output** (Tensor) -Tensor of shape :math:`(batch\_size, resolution, resolution, out\_channels)`.

    Raises:
        TypeError: If `in_channels` is not an int.
        TypeError: If `out_channels` is not an int.
        TypeError: If `resolution` is not an int.
        TypeError: If `modes` is not an int.
        ValueError: If `modes` is less than 1.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore.common.initializer import initializer, Normal
        >>> from mindflow.cell.neural_operators import FNO2D
        >>> B, H, W, C = 32, 64, 64, 1
        >>> input = initializer(Normal(), [B, H, W, C])
        >>> net = FNO2D(in_channels=1, out_channels=1, resolution=64, modes=12)
        >>> output = net(input)
        >>> print(output.shape)
        (32, 64, 64, 1)

    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 resolution,
                 modes,
                 channels=20,
                 depths=4,
                 mlp_ratio=4,
                 compute_dtype=mstype.float32):
        super().__init__()
        check_param_type(in_channels, "in_channels",
                         data_type=int, exclude_type=bool)
        check_param_type(out_channels, "out_channels",
                         data_type=int, exclude_type=bool)
        check_param_type(resolution, "resolution",
                         data_type=int, exclude_type=bool)
        check_param_type(modes, "modes", data_type=int, exclude_type=bool)
        if modes < 1:
            raise ValueError(
                "modes must at least 1, but got mode: {}".format(modes))

        self.modes1 = modes
        self.channels = channels
        self.fc_channel = mlp_ratio * channels
        self.fc0 = nn.Dense(in_channels + 2, self.channels,
                            has_bias=False).to_float(compute_dtype)
        self.layers = depths

        self.fno_seq = nn.SequentialCell()
        for _ in range(self.layers - 1):
            self.fno_seq.append(FNOBlock(self.channels, self.channels, modes1=self.modes1, resolution=resolution,
                                         compute_dtype=compute_dtype))
        self.fno_seq.append(
            FNOBlock(self.channels, self.channels, self.modes1, resolution=resolution, gelu=False,
                     compute_dtype=compute_dtype))

        self.fc1 = nn.Dense(self.channels, self.fc_channel,
                            has_bias=False).to_float(compute_dtype)
        self.fc2 = nn.Dense(self.fc_channel, out_channels,
                            has_bias=False).to_float(compute_dtype)

        self.grid = Tensor(get_grid_2d(resolution), dtype=mstype.float32)
        self.concat = ops.Concat(axis=-1)
        self.act = ops.GeLU()

    def construct(self, x: Tensor):
        batch_size = x.shape[0]

        grid = self.grid.repeat(batch_size, axis=0)
        x = P.Concat(-1)((x, grid))
        x = self.fc0(x)
        x = P.Transpose()(x, (0, 3, 1, 2))

        x = self.fno_seq(x)

        x = P.Transpose()(x, (0, 2, 3, 1))
        x = self.fc1(x)
        x = self.act(x)
        output = self.fc2(x)

        return output
