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

import mindspore.common.dtype as mstype
from mindspore import ops, nn, Tensor
from mindspore.ops import operations as P

from .dft import SpectralConv2dDft
from ...common.math import get_grid_2d
from ...utils.check_func import check_param_type


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
        channels (int): The number of channels after dimension lifting of the input. Default: ``20``.
        depths (int): The number of FNO layers. Default: ``4``.
        mlp_ratio (int): The number of channels lifting ratio of the decoder layer. Default: ``4``.
        compute_dtype (dtype.Number): The computation type of dense.
            Default: ``mindspore.common.dtype.float32``.
            Should be ``mindspore.common.dtype.float32`` or ``mindspore.common.dtype.float32``.
            float32 is recommended for the GPU backend, float16 is recommended for the Ascend backend.

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
