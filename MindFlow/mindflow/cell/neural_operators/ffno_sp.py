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
import mindspore as ms
import mindspore.common.dtype as mstype
from mindspore import nn, ops, Tensor, Parameter, ParameterTuple, mint
from mindspore.common.initializer import XavierNormal, initializer
from ...core.math import get_grid_1d, get_grid_2d, get_grid_3d
from .dft import dft1, idft1


class FeedForward(nn.Cell):
    """FeedForward cell"""

    def __init__(self, dim, factor, ff_weight_norm, n_layers, layer_norm, dropout):
        super().__init__()
        self.layers = nn.CellList()
        for i in range(n_layers):
            in_dim = dim if i == 0 else dim * factor
            out_dim = dim if i == n_layers - 1 else dim * factor
            layer = nn.SequentialCell([
                nn.Dense(in_dim, out_dim, has_bias=True) if not ff_weight_norm else nn.Identity(),
                nn.Dropout(p=dropout),
                nn.ReLU() if i < n_layers - 1 else nn.Identity(),
                nn.LayerNorm((out_dim,), epsilon=1e-5) if layer_norm and i == n_layers - 1 else nn.Identity()])
            self.layers.append(layer)

    def construct(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class SpectralConv(nn.Cell):
    """Base Class for Fourier Layer, including DFT, factorization, linear transform, and Inverse DFT"""

    def __init__(self, in_channels, out_channels, n_modes, resolutions, forecast_ff, backcast_ff,
                 fourier_weight, factor, ff_weight_norm, n_ff_layers, layer_norm, use_fork, dropout, filter_mode,
                 compute_dtype=mstype.float32):
        super().__init__()
        self.einsum_flag = tuple([int(s) for s in ms.__version__.split('.')]) >= (2, 5, 0)
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
        self.use_fork = use_fork
        self.fourier_weight = fourier_weight
        self.filter_mode = filter_mode

        if not self.fourier_weight:
            param_list = []
            for i, n_mode in enumerate(self.n_modes):
                weight_re = Tensor(ops.ones((in_channels, out_channels, n_mode)), mstype.float32)
                weight_im = Tensor(ops.ones((in_channels, out_channels, n_mode)), mstype.float32)

                w_re = Parameter(initializer(XavierNormal(), weight_re.shape, mstype.float32), name=f'w_re_{i}',
                                 requires_grad=True)
                w_im = Parameter(initializer(XavierNormal(), weight_im.shape, mstype.float32), name=f'w_im_{i}',
                                 requires_grad=True)

                param_list.append(w_re)
                param_list.append(w_im)

            self.fourier_weight = ParameterTuple([param for param in param_list])

        if use_fork:
            self.forecast_ff = forecast_ff
            if not self.forecast_ff:
                self.forecast_ff = FeedForward(
                    out_channels, factor, ff_weight_norm, n_ff_layers, layer_norm, dropout)

        self.backcast_ff = backcast_ff
        if not self.backcast_ff:
            self.backcast_ff = FeedForward(
                out_channels, factor, ff_weight_norm, n_ff_layers, layer_norm, dropout)

        self._positional_embedding, self._input_perm, self._output_perm = self._transpose(len(self.resolutions))

    def construct(self, x: Tensor):
        raise NotImplementedError()

    def _fourier_dimension(self, n, mode, n_dim):
        """" n- shape - 3D: S1 S2 S3 / 2D: M N / 1D: C
            mode - output length - n//2 +1
            dim -  3D: -1 -2 -3 / 2D: -1 -2  / 1D: -1 """
        dft_cell = dft1(shape=(n,), modes=mode, dim=(n_dim,), compute_dtype=self.compute_dtype)
        idft_cell = idft1(shape=(n,), modes=mode, dim=(n_dim,), compute_dtype=self.compute_dtype)

        return dft_cell, idft_cell

    def _einsum(self, inputs, weights, dim):
        """The Einstein multiplication function"""
        res_len = len(self.resolutions)

        if res_len not in [1, 2, 3]:
            raise ValueError(
                "The length of input resolutions dimensions should be in [1, 2, 3], but got: {}".format(res_len))

        if self.einsum_flag:
            expressions = {
                ('x', 1): 'bix,iox->box',
                ('x', 2): 'bixy,iox->boxy',
                ('y', 2): 'bixy,ioy->boxy',
                ('x', 3): 'bixyz,iox->boxyz',
                ('y', 3): 'bixyz,ioy->boxyz',
                ('z', 3): 'bixyz,ioz->boxyz'
            }

            key = (dim, res_len)
            if key not in expressions:
                raise ValueError(f"Unsupported type of the last dim of weight: {dim}")

            out = mint.einsum(expressions[key], inputs, weights)

        else:
            _, weight_out, weight_dim = weights.shape
            batch_size, inputs_in = inputs.shape[0], inputs.shape[1]
            weights_perm = (2, 0, 1)

            if res_len == 1:
                if dim == 'x':
                    input_perm = (2, 0, 1)
                    output_perm = (1, 2, 0)
                else:
                    raise ValueError(f"Unsupported type of the last dim of weight: {dim}")

                inputs = ops.transpose(inputs, input_perm=input_perm)
                weights = ops.transpose(weights, input_perm=weights_perm)
                out = ops.bmm(inputs, weights)
                out = ops.transpose(out, input_perm=output_perm)
            elif res_len == 2:
                if dim == 'y':
                    input_perm = (3, 0, 2, 1)
                    output_perm = (1, 3, 2, 0)
                elif dim == 'x':
                    input_perm = (2, 0, 3, 1)
                    output_perm = (1, 3, 0, 2)
                else:
                    raise ValueError(f"Unsupported type of the last dim of weight: {dim}")

                inputs = ops.transpose(inputs, input_perm=input_perm)
                inputs = ops.reshape(inputs, (weight_dim, -1, inputs_in))
                weights = ops.transpose(weights, input_perm=weights_perm)
                out = ops.bmm(inputs, weights)
                out = ops.reshape(out, (weight_dim, batch_size, -1, weight_out))
                out = ops.transpose(out, input_perm=output_perm)
            else:
                input_dim1, input_dim2, input_dim3 = inputs.shape[2], inputs.shape[3], inputs.shape[4]

                if dim == 'z':
                    input_perm = (4, 0, 2, 3, 1)
                    output_perm = (1, 4, 2, 3, 0)
                    reshape_dim = input_dim1
                elif dim == 'y':
                    input_perm = (3, 0, 4, 2, 1)
                    output_perm = (1, 4, 3, 0, 2)
                    reshape_dim = input_dim3
                elif dim == 'x':
                    input_perm = (2, 0, 3, 4, 1)
                    output_perm = (1, 4, 0, 2, 3)
                    reshape_dim = input_dim2
                else:
                    raise ValueError(f"Unsupported type of the last dim of weight: {dim}")

                inputs = ops.transpose(inputs, input_perm=input_perm)
                inputs = ops.reshape(inputs, (weight_dim, -1, inputs_in))
                weights = ops.transpose(weights, input_perm=weights_perm)
                out = ops.bmm(inputs, weights)
                out = ops.reshape(out, (weight_dim, batch_size, reshape_dim, -1, weight_out))
                out = ops.transpose(out, input_perm=output_perm)

        return out

    def _transpose(self, n_dim):
        """transpose tensor"""
        if n_dim == 1:
            positional_embedding = Tensor(get_grid_1d(resolution=self.resolutions))
            input_perm = (0, 2, 1)
            output_perm = (0, 2, 1)
        elif n_dim == 2:
            positional_embedding = Tensor(get_grid_2d(resolution=self.resolutions))
            input_perm = (0, 2, 3, 1)
            output_perm = (0, 3, 1, 2)
        elif n_dim == 3:
            positional_embedding = Tensor(get_grid_3d(resolution=self.resolutions))
            input_perm = (0, 2, 3, 4, 1)
            output_perm = (0, 4, 1, 2, 3)
        else:
            raise ValueError(
                "The length of input resolutions dimensions should be in [1, 2, 3], but got: {}".format(n_dim))
        return positional_embedding, input_perm, output_perm

    def _complex_mul(self, input_re, input_im, weight_re, weight_im, dim):
        """(a + bj) * (c + dj) = (ac - bd) + (ad + bc)j"""
        out_re = self._einsum(input_re, weight_re, dim) - self._einsum(input_im, weight_im, dim)
        out_im = self._einsum(input_re, weight_im, dim) + self._einsum(input_im, weight_re, dim)

        return out_re, out_im


class SpectralConv1d(SpectralConv):
    """1D Fourier layer. It does DFT, factorization, linear transform, and Inverse DFT."""

    def __init__(self, in_channels, out_channels, n_modes, resolutions, forecast_ff, backcast_ff,
                 fourier_weight, factor, ff_weight_norm, n_ff_layers, layer_norm, use_fork, dropout, r_padding,
                 filter_mode, compute_dtype=mstype.float32):
        super().__init__(in_channels, out_channels, n_modes, resolutions, forecast_ff, backcast_ff, fourier_weight,
                         factor, ff_weight_norm, n_ff_layers, layer_norm, use_fork, dropout, filter_mode)

        self._dft1_x_cell, self._idft1_x_cell = self._fourier_dimension(resolutions[0] + r_padding, n_modes[0], -1)

    def construct(self, x: Tensor):
        x = self.construct_fourier(x)
        b = self.backcast_ff(x)
        f = self.forecast_ff(x) if self.use_fork else None

        return b, f

    def construct_fourier(self, x):
        """1D Fourier layer."""
        x = ops.transpose(x, input_perm=self._output_perm)  # x shape: batch, in_dim, grid_size

        x_ft_re = x
        x_ft_im = ops.zeros_like(x_ft_re)

        x_ftx_re, x_ftx_im = self._dft1_x_cell((x_ft_re, x_ft_im))

        x_ftx_re_part = x_ftx_re[:, :, :self.n_modes[0]]
        x_ftx_im_part = x_ftx_im[:, :, :self.n_modes[0]]

        re0, re1, re2 = x_ftx_re.shape
        im0, im1, im2 = x_ftx_im.shape
        out_ftx_remain_re = ops.zeros((re0, re1, re2 - self.n_modes[0]))
        out_ftx_remain_im = ops.zeros((im0, im1, im2 - self.n_modes[0]))

        if self.filter_mode == 'full':
            ftx_re, ftx_im = self._complex_mul(
                x_ftx_re_part, x_ftx_im_part, self.fourier_weight[0], self.fourier_weight[1], 'x')
            out_ftx_re = ops.cat([ftx_re, out_ftx_remain_re], axis=2)
            out_ftx_im = ops.cat([ftx_im, out_ftx_remain_im], axis=2)
        elif self.filter_mode == 'low_pass':
            out_ftx_re = ops.cat([x_ftx_re_part, out_ftx_remain_re], axis=2)
            out_ftx_im = ops.cat([x_ftx_im_part, out_ftx_remain_im], axis=2)
        else:
            out_ftx_re = ops.zeros_like(x_ftx_re)
            out_ftx_im = ops.zeros_like(x_ftx_im)

        x, _ = self._idft1_x_cell((out_ftx_re, out_ftx_im))
        x = ops.transpose(x, input_perm=self._input_perm)

        return x


class SpectralConv2d(SpectralConv):
    """2D Fourier layer. It does DFT, factorization, linear transform, and Inverse DFT."""

    def __init__(self, in_channels, out_channels, n_modes, resolutions, forecast_ff, backcast_ff,
                 fourier_weight, factor, ff_weight_norm, n_ff_layers, layer_norm, use_fork, dropout, r_padding,
                 filter_mode, compute_dtype=mstype.float32):
        super().__init__(in_channels, out_channels, n_modes, resolutions, forecast_ff, backcast_ff, fourier_weight,
                         factor, ff_weight_norm, n_ff_layers, layer_norm, use_fork, dropout, filter_mode)

        self._dft1_x_cell, self._idft1_x_cell = self._fourier_dimension(resolutions[0] + r_padding, n_modes[0], -2)
        self._dft1_y_cell, self._idft1_y_cell = self._fourier_dimension(resolutions[1] + r_padding, n_modes[1], -1)

    def construct(self, x: Tensor):
        x = self.construct_fourier(x)
        b = self.backcast_ff(x)
        f = self.forecast_ff(x) if self.use_fork else None

        return b, f

    def construct_fourier(self, x):
        """2D Fourier layer."""
        x = ops.transpose(x, input_perm=self._output_perm)  # x shape: batch, in_dim, grid_size, grid_size

        x_ft_re = x
        x_ft_im = ops.zeros_like(x_ft_re)

        # Dimesion Y
        x_fty_re, x_fty_im = self._dft1_y_cell((x_ft_re, x_ft_im))

        x_fty_re_part = x_fty_re[:, :, :, :self.n_modes[1]]
        x_fty_im_part = x_fty_im[:, :, :, :self.n_modes[1]]

        re0, re1, re2, re3 = x_fty_re.shape
        im0, im1, im2, im3 = x_fty_im.shape
        out_fty_remain_re = ops.zeros((re0, re1, re2, re3 - self.n_modes[1]))
        out_fty_remain_im = ops.zeros((im0, im1, im2, im3 - self.n_modes[1]))

        if self.filter_mode == 'full':
            fty_re, fty_im = self._complex_mul(
                x_fty_re_part, x_fty_im_part, self.fourier_weight[2], self.fourier_weight[3], 'y')
            out_fty_re = ops.cat([fty_re, out_fty_remain_re], axis=3)
            out_fty_im = ops.cat([fty_im, out_fty_remain_im], axis=3)
        elif self.filter_mode == 'low_pass':
            out_fty_re = ops.cat([x_fty_re_part, out_fty_remain_re], axis=3)
            out_fty_im = ops.cat([x_fty_im_part, out_fty_remain_im], axis=3)
        else:
            out_fty_re = ops.zeros_like(x_fty_re)
            out_fty_im = ops.zeros_like(x_fty_im)

        xy, _ = self._idft1_y_cell((out_fty_re, out_fty_im))

        # Dimesion X
        x_ftx_re, x_ftx_im = self._dft1_x_cell((x_ft_re, x_ft_im))

        x_ftx_re_part = x_ftx_re[:, :, :self.n_modes[0], :]
        x_ftx_im_part = x_ftx_im[:, :, :self.n_modes[0], :]

        re0, re1, re2, re3 = x_ftx_re.shape
        im0, im1, im2, im3 = x_ftx_im.shape
        out_ftx_remain_re = ops.zeros((re0, re1, re2 - self.n_modes[0], re3))
        out_ftx_remain_im = ops.zeros((im0, im1, im2 - self.n_modes[0], im3))

        if self.filter_mode == 'full':
            ftx_re, ftx_im = self._complex_mul(
                x_ftx_re_part, x_ftx_im_part, self.fourier_weight[0], self.fourier_weight[1], 'x')
            out_ftx_re = ops.cat([ftx_re, out_ftx_remain_re], axis=2)
            out_ftx_im = ops.cat([ftx_im, out_ftx_remain_im], axis=2)
        elif self.filter_mode == 'low_pass':
            out_ftx_re = ops.cat([x_ftx_re_part, out_ftx_remain_re], axis=2)
            out_ftx_im = ops.cat([x_ftx_im_part, out_ftx_remain_im], axis=2)
        else:
            out_ftx_re = ops.zeros_like(x_ftx_re)
            out_ftx_im = ops.zeros_like(x_ftx_im)

        xx, _ = self._idft1_x_cell((out_ftx_re, out_ftx_im))

        # Combining Dimensions
        x = xx + xy

        x = ops.transpose(x, input_perm=self._input_perm)

        return x


class SpectralConv3d(SpectralConv):
    """3D Fourier layer. It does DFT, factorization, linear transform, and Inverse DFT."""

    def __init__(self, in_channels, out_channels, n_modes, resolutions, forecast_ff, backcast_ff,
                 fourier_weight, factor, ff_weight_norm, n_ff_layers, layer_norm, use_fork, dropout, r_padding,
                 filter_mode, compute_dtype=mstype.float32):
        super().__init__(in_channels, out_channels, n_modes, resolutions, forecast_ff, backcast_ff, fourier_weight,
                         factor, ff_weight_norm, n_ff_layers, layer_norm, use_fork, dropout, filter_mode)

        self._dft1_x_cell, self._idft1_x_cell = self._fourier_dimension(resolutions[0] + r_padding, n_modes[0], -3)
        self._dft1_y_cell, self._idft1_y_cell = self._fourier_dimension(resolutions[1] + r_padding, n_modes[1], -2)
        self._dft1_z_cell, self._idft1_z_cell = self._fourier_dimension(resolutions[2] + r_padding, n_modes[2], -1)

    def construct(self, x: Tensor):
        x = self.construct_fourier(x)
        b = self.backcast_ff(x)
        f = self.forecast_ff(x) if self.use_fork else None

        return b, f

    def construct_fourier(self, x):
        """3D Fourier layer."""
        x = ops.transpose(x, input_perm=self._output_perm)  # x shape: batch, in_dim, grid_size, grid_size, grid_size

        x_ft_re = x
        x_ft_im = ops.zeros_like(x_ft_re)

        # Dimesion Z
        x_ftz_re, x_ftz_im = self._dft1_z_cell((x_ft_re, x_ft_im))

        x_ftz_re_part = x_ftz_re[:, :, :, :, :self.n_modes[2]]
        x_ftz_im_part = x_ftz_im[:, :, :, :, :self.n_modes[2]]

        re0, re1, re2, re3, re4 = x_ftz_re.shape
        im0, im1, im2, im3, im4 = x_ftz_im.shape
        out_ftz_remain_re = ops.zeros((re0, re1, re2, re3, re4 - self.n_modes[2]))
        out_ftz_remain_im = ops.zeros((im0, im1, im2, im3, im4 - self.n_modes[2]))

        if self.filter_mode == 'full':
            ftz_re, ftz_im = self._complex_mul(
                x_ftz_re_part, x_ftz_im_part, self.fourier_weight[4], self.fourier_weight[5], 'z')
            out_ftz_re = ops.cat([ftz_re, out_ftz_remain_re], axis=4)
            out_ftz_im = ops.cat([ftz_im, out_ftz_remain_im], axis=4)
        elif self.filter_mode == 'low_pass':
            out_ftz_re = ops.cat([x_ftz_re_part, out_ftz_remain_re], axis=4)
            out_ftz_im = ops.cat([x_ftz_im_part, out_ftz_remain_im], axis=4)
        else:
            out_ftz_re = ops.zeros_like(x_ftz_re)
            out_ftz_im = ops.zeros_like(x_ftz_im)

        xz, _ = self._idft1_z_cell((out_ftz_re, out_ftz_im))

        # Dimesion Y
        x_fty_re, x_fty_im = self._dft1_y_cell((x_ft_re, x_ft_im))

        x_fty_re_part = x_fty_re[:, :, :, :self.n_modes[1], :]
        x_fty_im_part = x_fty_im[:, :, :, :self.n_modes[1], :]

        re0, re1, re2, re3, re4 = x_fty_re.shape
        im0, im1, im2, im3, im4 = x_fty_im.shape
        out_fty_remain_re = ops.zeros((re0, re1, re2, re3 - self.n_modes[1], re4))
        out_fty_remain_im = ops.zeros((im0, im1, im2, im3 - self.n_modes[1], im4))

        if self.filter_mode == 'full':
            fty_re, fty_im = self._complex_mul(
                x_fty_re_part, x_fty_im_part, self.fourier_weight[2], self.fourier_weight[3], 'y')
            out_fty_re = ops.cat([fty_re, out_fty_remain_re], axis=3)
            out_fty_im = ops.cat([fty_im, out_fty_remain_im], axis=3)
        elif self.filter_mode == 'low_pass':
            out_fty_re = ops.cat([x_fty_re_part, out_fty_remain_re], axis=3)
            out_fty_im = ops.cat([x_fty_im_part, out_fty_remain_im], axis=3)
        else:
            out_fty_re = ops.zeros_like(x_fty_re)
            out_fty_im = ops.zeros_like(x_fty_im)

        xy, _ = self._idft1_y_cell((out_fty_re, out_fty_im))

        # Dimesion X
        x_ftx_re, x_ftx_im = self._dft1_x_cell((x_ft_re, x_ft_im))

        x_ftx_re_part = x_ftx_re[:, :, :self.n_modes[0], :, :]
        x_ftx_im_part = x_ftx_im[:, :, :self.n_modes[0], :, :]

        re0, re1, re2, re3, re4 = x_ftx_re.shape
        im0, im1, im2, im3, im4 = x_ftx_im.shape
        out_ftx_remain_re = ops.zeros((re0, re1, re2 - self.n_modes[0], re3, re4))
        out_ftx_remain_im = ops.zeros((im0, im1, im2 - self.n_modes[0], im3, im4))

        if self.filter_mode == 'full':
            ftx_re, ftx_im = self._complex_mul(
                x_ftx_re_part, x_ftx_im_part, self.fourier_weight[0], self.fourier_weight[1], 'x')
            out_ftx_re = ops.cat([ftx_re, out_ftx_remain_re], axis=2)
            out_ftx_im = ops.cat([ftx_im, out_ftx_remain_im], axis=2)
        elif self.filter_mode == 'low_pass':
            out_ftx_re = ops.cat([x_ftx_re_part, out_ftx_remain_re], axis=2)
            out_ftx_im = ops.cat([x_ftx_im_part, out_ftx_remain_im], axis=2)
        else:
            out_ftx_re = ops.zeros_like(x_ftx_re)
            out_ftx_im = ops.zeros_like(x_ftx_im)

        xx, _ = self._idft1_x_cell((out_ftx_re, out_ftx_im))

        # Combining Dimensions
        x = xx + xy + xz

        x = ops.transpose(x, input_perm=self._input_perm)

        return x
