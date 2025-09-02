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
# pylint: disable=W0235

from mindspore import nn, ops, Tensor, Parameter, ParameterTuple, mint
from mindspore.common.initializer import XavierNormal, initializer
import mindspore.common.dtype as mstype

from .ffno_sp import SpectralConv1d, SpectralConv2d, SpectralConv3d
from ...common.math import get_grid_1d, get_grid_2d, get_grid_3d
from ...utils.check_func import check_param_type


class FFNOBlocks(nn.Cell):
    r"""
    The FFNOBlock, which usually accompanied by a Lifting Layer ahead and a Projection Layer behind,
    is a part of Factorized Fourier Neural Operator. It contains a Factorized Fourier Layer. The details can be found
    in `A. Tran, A. Mathews, et. al: FACTORIZED FOURIER NEURAL OPERATORS <https://arxiv.org/pdf/2010.08895.pdf>`_.

    Args:
        in_channels (int): The number of channels in the input space.
        out_channels (int): The number of channels in the output space.
        n_modes (Union[int, list(int)]): The number of modes reserved after linear transformation in Fourier Layer.
        resolutions (Union[int, list(int)]): The resolutions of the input tensor.
        factor (int): The number of neurons in the hidden layer of a feedforward network. Default: ``1``.
        n_ff_layers (int): The number of layers (hidden layers) in the feedforward neural network. Default: ``2``.
        ff_weight_norm (bool): Whether to do weight normalization in feedforward or not. Used as a reserved function
            interface, the weight normalization is not supported in feedforward. Default: ``False``.
        layer_norm (bool): Whether to do layer normalization in feedforward or not. Default: ``True``.
        dropout (float): The value of percent be dropped when applying dropout regularization. Default: ``0.0``.
        r_padding (int): The number used to pad a tensor on the right in a certain dimension. Pad the domain if
            input is non-periodic. Default: ``0``.
        use_fork (bool): Whether to perform forecasting or not. Default: ``False``.
        forecast_ff (Feedforward): The feedforward network of generating "backcast" output. Default: ``None``.
        backcast_ff (Feedforward): The feedforward network of generating "forecast" output. Default: ``None``.
        fourier_weight (ParameterTuple[Parmemter]): The fourier weight for transforming data in the frequency
            domain, with a ParameterTuple of Parmemter with a length of 2N.

            - Even indices (0, 2, 4, ...) represent the real parts of the complex parmemter.
            - Odd indices (1, 3, 5, ...) represent the imaginary parts of the complex parmemter.
            - Default: ``None``, meaning no data is provided.
        dft_compute_dtype (dtype.Number): The computation type of DFT in SpectralConv. Default: ``mstype.float32``.
        ffno_compute_dtype (dtype.Number): The computation type of MLP in ffno skip. Default: ``mstype.float16``.
            Should be ``mstype.float32`` or ``mstype.float16``. mstype.float32 is recommended for the GPU backend,
            mstype.float16 is recommended for the Ascend backend.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(batch\_size, in\_channels, resolution)`.

    Outputs:
        Tensor, the output of this FFNOBlocks.

        - **output** (Tensor) -Tensor of shape :math:`(batch\_size, out\_channels, resolution)`.

    Raises:
        TypeError: If `in_channels` is not an int.
        TypeError: If `out_channels` is not an int.
        TypeError: If `factor` is not an int.
        TypeError: If `n_ff_layers` is not an int.
        TypeError: If `ff_weight_norm` is not a Boolean value.
        ValueError: If `ff_weight_norm` is not ``False``.
        TypeError: If `layer_norm` is not a Boolean value.
        TypeError: If `dropout` is not a float.
        TypeError: If `r_padding` is not an int.
        TypeError: If `use_fork` is not a Boolean value.

    Supported Platforms:
        ``Ascend``

    Examples:`
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> import mindspore.common.dtype as mstype
        >>> from mindflow.cell.neural_operators import FFNOBlocks
        >>> data = Tensor(np.ones([2, 128, 128, 2]), mstype.float32)
        >>> net = FFNOBlocks(in_channels=2, out_channels=2, n_modes=[20, 20], resolutions=[128, 128])
        >>> out0, out1 = net(data)
        >>> print(data.shape, out0.shape, out1.shape)
        (2, 128, 128, 2) (2, 128, 128, 2) (2, 128, 128, 2)
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 n_modes,
                 resolutions,
                 factor=1,
                 n_ff_layers=2,
                 ff_weight_norm=False,
                 layer_norm=True,
                 dropout=0.0,
                 r_padding=0,
                 use_fork=False,
                 forecast_ff=None,
                 backcast_ff=None,
                 fourier_weight=None,
                 dft_compute_dtype=mstype.float32,
                 ffno_compute_dtype=mstype.float32
                 ):
        super().__init__()
        check_param_type(in_channels, "in_channels", data_type=int)
        check_param_type(out_channels, "out_channels", data_type=int)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_modes, self.resolutions = validate_and_expand_dimensions(
            1, n_modes, resolutions, False)

        check_param_type(factor, "factor", data_type=int)
        check_param_type(n_ff_layers, "n_ff_layers", data_type=int)
        check_param_type(ff_weight_norm, "ff_weight_norm", data_type=bool)
        check_param_type(layer_norm, "layer_norm", data_type=bool)
        check_param_type(dropout, "dropout", data_type=float)
        check_param_type(r_padding, 'r_padding', data_type=int)

        if ff_weight_norm:
            raise ValueError(
                f"The weight normalization is not supported in feedforward\
                but got value of ff_weight_norm {ff_weight_norm}")

        if r_padding < 0:
            raise ValueError(
                f"The right padding value cannot be negative\
                 but got value of r_padding {r_padding}")

        check_param_type(use_fork, "use_fork", data_type=bool)
        self.factor = factor
        self.ff_weight_norm = ff_weight_norm
        self.n_ff_layers = n_ff_layers
        self.layer_norm = layer_norm
        self.dropout = dropout
        self.r_padding = r_padding
        self.use_fork = use_fork
        self.forecast_ff = forecast_ff
        self.backcast_ff = backcast_ff
        self.fourier_weight = fourier_weight
        self.dft_compute_dtype = dft_compute_dtype
        self.ffno_compute_dtype = ffno_compute_dtype

        if len(self.resolutions) == 1:
            spectral_conv = SpectralConv1d
        elif len(self.resolutions) == 2:
            spectral_conv = SpectralConv2d
        elif len(self.resolutions) == 3:
            spectral_conv = SpectralConv3d
        else:
            raise ValueError(
                f"The length of input resolutions dimensions should be in [1, 2, 3], but got: {len(self.resolutions)}")

        self._convs = spectral_conv(self.in_channels,
                                    self.out_channels,
                                    self.n_modes,
                                    self.resolutions,
                                    forecast_ff=self.forecast_ff,
                                    backcast_ff=self.backcast_ff,
                                    fourier_weight=self.fourier_weight,
                                    factor=self.factor,
                                    ff_weight_norm=self.ff_weight_norm,
                                    n_ff_layers=self.n_ff_layers,
                                    layer_norm=self.layer_norm,
                                    use_fork=self.use_fork,
                                    dropout=self.dropout,
                                    r_padding=self.r_padding,
                                    compute_dtype=self.dft_compute_dtype,
                                    filter_mode='full')

    def construct(self, x: Tensor):
        b, _ = self._convs(x)
        x = ops.add(x, b)
        return x, b


def validate_and_expand_dimensions(dim, n_modes, resolutions, is_validate_dim=True):
    """validate and expand the dimension of inputs"""
    if isinstance(n_modes, int):
        n_modes = [n_modes] * dim
    if isinstance(resolutions, int):
        resolutions = [resolutions] * dim

    n_modes_num = len(n_modes)
    resolutions_num = len(resolutions)

    if is_validate_dim:
        if n_modes_num != dim:
            raise ValueError(
                f"The dimension of n_modes should be equal to {dim} when using FFNO{dim}D\
                 but got dimension of n_modes {n_modes_num}")
        if resolutions_num != dim:
            raise ValueError(
                f"The dimension of resolutions should be equal to {dim} when using FFNO{dim}D\
                 but got dimension of resolutions {resolutions_num}")
    if n_modes_num != resolutions_num:
        raise ValueError(
            f"The dimension of n_modes should be equal to that of resolutions\
             but got dimension of n_modes {n_modes_num} and dimension of resolutions {resolutions_num}")

    return n_modes, resolutions


class FFNO(nn.Cell):
    r"""
    The FFNO base class, which usually contains a Lifting Layer, a Factorized Fourier Block Layer and a Projection
    Layer. The details can be found in
    `A. Tran, A. Mathews, et. al: FACTORIZED FOURIER NEURAL OPERATORS <https://arxiv.org/pdf/2010.08895.pdf>`_.

    Args:
        in_channels (int): The number of channels in the input space.
        out_channels (int): The number of channels in the output space.
        n_modes (Union[int, list(int)]): The number of modes reserved after linear transformation in Fourier Layer.
        resolutions (Union[int, list(int)]): The resolutions of the input tensor.
        hidden_channels (int): The number of channels of the FNOBlock input and output. Default: ``20``.
        lifting_channels (int): The number of channels of the lifting layer mid channels. Default: None.
        projection_channels (int): The number of channels of the projection layer mid channels. Default: ``128``.
        factor (int): The number of neurons in the hidden layer of a feedforward network. Default: ``1``.
        n_layers (int): The number that Fourier Layer nests. Default: ``4``.
        n_ff_layers (int): The number of layers (hidden layers) in the feedforward neural network. Default: ``2``.
        ff_weight_norm (bool): Whether to do weight normalization in feedforward or not. Used as a reserved function
            interface, the weight normalization is not supported in feedforward. Default: ``False``.
        layer_norm (bool): Whether to do layer normalization in feedforward or not. Default: ``True``.
        share_weight (bool): Whether to share weights between SpectralConv layers or not. Default: ``False``.
        r_padding (int): The number used to pad a tensor on the right in a certain dimension. Pad the domain if
            input is non-periodic. Default: ``0``.
        data_format (str): The input data channel sequence. Default: ``channels_last``.
        positional_embedding (bool): Whether to embed positional information or not. Default: ``True``.
        dft_compute_dtype (dtype.Number): The computation type of DFT in SpectralConvDft. Default: ``mstype.float32``.
        ffno_compute_dtype (dtype.Number): The computation type of MLP in fno skip. Default: ``mstype.float16``.
         Should be ``mstype.float32`` or ``mstype.float16``. mstype.float32 is recommended for
         the GPU backend, mstype.float16 is recommended for the Ascend backend.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(batch\_size, resolution, in\_channels)`.

    Outputs:
        Tensor, the output of this FNOBlocks.

        - **output** (Tensor) -Tensor of shape :math:`(batch\_size, resolution, out\_channels)`.

    Raises:
        TypeError: If `in_channels` is not an int.
        TypeError: If `out_channels` is not an int.
        TypeError: If `hidden_channels` is not an int.
        TypeError: If `lifting_channels` is not an int.
        TypeError: If `projection_channels` is not an int.
        TypeError: If `factor` is not an int.
        TypeError: If `n_layers` is not an int.
        TypeError: If `n_ff_layers` is not an int.
        TypeError: If `ff_weight_norm` is not a Boolean value.
        ValueError: If `ff_weight_norm` is not ``False``.
        TypeError: If `layer_norm` is not a Boolean value.
        TypeError: If `share_weight` is not a Boolean value.
        TypeError: If `r_padding` is not an int.
        TypeError: If `data_format` is not a str.
        TypeError: If `positional_embedding` is not a bool.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> import mindspore.common.dtype as mstype
        >>> from mindflow.cell.neural_operators.ffno import FFNO
        >>> data = Tensor(np.ones([2, 128, 128, 2]), mstype.float32)
        >>> net = FFNO(in_channels=2, out_channels=2, n_modes=[20, 20], resolutions=[128, 128])
        >>> out = net(data)
        >>> print(data.shape, out.shape)
        (2, 128, 128, 2) (2, 128, 128, 2)
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            n_modes,
            resolutions,
            hidden_channels=20,
            lifting_channels=None,
            projection_channels=128,
            factor=1,
            n_layers=4,
            n_ff_layers=2,
            ff_weight_norm=False,
            layer_norm=True,
            share_weight=False,
            r_padding=0,
            data_format="channels_last",
            positional_embedding=True,
            dft_compute_dtype=mstype.float32,
            ffno_compute_dtype=mstype.float16
    ):
        super().__init__()
        check_param_type(in_channels, "in_channels", data_type=int, exclude_type=bool)
        check_param_type(out_channels, "out_channels", data_type=int, exclude_type=bool)
        check_param_type(hidden_channels, "hidden_channels", data_type=int, exclude_type=bool)
        check_param_type(factor, "factor", data_type=int, exclude_type=bool)
        check_param_type(n_layers, "n_layers", data_type=int, exclude_type=bool)
        check_param_type(n_ff_layers, "n_ff_layers", data_type=int, exclude_type=bool)
        check_param_type(ff_weight_norm, "ff_weight_norm", data_type=bool, exclude_type=str)
        check_param_type(layer_norm, "layer_norm", data_type=bool, exclude_type=str)
        check_param_type(share_weight, "share_weight", data_type=bool, exclude_type=str)
        check_param_type(r_padding, "r_padding", data_type=int, exclude_type=bool)
        check_param_type(data_format, "data_format", data_type=str, exclude_type=bool)
        check_param_type(positional_embedding, "positional_embedding", data_type=bool, exclude_type=str)

        if ff_weight_norm:
            raise ValueError(f"The weight normalization is not supported in feedforward\
                             but got value of ff_weight_norm {ff_weight_norm}")
        if r_padding < 0:
            raise ValueError(f"The right padding value cannot be negative but got value of r_padding {r_padding}")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.lifting_channels = lifting_channels
        self.projection_channels = projection_channels
        self.n_modes, self.resolutions = validate_and_expand_dimensions(1, n_modes, resolutions, False)
        self.n_layers = n_layers
        self.r_padding = r_padding
        self.data_format = data_format
        self.positional_embedding = positional_embedding
        if self.positional_embedding:
            self.in_channels += len(self.resolutions)
        self.dft_compute_dtype = dft_compute_dtype
        self.ffno_compute_dtype = ffno_compute_dtype
        self._concat = ops.Concat(axis=-1)
        self._positional_embedding = self._transpose(len(self.resolutions))
        self._padding = self._pad(len(self.resolutions))
        self._lifting = self.lift_channels(
            self.in_channels, self.hidden_channels, self.lifting_channels, self.ffno_compute_dtype)

        self.fourier_weight = None
        if share_weight:
            param_list = []
            for i, n_mode in enumerate(self.n_modes):
                weight_shape = [hidden_channels, hidden_channels, n_mode]
                w_re = Parameter(initializer(XavierNormal(), weight_shape, mstype.float32), name=f'base_w_re_{i}',
                                 requires_grad=True)
                w_im = Parameter(initializer(XavierNormal(), weight_shape, mstype.float32), name=f'base_w_im_{i}',
                                 requires_grad=True)
                param_list.append(w_re)
                param_list.append(w_im)

            self.fourier_weight = ParameterTuple([param for param in param_list])

        self.factor = factor
        self.ff_weight_norm = ff_weight_norm
        self.n_ff_layers = n_ff_layers
        self.layer_norm = layer_norm

        self._ffno_blocks = nn.CellList([FFNOBlocks(in_channels=self.hidden_channels,
                                                    out_channels=self.hidden_channels,
                                                    n_modes=self.n_modes,
                                                    resolutions=self.resolutions,
                                                    factor=self.factor,
                                                    n_ff_layers=self.n_ff_layers,
                                                    ff_weight_norm=self.ff_weight_norm,
                                                    layer_norm=self.layer_norm,
                                                    dropout=0.0, r_padding=self.r_padding,
                                                    use_fork=False, forecast_ff=None, backcast_ff=None,
                                                    fourier_weight=self.fourier_weight,
                                                    dft_compute_dtype=self.dft_compute_dtype
                                                    ) for _ in range(self.n_layers)])

        self._projection = self.lift_channels(
            self.hidden_channels, self.out_channels, self.projection_channels, self.ffno_compute_dtype)

    def lift_channels(self, in_c, out_c, mid_c=0, compute_dtype=mstype.float32):
        if mid_c:
            return nn.SequentialCell([
                nn.Dense(in_c, mid_c, has_bias=True).to_float(compute_dtype),
                nn.Dense(mid_c, out_c, has_bias=True).to_float(compute_dtype)
            ])
        return nn.SequentialCell(nn.Dense(in_c, out_c, has_bias=True).to_float(compute_dtype))

    def construct(self, x: Tensor):
        """construct"""
        batch_size = x.shape[0]
        grid = mint.repeat_interleave(self._positional_embedding.astype(x.dtype), repeats=batch_size, dim=0)

        if self.data_format != "channels_last":
            x = ops.movedim(x, 1, -1)

        if self.positional_embedding:
            x = self._concat((x, grid))

        x = self._lifting(x)
        if self.r_padding != 0:
            x = ops.movedim(x, -1, 1)
            x = ops.pad(x, self._padding)
            x = ops.movedim(x, 1, -1)

        b = Tensor(0, dtype=mstype.float32)
        for block in self._ffno_blocks:
            x, b = block(x)

        if self.r_padding != 0:
            b = self._remove_padding(len(self.resolutions), b)

        x = self._projection(b)

        if self.data_format != "channels_last":
            x = ops.movedim(x, -1, 1)

        return x

    def _transpose(self, n_dim):
        """transpose tensor"""
        if n_dim == 1:
            positional_embedding = Tensor(get_grid_1d(resolution=self.resolutions))
        elif n_dim == 2:
            positional_embedding = Tensor(get_grid_2d(resolution=self.resolutions))
        elif n_dim == 3:
            positional_embedding = Tensor(get_grid_3d(resolution=self.resolutions))
        else:
            raise ValueError(f"The length of input resolutions dimensions should be in [1, 2, 3], but got: {n_dim}")
        return positional_embedding

    def _pad(self, n_dim):
        """pad the domain if input is non-periodic"""
        if not n_dim in {1, 2, 3}:
            raise ValueError(f"The length of input resolutions dimensions should be in [1, 2, 3], but got: {n_dim}")
        return n_dim * [0, self.r_padding]

    def _remove_padding(self, n_dim, b_input):
        """remove pad domain"""
        if n_dim == 1:
            b = b_input[..., :-self.r_padding, :]
        elif n_dim == 2:
            b = b_input[..., :-self.r_padding, :-self.r_padding, :]
        elif n_dim == 3:
            b = b_input[..., :-self.r_padding, :-self.r_padding, :-self.r_padding, :]
        else:
            raise ValueError(f"The length of input resolutions dimensions should be in [1, 2, 3], but got: {n_dim}")
        return b


class FFNO1D(FFNO):
    r"""
    The 1D Factorized Fourier Neural Operator, which usually contains a Lifting Layer,
    a Factorized Fourier Block Layer and a Projection Layer. The details can be found in
    `A. Tran, A. Mathews, et. al: FACTORIZED FOURIER NEURAL OPERATORS <https://arxiv.org/pdf/2010.08895.pdf>`_.

    Args:
        in_channels (int): The number of channels in the input space.
        out_channels (int): The number of channels in the output space.
        n_modes (Union[int, list(int)]): The number of modes reserved after linear transformation in Fourier Layer.
        resolutions (Union[int, list(int)]): The resolutions of the input tensor.
        hidden_channels (int): The number of channels of the FNOBlock input and output. Default: ``20``.
        lifting_channels (int): The number of channels of the lifting layer mid channels. Default: None.
        projection_channels (int): The number of channels of the projection layer mid channels. Default: ``128``.
        factor (int): The number of neurons in the hidden layer of a feedforward network. Default: ``1``.
        n_layers (int): The number that Fourier Layer nests. Default: ``4``.
        n_ff_layers (int): The number of layers (hidden layers) in the feedforward neural network. Default: ``2``.
        ff_weight_norm (bool): Whether to do weight normalization in feedforward or not. Used as a reserved function
            interface, the weight normalization is not supported in feedforward. Default: ``False``.
        layer_norm (bool): Whether to do layer normalization in feedforward or not. Default: ``True``.
        share_weight (bool): Whether to share weights between SpectralConv layers or not. Default: ``False``.
        r_padding (int): The number used to pad a tensor on the right in a certain dimension. Default: ``0``.
        data_format (str): The input data channel sequence. Default: ``channels_last``.
        positional_embedding (bool): Whether to embed positional information or not. Default: ``True``.
        dft_compute_dtype (dtype.Number): The computation type of DFT in SpectralConvDft. Default: ``mstype.float32``.
        ffno_compute_dtype (dtype.Number): The computation type of MLP in fno skip. Default: ``mstype.float16``.
            Should be ``mstype.float32`` or ``mstype.float16``. mstype.float32 is recommended for
            the GPU backend, mstype.float16 is recommended for the Ascend backend.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(batch\_size, resolution, in\_channels)`.

    Outputs:
        Tensor, the output of this FNOBlocks.

        - **output** (Tensor) -Tensor of shape :math:`(batch\_size, resolution, out\_channels)`.

    Raises:
        TypeError: If `in_channels` is not an int.
        TypeError: If `out_channels` is not an int.
        TypeError: If `hidden_channels` is not an int.
        TypeError: If `lifting_channels` is not an int.
        TypeError: If `projection_channels` is not an int.
        TypeError: If `factor` is not an int.
        TypeError: If `n_layers` is not an int.
        TypeError: If `n_ff_layers` is not an int.
        TypeError: If `ff_weight_norm` is not a Boolean value.
        ValueError: If `ff_weight_norm` is not ``False``.
        TypeError: If `layer_norm` is not a Boolean value.
        TypeError: If `share_weight` is not a Boolean value.
        TypeError: If `r_padding` is not an int.
        TypeError: If `data_format` is not a str.
        TypeError: If `positional_embedding` is not a bool.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import numpy as np
        >>> import mindspore
        >>> import mindflow
        >>> from mindspore import Tensor
        >>> import mindspore.common.dtype as mstype
        >>> from mindflow.cell import FFNO1D
        >>> data = Tensor(np.ones([2, 128, 3]), mstype.float32)
        >>> net = FFNO1D(in_channels=3, out_channels=3, n_modes=[20], resolutions=[128])
        >>> out = net(data)
        >>> print(data.shape, out.shape)
        (2, 128, 3) (2, 128, 3)
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            n_modes,
            resolutions,
            hidden_channels=20,
            lifting_channels=None,
            projection_channels=128,
            factor=1,
            n_layers=4,
            n_ff_layers=2,
            ff_weight_norm=False,
            layer_norm=True,
            share_weight=False,
            r_padding=0,
            data_format="channels_last",
            positional_embedding=True,
            dft_compute_dtype=mstype.float32,
            ffno_compute_dtype=mstype.float16
    ):
        n_modes, resolutions = validate_and_expand_dimensions(1, n_modes, resolutions)
        super().__init__(
            in_channels,
            out_channels,
            n_modes,
            resolutions,
            hidden_channels,
            lifting_channels,
            projection_channels,
            factor,
            n_layers,
            n_ff_layers,
            ff_weight_norm,
            layer_norm,
            share_weight,
            r_padding,
            data_format,
            positional_embedding,
            dft_compute_dtype,
            ffno_compute_dtype
        )


class FFNO2D(FFNO):
    r"""
    The 2D Factorized Fourier Neural Operator, which usually contains a Lifting Layer,
    a Factorized Fourier Block Layer and a Projection Layer. The details can be found in
    `A. Tran, A. Mathews, et. al: FACTORIZED FOURIER NEURAL OPERATORS <https://arxiv.org/pdf/2010.08895.pdf>`_.

    Args:
        in_channels (int): The number of channels in the input space.
        out_channels (int): The number of channels in the output space.
        n_modes (Union[int, list(int)]): The number of modes reserved after linear transformation in Fourier Layer.
        resolutions (Union[int, list(int)]): The resolutions of the input tensor.
        hidden_channels (int): The number of channels of the FNOBlock input and output. Default: ``20``.
        lifting_channels (int): The number of channels of the lifting layer mid channels. Default: None.
        projection_channels (int): The number of channels of the projection layer mid channels. Default: ``128``.
        factor (int): The number of neurons in the hidden layer of a feedforward network. Default: ``1``.
        n_layers (int): The number that Fourier Layer nests. Default: ``4``.
        n_ff_layers (int): The number of layers (hidden layers) in the feedforward neural network. Default: ``2``.
        ff_weight_norm (bool): Whether to do weight normalization in feedforward or not. Used as a reserved function
            interface, the weight normalization is not supported in feedforward. Default: ``False``.
        layer_norm (bool): Whether to do layer normalization in feedforward or not. Default: ``True``.
        share_weight (bool): Whether to share weights between SpectralConv layers or not. Default: ``False``.
        r_padding (int): The number used to pad a tensor on the right in a certain dimension. Default: ``0``.
        data_format (str): The input data channel sequence. Default: ``channels_last``.
        positional_embedding (bool): Whether to embed positional information or not. Default: ``True``.
        dft_compute_dtype (dtype.Number): The computation type of DFT in SpectralConvDft. Default: ``mstype.float32``.
        ffno_compute_dtype (dtype.Number): The computation type of MLP in fno skip. Default: ``mstype.float16``.
            Should be ``mstype.float32`` or ``mstype.float16``. mstype.float32 is recommended for
            the GPU backend, mstype.float16 is recommended for the Ascend backend.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(batch\_size, resolution, in\_channels)`.

    Outputs:
        Tensor, the output of this FNOBlocks.

        - **output** (Tensor) -Tensor of shape :math:`(batch\_size, resolution, out\_channels)`.

    Raises:
        TypeError: If `in_channels` is not an int.
        TypeError: If `out_channels` is not an int.
        TypeError: If `hidden_channels` is not an int.
        TypeError: If `lifting_channels` is not an int.
        TypeError: If `projection_channels` is not an int.
        TypeError: If `factor` is not an int.
        TypeError: If `n_layers` is not an int.
        TypeError: If `n_ff_layers` is not an int.
        TypeError: If `ff_weight_norm` is not a Boolean value.
        ValueError: If `ff_weight_norm` is not ``False``.
        TypeError: If `layer_norm` is not a Boolean value.
        TypeError: If `share_weight` is not a Boolean value.
        TypeError: If `r_padding` is not an int.
        TypeError: If `data_format` is not a str.
        TypeError: If `positional_embedding` is not a bool.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import numpy as np
        >>> import mindspore
        >>> import mindflow
        >>> from mindspore import Tensor
        >>> import mindspore.common.dtype as mstype
        >>> from mindflow.cell import FFNO2D
        >>> data = Tensor(np.ones([2, 128, 128, 3]), mstype.float32)
        >>> net = FFNO2D(in_channels=3, out_channels=3, n_modes=[20, 20], resolutions=[128, 128])
        >>> out = net(data)
        >>> print(data.shape, out.shape)
        (2, 128, 128, 3) (2, 128, 128, 3)
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            n_modes,
            resolutions,
            hidden_channels=20,
            lifting_channels=None,
            projection_channels=128,
            factor=1,
            n_layers=4,
            n_ff_layers=2,
            ff_weight_norm=False,
            layer_norm=True,
            share_weight=False,
            r_padding=0,
            data_format="channels_last",
            positional_embedding=True,
            dft_compute_dtype=mstype.float32,
            ffno_compute_dtype=mstype.float16
    ):
        n_modes, resolutions = validate_and_expand_dimensions(2, n_modes, resolutions)
        super().__init__(
            in_channels,
            out_channels,
            n_modes,
            resolutions,
            hidden_channels,
            lifting_channels,
            projection_channels,
            factor,
            n_layers,
            n_ff_layers,
            ff_weight_norm,
            layer_norm,
            share_weight,
            r_padding,
            data_format,
            positional_embedding,
            dft_compute_dtype,
            ffno_compute_dtype
        )


class FFNO3D(FFNO):
    r"""
    The 3D Factorized Fourier Neural Operator, which usually contains a Lifting Layer,
    a Factorized Fourier Block Layer and a Projection Layer. The details can be found in
    `A. Tran, A. Mathews, et. al: FACTORIZED FOURIER NEURAL OPERATORS <https://arxiv.org/pdf/2010.08895.pdf>`_.

    Args:
        in_channels (int): The number of channels in the input space.
        out_channels (int): The number of channels in the output space.
        n_modes (Union[int, list(int)]): The number of modes reserved after linear transformation in Fourier Layer.
        resolutions (Union[int, list(int)]): The resolutions of the input tensor.
        hidden_channels (int): The number of channels of the FNOBlock input and output. Default: ``20``.
        lifting_channels (int): The number of channels of the lifting layer mid channels. Default: None.
        projection_channels (int): The number of channels of the projection layer mid channels. Default: ``128``.
        factor (int): The number of neurons in the hidden layer of a feedforward network. Default: ``1``.
        n_layers (int): The number that Fourier Layer nests. Default: ``4``.
        n_ff_layers (int): The number of layers (hidden layers) in the feedforward neural network. Default: ``2``.
        ff_weight_norm (bool): Whether to do weight normalization in feedforward or not. Used as a reserved function
            interface, the weight normalization is not supported in feedforward. Default: ``False``.
        layer_norm (bool): Whether to do layer normalization in feedforward or not. Default: ``True``.
        share_weight (bool): Whether to share weights between SpectralConv layers or not. Default: ``False``.
        r_padding (int): The number used to pad a tensor on the right in a certain dimension. Default: ``0``.
        data_format (str): The input data channel sequence. Default: ``channels_last``.
        positional_embedding (bool): Whether to embed positional information or not. Default: ``True``.
        dft_compute_dtype (dtype.Number): The computation type of DFT in SpectralConvDft. Default: ``mstype.float32``.
        ffno_compute_dtype (dtype.Number): The computation type of MLP in fno skip. Default: ``mstype.float16``.
            Should be ``mstype.float32`` or ``mstype.float16``. mstype.float32 is recommended for
            the GPU backend, mstype.float16 is recommended for the Ascend backend.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(batch\_size, resolution, in\_channels)`.

    Outputs:
        Tensor, the output of this FNOBlocks.

        - **output** (Tensor) -Tensor of shape :math:`(batch\_size, resolution, out\_channels)`.

    Raises:
        TypeError: If `in_channels` is not an int.
        TypeError: If `out_channels` is not an int.
        TypeError: If `hidden_channels` is not an int.
        TypeError: If `lifting_channels` is not an int.
        TypeError: If `projection_channels` is not an int.
        TypeError: If `factor` is not an int.
        TypeError: If `n_layers` is not an int.
        TypeError: If `n_ff_layers` is not an int.
        TypeError: If `ff_weight_norm` is not a Boolean value.
        ValueError: If `ff_weight_norm` is not ``False``.
        TypeError: If `layer_norm` is not a Boolean value.
        TypeError: If `share_weight` is not a Boolean value.
        TypeError: If `r_padding` is not an int.
        TypeError: If `data_format` is not a str.
        TypeError: If `positional_embedding` is not a bool.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import numpy as np
        >>> import mindspore
        >>> import mindflow
        >>> from mindspore import Tensor
        >>> import mindspore.common.dtype as mstype
        >>> from mindflow.cell import FFNO3D
        >>> data = Tensor(np.ones([2, 128, 128, 128, 3]), mstype.float32)
        >>> net = FFNO3D(in_channels=3, out_channels=3, n_modes=[20, 20, 20], resolutions=[128, 128, 128])
        >>> out = net(data)
        >>> print(data.shape, out.shape)
        (2, 128, 128, 128, 3) (2, 128, 128, 128, 3)
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            n_modes,
            resolutions,
            hidden_channels=20,
            lifting_channels=None,
            projection_channels=128,
            factor=1,
            n_layers=4,
            n_ff_layers=2,
            ff_weight_norm=False,
            layer_norm=True,
            share_weight=False,
            r_padding=0,
            data_format="channels_last",
            positional_embedding=True,
            dft_compute_dtype=mstype.float32,
            ffno_compute_dtype=mstype.float16
    ):
        n_modes, resolutions = validate_and_expand_dimensions(3, n_modes, resolutions)
        super().__init__(
            in_channels,
            out_channels,
            n_modes,
            resolutions,
            hidden_channels,
            lifting_channels,
            projection_channels,
            factor,
            n_layers,
            n_ff_layers,
            ff_weight_norm,
            layer_norm,
            share_weight,
            r_padding,
            data_format,
            positional_embedding,
            dft_compute_dtype,
            ffno_compute_dtype
        )
