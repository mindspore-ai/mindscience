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

from mindspore import nn, ops, Tensor
import mindspore.common.dtype as mstype

from .dft import SpectralConv1dDft, SpectralConv2dDft, SpectralConv3dDft
from ..activation import get_activation
from ...common.math import get_grid_1d, get_grid_2d, get_grid_3d
from ...utils.check_func import check_param_type


class FNOBlocks(nn.Cell):
    r"""
    The FNOBlock, which usually accompanied by a Lifting Layer ahead and a Projection Layer behind,
    is a part of Fourier Neural Operator. It contains a Fourier Layer and a FNO Skip Layer.
    The details can be found in `Zongyi Li, et. al: FOURIER NEURAL OPERATOR FOR PARAMETRIC PARTIAL
    DIFFERENTIAL EQUATIONS <https://arxiv.org/pdf/2010.08895.pdf>`_.

    Args:
        in_channels (int): The number of channels in the input space.
        out_channels (int): The number of channels in the output space.
        n_modes (Union[int, list(int)]): The number of modes reserved after linear transformation in Fourier Layer.
        resolutions (Union[int, list(int)]): The resolutions of the input tensor.
        act (Union[str, class]): The activation function, could be either str or class. Default: ``gelu``.
        add_residual (bool): Whether to add residual in FNOBlock or not. Default: ``False``.
        dft_compute_dtype (dtype.Number): The computation type of DFT in SpectralConvDft. Default: ``mstype.float32``.
        fno_compute_dtype (dtype.Number): The computation type of MLP in fno skip. Default: ``mstype.float16``.
            Should be ``mstype.float32`` or ``mstype.float16``. mstype.float32 is recommended for
            the GPU backend, mstype.float16 is recommended for the Ascend backend.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(batch\_size, in\_channels, resolution)`.

    Outputs:
        Tensor, the output of this FNOBlocks.

        - **output** (Tensor) -Tensor of shape :math:`(batch\_size, out\_channels, resolution)`.

    Raises:
        TypeError: If `in_channels` is not an int.
        TypeError: If `out_channels` is not an int.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> import mindspore.common.dtype as mstype
        >>> from mindflow.cell.neural_operators import FNOBlocks
        >>> data = Tensor(np.ones([2, 3, 128, 128]), mstype.float32)
        >>> net = FNOBlocks(in_channels=3, out_channels=3, n_modes=[20, 20], resolutions=[128, 128])
        >>> out = net(data)
        >>> print(data.shape, out.shape)
        (2, 3, 128, 128) (2, 3, 128, 128)
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 n_modes,
                 resolutions,
                 act="gelu",
                 add_residual=False,
                 dft_compute_dtype=mstype.float32,
                 fno_compute_dtype=mstype.float16
                 ):
        super().__init__()
        check_param_type(in_channels, "in_channels", data_type=int)
        check_param_type(out_channels, "out_channels", data_type=int)
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
                "The dimension of n_modes should be equal to that of resolutions\
                 but got dimension of n_modes {} and dimension of resolutions {}".format(len(self.n_modes),
                                                                                         len(self.resolutions)))
        self.act = get_activation(act) if isinstance(act, str) else act
        self.add_residual = add_residual
        self.dft_compute_dtype = dft_compute_dtype
        self.fno_compute_dtype = fno_compute_dtype

        if len(self.resolutions) == 1:
            self._convs = SpectralConv1dDft(
                self.in_channels,
                self.out_channels,
                self.n_modes,
                self.resolutions,
                compute_dtype=self.dft_compute_dtype
            )
            self._fno_skips = nn.Conv1d(
                self.in_channels,
                self.out_channels,
                kernel_size=1,
                has_bias=False,
                weight_init="HeUniform"
            ).to_float(self.fno_compute_dtype)
        elif len(self.resolutions) == 2:
            self._convs = SpectralConv2dDft(
                self.in_channels,
                self.out_channels,
                self.n_modes,
                self.resolutions,
                compute_dtype=self.dft_compute_dtype
            )
            self._fno_skips = nn.Conv2d(
                self.in_channels,
                self.out_channels,
                kernel_size=1,
                has_bias=False,
                weight_init="HeUniform"
            ).to_float(self.fno_compute_dtype)
        elif len(self.resolutions) == 3:
            self._convs = SpectralConv3dDft(
                self.in_channels,
                self.out_channels,
                self.n_modes,
                self.resolutions,
                compute_dtype=self.dft_compute_dtype
            )
            self._fno_skips = nn.Conv3d(
                self.in_channels,
                self.out_channels,
                kernel_size=1,
                has_bias=False,
                weight_init="HeUniform"
            ).to_float(self.fno_compute_dtype)
        else:
            raise ValueError("The length of input resolutions dimensions should be in [1, 2, 3], but got: {}".format(
                len(self.resolutions)))

    def construct(self, x: Tensor):
        if self.add_residual:
            x = self.act(self._convs(x) + self._fno_skips(x)) + x
        else:
            x = self.act(self._convs(x) + self._fno_skips(x))
        return x


class FNO(nn.Cell):
    r"""
    The FNO base class, which usually contains a Lifting Layer, a Fourier Block Layer and a Projection Layer.
    The details can be found in `Zongyi Li, et. al: FOURIER NEURAL OPERATOR FOR PARAMETRIC PARTIAL DIFFERENTIAL
    EQUATIONS <https://arxiv.org/pdf/2010.08895.pdf>`_.

    Args:
        in_channels (int): The number of channels in the input space.
        out_channels (int): The number of channels in the output space.
        n_modes (Union[int, list(int)]): The number of modes reserved after linear transformation in Fourier Layer.
        resolutions (Union[int, list(int)]): The resolutions of the input tensor.
        hidden_channels (int): The number of channels of the FNOBlock input and output. Default: ``20``.
        lifting_channels (int): The number of channels of the lifting layer mid channels. Default: None.
        projection_channels (int): The number of channels of the projection layer mid channels. Default: ``128``.
        n_layers (int): The number that Fourier Layer nests. Default: ``4``.
        data_format (str): The input data channel sequence. Default: ``channels_last``.
        fnoblock_act (Union[str, class]): The activation function for FNOBlock, could be either str or class.
            Default: ``identity``.
        mlp_act (Union[str, class]): The activation function for MLP layers, could be either str or class.
            Default: ``gelu``.
        add_residual (bool): Whether to add residual in FNOBlock or not. Default: ``False``.
        positional_embedding (bool): Whether to embed positional information or not. Default: ``True``.
        dft_compute_dtype (dtype.Number): The computation type of DFT in SpectralConvDft. Default: ``mstype.float32``.
        fno_compute_dtype (dtype.Number): The computation type of MLP in fno skip. Default: ``mstype.float16``.
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
        TypeError: If `n_layers` is not an int.
        TypeError: If `data_format` is not a str.
        TypeError: If `add_residual` is not an bool.
        TypeError: If `positional_embedding` is not an bool.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> import mindspore.common.dtype as mstype
        >>> from mindflow.cell.neural_operators.fno import FNO
        >>> data = Tensor(np.ones([2, 3, 128, 128]), mstype.float32)
        >>> net = FNO(in_channels=3, out_channels=3, n_modes=[20, 20], resolutions=[128, 128])
        >>> out = net(data)
        >>> print(data.shape, out.shape)
        (2, 3, 128, 128) (2, 3, 128, 128)
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
            n_layers=4,
            data_format="channels_last",
            fnoblock_act="identity",
            mlp_act="gelu",
            add_residual=False,
            positional_embedding=True,
            dft_compute_dtype=mstype.float32,
            fno_compute_dtype=mstype.float16
    ):
        super().__init__()
        check_param_type(in_channels, "in_channels", data_type=int, exclude_type=bool)
        check_param_type(out_channels, "out_channels", data_type=int, exclude_type=bool)
        check_param_type(hidden_channels, "hidden_channels", data_type=int, exclude_type=bool)
        check_param_type(projection_channels, "projection_channels", data_type=int, exclude_type=bool)
        check_param_type(n_layers, "n_layers", data_type=int, exclude_type=bool)
        check_param_type(data_format, "data_format", data_type=str, exclude_type=bool)
        check_param_type(positional_embedding, "positional_embedding", data_type=bool, exclude_type=str)
        check_param_type(add_residual, "add_residual", data_type=bool, exclude_type=str)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.lifting_channels = lifting_channels
        self.projection_channels = projection_channels
        if isinstance(n_modes, int):
            n_modes = [n_modes]
        self.n_modes = n_modes
        if isinstance(resolutions, int):
            resolutions = [resolutions]
        self.resolutions = resolutions
        if len(self.n_modes) != len(self.resolutions):
            raise ValueError(
                "The dimension of n_modes should be equal to that of resolutions\
                 but got dimension of n_modes {} and dimension of resolutions {}".format(len(self.n_modes),
                                                                                         len(self.resolutions)))
        self.n_layers = n_layers
        self.data_format = data_format
        if fnoblock_act == "identity":
            self.fnoblock_act = ops.Identity()
        else:
            self.fnoblock_act = get_activation(fnoblock_act) if isinstance(fnoblock_act, str) else fnoblock_act
        self.mlp_act = get_activation(mlp_act) if isinstance(mlp_act, str) else mlp_act
        self.add_residual = add_residual
        self.positional_embedding = positional_embedding
        if self.positional_embedding:
            self.in_channels += len(self.resolutions)
        self.dft_compute_dtype = dft_compute_dtype
        self.fno_compute_dtype = fno_compute_dtype
        self._concat = ops.Concat(axis=-1)
        self._positional_embedding, self._input_perm, self._output_perm = self._transpose(len(self.resolutions))
        if self.lifting_channels:
            self._lifting = nn.SequentialCell()
            self._lifting.append(nn.Dense(self.in_channels, self.lifting_channels, has_bias=False))
            self._lifting.append(self.mlp_act)
            self._lifting.append(nn.Dense(self.lifting_channels, self.hidden_channels, has_bias=False))
        else:
            self._lifting = nn.SequentialCell()
            self._lifting.append(nn.Dense(self.in_channels, self.hidden_channels, has_bias=False))
        self._fno_blocks = nn.SequentialCell()
        for _ in range(self.n_layers):
            self._fno_blocks.append(FNOBlocks(self.hidden_channels, self.hidden_channels, n_modes=self.n_modes,
                                              resolutions=self.resolutions, act=self.fnoblock_act,
                                              add_residual=self.add_residual, dft_compute_dtype=self.dft_compute_dtype,
                                              fno_compute_dtype=self.fno_compute_dtype))
        self._projection = nn.SequentialCell()
        self._projection.append(nn.Dense(self.hidden_channels, self.projection_channels, has_bias=False))
        self._projection.append(self.mlp_act)
        self._projection.append(nn.Dense(self.projection_channels, self.out_channels, has_bias=False))

    def construct(self, x: Tensor):
        """construct"""
        batch_size = x.shape[0]
        grid = self._positional_embedding.repeat(batch_size, axis=0).astype(x.dtype)
        x = self._concat((x, grid))
        x = self._lifting(x)
        if self.data_format == "channels_last":
            x = ops.transpose(x, input_perm=self._input_perm)
        x = self._fno_blocks(x)
        if self.data_format == "channels_last":
            x = ops.transpose(x, input_perm=self._output_perm)
        x = self._projection(x)
        return x

    def _transpose(self, n_dim):
        """transpose tensor"""
        if n_dim == 1:
            positional_embedding = Tensor(get_grid_1d(resolution=self.resolutions))
            input_perm = (0, 2, 1)
            output_perm = (0, 2, 1)
        elif n_dim == 2:
            positional_embedding = Tensor(get_grid_2d(resolution=self.resolutions))
            input_perm = (0, 3, 1, 2)
            output_perm = (0, 2, 3, 1)
        elif n_dim == 3:
            positional_embedding = Tensor(get_grid_3d(resolution=self.resolutions))
            input_perm = (0, 4, 1, 2, 3)
            output_perm = (0, 2, 3, 4, 1)
        else:
            raise ValueError(
                "The length of input resolutions dimensions should be in [1, 2, 3], but got: {}".format(n_dim))
        return positional_embedding, input_perm, output_perm


class FNO1D(FNO):
    r"""
    The 1D Fourier Neural Operator, which usually contains a Lifting Layer,
    a Fourier Block Layer and a Projection Layer. The details can be found in
    `Zongyi Li, et. al: FOURIER NEURAL OPERATOR FOR PARAMETRIC PARTIAL DIFFERENTIAL EQUATIONS
    <https://arxiv.org/pdf/2010.08895.pdf>`_.

    Args:
        in_channels (int): The number of channels in the input space.
        out_channels (int): The number of channels in the output space.
        n_modes (Union[int, list(int)]): The number of modes reserved after linear transformation in Fourier Layer.
        resolutions (Union[int, list(int)]): The resolutions of the input tensor.
        hidden_channels (int): The number of channels of the FNOBlock input and output. Default: ``20``.
        lifting_channels (int): The number of channels of the lifting layer mid channels. Default: None.
        projection_channels (int): The number of channels of the projection layer mid channels. Default: ``128``.
        n_layers (int): The number that Fourier Layer nests. Default: ``4``.
        data_format (str): The input data channel sequence. Default: ``channels_last``.
        fnoblock_act (Union[str, class]): The activation function for FNOBlock, could be either str or class.
            Default: ``identity``.
        mlp_act (Union[str, class]): The activation function for MLP layers, could be either str or class.
            Default: ``gelu``.
        add_residual (bool): Whether to add residual in FNOBlock or not. Default: ``False``.
        positional_embedding (bool): Whether to embed positional information or not. Default: ``True``.
        dft_compute_dtype (dtype.Number): The computation type of DFT in SpectralConvDft. Default: ``mstype.float32``.
        fno_compute_dtype (dtype.Number): The computation type of MLP in fno skip. Default: ``mstype.float16``.
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
        TypeError: If `n_layers` is not an int.
        TypeError: If `data_format` is not a str.
        TypeError: If `add_residual` is not an bool.
        TypeError: If `positional_embedding` is not an bool.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> import mindspore.common.dtype as mstype
        >>> from mindflow.cell.neural_operators import FNO1D
        >>> data = Tensor(np.ones([2, 128, 3]), mstype.float32)
        >>> net = FNO1D(in_channels=3, out_channels=3, n_modes=[20], resolutions=[128])
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
            n_layers=4,
            data_format="channels_last",
            fnoblock_act="identity",
            mlp_act="gelu",
            add_residual=False,
            positional_embedding=True,
            dft_compute_dtype=mstype.float32,
            fno_compute_dtype=mstype.float16
    ):
        super().__init__(
            in_channels,
            out_channels,
            n_modes,
            resolutions,
            hidden_channels,
            lifting_channels,
            projection_channels,
            n_layers,
            data_format,
            fnoblock_act,
            mlp_act,
            add_residual,
            positional_embedding,
            dft_compute_dtype,
            fno_compute_dtype
        )


class FNO2D(FNO):
    r"""
    The 2D Fourier Neural Operator, which usually contains a Lifting Layer,
    a Fourier Block Layer and a Projection Layer. The details can be found in
    `Zongyi Li, et. al: FOURIER NEURAL OPERATOR FOR PARAMETRIC PARTIAL DIFFERENTIAL EQUATIONS
    <https://arxiv.org/pdf/2010.08895.pdf>`_.

    Args:
        in_channels (int): The number of channels in the input space.
        out_channels (int): The number of channels in the output space.
        n_modes (Union[int, list(int)]): The number of modes reserved after linear transformation in Fourier Layer.
        resolutions (Union[int, list(int)]): The resolutions of the input tensor.
        hidden_channels (int): The number of channels of the FNOBlock input and output. Default: ``20``.
        lifting_channels (int): The number of channels of the lifting layer mid channels. Default: None.
        projection_channels (int): The number of channels of the projection layer mid channels. Default: ``128``.
        n_layers (int): The number that Fourier Layer nests. Default: ``4``.
        data_format (str): The input data channel sequence. Default: ``channels_last``.
        fnoblock_act (Union[str, class]): The activation function for FNOBlock, could be either str or class.
            Default: ``identity``.
        mlp_act (Union[str, class]): The activation function for MLP layers, could be either str or class.
            Default: ``gelu``.
        add_residual (bool): Whether to add residual in FNOBlock or not. Default: ``False``.
        positional_embedding (bool): Whether to embed positional information or not. Default: ``True``.
        dft_compute_dtype (dtype.Number): The computation type of DFT in SpectralConvDft. Default: ``mstype.float32``.
        fno_compute_dtype (dtype.Number): The computation type of MLP in fno skip. Default: ``mstype.float16``.
            Should be ``mstype.float32`` or ``mstype.float16``. mstype.float32 is recommended for
            the GPU backend, mstype.float16 is recommended for the Ascend backend.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(batch\_size, resolution[0], resolution[1], in\_channels)`.

    Outputs:
        Tensor, the output of this FNOBlocks.

        - **output** (Tensor) -Tensor of shape :math:`(batch\_size, resolution[0], resolution[1], out\_channels)`.

    Raises:
        TypeError: If `in_channels` is not an int.
        TypeError: If `out_channels` is not an int.
        TypeError: If `hidden_channels` is not an int.
        TypeError: If `lifting_channels` is not an int.
        TypeError: If `projection_channels` is not an int.
        TypeError: If `n_layers` is not an int.
        TypeError: If `data_format` is not a str.
        TypeError: If `add_residual` is not an bool.
        TypeError: If `positional_embedding` is not an bool.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> import mindspore.common.dtype as mstype
        >>> from mindflow.cell.neural_operators import FNO2D
        >>> data = Tensor(np.ones([2, 128, 128, 3]), mstype.float32)
        >>> net = FNO2D(in_channels=3, out_channels=3, n_modes=[20, 20], resolutions=[128, 128])
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
            n_layers=4,
            data_format="channels_last",
            fnoblock_act="identity",
            mlp_act="gelu",
            add_residual=False,
            positional_embedding=True,
            dft_compute_dtype=mstype.float32,
            fno_compute_dtype=mstype.float16
    ):
        super().__init__(
            in_channels,
            out_channels,
            n_modes,
            resolutions,
            hidden_channels,
            lifting_channels,
            projection_channels,
            n_layers,
            data_format,
            fnoblock_act,
            mlp_act,
            add_residual,
            positional_embedding,
            dft_compute_dtype,
            fno_compute_dtype
        )


class FNO3D(FNO):
    r"""
    The 3D Fourier Neural Operator, which usually contains a Lifting Layer,
    a Fourier Block Layer and a Projection Layer. The details can be found in
    `Zongyi Li, et. al: FOURIER NEURAL OPERATOR FOR PARAMETRIC PARTIAL DIFFERENTIAL EQUATIONS
    <https://arxiv.org/pdf/2010.08895.pdf>`_.

    Args:
        in_channels (int): The number of channels in the input space.
        out_channels (int): The number of channels in the output space.
        n_modes (Union[int, list(int)]): The number of modes reserved after linear transformation in Fourier Layer.
        resolutions (Union[int, list(int)]): The resolutions of the input tensor.
        hidden_channels (int): The number of channels of the FNOBlock input and output. Default: ``20``.
        lifting_channels (int): The number of channels of the lifting layer mid channels. Default: None.
        projection_channels (int): The number of channels of the projection layer mid channels. Default: ``128``.
        n_layers (int): The number that Fourier Layer nests. Default: ``4``.
        data_format (str): The input data channel sequence. Default: ``channels_last``.
        fnoblock_act (Union[str, class]): The activation function for FNOBlock, could be either str or class.
            Default: ``identity``.
        mlp_act (Union[str, class]): The activation function for MLP layers, could be either str or class.
            Default: ``gelu``.
        add_residual (bool): Whether to add residual in FNOBlock or not. Default: ``False``.
        positional_embedding (bool): Whether to embed positional information or not. Default: ``True``.
        dft_compute_dtype (dtype.Number): The computation type of DFT in SpectralConvDft. Default: ``mstype.float32``.
        fno_compute_dtype (dtype.Number): The computation type of MLP in fno skip. Default: ``mstype.float16``.
            Should be ``mstype.float32`` or ``mstype.float16``. mstype.float32 is recommended for
            the GPU backend, mstype.float16 is recommended for the Ascend backend.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(batch\_size, resolution[0], resolution[1], resolution[2], \
          in\_channels)`.

    Outputs:
        Tensor, the output of this FNOBlocks.

        - **output** (Tensor) -Tensor of shape :math:`(batch\_size, resolution[0], resolution[1],
          resolution[2], out\_channels)`.

    Raises:
        TypeError: If `in_channels` is not an int.
        TypeError: If `out_channels` is not an int.
        TypeError: If `hidden_channels` is not an int.
        TypeError: If `lifting_channels` is not an int.
        TypeError: If `projection_channels` is not an int.
        TypeError: If `n_layers` is not an int.
        TypeError: If `data_format` is not a str.
        TypeError: If `add_residual` is not an bool.
        TypeError: If `positional_embedding` is not an bool.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> import mindspore.common.dtype as mstype
        >>> from mindflow.cell.neural_operators import FNO2D
        >>> data = Tensor(np.ones([2, 128, 128, 128, 3]), mstype.float32)
        >>> net = FNO2D(in_channels=3, out_channels=3, n_modes=[20, 20, 20], resolutions=[128, 128, 128])
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
            n_layers=4,
            data_format="channels_last",
            fnoblock_act="identity",
            mlp_act="gelu",
            add_residual=False,
            positional_embedding=True,
            dft_compute_dtype=mstype.float32,
            fno_compute_dtype=mstype.float16
    ):
        super().__init__(
            in_channels,
            out_channels,
            n_modes,
            resolutions,
            hidden_channels,
            lifting_channels,
            projection_channels,
            n_layers,
            data_format,
            fnoblock_act,
            mlp_act,
            add_residual,
            positional_embedding,
            dft_compute_dtype,
            fno_compute_dtype
        )
