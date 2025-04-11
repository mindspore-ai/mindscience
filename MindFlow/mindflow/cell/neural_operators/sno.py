# Copyright 2024 Huawei Technologies Co., Ltd
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
"""sno"""
import mindspore.common.dtype as mstype
from mindspore import ops, nn

from .sp_transform import ConvCell, TransformCell, Dim
from ..activation import get_activation
from ..unet2d import UNet2D
from ...utils.check_func import check_param_type, check_param_type_value


class SNOKernelCell(nn.Cell):
    r"""
    The SNO Kernel, which performs polynomial transform, linear convolution in spectral space,
    and then inverse polynomial transform. Is a part of Spectral Neural Operator.
    It contains a spectral Layer and a skip Layer.

    Args:
        in_channels (int): The number of channels in the input space.
        out_channels (int): The number of channels in the output space.

        transforms (list(list(mindspore.Tensor))): The list of direct and inverse polynomial transforms
        on x, y and z axis, respectively. The list has the following structure:
        [[transform_x, inv_transform_x], ... [transform_z, inv_transform_z]]. For the detailed description, see `SNO`.

        kernel_size (int): Specifies the height and width of the convolution kernel in SNO layer.
        activation (mindspore.nn.Cell): The activation function.
        compute_dtype (dtype.Number): The computation type. Default: ``mstype.float32``.

    Inputs:
        - **x** (Tensor) - Tensor with shape :math:`(batch\_size, in\_channels, height, width)`.
    Outputs:
        - **output** (Tensor) -Tensor with shape :math:`(batch\_size, out\_channels, height, width)`.
    Raises:
        TypeError: If `in_channels` is not an int.
        TypeError: If `out_channels` is not an int.
        TypeError: If `kernel_size` is not an int.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> import mindspore.common.dtype as mstype
        >>> from mindflow.cell.neural_operators.sno import SNOKernelCell
        >>> resolution, modes = 100, 12
        >>> matr = Tensor(np.random.rand(modes, resolution), mstype.float32)
        >>> inv_matr = Tensor(np.random.rand(resolution, modes), mstype.float32)
        >>> x = Tensor(np.random.rand(19, 3, N_spatial, N_spatial), mstype.float32)
        >>> net = SNOKernelCell(in_channels=3, out_channels=5, transforms=[[matr,
        >>>                     inv_matr]]*2, kernel_size=5, activation=nn.GELU())
        >>> y = net(x)
        >>> print(x.shape, y.shape)
        (19, 3, 100, 100) (19, 5, 100, 100)
    """
    def __init__(self,
                 in_channels, out_channels,
                 transforms, kernel_size, activation,
                 compute_dtype=mstype.float32):

        super().__init__()
        check_param_type(in_channels, "in_channels", data_type=int, exclude_type=bool)
        check_param_type(out_channels, "out_channels", data_type=int, exclude_type=bool)
        check_param_type(kernel_size, "kernel_size", data_type=int, exclude_type=bool)

        dim = len(transforms)

        self.linear_conv = ConvCell(dim, in_channels, out_channels, kernel_size, compute_dtype)
        self.skip_conv = ConvCell(dim, in_channels, out_channels, 1, compute_dtype)
        axis = [Dim.x, Dim.y, Dim.z]

        self.direct_transforms = [TransformCell(dim, transforms[i][0],
                                                axis[i]) for i in range(dim)]
        self.inverse_transforms = [TransformCell(dim, transforms[i][1],
                                                 axis[i]) for i in range(dim)]
        self.act = activation

    def spectral_transform(self, x):
        '''polynomial transform, linear layer
        and inverse polynomial transform  '''
        a = x.copy()
        for transform in self.direct_transforms:
            a = transform(a)

        s = self.linear_conv(a)

        for inv_transform in self.inverse_transforms:
            s = inv_transform(s)
        return s

    def construct(self, x):
        s = self.spectral_transform(x)
        x = s + self.skip_conv(x)
        return self.act(x)


class USNOKernelCell2D(SNOKernelCell):
    r"""
    The 2D SNO Kernel with UNet skip block, which performs polynomial transform, linear convolution in spectral space,
    and then inverse polynomial transform. Is a part of Spectral Neural Operator.
    It contains a spectral layer, and UNet block is used as a skip layer.
    For the detailed description, see `SNOKernelCell`.
    Args:
        num_strides (int):
        The number of convolutional downsample blocks in UNet skip block.
    Raises:
        TypeError: If `num_strides` is not an int.
    """
    def __init__(self,
                 in_channels, out_channels,
                 transforms, kernel_size,
                 num_strides, activation,
                 compute_dtype=mstype.float32):

        super().__init__(
            in_channels, out_channels,
            transforms, kernel_size,
            activation, compute_dtype)

        check_param_type(num_strides, "num_strides", data_type=int, exclude_type=bool)
        self.unet = UNet2D(in_channels, out_channels, base_channels=in_channels,
                           n_layers=num_strides, data_format='NCHW',
                           activation=activation, enable_bn=False)
        self.skip_conv = None

    def construct(self, x):
        s = self.spectral_transform(x)
        x = s + self.unet(x)
        return self.act(x)


class SNO(nn.Cell):
    r"""
    The Spectral Neural Operator (SNO) base class, which contains a lifting layer (encoder),
    multiple spectral transform layers (linear transforms in spectral space) and a projection layer (decoder).
    This is a FNO-like architecture using polynomial transform (Chebyshev, Legendre, etc.) instead of Fourier transform.
    The details can be found in `Spectral Neural Operators <https://arxiv.org/pdf/2205.10573>`_.

    Args:
        in_channels (int): The number of channels in the input space.
        out_channels (int): The number of channels in the output space.
        hidden_channels (int): The number of channels of the SNO layers input and output. Default: ``64``.
        num_sno_layers (int): The number of spectral layers. Default: ``3``.
        data_format (str): The input data channel sequence. Default: ``channels_first``.
        transforms (list(list(mindspore.Tensor))): The list of direct and inverse polynomial transforms
            on x, y and z axis, respectively. The list has the following structure: [[transform_x, inv_transform_x],
            [transform_z, inv_transform_z]]. The shape of transformation matrix should be (n_modes, resolution),
            where n_modes is the number of polynomial transform modes, resolution is spatial resolution of input
            in the corresponding direction. The shape of inverse transformation is (resolution, n_modes).
            Default: ``None``.
        kernel_size (int): Specifies the height and width of the convolution kernel in SNO layers. Default: ``5``.
        num_usno_layers (int): The number of spectral layers with UNet skip blocks. Default: ``0``.
        num_unet_strides (int): The number of convolutional downsample blocks in UNet skip blocks. Default: ``1``.
        activation (Union[str, class]): The activation function, could be either str or class. Default: ``gelu``.
        compute_dtype (dtype.Number): The computation type. Default: ``mstype.float32``.
            Should be ``mstype.float32`` or ``mstype.float16``. mstype.float32 is recommended for
            the GPU backend, mstype.float16 is recommended for the Ascend backend.

    Inputs:
        - **x** (Tensor) - Tensor with shape :math:`(batch\_size, in_channels, resolution)`.

    Outputs:
        Tensor with shape :math:`(batch\_size, out_channels, resolution)`.

    Raises:
        TypeError: If `in_channels` is not an int.
        TypeError: If `out_channels` is not an int.
        TypeError: If `hidden_channels` is not an int.
        TypeError: If `num_sno_layers` is not an int.
        TypeError: If `transforms` is not a list.
        ValueError: If `len(transforms)` is not in (1, 2, 3).
        TypeError: If `num_usno_layers` is not an int.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> import mindspore.common.dtype as mstype
        >>> from mindflow.cell.neural_operators.sno import SNO
        >>> resolution, modes = 100, 12
        >>> matr = Tensor(np.random.rand(modes, resolution), mstype.float32)
        >>> inv_matr = Tensor(np.random.rand(resolution, modes), mstype.float32)
        >>> net = SNO(dim=2, in_channels=2, out_channels=5, transforms=[ [matr, inv_matr] * 2])
        >>> x = Tensor(np.random.rand(19, 2, resolution, resolution), mstype.float32)
        >>> y = net(x)
        >>> print(x.shape, y.shape)
        (19, 2, 100, 100) (19, 5, 100, 100)
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden_channels=64,
                 num_sno_layers=3,
                 data_format="channels_first",
                 transforms=None,
                 kernel_size=5,
                 num_usno_layers=0,
                 num_unet_strides=1,
                 activation="gelu",
                 compute_dtype=mstype.float32):

        super().__init__()
        check_param_type(in_channels, "in_channels", data_type=int, exclude_type=bool)
        check_param_type(out_channels, "out_channels", data_type=int, exclude_type=bool)
        check_param_type(hidden_channels, "hidden_channels", data_type=int, exclude_type=bool)
        check_param_type(num_sno_layers, "num_sno_layers", data_type=int, exclude_type=bool)
        check_param_type(transforms, "transforms", data_type=list)
        check_param_type_value(len(transforms), "dim", valid_value=(1, 2, 3), data_type=int)
        check_param_type(num_usno_layers, "num_usno_layers", data_type=int, exclude_type=bool)

        if data_format not in ("channels_first", "channels_last"):
            raise ValueError(
                "data_format must be 'channels_first' or 'channels_last', but got data_format: {}".format(data_format))
        self.dim = len(transforms)
        self.data_format = data_format
        self.permute_input = None
        self.permute_output = None
        if self.data_format == 'channels_last':
            self._transpose()

        if activation is not None:
            self.act = get_activation(activation) if isinstance(activation, str) else activation
        else:
            self.act = ops.Identity()

        self.encoder = nn.SequentialCell([
            ConvCell(self.dim, in_channels, hidden_channels, 1, compute_dtype),
            self.act
        ])

        if num_sno_layers == 0:
            raise ValueError("SNO should contain at least one spectral layer")

        if self.dim != 2:
            num_usno_layers = 0

        self.sno_kernel = nn.SequentialCell()
        for _ in range(num_sno_layers):
            self.sno_kernel.append(SNOKernelCell(
                hidden_channels, hidden_channels, transforms,
                kernel_size, self.act, compute_dtype))

        for _ in range(num_usno_layers):
            self.sno_kernel.append(USNOKernelCell2D(
                hidden_channels, hidden_channels,
                transforms, kernel_size,
                num_unet_strides, self.act, compute_dtype))

        self.decoder = nn.SequentialCell([
            ConvCell(self.dim, hidden_channels, hidden_channels, 1, compute_dtype),
            self.act,
            ConvCell(self.dim, hidden_channels, out_channels, 1, compute_dtype)
        ])

    def construct(self, x):
        if self.data_format == 'channels_last':
            x = ops.transpose(x, self.permute_input)
        x = self.encoder(x)
        x = self.sno_kernel(x)
        x = self.decoder(x)
        if self.data_format == 'channels_last':
            x = ops.transpose(x, self.permute_output)
        return x

    def _transpose(self):
        if self.dim == 1:
            self.permute_output = self.permute_input = (0, 2, 1)
        elif self.dim == 2:
            self.permute_input = (0, 3, 1, 2)
            self.permute_output = (0, 2, 3, 1)
        elif self.dim == 3:
            self.permute_input = (0, 4, 3, 1, 2)
            self.permute_output = (0, 2, 3, 4, 1)


class SNO1D(SNO):
    r"""
    The 1D SNO, which contains a lifting layer (encoder),
    multiple spectral transform layers and a projection layer (decoder).
    See documentation for base class, :class:`mindflow.cell.SNO`.

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> import mindspore.common.dtype as mstype
        >>> from mindflow.cell import SNO1D
        >>> resolution, modes = 100, 12
        >>> matr = Tensor(np.random.rand(modes, resolution), mstype.float32)
        >>> inv_matr = Tensor(np.random.rand(resolution, modes), mstype.float32)
        >>> net = SNO1D(in_channels=3, out_channels=7, transforms=[[matr, inv_matr]])
        >>> x = Tensor(np.random.rand(5, 3, resolution), mstype.float32)
        >>> y = net(x)
        >>> print(x.shape, y.shape)
        (5, 3, 100) (5, 7, 100)
    """
    def __init__(
            self,
            in_channels,
            out_channels,
            hidden_channels=64,
            num_sno_layers=3,
            data_format="channels_first",
            transforms=None,
            kernel_size=5,
            activation="gelu",
            compute_dtype=mstype.float32):

        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            num_sno_layers=num_sno_layers,
            data_format=data_format,
            transforms=transforms,
            kernel_size=kernel_size,
            activation=activation,
            compute_dtype=compute_dtype)


class SNO2D(SNO):
    r"""
    The 2D SNO, which contains a lifting layer (encoder),
    multiple spectral transform layers and a projection layer (decoder).
    See documentation for base class, :class:`mindflow.cell.SNO`.

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> import mindspore.common.dtype as mstype
        >>> from mindflow.cell import SNO2D
        >>> resolution, modes = 100, 12
        >>> matr = Tensor(np.random.rand(modes, resolution), mstype.float32)
        >>> inv_matr = Tensor(np.random.rand(resolution, modes), mstype.float32)
        >>> net = SNO2D(in_channels=2, out_channels=5, transforms=[[matr, inv_matr]] * 2,
        >>>             num_usno_layers=2, num_unet_strides=2)
        >>> x = Tensor(np.random.rand(19, 2, resolution, resolution), mstype.float32)
        >>> y = net(x)
        >>> print(x.shape, y.shape)
        (19, 2, 100, 100) (19, 5, 100, 100)
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            hidden_channels=64,
            num_sno_layers=3,
            data_format="channels_first",
            transforms=None,
            kernel_size=5,
            num_usno_layers=0,
            num_unet_strides=1,
            activation="gelu",
            compute_dtype=mstype.float32):

        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            num_sno_layers=num_sno_layers,
            data_format=data_format,
            transforms=transforms,
            kernel_size=kernel_size,
            num_usno_layers=num_usno_layers,
            num_unet_strides=num_unet_strides,
            activation=activation,
            compute_dtype=compute_dtype)


class SNO3D(SNO):
    r"""
    The 3D SNO, which contains a lifting layer (encoder),
    multiple spectral transform layers and a projection layer (decoder).
    See documentation for base class, :class:`mindflow.cell.SNO`.

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> import mindspore.common.dtype as mstype
        >>> from mindflow.cell import SNO3D
        >>> grid_size, grid_size_z, modes = 64, 40, 12
        >>> matr = Tensor(np.random.rand(modes, grid_size), mstype.float32)
        >>> inv_matr = Tensor(np.random.rand(grid_size, modes), mstype.float32)
        >>> matr_1 = Tensor(np.random.rand(modes, grid_size_z), mstype.float32)
        >>> inv_matr_1 = Tensor(np.random.rand(grid_size_z, modes), mstype.float32)
        >>> net = SNO3D(in_channels=10, out_channels=1,
        >>>             transforms=[[matr, inv_matr]] * 2 + [[matr_1, inv_matr_1]])
        >>> x = Tensor(np.random.rand(10, 10, resolution, resolution, grid_size_z), mstype.float32)
        >>> y = net(x)
        >>> print(x.shape, y.shape)
        (10, 10, 64, 64, 40) (10, 1, 64, 64, 40)
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            hidden_channels=64,
            num_sno_layers=3,
            data_format="channels_first",
            transforms=None,
            kernel_size=5,
            activation="gelu",
            compute_dtype=mstype.float32):

        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            num_sno_layers=num_sno_layers,
            data_format=data_format,
            transforms=transforms,
            kernel_size=kernel_size,
            activation=activation,
            compute_dtype=compute_dtype)
