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
from mindflow.cell.activation import get_activation
from mindflow.utils.check_func import check_param_type, check_param_type_value

from .sp_transform import ConvCell, TransformCell


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
        - **x** (Tensor) - Tensor of shape :math:`(batch\_size, in\_channels, height, width)`.
    Outputs:
        - **output** (Tensor) -Tensor of shape :math:`(batch\_size, out\_channels, height, width)`.
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
        >>> from sno import SNOKernelCell
        >>> resolution, modes = 100, 12
        >>> matr = Tensor(np.random.rand(modes, resolution), mstype.float32)
        >>> inv_matr = Tensor(np.random.rand(resolution, modes), mstype.float32)
        >>> x = Tensor(np.random.rand(19, 3, N_spatial), mstype.float32)
        >>> net = SNOKernelCell(in_channels=3, out_channels=5, transforms=[[matr,
        >>>                     inv_matr]], kernel_size=5, activation=nn.GELU())
        >>> y = net(x)
        >>> print(x.shape, y.shape)
        (19, 3, 100) (19, 5, 100)
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

        self.direct_transforms = [TransformCell(transforms[i][0]) for i in range(dim)]
        self.inverse_transforms = [TransformCell(transforms[i][1]) for i in range(dim)]
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


class SNO(nn.Cell):
    r"""
    The Spectral Neural Operator (SNO) base class, which contains a lifting layer (encoder),
    multiple spectral transform layers (linear transforms in spectral space) and a projection layer (decoder).
    This is a FNO-like architecture using polynomial transform (Chebyshev, Legendre, etc.) instead of Fourier transform.
    The details can be found in `Spectral Neural Operators <https://arxiv.org/pdf/2205.10573>`_.

    Args:
        in_channels (int): The number of channels in the input space.
        out_channels (int): The number of channels in the output space.
        hidden_channels (int): The number of channels of the SNO layers input and output. Default: 64.
        num_sno_layers (int): The number of spectral layers. Default: 3.
        data_format (str): The input data channel sequence. Default: ``channels_first``.

        transforms (list(list(mindspore.Tensor))): The list of direct and inverse polynomial transforms
        on x, y and z axis, respectively. The list has the following structure: [[transform_x, inv_transform_x],
        ... [transform_z, inv_transform_z]]. The shape of transformation matrix should be (n_modes, resolution),
        where n_modes is the number of polynomial transform modes, resolution is spatial resolution of input
        in the corresponding direction. The shape of inverse transformation is (resolution, n_modes). Default: None.

        kernel_size (int): Specifies the height and width of the convolution kernel in SNO layers. Default: 5.
        activation (Union[str, class]): The activation function, could be either str or class. Default: ``gelu``.
        compute_dtype (dtype.Number): The computation type. Default: mstype.float32.
            Should be ``mstype.float32`` or ``mstype.float16``. mstype.float32 is recommended for
            the GPU backend, mstype.float16 is recommended for the Ascend backend.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(batch\_size, in_channels, resolution)`

    Outputs:
        Tensor of shape :math:`(batch\_size, out_channels, resolution)`.

    Raises:
        TypeError: If `in_channels` is not an int.
        TypeError: If `out_channels` is not an int.
        TypeError: If `hidden_channels` is not an int.
        TypeError: If `num_sno_layers` is not an int.
        TypeError: If `transforms` is not a list.
        ValueError: If `len(transforms)` is not equal to 1.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> import mindspore.common.dtype as mstype
        >>> from sno import SNO
        >>> resolution, modes = 100, 12
        >>> matr = Tensor(np.random.rand(modes, resolution), mstype.float32)
        >>> inv_matr = Tensor(np.random.rand(resolution, modes), mstype.float32)
        >>> net = SNO(dim=2, in_channels=2, out_channels=5, transforms=[ [matr, inv_matr]])
        >>> x = Tensor(np.random.rand(19, 2, resolution), mstype.float32)
        >>> y = net(x)
        >>> print(x.shape, y.shape)
        (19, 2, 100) (19, 5, 100)
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden_channels=64,
                 num_sno_layers=3,
                 data_format="channels_first",
                 transforms=None,
                 kernel_size=5,
                 activation="gelu",
                 compute_dtype=mstype.float32):

        super().__init__()
        check_param_type(in_channels, "in_channels", data_type=int, exclude_type=bool)
        check_param_type(out_channels, "out_channels", data_type=int, exclude_type=bool)
        check_param_type(hidden_channels, "hidden_channels", data_type=int, exclude_type=bool)
        check_param_type(num_sno_layers, "num_sno_layers", data_type=int, exclude_type=bool)
        check_param_type(transforms, "transforms", data_type=list)
        check_param_type_value(len(transforms), "dim", valid_value=1, data_type=int)

        if data_format not in ("channels_first", "channels_last"):
            raise ValueError(
                "data_format must be 'channels_first' or 'channels_last', but got data_format: {}".format(data_format))
        self.dim = len(transforms)
        self.data_format = data_format
        self.permute = None
        if self.data_format == 'channels_last':
            self.permute = (0, 2, 1)

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

        self.sno_kernel = nn.SequentialCell()
        for _ in range(num_sno_layers):
            self.sno_kernel.append(SNOKernelCell(
                hidden_channels, hidden_channels, transforms,
                kernel_size, self.act, compute_dtype))

        self.decoder = nn.SequentialCell([
            ConvCell(self.dim, hidden_channels, hidden_channels, 1, compute_dtype),
            self.act,
            ConvCell(self.dim, hidden_channels, out_channels, 1, compute_dtype)
        ])

    def construct(self, x):
        if self.data_format == 'channels_last':
            x = ops.transpose(x, self.permute)
        x = self.encoder(x)
        x = self.sno_kernel(x)
        x = self.decoder(x)
        if self.data_format == 'channels_last':
            x = ops.transpose(x, self.permute)
        return x


class SNO1D(SNO):
    r"""
    The 1D SNO, which contains a lifting layer (encoder),
    multiple spectral transform layers and a projection layer (decoder).
    See documentation for base class, `SNO`.

    Example:
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> import mindspore.common.dtype as mstype
        >>> from sno import SNO1D
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
