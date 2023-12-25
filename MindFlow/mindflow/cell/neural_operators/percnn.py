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
"""physical informed recurrent convolutional nerual network"""
import numpy as np
import mindspore as ms
from mindspore import Parameter, Tensor, nn, ops, lazy_inline

from ...utils.check_func import check_param_type, check_param_type_value

def lazy_inline_wrapper(backend):
    """lazy inline wrapper"""
    if backend == "Ascend":
        def deco(f):
            f = lazy_inline(f)
            return f
    else:
        def deco(f):
            return f
    return deco


class PeriodicPadding(nn.Cell):
    r"""
    PeriodicPadding pads input tensor according to given kernel size.

    Args:
        dim (int): The physical dimension of input. Length of the shape of a 2D input is 4, of a
                    3D input is 5. Data follows `NCHW` or `NCDHW` format.
        kernel_size (int, tuple): Specifies the convolution kernel. If type of kernel_size is int,
                    last 2 or 3 dimension will be padded. If type of kernel_size is tuple, its
                    sequence should be (depth, height, width) for 3D and (height, width) for 2D.

    Inputs:
        **input** (Tensor) - Tensor of shape :math:`(batch\_size, channels, depth, height, width)` for 3D.
                               Tensor of shape :math:`(batch\_size, channels, height, width)` for 2D.

    Outputs:
        Tensor, has the same shape as `input`.

    Raises:
        TypeError: If `dim` is not an int.
        TypeError: If `kernel_size` is not a int/tuple/list.
        ValueError: If length of `kernel_size` is not the same as the value of `dim`.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore as ms
        >>> from mindspore import Tensor, ops, nn
        >>> x= np.zeros((1,2, 48, 48, 48))
        >>> pd = PeriodicPadding(3, (3,3,5))
        >>> x = pd(Tensor(x))
        >>> print(x.shape)

        (1, 2, 50, 50, 52)
    """

    def __init__(self, dim, kernel_size):
        super().__init__()
        check_param_type(dim, "dim", data_type=int)
        check_param_type(kernel_size, "kernel_size", data_type=(int, tuple, list))

        if isinstance(kernel_size, int):
            self.pad = [int((kernel_size - 1) / 2) for _ in range(dim * 2)]
        else:
            if len(kernel_size) != dim:
                raise ValueError("length of kernel size must be the same as dim")

            kernel_size = list(kernel_size)
            kernel_size.reverse()
            self.pad = []
            for size in kernel_size:
                self.pad.append(int((size - 1) / 2))
                self.pad.append(int((size - 1) / 2))

    def construct(self, x):
        x = ops.pad(x, padding=self.pad, mode="circular")
        return x

class PeRCNN(nn.Cell):
    r"""
    Recurrent convolutional neural network Cell.
    lazy_inline is used to accelerate the compile stage, but now it only functions in Ascend backends.
    PeRCNN currently supports input with two physical components. For inputs with different shape, users
    must manually add or remove corresponding parameters and pi_blocks.

    Args:
        dim (int): The physical dimension of input. Length of the shape of a 2D input is 4, of a
                    3D input is 5. Data follows `NCHW` or `NCDHW` format.
        kernel_size (int): Specifies the convolution kernel for parallel convolution layers.
        in_channels (int): The number of channels in the input space.
        hidden_channels (int): Number of channels in the output space of parallel convolution layers.
        dt (int, float): The time step of PeRCNN.
        nu (int, float): The coefficient of diffusion term.
        padding(str): Boundary padding. Now only periodic padding is supported. Default: ``periodic``
        laplace_kernel (mindspore.Tensor): For 3D, Set size of kernel is :math:`(\text{kernel_size[0]},
        \text{kernel_size[1]}, \text{kernel_size[2]})`, then the shape is :math:`(C_{out}, C_{in},
        \text{kernel_size[0]}, \text{kernel_size[1]}, \text{kernel_size[1]})`. For 2D, Tensor of shape
            :math:`(N, C_{in} / \text{groups}, \text{kernel_size[0]}, \text{kernel_size[1]})`, then the size of kernel
            is :math:`(\text{kernel_size[0]}, \text{kernel_size[1]})`.
        conv_layers_num (int): Number of parallel convolution layers. Default: ``3``.
        compute_dtype (dtype.Number): The data type of PeRCNN. Default: ``mindspore.float32``.
                Should be ``mindspore.float16`` or ``mindspore.float32``.
                mindspore.float32 is recommended for GPU backends,
                mindspore.float16 is recommended for Ascend backends.

    Inputs:
        **input** (Tensor) - Tensor of shape :math:`(batch\_size, channels, depth, height, width)` for 3D.
                               Tensor of shape :math:`(batch\_size, channels, height, width)` for 2D.

    Outputs:
        Tensor, has the same shape as `input`.

    Raises:
        TypeError: If `dim`, `in_channels`, `hidden_channels`, `kernel_size` is not an int.
        TypeError: If `dt` and `nu` is not an int nor a float.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore as ms
        >>> from mindflow.cell.neural_operators.percnn import PeRCNN

        >>> laplace_3d = [[[[[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0],
        >>>                 [0.0, 0.0, -0.08333333333333333, 0.0, 0.0],
        >>>                 [0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]],
        >>>                 [[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0],
        >>>                 [0.0, 0.0, 1.3333333333333333, 0.0, 0.0],
        >>>                 [0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]],
        >>>                 [[0.0, 0.0, -0.08333333333333333, 0.0, 0.0],
        >>>                 [0.0, 0.0, 1.3333333333333333, 0.0, 0.0],
        >>>                 [-0.08333333333333333, 1.3333333333333333, -7.5, 1.3333333333333333, -0.08333333333333333],
        >>>                 [0.0, 0.0, 1.3333333333333333, 0.0, 0.0],
        >>>                 [0.0, 0.0, -0.08333333333333333, 0.0, 0.0]],
        >>>                 [[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0],
        >>>                 [0.0, 0.0, 1.3333333333333333, 0.0, 0.0],
        >>>                 [0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]],
        >>>                 [[0.0, 0.0, 0.0, 0.0, 0.0],
        >>>                 [0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, -0.08333333333333333, 0.0, 0.0],
        >>>                 [0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]]]]]
        >>> laplace = np.array(laplace_3d)
        >>> grid_size = 48
        >>> field = 100
        >>> dx_3d = field / grid_size

        >>> laplace_3d_kernel = ms.Tensor(1 / dx_3d**2 * laplace, dtype=ms.float32)

        >>> rcnn_ms = PeRCNN(
        >>>     dim=3,
        >>>     in_channels=2,
        >>>     hidden_channels=2,
        >>>     kernel_size=1,
        >>>     dt=0.5,
        >>>     nu=0.274,
        >>>     laplace_kernel=laplace_3d_kernel,
        >>>     conv_layers_num=3,
        >>>     compute_dtype=ms.float32,
        >>>   )
        >>> input = np.random.randn(1, 2, 48, 48, 48)
        >>> input = ms.Tensor(input, ms.float32)
        >>> output = rcnn_ms(input)
        >>> print(output.shape)

        (1, 2, 48, 48, 48)
    """

    @lazy_inline_wrapper(ms.context.get_context(attr_key="device_target"))
    def __init__(
            self, dim, in_channels, hidden_channels, kernel_size, dt, nu,
            laplace_kernel=None, conv_layers_num=3, padding="periodic", compute_dtype=ms.float32):
        super().__init__()
        check_param_type_value(dim, "dim", valid_value=(2, 3), data_type=int, exclude_type=bool)
        check_param_type(in_channels, "in_channels", data_type=int, exclude_type=bool)
        check_param_type(hidden_channels, "hidden_channels", data_type=int, exclude_type=bool)
        check_param_type(kernel_size, "kernel_size", data_type=int, exclude_type=bool)
        check_param_type(conv_layers_num, "conv_layers_num", data_type=int, exclude_type=bool)
        check_param_type(dt, "dt", data_type=(int, float), exclude_type=bool)
        check_param_type(nu, "nu", data_type=(int, float), exclude_type=bool)

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.input_stride = 1
        self.compute_dtype = compute_dtype
        self.padding = padding
        self.dim = dim
        self.dt = dt
        self.nu = nu  # nu from 0 to upper bound (two times the estimate)
        self._nn_conv_table = {2: nn.Conv2d, 3: nn.Conv3d}
        self._ops_conv_table = {2: ops.conv2d, 3: ops.conv3d}

        if laplace_kernel is not None:
            self.w_laplace = laplace_kernel
            self.coef_u = Parameter(Tensor(np.random.rand(), dtype=self.compute_dtype), requires_grad=True)
            self.coef_v = Parameter(Tensor(np.random.rand(), dtype=self.compute_dtype), requires_grad=True)

        self.u_pi_block = nn.CellList()
        self.v_pi_block = nn.CellList()
        # Parallel conv layer for u
        for i in range(conv_layers_num):
            u_conv_layer = self._nn_conv_table[dim](
                in_channels=in_channels,
                out_channels=hidden_channels,
                kernel_size=kernel_size,
                stride=self.input_stride,
                pad_mode="valid",
                has_bias=True,
            ).to_float(self.compute_dtype)
            self.u_pi_block.append(u_conv_layer)
            u_conv_layer.update_parameters_name(f"u_{i}.")
        # 1x1 layer conv for u
        self.u_conv = self._nn_conv_table[dim](
            in_channels=hidden_channels,
            out_channels=1,
            kernel_size=1,
            stride=1,
            has_bias=True,
        ).to_float(self.compute_dtype)
        # Parallel conv layer for v
        for i in range(conv_layers_num):
            v_conv_layer = self._nn_conv_table[dim](
                in_channels=in_channels,
                out_channels=hidden_channels,
                kernel_size=kernel_size,
                stride=self.input_stride,
                pad_mode="valid",
                has_bias=True,
            ).to_float(self.compute_dtype)
            self.v_pi_block.append(v_conv_layer)
            v_conv_layer.update_parameters_name(f"v_{i}.")
        # 1x1 layer conv for v
        self.v_conv = self._nn_conv_table[dim](
            in_channels=hidden_channels,
            out_channels=1,
            kernel_size=1,
            stride=1,
            has_bias=True,
        ).to_float(self.compute_dtype)

    def construct(self, h):
        """construct function of PeRCNN"""
        if self.padding == "periodic":
            conv_padding = PeriodicPadding(self.dim, self.kernel_size)
        else:
            raise ValueError("unsupported padding type")
        h_conv = conv_padding(h)
        u_res = 1
        v_res = 1
        for conv in self.u_pi_block:
            u_res = u_res * conv(h_conv)
        for conv in self.v_pi_block:
            v_res = v_res * conv(h_conv)
        u_res = self.u_conv(u_res)
        v_res = self.v_conv(v_res)

        if self.w_laplace is not None:
            laplace_padding = PeriodicPadding(self.dim, self.w_laplace.shape[2:])
            h_lap = laplace_padding(h)

            u_pad = h_lap[:, 0:1, ...]
            v_pad = h_lap[:, 1:2, ...]
            u_res = u_res + self.nu * ops.sigmoid(self.coef_u) * self._ops_conv_table[
                self.dim
            ](u_pad, self.w_laplace)
            v_res = v_res + self.nu * ops.sigmoid(self.coef_v) * self._ops_conv_table[
                self.dim
            ](v_pad, self.w_laplace)

        # previous state
        if h.shape[1] < 2:
            raise ValueError("input field should have at least two physical components")
        u_prev = h[:, 0:1, ...]
        v_prev = h[:, 1:2, ...]

        u_next = u_prev + u_res * self.dt
        v_next = v_prev + v_res * self.dt
        out = ops.concat((u_next, v_next), axis=1)

        return out
