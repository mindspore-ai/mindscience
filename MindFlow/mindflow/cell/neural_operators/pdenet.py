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
"""pde net model"""
import mindspore.numpy as ms_np
import mindspore.common.dtype as mstype
from mindspore import nn, ops, Parameter

from .m2k import _M2K
from ...utils.check_func import check_param_type

__all__ = ['PDENet']


def _count_num_filter(max_order):
    count = 0
    for i in range(max_order + 1):
        for j in range(max_order + 1):
            if i + j <= max_order:
                count += 1
    return count


class PDENet(nn.Cell):
    r"""
    The PDE-Net model.

    PDE-Net is a feed-forward deep network to fulfill two objectives at the same time: to accurately predict dynamics of
    complex systems and to uncover the underlying hidden PDE models. The basic idea is to learn differential operators
    by learning convolution kernels (filters), and apply neural networks or other machine learning methods to
    approximate the unknown nonlinear responses. A special feature of the proposed PDE-Net is that all filters are
    properly constrained, which enables us to easily identify the governing PDE models while still maintaining the
    expressive and predictive power of the network. These constrains are carefully designed by fully exploiting the
    relation between the orders of differential operators and the orders of sum rules of filters (an important concept
    originated from wavelet theory).

    For more details, please refers to the paper `PDE-Net: Learning PDEs from Data
    <https://arxiv.org/pdf/1710.09668.pdf>`_.

    Args:
        height (int): The height number of the input and output tensor of the PDE-Net.
        width (int): The width number of the input and output tensor of the PDE-Net.
        channels (int): The channel number of the input and output tensor of the PDE-Net.
        kernel_size (int): Specifies the height and width of the 2D convolution kernel.
        max_order (int): The max order of the PDE models.
        dx (float): The spatial resolution of x dimension. Default: ``0.01``.
        dy (float): The spatial resolution of y dimension. Default: ``0.01``.
        dt (float): The time step of the PDE-Net. Default: ``0.01``.
        periodic (bool): Specifies whether periodic pad is used with convolution kernels. Default: ``True``.
        enable_moment (bool): Specifies whether the convolution kernels are constrained by moments. Default: ``True``.
        if_fronzen (bool): Specifies whether the moment is frozen. Default: ``False``.

    Inputs:
        - **input** (Tensor) - Tensor of shape :math:`(batch\_size, channels, height, width)`.

    Outputs:
        Tensor, has the same shape as `input` with data type of float32.

    Raises:
        TypeError: If `height`, `width`, `channels`, `kernel_size` or `max_order` is not an int.
        TypeError: If `periodic`, `enable_moment`, `if_fronzen` is not a bool.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> import mindspore.common.dtype as mstype
        >>> from mindflow.cell.neural_operators import PDENet
        >>> input = Tensor(np.random.rand(1, 2, 16, 16), mstype.float32)
        >>> net = PDENet(16, 16, 2, 5, 3, 2)
        >>> output = net(input)
        >>> print(output.shape)
        (1, 2, 16, 16)
    """

    def __init__(self,
                 height,
                 width,
                 channels,
                 kernel_size,
                 max_order,
                 dx=0.01,
                 dy=0.01,
                 dt=0.01,
                 periodic=True,
                 enable_moment=True,
                 if_fronzen=False):
        """Initialize PDE-Net."""
        super().__init__()
        check_param_type(height, "height", data_type=int)
        check_param_type(width, "width", data_type=int)
        check_param_type(channels, "channels", data_type=int)
        check_param_type(kernel_size, "kernel_size", data_type=int)
        check_param_type(max_order, "max_order", data_type=int)
        check_param_type(periodic, "periodic", data_type=bool)
        check_param_type(enable_moment, "enable_moment", data_type=bool)
        check_param_type(if_fronzen, "if_fronzen", data_type=bool)
        self.in_c = channels
        self.out_c = channels
        self.periodic = periodic
        self.kernel_size = kernel_size
        self.max_order = max_order
        self.delta_t = dt
        self.h = height
        self.w = width
        self.dtype = mstype.float32
        self.num_filter = _count_num_filter(max_order)
        self.dx = dx
        self.dy = dy
        self.enable_moment = enable_moment
        self.if_fronzen = if_fronzen
        self.padding = int((self.kernel_size - 1) / 2)
        if self.enable_moment:
            self._init_moment()
        else:
            self.id_conv = nn.Conv2d(self.in_c, self.out_c, kernel_size=self.kernel_size, pad_mode='valid')
            self.fd_conv = nn.Conv2d(self.in_c, self.num_filter - 1, kernel_size=self.kernel_size, pad_mode='valid')
        self.m2k = _M2K((self.kernel_size, self.kernel_size))
        self.idx2ij = {}
        if self.periodic:
            self.pad = [self.padding, self.padding, self.padding, self.padding]
            self.padding = 0
        self.coe_param = Parameter(ops.UniformReal(seed=2)((self.num_filter - 1, self.h, self.w)))

    def construct(self, x):
        return self._one_step_forward(x)

    @property
    def coe(self):
        return self.coe_param

    def _one_step_forward(self, x):
        if self.periodic:
            x = self._periodicpad(x)

        cast = ops.Cast()
        x = cast(x, self.dtype)

        if self.enable_moment:
            if self.if_fronzen:
                cur_moment = self.raw_moment
            else:
                cur_moment = self.moment * self.mask + self.raw_moment
            kernel = []
            for idx in range(cur_moment.shape[0]):
                kernel.append(self.m2k(cur_moment[idx]))
            kernel = ops.Stack()(kernel).astype(self.dtype)
            kernel = self.scale * kernel
            id_kernel = kernel[0].reshape((1, 1, self.kernel_size, self.kernel_size))
            fd_kernel = kernel[1:].reshape((self.num_filter - 1, 1, self.kernel_size, self.kernel_size))
            id_conv2d = ops.Conv2D(out_channel=id_kernel.shape[0], kernel_size=self.kernel_size, pad=self.padding)
            fd_conv2d = ops.Conv2D(out_channel=fd_kernel.shape[0], kernel_size=self.kernel_size, pad=self.padding)
            id_out = id_conv2d(x, id_kernel)
            fd_out = fd_conv2d(x, fd_kernel)
        else:
            id_out = self.id_conv(x)
            fd_out = self.fd_conv(x)

        f = 0
        for idx in range(fd_out.shape[1]):
            if idx == 0:
                f = self.coe[idx] * fd_out[:, 0:1, :, :]
            else:
                f = f + self.coe[idx] * fd_out[:, idx:(idx + 1), :, :]

        out = id_out + f * self.delta_t
        return out

    def _init_moment(self):
        raw_moment = ms_np.zeros((self.num_filter, self.kernel_size, self.kernel_size))
        mask = ms_np.ones((self.num_filter, self.kernel_size, self.kernel_size))
        scale = ms_np.ones((self.num_filter,))

        self.idx2ij = {}
        idx = 0
        for o1 in range(self.max_order + 1):
            for o2 in range(o1 + 1):
                i = o1 - o2
                j = o2
                self.idx2ij[str(idx)] = (i, j,)
                raw_moment[idx, j, i] = 1
                scale[idx] = 1.0 / (self.dx ** i * self.dy ** j)
                for p in range(i + j + 1):
                    for q in range(i + j + 1):
                        if p + q <= (i + j):
                            mask[idx, p, q] = 0
                idx += 1

        scale = scale.reshape([self.num_filter, 1, 1])
        self.raw_moment = raw_moment
        self.mask = mask
        self.scale = scale
        self.moment = Parameter(raw_moment)

    def _periodicpad(self, x):
        cast = ops.Cast()
        x = cast(x, self.dtype)
        x_dim = len(x.shape)
        inputs = ops.Transpose()(x, tuple(range(x_dim - 1, -1, -1)))
        i = 0
        periodic_pad = self.pad
        for _ in periodic_pad:
            if i + 2 >= len(periodic_pad):
                break
            pad_value = periodic_pad[i]
            pad_next_value = periodic_pad[i + 1]
            permute = list(range(x_dim))
            permute[i] = 0
            permute[0] = i
            permute_tuple = tuple(permute)
            inputs = ops.Transpose()(inputs, permute_tuple)
            inputlist = [inputs,]
            if pad_value > 0:
                inputlist = [inputs[-pad_value:, :, :, :], inputs]
            if pad_next_value > 0:
                inputlist = inputlist + [inputs[0:pad_next_value, :, :, :],]
            if pad_value + pad_next_value > 0:
                inputs = ops.Concat()(inputlist)
            inputs = ops.Transpose()(inputs, permute_tuple)
            i += 1
        x = ops.Transpose()(inputs, tuple(range(x_dim - 1, -1, -1)))
        return x
