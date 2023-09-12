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
# ==============================================================================
"""dft"""
import mindspore as ms
import mindspore.numpy as mnp
from mindspore import nn, ops, Tensor, Parameter
import numpy as np
from scipy.linalg import dft

from sciai.utils.check_utils import _check_param_no_greater, _check_value_in, _check_type, _check_param_even


class DFT1d(nn.Cell):
    """One dimensional Discrete Fourier Transformation"""

    def __init__(self, n, modes, last_index, idx=0, inv=False, dtype=ms.float32):
        super().__init__()
        self.n = n
        self.dft_mat = dft(n, scale="sqrtn")
        self.modes = modes
        self.last_index = last_index
        self.inv = inv
        self.idx = idx

        self.dft_mode_mat_upper = self.dft_mat[:, :modes]
        self.a_re_upper = Tensor(self.dft_mode_mat_upper.real, dtype=dtype)
        self.a_im_upper = Tensor(self.dft_mode_mat_upper.imag, dtype=dtype)

        self.dft_mode_mat_lower = self.dft_mat[:, -modes:]
        self.a_re_lower = Tensor(self.dft_mode_mat_lower.real, dtype=dtype)
        self.a_im_lower = Tensor(self.dft_mode_mat_lower.imag, dtype=dtype)
        self.concat = ops.Concat(axis=-1)

        if self.inv:
            self.a_re_upper = self.a_re_upper.T
            self.a_im_upper = -self.a_im_upper.T
            if last_index:
                if modes == n // 2 + 1:
                    self.dft_mat_res = self.dft_mat[:, -modes + 2:]
                else:
                    self.dft_mat_res = self.dft_mat[:, -modes + 1:]

                mat = Tensor(np.zeros(n), dtype=dtype).reshape(n, 1)
                self.a_re_res = mnp.flip(Tensor(self.dft_mat_res.real, dtype=dtype), axis=-1)
                self.a_im_res = mnp.flip(Tensor(self.dft_mat_res.imag, dtype=dtype), axis=-1)
                if modes == n // 2 + 1:
                    self.a_re_res = self.concat((mat, self.a_re_res, mat))
                    self.a_im_res = self.concat((mat, self.a_im_res, mat))
                else:
                    self.a_re_res = self.concat((mat, self.a_re_res))
                    self.a_im_res = self.concat((mat, self.a_im_res))

                self.a_re_res = self.a_re_res.T
                self.a_im_res = -self.a_im_res.T
            else:
                self.a_re_res = self.a_re_lower.T
                self.a_im_res = -self.a_im_lower.T

        if (self.n - 2 * self.modes) > 0:
            self.mat = ops.zeros((self.n - 2 * self.modes), dtype)

    def swap_axes(self, x_re, x_im):
        return x_re.swapaxes(-1, self.idx), x_im.swapaxes(-1, self.idx)

    def complex_matmul(self, x_re, x_im, a_re, a_im):
        y_re = ops.matmul(x_re, a_re) - ops.matmul(x_im, a_im)
        y_im = ops.matmul(x_im, a_re) + ops.matmul(x_re, a_im)
        return y_re, y_im

    def construct(self, x):
        """construct"""
        x_re, x_im = x

        if not self.inv:
            x_re, x_im = self.swap_axes(x_re, x_im)
            y_re, y_im = self.complex_matmul(x_re=x_re, x_im=x_im, a_re=self.a_re_upper, a_im=self.a_im_upper)

            if not self.last_index:
                y_re2, y_im2 = self.complex_matmul(x_re=x_re, x_im=x_im, a_re=self.a_re_lower, a_im=self.a_im_lower)

                if self.n == self.modes * 2:
                    y_re = self.concat((y_re, y_re2))
                    y_im = self.concat((y_im, y_im2))
                else:
                    dims = x_re.shape[:-1]
                    length = len(dims)
                    mat = self.mat
                    for i in range(length - 1, -1, -1):
                        mat = mat.expand_dims(0).repeat(dims[i], 0)
                    y_re = self.concat((y_re, mat, y_re2))
                    y_im = self.concat((y_im, mat, y_im2))

            y_re, y_im = self.swap_axes(y_re, y_im)
            return y_re, y_im

        x_re, x_im = self.swap_axes(x_re, x_im)
        y_re, y_im = self.complex_matmul(x_re=x_re[..., :self.modes], x_im=x_im[..., :self.modes],
                                         a_re=self.a_re_upper, a_im=self.a_im_upper)
        y_re, y_im = self.swap_axes(y_re, y_im)

        if self.last_index:
            y_re_res, y_im_res = self.complex_matmul(x_re=x_re, x_im=x_im, a_re=self.a_re_res, a_im=-self.a_im_res)
        else:
            y_re_res, y_im_res = self.complex_matmul(x_re=x_re[..., -self.modes:], x_im=x_im[..., -self.modes:],
                                                     a_re=self.a_re_res, a_im=self.a_im_res)

        y_re_res, y_im_res = self.swap_axes(y_re_res, y_im_res)
        return y_re + y_re_res, y_im + y_im_res


class DFTn(nn.Cell):
    """N dimensional Discrete Fourier Transformation"""

    def __init__(self, shape, modes, dim=None, inv=False, dtype=ms.float32):
        super().__init__()
        if dim is None:
            dim = range(len(shape))
        self.dft1_seq = nn.SequentialCell()
        last_index = [False for _ in range(len(shape))]
        last_index[-1] = True
        for dim_id, idx in enumerate(dim):
            dft1d = DFT1d(n=shape[dim_id], modes=modes[dim_id], last_index=last_index[dim_id], idx=idx, inv=inv,
                          dtype=dtype)
            self.dft1_seq.append(dft1d)

    def construct(self, x):
        """construct"""
        return self.dft1_seq(x)


def _dftn(shape, modes, dim=None, dtype=ms.float32):
    return DFTn(shape=shape, modes=modes, dim=dim, inv=False, dtype=dtype)


def _idftn(shape, modes, dim=None, dtype=ms.float32):
    return DFTn(shape=shape, modes=modes, dim=dim, inv=True, dtype=dtype)


def dft3(shape, modes, dim=(-3, -2, -1), dtype=ms.float32):
    r"""
    Calculate three-dimensional discrete Fourier transform. Corresponding to the rfftn operator in torch.

    Args:
        shape (tuple): Dimension of the input 'x'.
        modes (tuple): The length of the output transform axis. The `modes` must be no greater than half of the
            dimension of input 'x'.
        dim (tuple): Dimensions to be transformed.
        dtype (ms.dtype): The type of input tensor. Default: ms.float32.

    Inputs:
        - **x** (Tensor, Tensor): The input data. It's 3-D tuple of Tensor. It's a complex,
          including x real and imaginary. Tensor of shape :math:`(*, *)`.

    Returns:
        Complex tensor with the same shape of input x.

    Raises:
        TypeError: If `shape` is not a tuple.
        ValueError: If the length of `shape` is no equal to 3.

    Examples:
        >>> import numpy as np
        >>> import mindspore as ms
        >>> from mindspore import Tensor, ops
        >>> from sciai.architecture.neural_operators.dft import dft3
        >>> array = np.ones((6, 6, 6)) * np.arange(1, 7)
        >>> x_re = Tensor(array, dtype=ms.float32)
        >>> x_im = x_re
        >>> dft3_cell = dft3(shape=array.shape, modes=(2, 2, 2), dtype=ms.float32)
        >>> ret, _ = dft3_cell((x_re, x_im))
        >>> print(ret)
        [[[ 5.1439293e+01 -2.0076393e+01]
        [ 7.9796671e-08 -1.9494735e-08]
        [ 0.0000000e+00  0.0000000e+00]
        [ 0.0000000e+00  0.0000000e+00]
        [ 9.0537789e-08  1.0553553e-07]
        [ 3.3567730e-07  1.0368046e-07]]

        [[ 4.7683722e-07 -3.1770034e-07]
        [ 6.5267522e-15 -2.7775875e-15]
        [ 0.0000000e+00  0.0000000e+00]
        [ 0.0000000e+00  0.0000000e+00]
        [-2.1755840e-15 -1.5215135e-15]
        [ 3.6259736e-15 -4.0336615e-15]]

        [[ 0.0000000e+00  0.0000000e+00]
        [ 0.0000000e+00  0.0000000e+00]
        [ 0.0000000e+00  0.0000000e+00]
        [ 0.0000000e+00  0.0000000e+00]
        [ 0.0000000e+00  0.0000000e+00]
        [ 0.0000000e+00  0.0000000e+00]]

        [[ 0.0000000e+00  0.0000000e+00]
        [ 0.0000000e+00  0.0000000e+00]
        [ 0.0000000e+00  0.0000000e+00]
        [ 0.0000000e+00  0.0000000e+00]
        [ 0.0000000e+00  0.0000000e+00]
        [ 0.0000000e+00  0.0000000e+00]]

        [[ 1.1920930e-07 -5.1619136e-08]
        [-3.6259733e-16 -1.0747753e-15]
        [ 0.0000000e+00  0.0000000e+00]
        [ 0.0000000e+00  0.0000000e+00]
        [ 3.6259733e-16 -1.8129867e-16]
        [ 3.6259733e-16 -1.4373726e-15]]

        [[ 5.9604650e-07 -2.5809570e-07]
        [ 8.7023360e-15 -1.9812689e-15]
        [ 0.0000000e+00  0.0000000e+00]
        [ 0.0000000e+00  0.0000000e+00]
        [ 2.9007787e-15  7.2519467e-16]
        [ 8.7023360e-15 -1.7869532e-15]]]
    """
    _check_type(shape, "shape", target_type=tuple)
    _check_type(modes, "modes", target_type=tuple)
    _check_value_in(len(shape), "shape length", 3)
    _check_value_in(len(modes), "modes length", 3)
    _check_param_even(shape, "shape")
    _check_param_no_greater(modes[0], "mode1", shape[0] // 2)
    _check_param_no_greater(modes[1], "mode2", shape[1] // 2)
    _check_param_no_greater(modes[2], "mode3", shape[2] // 2 + 1)
    return _dftn(shape, modes, dim=dim, dtype=dtype)


def idft3(shape, modes, dim=(-3, -2, -1), dtype=ms.float32):
    r"""
    Calculate three-dimensional discrete Fourier transform. Corresponding to the irfftn operator in torch.

    Args:
        shape (tuple): Dimension of the input 'x'.
        modes (tuple): The length of the output transform axis. The `modes` must be no greater than half of the
            dimension of input 'x'.
        dim (tuple): Dimensions to be transformed.
        dtype (ms.dtype): The type of input tensor. Default: ms.float32.

    Inputs:
        - **x** (Tensor, Tensor): The input data. It's 3-D tuple of Tensor. It's a complex, including x real and
          imaginary. Tensor of shape :math:`(*, *)`.

    Returns:
        Complex tensor with the same shape of input x.

    Raises:
        TypeError: If `shape` is not a tuple.
        ValueError: If the length of `shape` is no equal to 3.

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> import mindspore.common.dtype as ms
        >>> from sciai.architecture.neural_operators.dft import idft3
        >>> array = np.ones((2, 2, 2)) * np.arange(1, 3)
        >>> x_re = Tensor(array, dtype=ms.float32)
        >>> x_im = ops.zeros_like(x_re)
        >>> idft3_cell = idft3(shape=(6, 6, 6), modes=(2, 2, 2), dtype=ms.float32)
        >>> ret, _ = idft3_cell((x_re, x_im))
        >>> print(ret)
        [[[ 5.44331074e+00  3.26598644e+00 -1.08866215e+00 -3.26598644e+00 -1.08866215e+00  3.26598644e+00]
        [ 2.04124165e+00  2.04124165e+00  4.08248246e-01 -1.22474492e+00 -1.22474492e+00  4.08248365e-01]
        [-6.80413842e-01 -1.22474492e+00 -6.80413783e-01  4.08248305e-01 9.52579379e-01  4.08248246e-01]
        [ 0.00000000e+00 -2.30921616e-16 -2.30921616e-16  6.53092730e-32 2.30921616e-16  2.30921616e-16]
        [-6.80413842e-01  4.08248246e-01  9.52579379e-01  4.08248305e-01 -6.80413783e-01 -1.22474492e+00]
        [ 2.04124165e+00  4.08248365e-01 -1.22474492e+00 -1.22474492e+00 4.08248246e-01  2.04124165e+00]]
        ......
        [[ 2.04124165e+00  4.08248544e-01 -1.22474492e+00 -1.22474504e+00 4.08248186e-01  2.04124165e+00]
        [ 1.02062082e+00  6.12372518e-01 -2.04124182e-01 -6.12372518e-01 -2.04124182e-01  6.12372518e-01]
        [-5.10310411e-01 -5.10310411e-01 -1.02062061e-01  3.06186229e-01 3.06186229e-01 -1.02062091e-01]
        [-7.21630050e-17 -1.29893429e-16 -7.21630183e-17  4.32978030e-17 1.01028220e-16  4.32978163e-17]
        [-6.08337416e-08  4.08248246e-01  4.08248305e-01  3.65002428e-08 -4.08248246e-01 -4.08248305e-01]
        [ 5.10310471e-01 -3.06186140e-01 -7.14434564e-01 -3.06186318e-01 5.10310352e-01  9.18558717e-01]]]

    """
    _check_type(shape, "shape", target_type=tuple)
    _check_type(modes, "modes", target_type=tuple)
    _check_value_in(len(shape), "shape length", 3)
    _check_value_in(len(modes), "modes length", 3)
    _check_param_even(shape, "shape")
    _check_param_no_greater(modes[0], "mode1", shape[0] // 2)
    _check_param_no_greater(modes[1], "mode2", shape[1] // 2)
    _check_param_no_greater(modes[2], "mode3", shape[2] // 2 + 1)
    return _idftn(shape, modes, dim=dim, dtype=dtype)


def dft2(shape, modes, dim=(-2, -1), dtype=ms.float32):
    """
    Calculate two-dimensional discrete Fourier transform. Corresponding to the rfft2 operator in torch.

    Args:
        shape (tuple): Dimension of the input 'x'.
        modes (tuple): The length of the output transform axis. The `modes` must be no greater than half of the
            dimension of input 'x'.
        dim (tuple): Dimensions to be transformed.
        dtype (:class:`ms.dtype`): The type of input tensor. Default: ms.float32.

    Inputs:
        - **x** (Tensor, Tensor): The input data. It's 2-D tuple of Tensor. It's a complex,
          including x real and imaginary. Tensor of shape :math:`(*, *)`.

    Returns:
        Complex tensor with the same shape of input x.

    Raises:
        TypeError: If `shape` is not a tuple.
        ValueError: If the length of `shape` is no equal to 2.

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> import mindspore.common.dtype as ms
        >>> from sciai.architecture.neural_operators.dft import dft2
        >>> array = np.ones((5, 5)) * np.arange(1, 6)
        >>> x_re = Tensor(array, dtype=ms.float32)
        >>> x_im = x_re
        >>> dft2_cell = dft2(shape=array.shape, modes=(2, 2), dtype=ms.float32)
        >>> ret, _ = dft2_cell((x_re, x_im))
        >>> print(ret)
        [[ 1.5000000e+01 -5.9409552e+00]
         [-2.4656805e-07  7.6130398e-08]
         [ 0.0000000e+00  0.0000000e+00]
         [-1.9992007e-07  7.3572544e-08]
         [-2.4656805e-07  7.6130398e-08]]

    """
    _check_type(shape, "shape", target_type=tuple)
    _check_type(modes, "modes", target_type=tuple)
    _check_value_in(len(shape), "shape length", 2)
    _check_value_in(len(modes), "modes length", 2)
    _check_param_even(shape, "shape")
    _check_param_no_greater(modes[0], "mode1", shape[0] // 2)
    _check_param_no_greater(modes[1], "mode2", shape[1] // 2 + 1)
    return _dftn(shape, modes, dim=dim, dtype=dtype)


def idft2(shape, modes, dim=(-2, -1), dtype=ms.float32):
    """
    Calculate two-dimensional discrete Fourier transform. Corresponding to the irfft2 operator in torch.

    Args:
        shape (tuple): Dimension of the input 'x'.
        modes (tuple): The length of the output transform axis. The `modes` must be no greater than half of the
            dimension of input 'x'.
        dim (tuple): Dimensions to be transformed.
        dtype (:class:`ms.dtype`): The type of input tensor. Default: ms.float32.

    Inputs:
        - **x** (Tensor, Tensor): The input data. It's 2-D tuple of Tensor. It's a complex,
          including x real and imaginary. Tensor of shape :math:`(*, *)`.

    Returns:
        Complex tensor with the same shape of input x.

    Raises:
        TypeError: If `shape` is not a tuple.
        ValueError: If the length of `shape` is no equal to 2.

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> import mindspore.common.dtype as ms
        >>> from sciai.architecture.neural_operators.dft import idft2
        >>> array = np.ones((2, 2)) * np.arange(1, 3)
        >>> x_re = Tensor(array, dtype=ms.float32)
        >>> x_im = ops.zeros_like(x_re)
        >>> idft2_cell = idft2(shape=(5, 5), modes=(2, 2), dtype=ms.float32)
        >>> ret, _ = idft2_cell((x_re, x_im))
        >>> print(ret)
        [[ 3.9999998   1.7888544  -1.7888546  -1.7888546   1.7888544 ]
         [ 0.80901694  0.80901694 -0.08541022 -0.6381966  -0.08541021]
         [-0.30901706 -0.8618034  -0.30901694  0.5854102   0.5854101 ]
         [-0.30901706  0.5854101   0.5854102  -0.30901694 -0.8618034 ]
         [ 0.80901694 -0.08541021 -0.6381966  -0.08541022  0.80901694]]

    """
    _check_type(shape, "shape", target_type=tuple)
    _check_type(modes, "modes", target_type=tuple)
    _check_value_in(len(shape), "shape length", 2)
    _check_value_in(len(modes), "modes length", 2)
    _check_param_even(shape, "shape")
    _check_param_no_greater(modes[0], "mode1", shape[0] // 2)
    _check_param_no_greater(modes[1], "mode2", shape[1] // 2 + 1)
    return _idftn(shape, modes, dim=dim, dtype=dtype)


def dft1(shape, modes, dim=(-1,), dtype=ms.float32):
    """
    Calculate one-dimensional discrete Fourier transform. Corresponding to the rfft operator in torch.

    Args:
       shape (tuple): Dimension of the input 'x'.
       modes (int): The length of the output transform axis. The `modes` must be no greater than half of the
            dimension of input 'x'.
       dim (tuple): Dimensions to be transformed.
       dtype (:class:`ms.dtype`): The type of input tensor.
         Default: ms.float32.

    Inputs:
       - **x** (Tensor, Tensor): The input data. It's 2-D tuple of Tensor. It's a complex,
         including x real and imaginary. Tensor of shape :math:`(*, *)`.

    Returns:
       Complex tensor with the same shape of input x.

    Raises:
       TypeError: If `shape` is not a tuple.
       ValueError: If the length of `shape` is no equal to 1.

    Examples:
       >>> from mindspore import Tensor, ops
       >>> import mindspore.common.dtype as ms
       >>> from sciai.architecture.neural_operators.dft import dft1
       >>> array = [_ for _ in range(5)]
       >>> x_re = Tensor(array, dtype=ms.float32)
       >>> x_im = ops.zeros_like(x_re)
       >>> dft1_cell = dft1(shape=(len(x_re),), modes=2, dtype=ms.float32)
       >>> ret, _ = dft1_cell((x_re, x_im))
       >>> print(ret)
       [ 4.4721355 -1.1180341]

    """
    _check_type(shape, "shape", target_type=tuple)
    _check_type(modes, "modes", target_type=int)
    _check_value_in(len(shape), "shape length", 1)
    _check_param_even(shape, "shape")
    _check_param_no_greater(modes, "mode1", shape[0] // 2 + 1)
    modes = (modes,)
    return _dftn(shape, modes, dim=dim, dtype=dtype)


def idft1(shape, modes, dim=(-1,), dtype=ms.float32):
    """
    Calculate one-dimensional discrete Fourier transform. Corresponding to the irfft operator in torch.

    Args:
        shape (tuple): Dimension of the input 'x'.
        modes (int): The length of the output transform axis. The `modes` must be no greater than half of the
            dimension of input 'x'.
        dim (tuple): Dimensions to be transformed.
        dtype (:class:`ms.dtype`): The type of input tensor. Default: ms.float32.

    Inputs:
        - **x** (Tensor, Tensor): The input data. It's 2-D tuple of Tensor. It's a complex,
          including x real and imaginary. Tensor of shape :math:`(*, *)`.

    Returns:
        Complex tensor with the same shape of input x.

    Raises:
        TypeError: If `shape` is not a tuple.
        ValueError: If the length of `shape` is no equal to 1.

    Examples:
        >>> from mindspore import Tensor, ops
        >>> import mindspore.common.dtype as ms
        >>> from sciai.architecture.neural_operators.dft import idft1
        >>> array = [_ for _ in range(2)]
        >>> x_re = Tensor(array, dtype=ms.float32)
        >>> x_im = x_re
        >>> idft1_cell = idft1(shape=(len(x_re),), modes=2, dtype=ms.float32)
        >>> ret, _ = idft1_cell((x_re, x_im))
        >>> print(ret)
        [ 0.8944272  -0.5742576  -1.2493379  -0.19787574  1.127044  ]

    """
    _check_type(shape, "shape", target_type=tuple)
    _check_type(modes, "modes", target_type=int)
    _check_value_in(len(shape), "shape length", 1)
    _check_param_even(shape, "shape")
    _check_param_no_greater(modes, "mode1", shape[0] // 2 + 1)
    modes = (modes,)
    return _idftn(shape, modes, dim=dim, dtype=dtype)


class SpectralConv1dDft(nn.Cell):
    """1D Fourier layer. It does DFT, linear transform, and Inverse DFT."""

    def __init__(self, in_channels, out_channels, modes1, resolution, dtype=ms.float32):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.resolution = resolution
        self.dtype = dtype
        self.cast = ops.Cast()

        self.scale = (1. / (in_channels * out_channels))
        self.w_re, self.w_im = [_base_param(self.scale, in_channels, out_channels, dtype, modes1) for _ in range(2)]
        self.dft1_cell = dft1(shape=(self.resolution,), modes=modes1, dtype=dtype)
        self.idft1_cell = idft1(shape=(self.resolution,), modes=modes1, dtype=dtype)

    @staticmethod
    def mul1d(inputs, weights):
        weights = weights.expand_dims(0)
        inputs = inputs.expand_dims(2)
        out = inputs * weights
        return out.sum(1)

    def construct(self, x: Tensor):
        """construct"""
        x_re = x
        x_im = ops.zeros_like(x_re)
        x_ft_re, x_ft_im = self.dft1_cell((x_re, x_im))
        w_re = self.cast(self.w_re, self.dtype)
        w_im = self.cast(self.w_im, self.dtype)
        out_ft_re = self.mul1d(x_ft_re[:, :, :self.modes1], w_re) - self.mul1d(x_ft_im[:, :, :self.modes1], w_im)
        out_ft_im = self.mul1d(x_ft_re[:, :, :self.modes1], w_im) + self.mul1d(x_ft_im[:, :, :self.modes1], w_re)

        x, _ = self.idft1_cell((out_ft_re, out_ft_im))
        return x


class SpectralConv2dDft(nn.Cell):
    """2D Fourier layer. It does DFT, linear transform, and Inverse DFT."""

    def __init__(self, in_channels, out_channels, modes1, modes2, column_resolution, raw_resolution, dtype=ms.float16):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.column_resolution = column_resolution
        self.raw_resolution = raw_resolution
        self.dtype = dtype

        self.scale = (1. / (in_channels * out_channels))

        self.w_re1, self.w_im1, self.w_re2, self.w_im2 \
            = [_base_param(self.scale, in_channels, out_channels, dtype, modes1, modes2) for _ in range(4)]

        self.dft2_cell = dft2(shape=(column_resolution, raw_resolution), modes=(modes1, modes2), dtype=dtype)
        self.idft2_cell = idft2(shape=(column_resolution, raw_resolution), modes=(modes1, modes2), dtype=dtype)
        self.mat = ops.zeros((1, out_channels, column_resolution - 2 * modes1, modes2), dtype)
        self.concat = ops.Concat(-2)

    @staticmethod
    def mul2d(inputs, weights):
        weight = weights.expand_dims(0)
        data = inputs.expand_dims(2)
        out = weight * data
        return out.sum(1)

    def construct(self, x: Tensor):
        """construct"""
        x_re = x
        x_im = ops.zeros_like(x_re)
        x_ft_re, x_ft_im = self.dft2_cell((x_re, x_im))

        out_ft_re1 = self.mul2d(x_ft_re[:, :, :self.modes1, :self.modes2], self.w_re1) \
                     - self.mul2d(x_ft_im[:, :, :self.modes1, :self.modes2], self.w_im1)
        out_ft_im1 = self.mul2d(x_ft_re[:, :, :self.modes1, :self.modes2], self.w_im1) \
                     + self.mul2d(x_ft_im[:, :, :self.modes1, :self.modes2], self.w_re1)

        out_ft_re2 = self.mul2d(x_ft_re[:, :, -self.modes1:, :self.modes2], self.w_re2) \
                     - self.mul2d(x_ft_im[:, :, -self.modes1:, :self.modes2], self.w_im2)
        out_ft_im2 = self.mul2d(x_ft_re[:, :, -self.modes1:, :self.modes2], self.w_im2) \
                     + self.mul2d(x_ft_im[:, :, -self.modes1:, :self.modes2], self.w_re2)

        batch_size = x.shape[0]
        mat = self.mat.repeat(batch_size, 0)
        out_re = self.concat((out_ft_re1, mat, out_ft_re2))
        out_im = self.concat((out_ft_im1, mat, out_ft_im2))

        x, _ = self.idft2_cell((out_re, out_im))
        return x


class SpectralConv3d(nn.Cell):
    """3D Fourier layer. It does DFT, linear transform, and Inverse DFT."""

    def __init__(self, in_channels, out_channels, modes1, modes2, modes3,
                 column_resolution, row_resolution, bar_resolution, dtype=ms.float32):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.column_resolution = column_resolution
        self.row_resolution = row_resolution
        self.bar_resolution = bar_resolution
        self.dtype = dtype

        self.scale = (1 / (in_channels * out_channels))

        self.w_re1, self.w_im1, self.w_re2, self.w_im2, self.w_re3, self.w_im3, self.w_re4, self.w_im4 \
            = [_base_param(self.scale, in_channels, out_channels, dtype, modes1, modes2, modes3) for _ in range(8)]

        self.dft3_cell = dft3(shape=(column_resolution, row_resolution, bar_resolution),
                              modes=(modes1, modes2, modes3),
                              dtype=dtype)
        self.idft3_cell = idft3(shape=(column_resolution, row_resolution, bar_resolution),
                                modes=(modes1, modes2, modes3),
                                dtype=dtype)
        self.mat_x = ops.zeros((1, out_channels, column_resolution - 2 * modes1, modes2, modes3), dtype)
        self.mat_y = ops.zeros((1, out_channels, column_resolution, row_resolution - 2 * modes2, modes3), dtype)
        self.concat = ops.Concat(-2)

    def _base_parameter(self):
        return Parameter(Tensor(
            self.scale * np.random.rand(self.in_channels, self.out_channels, self.modes1, self.modes2, self.modes3),
            dtype=self.dtype), requires_grad=True)

    # Complex multiplication
    def mul3d(self, inputs, weights):
        weight = weights.expand_dims(0)
        data = inputs.expand_dims(2)
        out = weight * data
        return out.sum(1)

    def construct(self, x: Tensor):
        """construct"""
        x_re = x
        x_im = ops.zeros_like(x_re)
        x_ft_re, x_ft_im = self.dft3_cell((x_re, x_im))

        out_ft_re1 = self.mul3d(x_ft_re[:, :, :self.modes1, :self.modes2, :self.modes3], self.w_re1) \
                     - self.mul3d(x_ft_im[:, :, :self.modes1, :self.modes2, :self.modes3], self.w_im1)
        out_ft_im1 = self.mul3d(x_ft_re[:, :, :self.modes1, :self.modes2, :self.modes3], self.w_im1) \
                     + self.mul3d(x_ft_im[:, :, :self.modes1, :self.modes2, :self.modes3], self.w_re1)

        out_ft_re2 = self.mul3d(x_ft_re[:, :, -self.modes1:, :self.modes2, :self.modes3], self.w_re2) \
                     - self.mul3d(x_ft_im[:, :, -self.modes1:, :self.modes2, :self.modes3], self.w_im2)
        out_ft_im2 = self.mul3d(x_ft_re[:, :, -self.modes1:, :self.modes2, :self.modes3], self.w_im2) \
                     + self.mul3d(x_ft_im[:, :, -self.modes1:, :self.modes2, :self.modes3], self.w_re2)

        out_ft_re3 = self.mul3d(x_ft_re[:, :, :self.modes1, -self.modes2:, :self.modes3], self.w_re3) \
                     - self.mul3d(x_ft_im[:, :, :self.modes1, -self.modes2:, :self.modes3], self.w_im3)
        out_ft_im3 = self.mul3d(x_ft_re[:, :, :self.modes1, -self.modes2:, :self.modes3], self.w_im3) \
                     + self.mul3d(x_ft_im[:, :, :self.modes1, -self.modes2:, :self.modes3], self.w_re3)

        out_ft_re4 = self.mul3d(x_ft_re[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.w_re4) \
                     - self.mul3d(x_ft_im[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.w_im4)
        out_ft_im4 = self.mul3d(x_ft_re[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.w_im4) \
                     + self.mul3d(x_ft_im[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.w_re4)

        batch_size = x.shape[0]
        mat_x = self.mat_x.repeat(batch_size, 0)
        mat_y = self.mat_y.repeat(batch_size, 0)

        out_re1 = ops.concat((out_ft_re1, mat_x, out_ft_re2), -3)
        out_im1 = ops.concat((out_ft_im1, mat_x, out_ft_im2), -3)

        out_re2 = ops.concat((out_ft_re3, mat_x, out_ft_re4), -3)
        out_im2 = ops.concat((out_ft_im3, mat_x, out_ft_im4), -3)
        out_re = ops.concat((out_re1, mat_y, out_re2), -2)
        out_im = ops.concat((out_im1, mat_y, out_im2), -2)
        x, _ = self.idft3_cell((out_re, out_im))
        return x


def _base_param(scale, in_channels, out_channels, dtype, *modes):
    return Parameter(Tensor(scale * np.random.rand(in_channels, out_channels, *modes), dtype=dtype), requires_grad=True)
