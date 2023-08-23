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
"""dft"""
import numpy as np
from scipy.linalg import dft

import mindspore
from mindspore import nn, ops, Tensor
from mindspore.common.initializer import Zero

# MindSpore 2.0 has changed the APIs of _checkparam, the following try except is for compatibility
try:
    from mindspore._checkparam import Validator as validator
except ImportError:
    import mindspore._checkparam as validator


class DFT1d(nn.Cell):
    '''One dimensional Discrete Fourier Transformation'''

    def __init__(self, n, modes, last_index, idx=0, inv=False, compute_dtype=mindspore.float32):
        super().__init__()

        self.n = n
        self.dft_mat = dft(n, scale="sqrtn")
        self.modes = modes
        self.last_index = last_index
        self.inv = inv
        self.idx = idx

        self.dft_mode_mat_upper = self.dft_mat[:, :modes]
        self.a_re_upper = Tensor(self.dft_mode_mat_upper.real, dtype=compute_dtype)
        self.a_im_upper = Tensor(self.dft_mode_mat_upper.imag, dtype=compute_dtype)

        self.dft_mode_mat_lower = self.dft_mat[:, -modes:]
        self.a_re_lower = Tensor(self.dft_mode_mat_lower.real, dtype=compute_dtype)
        self.a_im_lower = Tensor(self.dft_mode_mat_lower.imag, dtype=compute_dtype)
        self.concat = ops.Concat(axis=-1)

        if self.inv:
            self.a_re_upper = self.a_re_upper.T
            self.a_im_upper = -self.a_im_upper.T
            if last_index:
                if modes == n // 2 + 1:
                    self.dft_mat_res = self.dft_mat[:, -modes + 2:]
                else:
                    self.dft_mat_res = self.dft_mat[:, -modes + 1:]

                mat = Tensor(np.zeros(n,), dtype=compute_dtype).reshape(n, 1)
                self.a_re_res = mindspore.numpy.flip(Tensor(self.dft_mat_res.real, dtype=compute_dtype), axis=-1)
                self.a_im_res = mindspore.numpy.flip(Tensor(self.dft_mat_res.imag, dtype=compute_dtype), axis=-1)
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
            self.mat = Tensor(shape=(self.n - 2 * self.modes), dtype=compute_dtype, init=Zero())

    def swap_axes(self, x_re, x_im):
        return x_re.swapaxes(-1, self.idx), x_im.swapaxes(-1, self.idx)

    def complex_matmul(self, x_re, x_im, a_re, a_im):
        y_re = ops.matmul(x_re, a_re) - ops.matmul(x_im, a_im)
        y_im = ops.matmul(x_im, a_re) + ops.matmul(x_re, a_im)
        return y_re, y_im

    def construct(self, x):
        '''construct'''
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

        else:
            x_re, x_im = self.swap_axes(x_re, x_im)
            y_re, y_im = self.complex_matmul(x_re=x_re[..., :self.modes], x_im=x_im[..., :self.modes],
                                             a_re=self.a_re_upper,
                                             a_im=self.a_im_upper)
            y_re, y_im = self.swap_axes(y_re, y_im)

            if self.last_index:
                y_re_res, y_im_res = self.complex_matmul(x_re=x_re, x_im=x_im, a_re=self.a_re_res, a_im=-self.a_im_res)
            else:
                y_re_res, y_im_res = self.complex_matmul(x_re=x_re[..., -self.modes:], x_im=x_im[..., -self.modes:],
                                                         a_re=self.a_re_res, a_im=self.a_im_res)

            y_re_res, y_im_res = self.swap_axes(y_re_res, y_im_res)
            y_re = y_re + y_re_res
            y_im = y_im + y_im_res
        return y_re, y_im


class DFTn(nn.Cell):
    '''N dimensional Discrete Fourier Transformation'''

    def __init__(self, shape, modes, dim=None, inv=False, compute_dtype=mindspore.float32):
        super().__init__()

        if dim is None:
            dim = range(len(shape))
        self.dft1_seq = nn.SequentialCell()
        last_index = [False for _ in range(len(shape))]
        last_index[-1] = True
        for dim_id, idx in enumerate(dim):
            self.dft1_seq.append(
                DFT1d(n=shape[dim_id], modes=modes[dim_id], last_index=last_index[dim_id], idx=idx, inv=inv,
                      compute_dtype=compute_dtype))

    def construct(self, x):
        return self.dft1_seq(x)


def _check_param_even(param, param_name):
    """ Check whether the param is an odd number"""
    for value in param:
        if value % 2 != 0:
            raise ValueError("The value of {} should be an even number, but got {}".format(
                param_name, param))


def _dftn(shape, modes, dim=None, compute_dtype=mindspore.float32):
    dftn_ = DFTn(shape=shape, modes=modes, dim=dim, inv=False, compute_dtype=compute_dtype)
    return dftn_


def _idftn(shape, modes, dim=None, compute_dtype=mindspore.float32):
    idftn_ = DFTn(shape=shape, modes=modes, dim=dim, inv=True, compute_dtype=compute_dtype)
    return idftn_


def dft2(shape, modes, dim=(-2, -1), compute_dtype=mindspore.float32):
    """
    Calculate two-dimensional discrete Fourier transform. Corresponding to the rfft2 operator in torch.

    Args:
        shape (tuple): Dimension of the input 'x'.
        modes (tuple): The length of the output transform axis. The `modes` must be no greater than half of the
            dimension of input 'x'.
        dim (tuple): Dimensions to be transformed.
        compute_dtype (:class:`mindspore.dtype`): The type of input tensor. Default: mindspore.float32.

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
        >>> import mindspore.common.dtype as mstype
        >>> from mindearth.cell.neural_operators.dft import dft2
        >>> array = np.ones((5, 5)) * np.arange(1, 6)
        >>> x_re = Tensor(array, dtype=mstype.float32)
        >>> x_im = x_re
        >>> dft2_cell = dft2(shape=array.shape, modes=(2, 2), compute_dtype=mstype.float32)
        >>> ret, _ = dft2_cell((x_re, x_im))
        >>> print(ret)
        [[ 1.5000000e+01 -5.9409552e+00]
         [-2.4656805e-07  7.6130398e-08]
         [ 0.0000000e+00  0.0000000e+00]
         [-1.9992007e-07  7.3572544e-08]
         [-2.4656805e-07  7.6130398e-08]]

    """
    validator.check_value_type("shape", shape, tuple)
    validator.check_value_type("modes", modes, tuple)
    validator.check("shape_length", len(shape), "expected_shape_length", 2, rel=validator.EQ)
    validator.check("modes_length", len(modes), "nessary_modes_length", 2, rel=validator.EQ)
    _check_param_even(shape, "shape")
    validator.check("mode1", modes[0], "shape0", shape[0] // 2, rel=validator.LE)
    validator.check("mode2", modes[1], "shape1", shape[1] // 2 + 1, rel=validator.LE)
    return _dftn(shape, modes, dim=dim, compute_dtype=compute_dtype)


def idft2(shape, modes, dim=(-2, -1), compute_dtype=mindspore.float32):
    """
    Calculate two-dimensional discrete Fourier transform. Corresponding to the irfft2 operator in torch.

    Args:
        shape (tuple): Dimension of the input 'x'.
        modes (tuple): The length of the output transform axis. The `modes` must be no greater than half of the
            dimension of input 'x'.
        dim (tuple): Dimensions to be transformed.
        compute_dtype (:class:`mindspore.dtype`): The type of input tensor. Default: mindspore.float32.

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
        >>> import mindspore.common.dtype as mstype
        >>> from mindearth.cell.neural_operators.dft import idft2
        >>> array = np.ones((2, 2)) * np.arange(1, 3)
        >>> x_re = Tensor(array, dtype=mstype.float32)
        >>> x_im = ops.zeros_like(x_re)
        >>> idft2_cell = idft2(shape=(5, 5), modes=(2, 2), compute_dtype=mstype.float32)
        >>> ret, _ = idft2_cell((x_re, x_im))
        >>> print(ret)
        [[ 3.9999998   1.7888544  -1.7888546  -1.7888546   1.7888544 ]
         [ 0.80901694  0.80901694 -0.08541022 -0.6381966  -0.08541021]
         [-0.30901706 -0.8618034  -0.30901694  0.5854102   0.5854101 ]
         [-0.30901706  0.5854101   0.5854102  -0.30901694 -0.8618034 ]
         [ 0.80901694 -0.08541021 -0.6381966  -0.08541022  0.80901694]]

    """
    validator.check_value_type("shape", shape, tuple)
    validator.check_value_type("modes", modes, tuple)
    validator.check("shape_length", len(shape), "expected_shape_length", 2, rel=validator.EQ)
    validator.check("modes_length", len(modes), "nessary_modes_length", 2, rel=validator.EQ)
    _check_param_even(shape, "shape")
    validator.check("mode1", modes[0], "shape0", shape[0] // 2, rel=validator.LE)
    validator.check("mode2", modes[1], "shape1", shape[1] // 2 + 1, rel=validator.LE)
    return _idftn(shape, modes, dim=dim, compute_dtype=compute_dtype)


def dft1(shape, modes, dim=(-1,), compute_dtype=mindspore.float32):
    """
    Calculate one-dimensional discrete Fourier transform. Corresponding to the rfft operator in torch.

    Args:
       shape (tuple): Dimension of the input 'x'.
       modes (int): The length of the output transform axis. The `modes` must be no greater than half of the
            dimension of input 'x'.
       dim (tuple): Dimensions to be transformed.
       compute_dtype (:class:`mindspore.dtype`): The type of input tensor.
         Default: mindspore.float32.

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
       >>> import mindspore.common.dtype as mstype
       >>> from mindearth.cell.neural_operators.dft import dft1
       >>> array = [i for i in range(5)]
       >>> x_re = Tensor(array, dtype=mstype.float32)
       >>> x_im = ops.zeros_like(x_re)
       >>> dft1_cell = dft1(shape=(len(x_re),), modes=2, compute_dtype=mstype.float32)
       >>> ret, _ = dft1_cell((x_re, x_im))
       >>> print(ret)
       [ 4.4721355 -1.1180341]

    """
    validator.check_value_type("shape", shape, tuple)
    validator.check_value_type("modes", modes, int)
    validator.check("shape_length", len(shape), "expected_shape_length", 1, rel=validator.EQ)
    _check_param_even(shape, "shape")
    validator.check("mode1", modes, "shape0", shape[0] // 2 + 1, rel=validator.LE)
    modes = (modes,)
    return _dftn(shape, modes, dim=dim, compute_dtype=compute_dtype)


def idft1(shape, modes, dim=(-1,), compute_dtype=mindspore.float32):
    """
    Calculate one-dimensional discrete Fourier transform. Corresponding to the irfft operator in torch.

    Args:
        shape (tuple): Dimension of the input 'x'.
        modes (int): The length of the output transform axis. The `modes` must be no greater than half of the
            dimension of input 'x'.
        dim (tuple): Dimensions to be transformed.
        compute_dtype (:class:`mindspore.dtype`): The type of input tensor. Default: mindspore.float32.

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
        >>> import mindspore.common.dtype as mstype
        >>> from mindearth.cell.neural_operators.dft import idft1
        >>> array = [i for i in range(2)]
        >>> x_re = Tensor(array, dtype=mstype.float32)
        >>> x_im = x_re
        >>> idft1_cell = idft1(shape=(len(x_re),), modes=2, compute_dtype=mstype.float32)
        >>> ret, _ = idft1_cell((x_re, x_im))
        >>> print(ret)
        [ 0.8944272  -0.5742576  -1.2493379  -0.19787574  1.127044  ]

    """
    validator.check_value_type("shape", shape, tuple)
    validator.check_value_type("modes", modes, int)
    validator.check("shape_length", len(shape), "expected_shape_length", 1, rel=validator.EQ)
    _check_param_even(shape, "shape")
    validator.check("mode1", modes, "shape0", shape[0] // 2 + 1, rel=validator.LE)
    modes = (modes,)
    return _idftn(shape, modes, dim=dim, compute_dtype=compute_dtype)
