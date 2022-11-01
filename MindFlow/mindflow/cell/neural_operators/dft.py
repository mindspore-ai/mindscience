''''
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
'''

from scipy.linalg import dft

import mindspore
from mindspore import nn, ops, Tensor

from ...utils.check_func import check_param_no_greater, check_param_value, check_param_type


class DFT1d(nn.Cell):
    '''One dimensional Discrete Fourier Transformation'''
    def __init__(self, n, modes, idx=0, inv=False, compute_dtype=mindspore.float32):
        super().__init__()

        dft_mat = dft(n, scale="sqrtn")
        self.modes = modes
        self.dft_mode_mat = dft_mat[:, :modes]
        self.inv = inv
        self.idx = idx
        self.a_re = Tensor(self.dft_mode_mat.real, dtype=compute_dtype)
        self.a_im = Tensor(self.dft_mode_mat.imag, dtype=compute_dtype)
        if self.inv:
            self.a_re = self.a_re.T
            self.a_im = -self.a_im.T

        self.batch_matmul = ops.BatchMatMul()

    def swap_axes(self, x_re, x_im):
        return x_re.swapaxes(-1, self.idx), x_im.swapaxes(-1, self.idx)

    def complex_matmul(self, x_re, x_im):
        y_re = ops.matmul(x_re, self.a_re) - ops.matmul(x_im, self.a_im)
        y_im = ops.matmul(x_im, self.a_re) + ops.matmul(x_re, self.a_im)
        return y_re, y_im

    def construct(self, x):
        x_re, x_im = x
        x_re, x_im = self.swap_axes(x_re, x_im)
        y_re, y_im = self.complex_matmul(x_re=x_re, x_im=x_im)
        y_re, y_im = self.swap_axes(y_re, y_im)

        return y_re, y_im


class DFTn(nn.Cell):
    '''N dimensional Discrete Fourier Transformation'''
    def __init__(self, shape, modes, dftn_idx=None, inv=False, compute_dtype=mindspore.float32):
        super().__init__()

        if dftn_idx is None:
            dftn_idx = range(len(shape))
        self.dftn_idx = dftn_idx
        self.dft1_seq = nn.SequentialCell()
        for idx in dftn_idx:
            self.dft1_seq.append(DFT1d(n=shape[idx], modes=modes, idx=idx, inv=inv, compute_dtype=compute_dtype))

    def construct(self, x):
        return self.dft1_seq(x)


def _dftn(shape, modes, dftn_idx=None, compute_dtype=mindspore.float32):
    dftn_ = DFTn(shape=shape, modes=modes, dftn_idx=dftn_idx, inv=False, compute_dtype=compute_dtype)
    return dftn_


def _idftn(shape, modes, dftn_idx=None, compute_dtype=mindspore.float32):
    idftn_ = DFTn(shape=shape, modes=modes, dftn_idx=dftn_idx, inv=True, compute_dtype=compute_dtype)
    return idftn_


def dft2(shape, modes, compute_dtype=mindspore.float32):
    """
    Calculate two-dimensional discrete Fourier transform.

    Args:
        shape (tuple): Dimension of the input 'x'
        modes (int): The length of the output transform axis. The `modes` must be no greater than the
            minimum dimension of input 'x'.
        compute_dtype (:class:`mindspore.dtype`): The type of input tensor. Default: mindspore.float32.

    Inputs:
        - **x** (Tensor, Tensor): The input data. It's 2-D tuple of Tensor. It's a complex,
          including x real and imaginary. Tensor of shape :math:`(*, *)`.

    Returns:
        Complex tensor with the same shape of input x.

    Raises:
        TypeError: If `shape` is not a tuple.
        TypeError: If `modes` is not an int.
        ValueError: If the length of `shape` is no equal to 2.
        ValueError: If the minimum dimension of input 'x' less than `modes`.

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> import mindspore.common.dtype as mstype
        >>> from mindflow.cell.neural_operators.dft import dft2
        >>> array = np.ones((5, 5)) * np.arang(1, 6)
        >>> x_re = Tensor(array, dtype=mstype.float32)
        >>> x_im = x_re
        >>> dft2_cell = dft2(shape=array.shape, modes=5, compute_dtype=mstype.float32)
        >>> ret, _ = dft2_cell((x_re, x_im))
        >>> print(ret)
        [[ 1.5000000e+01 -5.9409552e+00 -3.3122988e+00 -1.6877017e+00 9.4095492e-01]
         [-2.4656805e-07  7.6130398e-08 -2.3336057e-08 -7.6680493e-09 1.0148177e-07]
         [-1.9992007e-07  7.3572544e-08  3.4086860e-08  2.7508431e-09 2.2869806e-08]
         [-1.9992007e-07  7.3572544e-08  3.4086860e-08  2.7508431e-09 2.2869806e-08]
         [-2.4656805e-07  7.6130398e-08 -2.3336057e-08 -7.6680493e-09 1.0148177e-07]]

    """
    check_param_type(shape, "shape", data_type=tuple)
    check_param_type(modes, "modes", data_type=int)
    check_param_value(len(shape), "shape length", 2)
    check_param_no_greater(modes, "modes", min(shape))
    return _dftn(shape, modes, dftn_idx=(-2, -1), compute_dtype=compute_dtype)


def idft2(shape, modes, compute_dtype=mindspore.float32):
    """
    Calculate two-dimensional discrete Fourier transform.

    Args:
        shape (tuple): Dimension of the input 'x'
        modes (int): The length of the output transform axis. The `modes` must be no greater than the
            minimum dimension of input 'x'.
        compute_dtype (:class:`mindspore.dtype`): The type of input tensor. Default: mindspore.float32.

    Inputs:
        - **x** (Tensor, Tensor): The input data. It's 2-D tuple of Tensor. It's a complex,
          including x real and imaginary. Tensor of shape :math:`(*, *)`.

    Returns:
        Complex tensor with the same shape of input x.

    Raises:
        TypeError: If `shape` is not a tuple.
        TypeError: If `modes` is not an int.
        ValueError: If the length of `shape` is no equal to 2.
        ValueError: If the minimum dimension of input 'x' less than `modes`.

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> import mindspore.common.dtype as mstype
        >>> from mindflow.cell.neural_operators.dft import idft2
        >>> array = np.ones((5, 5)) * np.arang(1, 6)
        >>> x_re = Tensor(array, dtype=mstype.float32)
        >>> x_im = ops.zeros_like(x_re)
        >>> idft2_cell = idft2(shape=array.shape, modes=5, compute_dtype=mstype.float32)
        >>> ret, _ = idft2_cell((x_re, x_im))
        >>> print(ret)
        [[ 1.5000000e+01  9.4095492e-01 -1.6877017e+00 -3.3122988e+00 -5.9409552e+00]
         [-2.4656805e-07  1.0148177e-07 -7.6680493e-09 -2.3336057e-08 7.6130398e-08]
         [-1.9992007e-07  2.2869806e-08  2.7508431e-09  3.4086860e-08 7.3572544e-08]
         [-1.9992007e-07  2.2869806e-08  2.7508431e-09  3.4086860e-08 7.3572544e-08]
         [-2.4656805e-07  1.0148177e-07 -7.6680493e-09 -2.3336057e-08 7.6130398e-08]]

    """
    check_param_type(shape, "shape", data_type=tuple)
    check_param_type(modes, "modes", data_type=int)
    check_param_value(len(shape), "shape length", 2)
    check_param_no_greater(modes, "modes", min(shape))
    return _idftn(shape, modes, dftn_idx=(-2, -1), compute_dtype=compute_dtype)


def dft1(shape, modes, compute_dtype=mindspore.float32):
    """
    Calculate one-dimensional discrete Fourier transform.

    Args:
        shape (tuple): Dimension of the input 'x'
        modes (int): The length of the output transform axis. The `modes` must be no greater than the
            minimum dimension of input 'x'.
        compute_dtype (:class:`mindspore.dtype`): The type of input tensor.
          Default: mindspore.float32.

    Inputs:
        - **x** (Tensor, Tensor): The input data. It's 2-D tuple of Tensor. It's a complex,
          including x real and imaginary. Tensor of shape :math:`(*, *)`.

    Returns:
        Complex tensor with the same shape of input x.

    Raises:
        TypeError: If `shape` is not a tuple.
        TypeError: If `modes` is not an int.
        ValueError: If the length of `shape` is no equal to 1.
        ValueError: If the minimum dimension of input 'x' less than `modes`.

    Examples:
        >>> from mindspore import Tensor, ops
        >>> import mindspore.common.dtype as mstype
        >>> from mindflow.cell.neural_operators.dft import dft1
        >>> array = [i for i in range(5)]
        >>> x_re = Tensor(array, dtype=mstype.float32)
        >>> x_im = ops.zeros_like(x_re)
        >>> dft1_cell = dft1(shape=(len(x_re),), modes=5, compute_dtype=mstype.float32)
        >>> ret, _ = dft1_cell((x_re, x_im))
        >>> print(ret)
        [ 4.4721355 -1.1180341 -1.1180341 -1.1180341 -1.1180341]

    """
    check_param_type(shape, "shape", data_type=tuple)
    check_param_type(modes, "modes", data_type=int)
    check_param_value(len(shape), "shape length", 1)
    check_param_no_greater(modes, "modes", shape[0])
    return _dftn(shape, modes, dftn_idx=(-1,), compute_dtype=compute_dtype)


def idft1(shape, modes, compute_dtype=mindspore.float32):
    """
    Calculate one-dimensional discrete Fourier transform.

    Args:
        shape (tuple): Dimension of the input 'x'
        modes (int): The length of the output transform axis. The `modes` must be no greater than the
            minimum dimension of input 'x'.
        compute_dtype (:class:`mindspore.dtype`): The type of input tensor. Default: mindspore.float32.

    Inputs:
        - **x** (Tensor, Tensor): The input data. It's 2-D tuple of Tensor. It's a complex,
          including x real and imaginary. Tensor of shape :math:`(*, *)`.

    Returns:
        Complex tensor with the same shape of input x.


    Raises:
        TypeError: If `shape` is not a tuple.
        TypeError: If `modes` is not an int.
        ValueError: If the length of `shape` is no equal to 1.
        ValueError: If the minimum dimension of input 'x' less than `modes`.

    Examples:
        >>> from mindspore import Tensor, ops
        >>> import mindspore.common.dtype as mstype
        >>> from mindflow.cell.neural_operators.dft import idft1
        >>> array = [i for i in range(5)]
        >>> x_re = Tensor(array, dtype=mstype.float32)
        >>> x_im = x_re
        >>> idft1_cell = idft1(shape=(len(x_re),), modes=5, compute_dtype=mstype.float32)
        >>> ret, _ = idft1_cell((x_re, x_im))
        >>> print(ret)
        [ 4.4721355 -2.6568758 -1.4813054 -0.7547629  0.4208076]

    """
    check_param_type(shape, "shape", data_type=tuple)
    check_param_type(modes, "modes", data_type=int)
    check_param_value(len(shape), "shape length", 1)
    check_param_no_greater(modes, "modes", shape[0])
    return _idftn(shape, modes, dftn_idx=(-1,), compute_dtype=compute_dtype)
