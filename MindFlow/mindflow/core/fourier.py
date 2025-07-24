# Copyright 2025 Huawei Technologies Co., Ltd
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
''' provide complex dft based on the real dft API in mindflow.dft '''
import numpy as np
from scipy.linalg import dft
import mindspore as ms
import mindspore.common.dtype as mstype
from mindspore import nn, ops, Tensor, mint
from mindspore.common.initializer import Zero
from mindspore.ops import operations as P

from ..utils.check_func import check_param_no_greater, check_param_value, check_param_type, check_param_even


class MyRoll(nn.Cell):
    ''' Custom defined roll operator to avoid bug in MindSpore '''
    def __init__(self):
        super().__init__()

        if ms.get_context('device_target') == 'Ascend' and ms.get_context('mode') == ms.GRAPH_MODE:
            self.roller = mint.roll
        else:
            self.roller = None

    def construct(self, x, shifts, dims):
        ''' Same as mint.roll '''
        shifts = np.atleast_1d(shifts).astype(int).tolist()
        dims = np.atleast_1d(dims).astype(int).tolist()

        if self.roller:
            return self.roller(x, shifts, dims)

        for i, j in zip(shifts, dims):
            n = x.shape[j]
            x = ops.swapaxes(x, j, 0)
            x = ops.cat([x[n - i % n:], x[:n - i % n]], axis=0)
            x = ops.swapaxes(x, j, 0)
        return x

class MyFlip(nn.Cell):
    ''' Custom defined flip operator to avoid bug in MindSpore '''
    def __init__(self):
        super().__init__()
        msver = tuple([int(s) for s in ms.__version__.split('.')])

        if msver <= (2, 4, 0) and \
           ms.get_context('device_target') == 'Ascend' and \
           ms.get_context('mode') == ms.PYNATIVE_MODE:
            self.fliper = None
        else:
            self.fliper = mint.flip

    def construct(self, x, dims):
        ''' same as mint.flip '''
        dims = np.atleast_1d(dims).astype(int).tolist()

        if self.fliper:
            return self.fliper(x, dims)

        for j in dims:
            x = ops.swapaxes(x, j, 0)
            x = x[::-1]
            x = ops.swapaxes(x, j, 0)
        return x


def convert_params(shape, modes, dim):
    ''' convert input arguments to suitable format '''
    if dim is None:
        ndim = len(shape)
        dim = tuple([n - ndim for n in range(ndim)])
    else:
        dim = tuple(np.atleast_1d(dim).astype(int).tolist())

    shape = tuple(np.atleast_1d(shape).astype(int).tolist())
    modes = tuple(np.atleast_1d(modes).astype(int).tolist())

    return shape, modes, dim


def check_params(shape, modes, dim):
    ''' check lawfulness of input arguments '''
    check_param_type(dim, "dim", data_type=tuple)
    check_param_type(shape, "shape", data_type=tuple)
    check_param_type(modes, "modes", data_type=tuple)
    check_param_no_greater(len(dim), "dim length", 3)
    check_param_value(len(shape), "shape length", len(dim))
    check_param_value(len(modes), "modes length", len(dim))
    check_param_even(shape, "shape")
    for i, (m, n) in enumerate(zip(modes, shape)):
        check_param_no_greater(m, f'mode{i+1}', n // 2 + (i == len(dim) - 1))


class _DFT1d(nn.Cell):
    '''One dimensional Discrete Fourier Transformation'''

    def __init__(self, n, modes, last_index, idx=0, inv=False, compute_dtype=mstype.float32):
        super().__init__()

        self.n = n
        self.dft_mat = dft(n, scale="sqrtn")
        self.modes = modes
        self.last_index = last_index
        self.inv = inv
        self.idx = idx
        self.compute_dtype = compute_dtype

        self.dft_mode_mat_upper = self.dft_mat[:, :modes]
        self.a_re_upper = Tensor(
            self.dft_mode_mat_upper.real, dtype=compute_dtype)
        self.a_im_upper = Tensor(
            self.dft_mode_mat_upper.imag, dtype=compute_dtype)

        self.dft_mode_mat_lower = self.dft_mat[:, -modes:]
        self.a_re_lower = Tensor(
            self.dft_mode_mat_lower.real, dtype=compute_dtype)
        self.a_im_lower = Tensor(
            self.dft_mode_mat_lower.imag, dtype=compute_dtype)
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
                self.a_re_res = MyFlip()(Tensor(self.dft_mat_res.real, dtype=compute_dtype), dims=-1)
                self.a_im_res = MyFlip()(Tensor(self.dft_mat_res.imag, dtype=compute_dtype), dims=-1)
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
            self.mat = Tensor(shape=(self.n - 2 * self.modes),
                              dtype=compute_dtype, init=Zero())

    def swap_axes(self, x_re, x_im):
        return x_re.swapaxes(-1, self.idx), x_im.swapaxes(-1, self.idx)

    def complex_matmul(self, x_re, x_im, a_re, a_im):
        y_re = ops.matmul(x_re, a_re) - ops.matmul(x_im, a_im)
        y_im = ops.matmul(x_im, a_re) + ops.matmul(x_re, a_im)
        return y_re, y_im

    def zero_mat(self, dims):
        length = len(dims)
        mat = self.mat
        for i in range(length - 1, -1, -1):
            mat = mint.repeat_interleave(mat.expand_dims(0), dims[i], 0)
        return mat

    def compute_forward(self, x_re, x_im):
        ''' Forward transform for rdft '''
        y_re, y_im = self.complex_matmul(
            x_re=x_re, x_im=x_im, a_re=self.a_re_upper, a_im=self.a_im_upper)

        if self.last_index:
            return y_re, y_im

        y_re2, y_im2 = self.complex_matmul(
            x_re=x_re, x_im=x_im, a_re=self.a_re_lower, a_im=self.a_im_lower)

        if self.n == self.modes * 2:
            y_re = self.concat((y_re, y_re2))
            y_im = self.concat((y_im, y_im2))
        else:
            mat = self.zero_mat(x_re.shape[:-1])
            y_re = self.concat((y_re, mat, y_re2))
            y_im = self.concat((y_im, mat, y_im2))

        return y_re, y_im

    def compute_inverse(self, x_re, x_im):
        ''' Inverse transform for irdft '''
        y_re, y_im = self.complex_matmul(x_re=x_re[..., :self.modes],
                                         x_im=x_im[..., :self.modes],
                                         a_re=self.a_re_upper,
                                         a_im=self.a_im_upper)
        if self.last_index:
            y_re_res, y_im_res = self.complex_matmul(x_re=x_re,
                                                     x_im=x_im,
                                                     a_re=self.a_re_res,
                                                     a_im=-self.a_im_res)
        else:
            y_re_res, y_im_res = self.complex_matmul(x_re=x_re[..., -self.modes:],
                                                     x_im=x_im[..., -self.modes:],
                                                     a_re=self.a_re_res,
                                                     a_im=self.a_im_res)
        return y_re + y_re_res, y_im + y_im_res

    def construct(self, x):
        ''' perform 1d rdft/irdft with matmul operations '''
        x_re, x_im = x
        x_re, x_im = P.Cast()(x_re, self.compute_dtype), P.Cast()(x_im, self.compute_dtype)
        x_re, x_im = self.swap_axes(x_re, x_im)
        if self.inv:
            y_re, y_im = self.compute_inverse(x_re, x_im)
        else:
            y_re, y_im = self.compute_forward(x_re, x_im)
        y_re, y_im = self.swap_axes(y_re, y_im)
        return y_re, y_im


class _DFTn(nn.Cell):
    r"""
    N dimensional Discrete Fourier Transformation

    Args:
        shape (tuple): Dimension of the input 'x'.
        modes (tuple): The length of the output transform axis. The `modes` must be no greater than half of the
            dimension of input 'x'.
        dim (tuple): Dimensions to be transformed. Default: None, the leading dimensions will be transformed.
        inv (bool): Whether to compute inverse transformation. Default: False.
        compute_dtype (mindspore.dtype): The type of input tensor. Default: mindspore.float32.

    Inputs:
        - **x** (Tensor, Tensor): The input data. It's 3-D tuple of Tensor. It's a complex,
          including x real and imaginary. Tensor of shape :math:`(*, *)`.

    Returns:
        Complex tensor with the same shape of input x.

    Raises:
        TypeError: If `shape` is not a tuple.
        ValueError: If the length of `shape` is greater than 3.
    """
    def __init__(self, shape, modes, dim=None, inv=False, compute_dtype=mstype.float32):
        super().__init__()

        if dim is None:
            dim = range(len(shape))
        self.dft1_seq = nn.SequentialCell()
        last_index = [False for _ in range(len(shape))]
        last_index[-1] = True
        for dim_id, idx in enumerate(dim):
            self.dft1_seq.append(
                _DFT1d(n=shape[dim_id], modes=modes[dim_id], last_index=last_index[dim_id], idx=idx, inv=inv,
                       compute_dtype=compute_dtype))

    def construct(self, x):
        return self.dft1_seq(x)


class RDFTn(nn.Cell):
    r"""
    1/2/3D discrete real Fourier transformation on real number. The results should be same as
    `scipy.fft.rfftn() <https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.rfftn.html>`_ .

    Args:
        shape (tuple): The shape of the dimensions to be transformed, other dimensions need not be included.
        compute_dtype (mindspore.dtype): The type of input tensor. Default: mindspore.float32.

    Inputs:
        - **ar** (Tensor) - The real tensor to be transformed, with trailing dimensions aligned with `shape`.

    Outputs:
        - **br** (Tensor) - Real part of the output tensor, with trailing dimensions aligned with `shape`,
          except for the last dimension, which should be shape[-1] / 2 + 1.
        - **bi** (Tensor) - Imag part of the output tensor, with trailing dimensions aligned with `shape`,
          except for the last dimension, which should be shape[-1] / 2 + 1.

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> from mindspore import ops
        >>> from mindflow.core import RDFTn
        >>> ar = ops.rand((2, 32, 512))
        >>> dft_cell = RDFTn(x.shape[-2:])
        >>> br, bi = dft_cell(ar)
        >>> print(br.shape)
        (2, 32, 257)
    """
    def __init__(self, shape, compute_dtype=mstype.float32):
        super().__init__()

        n = shape[-1]
        ndim = len(shape)
        modes = tuple([_ // 2 for _ in shape[-ndim:-1]] + [n // 2 + 1]) if ndim > 1 else n // 2 + 1

        shape, modes, dim = convert_params(shape, modes, dim=None)
        check_params(shape, modes, dim)

        self.n = n
        self.ndim = ndim
        self.shape = shape
        self.scale = float(np.prod(shape) ** .5)

        self.dft_cell = _DFTn(shape, modes, dim, inv=False, compute_dtype=compute_dtype)

    def construct(self, ar):
        ''' perform n-dimensional rDFT on real tensor '''
        # n-D Fourier transform with last axis being real-transformed, output dimension (..., m, n//2+1)
        br, bi = self.dft_cell((ar, ar * 0)) # the last ndim dimensions of ar must accord with shape
        return br * self.scale, bi * self.scale


class IRDFTn(nn.Cell):
    r"""
    1/2/3D discrete inverse real Fourier transformation on complex number. The results should be same as
    `scipy.fft.irfftn() <https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.irfftn.html>`_ .

    Args:
        shape (tuple): The shape of the dimensions to be transformed, other dimensions need not be included.
        compute_dtype (mindspore.dtype): The type of input tensor. Default: mindspore.float32.

    Inputs:
        - **ar** (Tensor) - Real part of the tensor to be transformed, with trailing dimensions aligned with `shape`,
          except for the last dimension, which should be shape[-1] / 2 + 1.
        - **ai** (Tensor) - Imag part of the tensor to be transformed, with trailing dimensions aligned with `shape`,
          except for the last dimension, which should be shape[-1] / 2 + 1.

    Outputs:
        - **br** (Tensor) - The output real tensor, with trailing dimensions aligned with `shape`.

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> from mindspore import ops
        >>> from mindflow.core import IRDFTn
        >>> ar = ops.rand((2, 32, 257))
        >>> ai = ops.rand((2, 32, 257))
        >>> dft_cell = IRDFTn(x.shape[-2:])
        >>> br = dft_cell(ar)
        >>> print(br.shape)
        (2, 32, 512)
    """
    def __init__(self, shape, compute_dtype=mstype.float32):
        super().__init__()

        n = shape[-1]
        ndim = len(shape)
        modes = tuple([_ // 2 for _ in shape[-ndim:-1]] + [n // 2 + 1]) if ndim > 1 else n // 2 + 1

        shape, modes, dim = convert_params(shape, modes, dim=None)
        check_params(shape, modes, dim)

        self.n = n
        self.ndim = ndim
        self.shape = shape
        self.scale = float(np.prod(self.shape) ** .5)

        self.idft_cell = _DFTn(shape, modes, dim, inv=True, compute_dtype=compute_dtype)

    def construct(self, ar, ai):
        ''' perform n-dimensional irDFT on complex tensor and output real tensor '''
        br, _ = self.idft_cell((ar, ai))
        return br / self.scale


class DFTn(nn.Cell):
    r"""
    1/2/3D discrete Fourier transformation on complex number. The results should be same as
    `scipy.fft.fftn() <https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.fftn.html#scipy.fft.fftn>`_ .

    Args:
        shape (tuple): The shape of the dimensions to be transformed, other dimensions need not be included.
        compute_dtype (mindspore.dtype): The type of input tensor. Default: mindspore.float32.

    Inputs:
        - **ar** (Tensor) - Real part of the tensor to be transformed, with trailing dimensions aligned with `shape`.
        - **ai** (Tensor) - Imag part of the tensor to be transformed, with trailing dimensions aligned with `shape`.

    Outputs:
        - **br** (Tensor) - Real part of the output tensor, with trailing dimensions aligned with `shape`.
        - **bi** (Tensor) - Imag part of the output tensor, with trailing dimensions aligned with `shape`.

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> from mindspore import ops
        >>> from mindflow.cell import DFTn
        >>> ar = ops.rand((2, 32, 512))
        >>> ai = ops.rand((2, 32, 512))
        >>> dft_cell = DFTn(x.shape[-2:])
        >>> br, bi = dft_cell(ar, ai)
        >>> print(br.shape)
        (2, 32, 512)
    """
    def __init__(self, shape, compute_dtype=mstype.float32):
        super().__init__()

        n = shape[-1]
        ndim = len(shape)
        modes = tuple([_ // 2 for _ in shape[-ndim:-1]] + [n // 2 + 1]) if ndim > 1 else n // 2 + 1

        shape, modes, dim = convert_params(shape, modes, dim=None)
        check_params(shape, modes, dim)

        self.n = n
        self.ndim = ndim
        self.shape = shape
        self.scale = float(np.prod(shape) ** .5)

        self.dft_cell = RDFTn(shape, compute_dtype)

        # use mask to assemble slices of Tensors, avoiding dynamic shape
        mask_x0 = np.ones(self.n//2 + 1)
        mask_xm = np.ones(self.n//2 + 1)
        mask_y0 = np.ones(self.shape)
        mask_z0 = np.ones(self.shape)
        mask_x0[0] = 0
        mask_xm[-1] = 0
        if self.ndim > 1:
            mask_y0[..., 0, :] = 0
        if self.ndim > 2:
            mask_z0[..., 0, :, :] = 0

        self.mask_x0 = Tensor(mask_x0, dtype=compute_dtype, const_arg=True)
        self.mask_xm = Tensor(mask_xm, dtype=compute_dtype, const_arg=True)
        self.mask_y0 = Tensor(mask_y0, dtype=compute_dtype, const_arg=True)
        self.mask_z0 = Tensor(mask_z0, dtype=compute_dtype, const_arg=True)

        self.fliper = MyFlip()
        self.roller = MyRoll()

    def construct(self, ar, ai):
        ''' perform n-dimensional DFT on complex tensor '''
        # n-D complex Fourier transform, output dimension (..., m, n)
        # call dft for real & imag parts separately and then assemble
        brr, bri = self.dft_cell(ar) # ar and ai must have same shape
        bir, bii = self.dft_cell(ai) # the last ndim dimensions of ai must accord with shape

        n = self.n

        br_half1 = ops.pad((brr - bii) * self.mask_xm, [0, n//2 - 1])
        bi_half1 = ops.pad((bri + bir) * self.mask_xm, [0, n//2 - 1])

        br_half2 = ops.pad((brr + bii) * self.mask_x0, [n//2 - 1, 0])
        bi_half2 = ops.pad((bir - bri) * self.mask_x0, [n//2 - 1, 0])
        br_half2 = self.roller(self.fliper(br_half2, dims=-1), n//2, dims=-1)
        bi_half2 = self.roller(self.fliper(bi_half2, dims=-1), n//2, dims=-1)

        if self.ndim > 1:
            br_half2_1 = br_half2 * (1 - self.mask_y0)
            bi_half2_1 = bi_half2 * (1 - self.mask_y0)
            br_half2_2 = br_half2 * self.mask_y0
            bi_half2_2 = bi_half2 * self.mask_y0
            br_half2 = br_half2_1 + self.roller(self.fliper(br_half2_2, dims=-2), 1, dims=-2)
            bi_half2 = bi_half2_1 + self.roller(self.fliper(bi_half2_2, dims=-2), 1, dims=-2)

        if self.ndim > 2:
            br_half2_1 = br_half2 * (1 - self.mask_z0)
            bi_half2_1 = bi_half2 * (1 - self.mask_z0)
            br_half2_2 = br_half2 * self.mask_z0
            bi_half2_2 = bi_half2 * self.mask_z0
            br_half2 = br_half2_1 + self.roller(self.fliper(br_half2_2, dims=-3), 1, dims=-3)
            bi_half2 = bi_half2_1 + self.roller(self.fliper(bi_half2_2, dims=-3), 1, dims=-3)

        br = br_half1 + br_half2
        bi = bi_half1 + bi_half2

        return br, bi


class IDFTn(nn.Cell):
    r"""
    1/2/3D discrete inverse Fourier transformation on complex number. The results should be same as
    `scipy.fft.ifftn() <https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.ifftn.html#scipy.fft.ifftn>`_ .

    Args:
        shape (tuple): The shape of the dimensions to be transformed, other dimensions need not be included.
        compute_dtype (mindspore.dtype): The type of input tensor. Default: mindspore.float32.

    Inputs:
        - **ar** (Tensor) - Real part of the tensor to be transformed, with trailing dimensions aligned with `shape`.
        - **ai** (Tensor) - Imag part of the tensor to be transformed, with trailing dimensions aligned with `shape`.

    Outputs:
        - **br** (Tensor) - Real part of the output tensor, with trailing dimensions aligned with `shape`.
        - **bi** (Tensor) - Imag part of the output tensor, with trailing dimensions aligned with `shape`.

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> from mindspore import ops
        >>> from mindflow.cell import DFTn
        >>> ar = ops.rand((2, 32, 512))
        >>> ai = ops.rand((2, 32, 512))
        >>> dft_cell = DFTn(x.shape[-2:])
        >>> br, bi = dft_cell(ar, ai)
        >>> print(br.shape)
        (2, 32, 512)
    """
    def __init__(self, shape, compute_dtype=mstype.float32):
        super().__init__()
        self.dft_cell = DFTn(shape, compute_dtype)

    def construct(self, ar, ai):
        ''' perform n-dimensional iDFT on complex tensor '''
        scale = self.dft_cell.scale**2
        br, bi = self.dft_cell(ar, -ai)
        return br / scale, -bi / scale


class DCT(nn.Cell):
    r"""
    1D discrete cosine transformation on real number on the last axis. The results should be same as
    `scipy.fft.dct() <https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.dct.html#scipy.fft.dct>`_ .
    Reference: `Type 2 DCT using N FFT (Makhoul) <https://dsp.stackexchange.com/a/10606>`_ .

    Args:
        shape (tuple): The shape of the dimensions to be transformed, other dimensions need not be included.
            Must be a length-1 tuple.
        compute_dtype (mindspore.dtype): The type of input tensor. Default: mindspore.float32.

    Inputs:
        - **a** (Tensor) - The real tensor to be transformed, with trailing dimensions aligned with `shape`.

    Outputs:
        - **b** (Tensor) - The output real tensor, with trailing dimensions aligned with `shape`.

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> from mindspore import ops
        >>> from mindflow.cell import DCT
        >>> a = ops.rand((2, 32, 512))
        >>> dft_cell = DCT(x.shape[-1:])
        >>> b = dft_cell(a)
        >>> print(b.shape)
        (2, 32, 512)
    """
    def __init__(self, shape, compute_dtype=mstype.float32):
        super().__init__()
        self.dft_cell = DFTn(shape, compute_dtype)
        assert self.dft_cell.ndim == 1, 'only support 1D dct'
        n, = self.dft_cell.shape
        w = Tensor(np.arange(n) * np.pi / (2 * n), dtype=compute_dtype)
        self.cosw = ops.cos(w)
        self.sinw = ops.sin(w)

        self.fliper = MyFlip()

    def construct(self, a):
        ''' perform 1-dimensional DCT on real tensor '''
        b_half1 = a[..., ::2]
        b_half2 = self.fliper(a[..., 1::2], dims=-1)
        b = ops.cat([b_half1, b_half2], axis=-1)
        cr, ci = self.dft_cell(b, b * 0)
        return 2 * (cr * self.cosw + ci * self.sinw)


class IDCT(nn.Cell):
    r"""
    1D inverse discrete cosine transformation on real number on the last axis. The results should be same as
    `scipy.fft.dct() <https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.dct.html#scipy.fft.dct>`_ .
    Reference: `A fast cosine transform in one and two dimensions
        <https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1163351>`_ .

    Args:
        shape (tuple): The shape of the dimensions to be transformed, other dimensions need not be included.
            Must be a length-1 tuple.
        compute_dtype (mindspore.dtype): The type of input tensor. Default: mindspore.float32.

    Inputs:
        - **a** (Tensor) - The real tensor to be transformed, with trailing dimensions aligned with `shape`.

    Outputs:
        - **b** (Tensor) - The output real tensor, with trailing dimensions aligned with `shape`.

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> from mindspore import ops
        >>> from mindflow.cell import IDCT
        >>> a = ops.rand((2, 32, 512))
        >>> dft_cell = IDCT(x.shape[-1:])
        >>> b = dft_cell(a)
        >>> print(b.shape)
        (2, 32, 512)
    """
    def __init__(self, shape, compute_dtype=mstype.float32):
        super().__init__()

        self.dft_cell = IRDFTn(shape, compute_dtype)
        assert self.dft_cell.ndim == 1, 'only support 1D dct'
        n, = self.dft_cell.shape
        assert n % 2 == 0, 'only support even length' # n has to be even, or IRDFTn would fail

        w = Tensor(np.arange(0, n // 2 + 1, 1) * np.pi / (2 * n), dtype=compute_dtype)
        self.cosw = ops.cos(w)
        self.sinw = ops.sin(w)

        self.fliper = MyFlip()

    def construct(self, a):
        ''' perform 1-dimensional iDCT on real tensor '''
        n = a.shape[-1]

        br = a[..., :n // 2 + 1]
        bi = ops.pad(self.fliper(- a[..., -(n // 2):], dims=-1), (1, 0))
        vr = (br * self.cosw - bi * self.sinw) / 2
        vi = (bi * self.cosw + br * self.sinw) / 2

        c = self.dft_cell(vr, vi) # (..., n)
        c1 = c[..., :(n + 1) // 2]
        c2 = self.fliper(c[..., (n + 1) // 2:], dims=-1)
        d1 = ops.pad(c1.reshape(-1)[..., None], (0, 1)).reshape(*c1.shape[:-1], -1)
        d2 = ops.pad(c2.reshape(-1)[..., None], (1, 0)).reshape(*c2.shape[:-1], -1)
        return d1 + d2


class DST(nn.Cell):
    r"""
    1D discrete sine transformation on real number on the last axis. The results should be same as
    `scipy.fft.dct() <https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.dct.html#scipy.fft.dct>`_ .
    Reference: `Wikipedia <https://en.wikipedia.org/wiki/Discrete_sine_transform#Computation>`_ .

    Args:
        shape (tuple): The shape of the dimensions to be transformed, other dimensions need not be included.
            Must be a length-1 tuple.
        compute_dtype (mindspore.dtype): The type of input tensor. Default: mindspore.float32.

    Inputs:
        - **a** (Tensor) - The real tensor to be transformed, with trailing dimensions aligned with `shape`.

    Outputs:
        - **b** (Tensor) - The output real tensor, with trailing dimensions aligned with `shape`.

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> from mindspore import ops
        >>> from mindflow.cell import DST
        >>> a = ops.rand((2, 32, 512))
        >>> dft_cell = DST(x.shape[-1:])
        >>> b = dft_cell(a)
        >>> print(b.shape)
        (2, 32, 512)
    """
    def __init__(self, shape, compute_dtype=mstype.float32):
        super().__init__()
        self.dft_cell = DCT(shape, compute_dtype)
        multiplier = np.ones(shape)
        multiplier[..., 1::2] *= -1
        self.multiplier = Tensor(multiplier, dtype=compute_dtype)

    def construct(self, a):
        ''' perform 1-dimensional DST on real tensor '''
        return self.dft_cell.fliper(self.dft_cell(a * self.multiplier), dims=-1)


class IDST(nn.Cell):
    r"""
    1D inverse discrete sine transformation on real number on the last axis. The results should be same as
    `scipy.fft.dct() <https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.dct.html#scipy.fft.dct>`_ .
    Reference: `Wikipedia <https://en.wikipedia.org/wiki/Discrete_sine_transform#Computation>`_ .

    Args:
        shape (tuple): The shape of the dimensions to be transformed, other dimensions need not be included.
            Must be a length-1 tuple.
        compute_dtype (mindspore.dtype): The type of input tensor. Default: mindspore.float32.

    Inputs:
        - **a** (Tensor) - The real tensor to be transformed, with trailing dimensions aligned with `shape`.

    Outputs:
        - **b** (Tensor) - The output real tensor, with trailing dimensions aligned with `shape`.

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> from mindspore import ops
        >>> from mindflow.cell import IDST
        >>> a = ops.rand((2, 32, 512))
        >>> dft_cell = IDST(x.shape[-1:])
        >>> b = dft_cell(a)
        >>> print(b.shape)
        (2, 32, 512)
    """
    def __init__(self, shape, compute_dtype=mstype.float32):
        super().__init__()
        self.dft_cell = IDCT(shape, compute_dtype)
        multiplier = np.ones(shape)
        multiplier[..., 1::2] *= -1
        self.multiplier = Tensor(multiplier, dtype=compute_dtype)

    def construct(self, a):
        ''' perform 1-dimensional iDST on real tensor '''
        return self.dft_cell(self.dft_cell.fliper(a, dims=-1)) * self.multiplier
