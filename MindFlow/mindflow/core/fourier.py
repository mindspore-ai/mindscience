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
import scipy
import mindspore as ms
import mindspore.common.dtype as mstype
from mindspore import nn, ops, Tensor, mint
from mindspore.common.initializer import Zero
from mindspore.ops import operations as P

from ..utils.check_func import check_param_no_greater, check_param_value


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


def convert_shape(shape):
    ''' convert shape to suitable format '''
    if isinstance(shape, int):
        n = shape
    elif len(shape) == 1:
        n, = shape
    else:
        raise TypeError("Only support 1D dct/dst, but got shape {}".format(shape))
    return n


def convert_params(shape, modes, dim):
    ''' convert input arguments to suitable format '''
    shape = tuple(np.atleast_1d(shape).astype(int).tolist())
    ndim = len(shape)

    if dim is None:
        dim = tuple([n - ndim for n in range(ndim)])
    else:
        dim = tuple(np.atleast_1d(dim).astype(int).tolist())

    if modes is None or isinstance(modes, int):
        modes = tuple([modes] * ndim)
    else:
        modes = tuple(np.atleast_1d(modes).astype(int).tolist())

    return shape, modes, dim


def check_params(shape, modes, dim):
    ''' check lawfulness of input arguments '''
    check_param_no_greater(len(dim), "dim length", 3)
    check_param_value(len(shape), "shape length", len(dim))
    check_param_value(len(modes), "modes length", len(dim))
    if np.any(modes):
        for i, (m, n) in enumerate(zip(modes, shape)):
            # if for last axis mode need to be n//2+1, mode should be set to None
            check_param_no_greater(m, f'mode{i+1}', n // 2)


class _DFT1d(nn.Cell):
    '''One dimensional Discrete Fourier Transformation'''

    def __init__(self, n, mode, last_index, idx=0, scale='sqrtn', inv=False, compute_dtype=mstype.float32):
        super().__init__()

        self.n = n
        self.dft_mat = scipy.linalg.dft(n, scale=scale)
        self.last_index = last_index
        self.inv = inv
        self.odd = bool(n % 2)
        self.idx = idx
        self.mode_upper = mode if mode else n // 2 + (self.last_index or self.odd)
        self.mode_lower = mode if mode else n - self.mode_upper
        self.compute_dtype = compute_dtype

        # generate DFT matrix for positive and negative frequencies
        dft_mat_mode = self.dft_mat[:, :self.mode_upper]
        self.a_re_upper = Tensor(dft_mat_mode.real, dtype=compute_dtype)
        self.a_im_upper = Tensor(dft_mat_mode.imag, dtype=compute_dtype)

        dft_mat_mode = self.dft_mat[:, -self.mode_lower:]
        self.a_re_lower = Tensor(dft_mat_mode.real, dtype=compute_dtype)
        self.a_im_lower = Tensor(dft_mat_mode.imag, dtype=compute_dtype)

        # the zero matrix to fill the un-transformed modes
        m = self.n - (self.mode_upper + self.mode_lower)
        if m > 0:
            self.mat = Tensor(shape=m, dtype=compute_dtype, init=Zero())

        self.concat = ops.Concat(axis=-1)
        self.cast = P.Cast()

        if self.inv:
            self.a_re_upper = self.a_re_upper.T
            self.a_im_upper = -self.a_im_upper.T
            self.a_re_lower = self.a_re_lower.T
            self.a_im_lower = -self.a_im_lower.T

            # last axis is real-transformed, so the inverse is conjugate of the positive frequencies
            if last_index:
                mode_res = min(self.mode_lower, self.mode_upper - 1)
                dft_mat_res = self.dft_mat[:, -mode_res:]
                a_re_res = MyFlip()(Tensor(dft_mat_res.real, dtype=compute_dtype), dims=-1)
                a_im_res = MyFlip()(Tensor(dft_mat_res.imag, dtype=compute_dtype), dims=-1)

                a_re_res = ops.pad(a_re_res, (1, self.mode_upper - mode_res - 1))
                a_im_res = ops.pad(a_im_res, (1, self.mode_upper - mode_res - 1))

                self.a_re_upper += a_re_res.T
                self.a_im_upper += a_im_res.T

    def swap_axes(self, x_re, x_im):
        return x_re.swapaxes(-1, self.idx), x_im.swapaxes(-1, self.idx)

    def complex_matmul(self, x_re, x_im, a_re, a_im):
        y_re = ops.matmul(x_re, a_re) - ops.matmul(x_im, a_im)
        y_im = ops.matmul(x_im, a_re) + ops.matmul(x_re, a_im)
        return y_re, y_im

    def zero_mat(self, dims):
        mat = self.mat
        for n in dims[::-1]:
            mat = mint.repeat_interleave(mat.expand_dims(0), n, 0)
        return mat

    def compute_forward(self, x_re, x_im):
        ''' Forward transform for rdft '''
        y_re, y_im = self.complex_matmul(
            x_re=x_re, x_im=x_im, a_re=self.a_re_upper, a_im=self.a_im_upper)

        if self.last_index:
            return y_re, y_im

        y_re2, y_im2 = self.complex_matmul(
            x_re=x_re, x_im=x_im, a_re=self.a_re_lower, a_im=self.a_im_lower)

        if self.n == self.mode_upper + self.mode_lower:
            y_re = self.concat((y_re, y_re2))
            y_im = self.concat((y_im, y_im2))
        else:
            mat = self.zero_mat(x_re.shape[:-1])
            y_re = self.concat((y_re, mat, y_re2))
            y_im = self.concat((y_im, mat, y_im2))

        return y_re, y_im

    def compute_inverse(self, x_re, x_im):
        ''' Inverse transform for irdft '''
        y_re, y_im = self.complex_matmul(x_re=x_re[..., :self.mode_upper],
                                         x_im=x_im[..., :self.mode_upper],
                                         a_re=self.a_re_upper,
                                         a_im=self.a_im_upper)
        if self.last_index:
            return y_re, y_im

        y_re_res, y_im_res = self.complex_matmul(x_re=x_re[..., -self.mode_lower:],
                                                 x_im=x_im[..., -self.mode_lower:],
                                                 a_re=self.a_re_lower,
                                                 a_im=self.a_im_lower)
        return y_re + y_re_res, y_im + y_im_res

    def construct(self, x):
        ''' perform 1d rdft/irdft with matmul operations '''
        x_re, x_im = x
        x_re, x_im = self.cast(x_re, self.compute_dtype), self.cast(x_im, self.compute_dtype)
        x_re, x_im = self.swap_axes(x_re, x_im)
        if self.inv:
            y_re, y_im = self.compute_inverse(x_re, x_im)
        else:
            y_re, y_im = self.compute_forward(x_re, x_im)
        y_re, y_im = self.swap_axes(y_re, y_im)
        return y_re, y_im


class _DFTn(nn.Cell):
    ''' Base class for n-D DFT transform '''
    def __init__(self, shape, dim=None, norm='backward', modes=None, compute_dtype=mstype.float32):
        super().__init__()

        shape, modes, dim = convert_params(shape, modes, dim)
        check_params(shape, modes, dim)

        ndim = len(shape)
        inv, scale, r2c_flags = self.set_options(ndim, norm)
        self.dft1_seq = nn.SequentialCell()
        for n, m, r, d in zip(shape, modes, r2c_flags, dim):
            self.dft1_seq.append(_DFT1d(
                n=n, mode=m, last_index=r, idx=d, scale=scale, inv=inv, compute_dtype=compute_dtype))

    def set_options(self, ndim, norm):
        '''
        Choose the dimensions, normalization, and transformation mode (forward/backward).
        Derivative APIs overwrite the options to achieve their specific goals.
        '''
        inv = False
        scale = {
            'backward': None,
            'forward': 'n',
            'ortho': 'sqrtn',
        }[norm]
        r2c_flags = np.zeros(ndim, dtype=bool).tolist()
        r2c_flags[-1] = True
        return inv, scale, r2c_flags

    def construct(self, *args, **kwargs):
        raise NotImplementedError


class RDFTn(_DFTn):
    r"""
    1/2/3D discrete real Fourier transformation on real number. The results should be same as
    `scipy.fft.rfftn() <https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.rfftn.html>`_ .

    Args:
        shape (tuple): The shape of the dimensions to be transformed, other dimensions need not be included.
        dim (tuple): Dimensions to be transformed. Default: None, the leading dimensions will be transformed.
        norm (str): Normalization mode, should be one of 'forward', 'backward', 'ortho'. Default: 'backward',
            same as torch.fft.rfftn
        modes (tuple, int, None): The length of the output transform axis. The `modes` must be no greater than half of the
            dimension of input 'x'.
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
    def construct(self, ar):
        ''' perform n-dimensional rDFT on real tensor '''
        # n-D Fourier transform with last axis being real-transformed, output dimension (..., m, n//2+1)
        # the last ndim dimensions of ar must accord with shape
        return self.dft1_seq((ar, ar * 0))


class IRDFTn(_DFTn):
    r"""
    1/2/3D discrete inverse real Fourier transformation on complex number. The results should be same as
    `scipy.fft.irfftn() <https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.irfftn.html>`_ .

    Args:
        shape (tuple): The shape of the dimensions to be transformed, other dimensions need not be included.
        dim (tuple): Dimensions to be transformed. Default: None, the leading dimensions will be transformed.
        norm (str): Normalization mode, should be one of 'forward', 'backward', 'ortho'. Default: 'backward',
            same as torch.fft.irfftn
        modes (tuple, int, None): The length of the output transform axis. The `modes` must be no greater than half of the
            dimension of input 'x'.
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
    def set_options(self, ndim, norm):
        inv = True
        scale = {
            'forward': None,
            'backward': 'n',
            'ortho': 'sqrtn',
        }[norm]
        r2c_flags = np.zeros(ndim, dtype=bool).tolist()
        r2c_flags[-1] = True
        return inv, scale, r2c_flags

    def construct(self, ar, ai):
        ''' perform n-dimensional irDFT on complex tensor and output real tensor '''
        return self.dft1_seq((ar, ai))[0]


class DFTn(_DFTn):
    r"""
    1/2/3D discrete Fourier transformation on complex number. The results should be same as
    `scipy.fft.fftn() <https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.fftn.html#scipy.fft.fftn>`_ .

    Args:
        shape (tuple): The shape of the dimensions to be transformed, other dimensions need not be included.
        dim (tuple): Dimensions to be transformed. Default: None, the leading dimensions will be transformed.
        norm (str): Normalization mode, should be one of 'forward', 'backward', 'ortho'. Default: 'backward',
            same as torch.fft.irfftn
        modes (tuple, int, None): The length of the output transform axis. The `modes` must be no greater than half of the
            dimension of input 'x'.
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
    def set_options(self, ndim, norm):
        inv = False
        scale = {
            'forward': 'n',
            'backward': None,
            'ortho': 'sqrtn',
        }[norm]
        r2c_flags = np.zeros(ndim, dtype=bool).tolist()
        return inv, scale, r2c_flags

    def construct(self, ar, ai):
        ''' perform n-dimensional DFT on complex tensor '''
        # n-D complex Fourier transform, output dimension (..., m, n)
        return self.dft1_seq((ar, ai))


class IDFTn(DFTn):
    r"""
    1/2/3D discrete inverse Fourier transformation on complex number. The results should be same as
    `scipy.fft.ifftn() <https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.ifftn.html#scipy.fft.ifftn>`_ .

    Args:
        shape (tuple): The shape of the dimensions to be transformed, other dimensions need not be included.
        dim (tuple): Dimensions to be transformed. Default: None, the leading dimensions will be transformed.
        norm (str): Normalization mode, should be one of 'forward', 'backward', 'ortho'. Default: 'backward',
            same as torch.fft.irfftn
        modes (tuple, int, None): The length of the output transform axis. The `modes` must be no greater than half of the
            dimension of input 'x'.
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
    def set_options(self, ndim, norm):
        inv = True
        scale = {
            'forward': None,
            'backward': 'n',
            'ortho': 'sqrtn',
        }[norm]
        r2c_flags = np.zeros(ndim, dtype=bool).tolist()
        return inv, scale, r2c_flags


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

        n = convert_shape(shape)

        self.dft_cell = DFTn(n, compute_dtype=compute_dtype)

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

        n = convert_shape(shape)

        # assert n % 2 == 0, 'only support even length' # n has to be even, or IRDFTn would fail

        self.dft_cell = IRDFTn(n, compute_dtype=compute_dtype)

        w = Tensor(np.arange(n // 2 + 1) * np.pi / (2 * n), dtype=compute_dtype)
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
        # in case n is odd, d1 and d2 need to be aligned
        d1 = d1[..., :n]
        d2 = ops.pad(d2, (0, n % 2))
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
        n = convert_shape(shape)
        self.dft_cell = DCT(n, compute_dtype=compute_dtype)
        multiplier = np.ones(n)
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
        n = convert_shape(shape)
        self.dft_cell = IDCT(n, compute_dtype=compute_dtype)
        multiplier = np.ones(n)
        multiplier[..., 1::2] *= -1
        self.multiplier = Tensor(multiplier, dtype=compute_dtype)

    def construct(self, a):
        ''' perform 1-dimensional iDST on real tensor '''
        return self.dft_cell(self.dft_cell.fliper(a, dims=-1)) * self.multiplier
