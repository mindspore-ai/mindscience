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
'''Module providing for SKNO block'''
import numpy as np

import mindspore
import mindspore.numpy as mnp
import mindspore.nn.probability.distribution as msd
from mindspore import ops, nn, Tensor, Parameter
from mindspore import dtype as mstype
from mindspore.common.initializer import initializer, Normal
from mindearth.cell.neural_operators.dft import dft1, idft1

from .callback import divide_function


def legendre_gauss_weights(length, lower_bound=-1.0, upper_bound=1.0):
    r"""
    Helper routine which returns the Legendre-Gauss nodes and weights
    on the interval [lower_bound, b]

    Args:
        length (int): The length of the Legendre-Gauss nodes and weights.
        lower_bound (float): The lower bound of the value interval. Default: -1.0.
        upper_bound (float): The upper bound of the value interval. Default: 1.0.

    Returns:
        Tensors, `xlg` is the cost of the Legendre-Gauss, `wlg` is the weight of the Legendre-Gauss.
    """
    xlg, wlg = np.polynomial.legendre.leggauss(length)
    xlg = (upper_bound - lower_bound) * 0.5 * xlg + (upper_bound + lower_bound) * 0.5
    wlg = wlg * (upper_bound - lower_bound) * 0.5
    return xlg, wlg


def lobatto_weights(length, lower_bound=-1.0, upper_bound=1.0, tol=1e-16, maxiter=100):
    r"""
    Helper routine which returns the Legendre-Gauss-Lobatto nodes and weights
    on the interval [lower_bound, b]

    Args:
        length (int): The length of the Legendre-Gauss nodes and weights.
        lower_bound (float): The lower bound of the value interval. Default: -1.0.
        upper_bound (float): The upper bound of the value interval. Default: 1.0.
        tol (float): Maximum difference that can be tolerated. Default: 1e-16.
        maxiter (int): Number of iterations.

    Returns:
        Tensors, `xlg` is the cost of the Legendre-Gauss, `wlg` is the weight of the Legendre-Gauss.
    """
    wlg = np.zeros((length,))
    tlg = np.zeros((length,))
    tmp = np.zeros((length,))
    # Vandermonde Matrix
    vdm = np.zeros((length, length))
    # initialize Chebyshev nodes as first guess
    for i in range(length):
        tlg[i] = -np.cos(divide_function(np.pi * i, length - 1))
    tmp = 2.0
    for _ in range(maxiter):
        tmp = tlg
        vdm[:, 0] = 1.0
        vdm[:, 1] = tlg
        for k in range(2, length):
            vdm[:, k] = divide_function((2 * k - 1) * tlg * vdm[:, k - 1] - (k - 1) * vdm[:, k - 2], k)
        tlg = tmp - divide_function(tlg * vdm[:, length - 1] - vdm[:, length - 2], length * vdm[:, length - 1])
        if max(abs(tlg - tmp).flatten()) < tol:
            break
    wlg = divide_function(2.0, (length * (length - 1)) * (vdm[:, length - 1] ** 2))
    # rescale
    tlg = (upper_bound - lower_bound) * 0.5 * tlg + (upper_bound + lower_bound) * 0.5
    wlg = wlg * (upper_bound - lower_bound) * 0.5
    return tlg, wlg


def clenshaw_curtiss_weights(length, lower_bound=-1.0, upper_bound=1.0):
    r"""
    Computation of the Clenshaw-Curtis quadrature nodes and weights.
    This implementation follows

    [1] Joerg Waldvogel, Fast Construction of the Fejer and Clenshaw-Curtis Quadrature Rules; BIT Numerical Mathematics,
    Vol. 43, No. 1, pp. 001–018.

    Args:
        length (int): The length of the Legendre-Gauss nodes and weights.
        lower_bound (float): The lower bound of the value interval. Default: -1.0.
        upper_bound (float): The upper bound of the value interval. Default: 1.0.

    Returns:
        Tensors, `xlg` is the cost of the Legendre-Gauss, `wlg` is the weight of the Legendre-Gauss.
    """
    tcc = np.cos(np.linspace(np.pi, 0, length))

    if length == 2:
        wcc = np.array([1., 1.])
    else:
        length1 = length - 1
        n = np.arange(1, length1, 2)
        l = len(n)
        m = length1 - l
        v = np.concatenate([divide_function(divide_function(2, n), n - 2), divide_function(1, n[-1:]), np.zeros(m)])
        v = - v[:-1] - v[-1:0:-1]
        g0 = -np.ones(length1)
        g0[l] = g0[l] + length1
        g0[m] = g0[m] + length1
        g = divide_function(g0, length1 ** 2 - 1 + (length1 % 2))
        wcc = np.fft.ifft(v + g).real
        wcc = np.concatenate((wcc, wcc[:1]))
    # rescale
    tcc = (upper_bound - lower_bound) * 0.5 * tcc + (upper_bound + lower_bound) * 0.5
    wcc = wcc * (upper_bound - lower_bound) * 0.5
    return tcc, wcc


def precompute_legpoly(mmax, lmax, tq, norm="ortho", inverse=False, csphase=True):
    r"""
    Computes the values of (-1)^m c^l_m P^l_m(\cos \theta) at the positions specified by x (theta)
    The resulting tensor has shape (mmax, lmax, len(x)).
    The Condon-Shortley Phase (-1)^m can be turned off optionally.

    Args:
        lmax (int): The max size of latitude.
        mmax (int): The max size of longitude.
        tq (float): The third vector of the spherical harmonic function.
        norm (str): The type of normalization in legpoly precomputation. Default: "ortho".
        inverse (bool): If inverse SHT is applied. Default: False.
        csphase (bool, optional): If changing the phase in legpoly precomputation. Default: True.

    Returns:
        Tensors, the Legendre weight matrices, which is pre-computed and stored.
    """
    nmax = max(mmax, lmax)
    pct = np.zeros((nmax, nmax, len(tq)), dtype=np.float64)

    cost = np.cos(tq)

    norm_factor = 1. if norm == "ortho" else np.sqrt(4 * np.pi)
    norm_factor = divide_function(1., norm_factor) if inverse else norm_factor

    pct[0, 0, :] = divide_function(norm_factor, np.sqrt(4 * np.pi))

    for l in range(1, nmax):
        pct[l - 1, l, :] = np.sqrt(2 * l + 1) * cost * pct[l - 1, l - 1, :]
        pct[l, l, :] = np.sqrt((2 * l + 1) * (1 + cost) \
                               * divide_function(divide_function(1 - cost, 2), l)) * pct[l - 1, l - 1, :]

    for l in range(2, nmax):
        for m in range(0, l - 1):
            pct[m, l, :] = cost * np.sqrt(divide_function(2 * l - 1, l - m) * divide_function(2 * l + 1, l + m)) \
                           * pct[m, l - 1, :] - np.sqrt(divide_function(l + m - 1, l - m) \
                           * divide_function(2 * l + 1, 2 * l - 3) \
                           * divide_function(l - m - 1, l + m)) * pct[m, l - 2, :]

    if norm == "schmidt":
        for l in range(0, nmax):
            if inverse:
                pct[:, l, :] = pct[:, l, :] * np.sqrt(2 * l + 1)
            else:
                pct[:, l, :] = divide_function(pct[:, l, :], np.sqrt(2 * l + 1))

    pct = pct[:mmax, :lmax]

    if csphase:
        for m in range(1, mmax, 2):
            pct[m] *= -1

    return mindspore.Tensor(pct, dtype=mindspore.float32)


class RealSHT(nn.Cell):
    r"""
    The forward transformation of SHT. The input features are decomposed by spherical harmonic transformation.

    Args:
        nlat (int): The latitude size of the input feature.
        nlon (int): The longitude size of the input feature.
        lmax (int): The max size of latitude.
        mmax (int): The max size of longitude.
        grid (str): The type of grid. Default: "lobatto".
        norm (str): The type of normalization in legpoly precomputation. Default: "ortho".
        csphase (bool, optional): If changing the phase in legpoly precomputation. Default: True.

    Inputs:
        - **input** (Tensor) - Tensor of shape :math:`(batch\_size, latent\_dims, nlat, nlon)`.

    Outputs:
        - **output[0]** (Tensor) - Tensor of shape :math:`(batch\_size, latent\_dims, lmax, mmax)`.
        - **output[1]** (Tensor) - Tensor of shape :math:`(batch\_size, latent\_dims, lmax, mmax)`.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore
        >>> from mindspore import Tensor
        >>> inputs = Tensor(np.random.randn(1,768,128,256), dtype=mindspore.float32)
        >>> net = RealSHT(nlat=128, nlon=256, lmax=128, mmax=129)
        >>> output = net(inputs)
        >>> print(output[0].shape, output[1].shape)
        (1,768,128,129), (1,768,128,129)

    """

    def __init__(self, nlat, nlon, lmax=None, mmax=None, grid="lobatto", norm="ortho", csphase=True):
        super(RealSHT, self).__init__()
        self.nlat = nlat
        self.nlon = nlon
        self.grid = grid
        self.norm = norm
        self.csphase = csphase

        if self.grid == "legendre-gauss":
            cost, w = legendre_gauss_weights(nlat, -1, 1)
            self.lmax = lmax or self.nlat
        elif self.grid == "lobatto":
            cost, w = lobatto_weights(nlat, -1, 1)
            self.lmax = lmax or self.nlat - 1
        elif self.grid == "equiangular":
            cost, w = clenshaw_curtiss_weights(nlat, -1, 1)
            self.lmax = lmax or self.nlat
        else:
            raise ValueError("Unknown quadrature mode")

        self.tq = np.flip(np.arccos(cost))

        self.mmax = mmax or self.nlon // 2 + 1

        self.w = Tensor(w, dtype=mindspore.float32)[None, None, :]
        self.pct = precompute_legpoly(self.nlon // 2 + 1, self.lmax, self.tq, norm=self.norm, csphase=self.csphase)
        self.weight = self.pct * self.w
        self.weight = self.weight.transpose((2, 1, 0))

        # fft
        self.dft1_cell = dft1(shape=(self.nlon,), modes=self.nlon // 2 + 1, dim=(-1,))

        self.weights = Parameter(self.weight[None, None, :], name="weights")

    def construct(self, x: Tensor):
        '''RealSHT construct function

        Args:
            x (Tensor): Input Tensor.
        '''
        x_re = x
        x_im = ops.zeros_like(x_re)
        x_re, x_im = self.dft1_cell((x_re, x_im))
        x_re, x_im = 2.0 * mnp.pi * x_re, 2.0 * mnp.pi * x_im
        x_re = x_re[:, :, :, None,]
        x_re_out = (x_re * self.weights).sum(axis=2)
        x_im = x_im[:, :, :, None,]
        x_im_out = (x_im * self.weights).sum(axis=2)
        return x_re_out, x_im_out


class InverseRealSHT(nn.Cell):
    r"""
    The forward transformation of SHT. The input features are decomposed by spherical harmonic transformation.

    Args:
        nlat (int): The latitude size of the input feature.
        nlon (int): The longitude size of the input feature.
        lmax (int): The max size of latitude.
        mmax (int): The max size of longitude.
        grid (str): The type of grid. Default: "lobatto".
        norm (str): The type of normalization in legpoly precomputation. Default: "ortho".
        csphase (bool, optional): If changing the phase in legpoly precomputation. Default: True.

    Inputs:
        - **input** Tuple(Tensor, Tensor) - Two Tensors have the same shape :math:
          `(batch\_size, latent\_dims, lmax, mmax)`.

    Outputs:
        - **output** (Tensor) - Tensor of shape :math:`(batch\_size, latent\_dims, nlat, nlon)`.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore
        >>> from mindspore import Tensor
        >>> input_re = Tensor(np.random.randn(1,768,128,129), dtype=mindspore.float32)
        >>> input_im = Tensor(np.random.randn(1,768,128,129), dtype=mindspore.float32)
        >>> net = InverseRealSHT(nlat=128, nlon=256, lmax=128, mmax=129)
        >>> output = net((input_re, input_im))
        >>> print(output.shape)
        (1,768,128,256)

    """

    def __init__(self, nlat, nlon, lmax=None, mmax=None, grid="lobatto", norm="ortho", csphase=True):
        super(InverseRealSHT, self).__init__()
        self.nlat = nlat
        self.nlon = nlon
        self.grid = grid
        self.norm = norm
        self.csphase = csphase

        if self.grid == "legendre-gauss":
            cost, _ = legendre_gauss_weights(nlat, -1, 1)
            self.lmax = lmax or self.nlat
        elif self.grid == "lobatto":
            cost, _ = lobatto_weights(nlat, -1, 1)
            self.lmax = lmax or self.nlat - 1
        elif self.grid == "equiangular":
            cost, _ = clenshaw_curtiss_weights(nlat, -1, 1)
            self.lmax = lmax or self.nlat
        else:
            raise ValueError("Unknown quadrature mode")

        t = np.flip(np.arccos(cost))

        self.mmax = mmax or self.nlon // 2 + 1

        pct = precompute_legpoly(self.mmax, self.lmax, t, norm=self.norm, inverse=True, csphase=self.csphase)

        self.idft1_cell = idft1(shape=(self.nlon,), modes=self.nlon // 2 + 1, dim=(-1,))

        self.pct = Tensor(pct, dtype=mindspore.float32)
        self.pct = self.pct.transpose((2, 1, 0))
        self.weights = Parameter(self.pct[None, None, :], name="weights")

    def extra_repr(self):
        return f'nlat={self.nlat}, nlon={self.nlon},\n lmax={self.lmax}, \
                 mmax={self.mmax},\n grid={self.grid}, csphase={self.csphase}'

    def construct(self, x: Tensor):
        '''InverseRealSHT construct function

        Args:
            x (Tensor): Input Tensor.
        '''
        x_re, x_im = x
        x_re = x_re[:, :, None,]
        x_re_out = (x_re * self.weights).sum(axis=3)
        x_im = x_im[:, :, None,]
        x_im_out = (x_im * self.weights).sum(axis=3)
        x, _ = self.idft1_cell((x_re_out, x_im_out))
        return x


class DropPath(nn.Cell):
    r"""
    The Dropout operation during training. DropPath prevents co-adaptation of parallel paths in networks
        by randomly dropping operands of the join layers.

    Args:
        drop_prob (float, optional): The rate of drop path layer. Default: 0.0.
            scale_by_keep (bool, optional): If keeping the feature size after dropping out. Default: True.

    Inputs:
        - **input** (Tensor) - Tensor of shape :math:`(batch\_size, h\_size*w\_size, latent\_dims)`.

    Outputs:
        - **output** (Tensor) - Tensor of shape :math:`(batch\_size, h\_size*w\_size, latent\_dims)`.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore
        >>> from mindspore import Tensor
        >>> inputs = Tensor(np.random.randn(1,16*32,768), dtype=mindspore.float32)
        >>> net = DropPath(drop_prob=0.0, scale_by_keep=True)
        >>> output = net(inputs)
        >>> print(output.shape)
        (2, 512, 768)

    """

    def __init__(self, drop_prob=0.0, scale_by_keep=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.keep_prob = 1.0 - self.drop_prob
        self.scale_by_keep = scale_by_keep
        self.bernoulli = msd.Bernoulli(probs=self.keep_prob)
        self.div = ops.Div()

    def construct(self, x: Tensor):
        '''DropPath construct function

        Args:
            x (Tensor): Input Tensor.
        '''
        if self.drop_prob > 0.0:
            random_tensor = self.bernoulli.sample((x.shape[0],) + (1,) * (x.ndim - 1))
            if self.keep_prob > 0.0 and self.scale_by_keep:
                random_tensor = self.div(random_tensor, self.keep_prob)
            x = x * random_tensor
        return x


class SKNO1D(nn.Cell):
    r"""
    The 1D SKNO Operaotr. Decomposing the input feature with the SHT operator.
        Enhancing the decomposed feature with Linear Layers.
        Restoring the decomposed feature to the original feature with the iSHT operator.

    Args:
        h_size (int): The height (latitude) size of the input feature.
        w_size (int): The width (longitude) size of the input feature.
        latent_dims (int): The number of input layer channel.
        num_blocks: (int, optional): The number of blocks. Default: 16.
        high_freq (bool, optional): If high-frequency information complement is applied. Default: True.
        compute_dtype (:class:`mindspore.dtype`): The type of input tensor. Default: mindspore.float32.

    Inputs:
        - **input** (Tensor) - Tensor of shape :math:`(batch\_size, h\_size*w\_size, latent\_dims)`.

    Outputs:
        - **output** (Tensor) - Tensor of shape :math:`(batch\_size, h\_size*w\_size, latent\_dims)`.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore
        >>> from mindspore import Tensor
        >>> inputs = Tensor(np.random.randn(1,16*32,768), dtype=mindspore.float32)
        >>> net = SKNO1D(h_size=16, w_size=32, latent_dims=768, num_blocks=16, high_freq=True)
        >>> output = net(inputs)
        >>> print(output.shape)
        (2, 512, 768)

    """

    def __init__(self,
                 h_size=32,
                 w_size=64,
                 latent_dims=768,
                 num_blocks=16,
                 high_freq=False,
                 compute_dtype=mstype.float32):
        super().__init__()

        self.compute_type = compute_dtype

        self.fno_seq = nn.SequentialCell()
        self.concat = ops.Concat(axis=-1)
        self.act = ops.GeLU()

        self.h_size = h_size
        self.w_size = w_size

        self.sht_cell = RealSHT(nlat=h_size, nlon=w_size, lmax=h_size, mmax=w_size // 2 + 1)
        self.isht_cell = InverseRealSHT(nlat=h_size, nlon=w_size, lmax=h_size, mmax=w_size // 2 + 1)

        self.scale = 0.02
        self.num_blocks = num_blocks
        self.block_size = latent_dims // self.num_blocks
        self.hidden_size_factor = 1
        w1 = self.scale * Tensor(np.random.randn(
            2, self.num_blocks, self.block_size, self.block_size * self.hidden_size_factor), dtype=compute_dtype)
        b1 = self.scale * Tensor(np.random.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor),
                                 dtype=compute_dtype)
        w2 = self.scale * Tensor(np.random.randn(
            2, self.num_blocks, self.block_size * self.hidden_size_factor, self.block_size), dtype=compute_dtype)
        b2 = self.scale * Tensor(np.random.randn(2, self.num_blocks, self.block_size), dtype=compute_dtype)

        self.w1 = Parameter(w1, requires_grad=True)
        self.b1 = Parameter(b1, requires_grad=True)
        self.w2 = Parameter(w2, requires_grad=True)
        self.b2 = Parameter(b2, requires_grad=True)

        self.relu = ops.ReLU()

        self.sparsity_threshold = 0.01
        self.hard_thresholding_fraction = 1.0

        self.high_freq = high_freq
        self.w = nn.Conv2d(latent_dims, latent_dims, 1)  # High Frequency

        self.cast = ops.Cast()

    @staticmethod
    def mul2d(inputs, weights):
        weight = weights.expand_dims(0)
        data = inputs.expand_dims(5)
        out = weight * data
        return out.sum(4)

    def construct(self, x: Tensor):
        '''SKNO1D construct function

        Args:
            x (Tensor): Input Tensor.
        '''
        if self.high_freq:
            b, n, c = x.shape
            h, w = self.h_size, self.w_size
            x = x.reshape(b, h, w, c).transpose((0, 3, 1, 2))
            bias = self.w(x).transpose((0, 2, 3, 1))
            bias = bias.reshape(b, h * w, c)
        else:
            bias = x
            b, n, c = x.shape
            h, w = self.h_size, self.w_size
            x = x.reshape(b, h, w, c).transpose((0, 3, 1, 2))

        x_ft_re, x_ft_im = self.sht_cell(x)
        x_ft_re = x_ft_re.transpose((0, 2, 3, 1))
        x_ft_im = x_ft_im.transpose((0, 2, 3, 1))

        x_ft_re = x_ft_re.reshape(b, x_ft_re.shape[1], x_ft_re.shape[2], self.num_blocks, self.block_size)
        x_ft_im = x_ft_im.reshape(b, x_ft_im.shape[1], x_ft_im.shape[2], self.num_blocks, self.block_size)

        kept_modes = h // 2 + 1

        o1_real = self.relu(self.mul2d(x_ft_re, self.w1[0]) - self.mul2d(x_ft_im, self.w1[1]) + self.b1[0])
        o1_real[:, :, kept_modes:] = 0.0

        o1_imag = self.relu(self.mul2d(x_ft_im, self.w1[0]) + self.mul2d(x_ft_re, self.w1[1]) + self.b1[1])
        o1_imag[:, :, kept_modes:] = 0.0

        o2_real = (self.mul2d(o1_real, self.w2[0]) - self.mul2d(o1_imag, self.w2[1]) + self.b2[0])
        o2_real[:, :, kept_modes:] = 0.0

        o2_imag = (self.mul2d(o1_imag, self.w2[0]) + self.mul2d(o1_real, self.w2[1]) + self.b2[1])
        o2_imag[:, :, kept_modes:] = 0.0

        o2_real = self.cast(o2_real, self.compute_type)
        o2_imag = self.cast(o2_imag, self.compute_type)

        o2_real = ops.softshrink(o2_real, lambd=self.sparsity_threshold)
        o2_imag = ops.softshrink(o2_imag, lambd=self.sparsity_threshold)

        o2_real = o2_real.reshape(b, o2_real.shape[1], o2_real.shape[2], c)
        o2_imag = o2_imag.reshape(b, o2_imag.shape[1], o2_imag.shape[2], c)

        o2_real, o2_imag = o2_real.transpose((0, 3, 1, 2)), o2_imag.transpose((0, 3, 1, 2))
        x = self.isht_cell((o2_real, o2_imag))

        x = x.transpose((0, 2, 3, 1)).reshape(b, n, c)
        return x + bias


class MLPNet(nn.Cell):
    r"""
    The MLPNet Network. Applies a series of fully connected layers to the incoming data among which hidden layers have
        mlp_ratio times number of dims.

    Args:
        latent_dims (int): the number of input layer channel.
        mlp_ratio (int): The rate of mlp layer.
        dropout_rate (float, optional): The rate of dropout layer. Default: 1.0.
        compute_dtype (:class:`mindspore.dtype`): The type of input tensor. Default: mindspore.float32.

    Inputs:
        - **input** (Tensor) - Tensor of shape :math:`(batch\_size, h\_size*w\_size, latent\_dims)`.

    Outputs:
        - **output** (Tensor) - Tensor of shape :math:`(batch\_size, h\_size*w\_size, latent\_dims)`.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore
        >>> from mindspore import Tensor
        >>> inputs = Tensor(np.random.randn(2,16*32,768), dtype=mindspore.float32)
        >>> net = MLPNet(latent_dims=768, mlp_ratio=4, latent_dims=32)
        >>> output = net(inputs)
        >>> print(output.shape)
        (2, 512, 768)

    """

    def __init__(self,
                 latent_dims,
                 mlp_ratio,
                 dropout_rate=1.0,
                 compute_dtype=mstype.float32):
        super(MLPNet, self).__init__()
        self.fc1 = nn.Dense(latent_dims, latent_dims * mlp_ratio,
                            weight_init=initializer(Normal(sigma=0.02), shape=(latent_dims * mlp_ratio, latent_dims)),
                            ).to_float(compute_dtype)
        self.fc2 = nn.Dense(latent_dims * mlp_ratio, latent_dims,
                            weight_init=initializer(Normal(sigma=0.02), shape=(latent_dims, latent_dims * mlp_ratio)),
                            ).to_float(compute_dtype)

        self.act_fn = nn.GELU()
        self.dropout = nn.Dropout(keep_prob=dropout_rate)

    def construct(self, x: Tensor):
        '''MLPNet construct function

        Args:
            x (Tensor): Input Tensor.
        '''
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)

        return x


class PatchEmbed(nn.Cell):
    r"""
    The Patch Embedding Operation. Encoding the input feature to the patch embedding with the convolutional layer, where
        the stride is equal to the patch size.

    Args:
        in_channels (int): the number of input layer channel.
        latent_dims (int): the number of dims of hidden layers.
        patch_size (int, optional): The patch size of image. Default: 8.
        compute_dtype (:class:`mindspore.dtype`): The type of input tensor. Default: mindspore.float32.

    Inputs:
        - **input** (Tensor) - Tensor of shape :math:`(batch\_size, in\_channels, h\_size, w\_size)`.

    Outputs:
        - **output** (Tensor) - Tensor of shape :math:`(batch\_size, h\_size*w\_size, latent\_dims)`.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore
        >>> from mindspore import Tensor
        >>> inputs = Tensor(np.random.randn(2,69,128,256), dtype=mindspore.float32)
        >>> net = PatchEmbed(in_channels=69, latent_dims=768, patch_size=8)
        >>> output = net(inputs)
        >>> print(output.shape)
        (2, 512, 768)

    """

    def __init__(self,
                 in_channels,
                 latent_dims,
                 patch_size=8,
                 compute_dtype=mstype.float32):
        super(PatchEmbed, self).__init__()
        self.compute_dtype = compute_dtype
        self.proj = nn.Conv2d(in_channels=in_channels,
                              out_channels=latent_dims,
                              kernel_size=patch_size,
                              stride=patch_size,
                              has_bias=True,
                              bias_init='normal'
                              ).to_float(compute_dtype)

    def construct(self, x: Tensor):
        '''PatchEmbed construct function

        Args:
            x (Tensor): Input Tensor.
        '''
        x = self.proj(x)
        x = ops.Reshape()(x, (x.shape[0], x.shape[1], x.shape[2] * x.shape[3]))
        x = ops.Transpose()(x, (0, 2, 1))
        return x


class SKNOBlock(nn.Cell):
    r"""
    The block of the SKNO metwork. During training, the SKNOBlock services as the main part of SKNO algorithm.

    Args:
        latent_dims (int): The number of input layer channel.
        mlp_ratio (int): The rate of mlp layer.
        dropout_rate (float, optional): The rate of dropout layer. Default: 1.0.
        drop_path (float, optional): The rate of drop path layer. Default: 0.0.
        h_size (int): The height (latitude) size of the input feature.
        w_size (int): The width (longitude) size of the input feature.
        patch_size (int, optional): The patch size of image. Default: 8.
        num_blocks: (int, optional): The number of blocks. Default: 16.
        high_freq (bool, optional): If high-frequency information complement is applied. Default: True.
        compute_dtype (:class:`mindspore.dtype`): The type of input tensor. Default: mindspore.float32.

    Inputs:
        - **input** (Tensor) - Tensor of shape :math:`(batch\_size, h\_size*w\_size, latent\_dims)`.

    Outputs:
        - **output** (Tensor) - Tensor of shape :math:`(batch\_size, h\_size*w\_size, latent\_dims)`.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore
        >>> from mindspore import Tensor
        >>> inputs = Tensor(np.random.randn(1,16*32,768), dtype=mindspore.float32)
        >>> net = SKNOBlock(latent_dims=768, mlp_ratio=1)
        >>> output = net(inputs)
        >>> print(output.shape)
        (2, 512, 768)

    """

    def __init__(self,
                 latent_dims,
                 mlp_ratio,
                 dropout_rate=1.0,
                 drop_path=0.,
                 h_size=128,
                 w_size=256,
                 patch_size=8,
                 num_blocks=16,
                 high_freq=False,
                 compute_dtype=mstype.float32):
        super(SKNOBlock, self).__init__()
        self.latent_dims = latent_dims
        self.layer_norm = nn.LayerNorm([latent_dims], epsilon=1e-6).to_float(compute_dtype)

        self.ffn_norm = nn.LayerNorm([latent_dims], epsilon=1e-6).to_float(compute_dtype)
        self.mlp = MLPNet(latent_dims, mlp_ratio, dropout_rate, compute_dtype=compute_dtype)
        self.filter = SKNO1D(h_size=h_size // patch_size,
                             w_size=w_size // patch_size,
                             latent_dims=latent_dims,
                             num_blocks=num_blocks,
                             high_freq=high_freq,
                             compute_dtype=compute_dtype)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def construct(self, x: Tensor):
        '''SKNOBlock construct function

        Args:
            x (Tensor): Input Tensor.
        '''
        res_x = x
        x = self.layer_norm(x)
        x = self.filter(x)
        x = x + res_x

        res_x = x
        x = self.ffn_norm(x)
        x = self.mlp(x)
        x = self.drop_path(x)
        x = x + res_x
        return x
