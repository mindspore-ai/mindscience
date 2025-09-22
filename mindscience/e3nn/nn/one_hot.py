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
"""OneHot"""
import math

import numpy as np

from mindspore import Tensor, ops, nn, float32, float16
from mindspore import numpy as mnp

from ..o3.irreps import Irreps

TMAP = {"MixedPrecisionType.FP16": float16, "MixedPrecisionType.FP32": float32}

def soft_unit_step(x):
    r"""
    Smooth version of the unit step function.

    .. math::
        x \mapsto \theta(x) e^{-1/x}

    Args:
        x (Tensor): the input tensor.

    Returns:
        Tensor, the output of the unit step function.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> from mindchemistry.e3.nn import soft_unit_step
        >>> from mindspore import ops, set_context, Tensor
        >>> x = Tensor(ops.linspace(-1.0, 10.0, 1000))
        >>> outputs = soft_unit_step(x)
        >>> print(outputs.shape)
        (1000,)

    """
    return ops.relu(x) * ops.exp(- 1 / x) / x


class OneHot(nn.Cell):
    r"""
    One-hot embedding.
    """

    def __init__(self, num_types, dtype=float32):
        super().__init__()
        self.num_types = num_types
        self.irreps_output = Irreps([(self.num_types, (0, 1))])

        self.one_hot = ops.OneHot()
        self.on_off = (Tensor(1., dtype=dtype), Tensor(0., dtype=dtype))

    def construct(self, atom_type):
        type_numbers = atom_type
        one_hot = self.one_hot(type_numbers, self.num_types, *self.on_off)
        return one_hot

    def __repr__(self):
        return f'OneHot [num_types: {self.num_types}] ( -> {self.irreps_output})'


# pylint: disable=C0103
# pylint: disable=R1705
class SoftOneHotLinspace(nn.Cell):
    r"""
    Projection on a basis of functions. Returns a set of :math:`\{y_i(x)\}_{i=1}^N`,

    .. math::
        y_i(x) = \frac{1}{Z} f_i(x)

    where :math:`x` is the input and :math:`f_i` is the ith basis function.
    :math:`Z` is a constant defined (if possible) such that,

    .. math::
        \langle \sum_{i=1}^N y_i(x)^2 \rangle_x \approx 1

    Note that `bessel` basis cannot be normalized.

    Args:
        start (float): minimum value span by the basis.
        end (float): maximum  value span by the basis.
        number (int): number of basis functions :math:`N`.
        basis (str): {'gaussian', 'cosine', 'smooth_finite', 'fourier', 'bessel'}, the basis family.
            Default: ``'smooth_finite'``.
        cutoff (bool): whether require the :math:`y_i(x)` from the outside domain of (`start`, `end`) to be
            vanished. Default: ``True``.
        dtype (mindspore.dtype): The type of input tensor. Default: ``mindspore.float32``.

    Inputs:
        - **x** (Tensor) - The shape of Tensor is :math:`(...)`.

    Outputs:
        - **output** (Tensor) - The shape of Tensor is :math:`(..., N)`.

    Raises:
        ValueError: If `basis` is not in {'gaussian', 'cosine', 'smooth_finite', 'fourier', 'bessel'}.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> from mindchemistry.e3.nn import SoftOneHotLinspace
        >>> from mindspore import ops, Tensor
        >>> soft_one_hot_linspace = SoftOneHotLinspace(-0.5, 1.5, number=4)
        >>> x = Tensor(ops.ones((4, 6)))
        >>> outputs = soft_one_hot_linspace(x)
        >>> print(outputs.shape)
        (4, 6, 4)

    """

    def __init__(self, start, end, number, basis='smooth_finite', cutoff=True, dtype=float32):
        super().__init__()

        self.start = Tensor(start, dtype=dtype)
        self.end = Tensor(end, dtype=dtype)
        self.number = number
        self.basis = basis
        self.cutoff = cutoff

        if self.cutoff:
            self.values = Tensor(np.linspace(start, end, number), dtype=dtype)
            self.step = self.values[1] - self.values[0]
        else:
            self.values = Tensor(np.linspace(start, end, number + 2), dtype=dtype)
            self.step = self.values[1] - self.values[0]
            self.values = self.values[1:-1]

        self.PI = Tensor(math.pi, dtype=dtype)
        self.c = self.end - self.start
        self.consts = [
            ops.exp(Tensor(2.0, dtype=dtype)),
            ops.sqrt(Tensor(0.25 + self.number / 2, dtype=dtype)),
            ops.sqrt(Tensor(2. / self.c, dtype=dtype))
        ]
        self.bessel_roots = mnp.arange(1, self.number + 1) * self.PI

    def construct(self, x):
        """construct"""
        diff = (x.expand_dims(-1) - self.values) / self.step

        if self.basis == 'gaussian':
            return ops.exp(-diff.pow(2)) / 1.12

        elif self.basis == 'cosine':
            return ops.cos(self.PI / 2 * diff) * (diff < 1) * (-1 < diff)

        elif self.basis == 'smooth_finite':
            return 1.14136 * self.consts[0] * soft_unit_step(diff + 1.) * soft_unit_step(1. - diff)

        elif self.basis == 'fourier':
            x = (x.expand_dims(-1) - self.start) / (self.end - self.start)
            if not self.cutoff:
                i = mnp.arange(0, self.number)
                return ops.cos(self.PI * i * x) / self.consts[1]
            else:
                i = mnp.arange(1, self.number + 1)
                return ops.sin(self.PI * i * x) / self.consts[1] * (x > 0) * (x < 1)

        if self.basis == 'bessel':
            x = x.expand_dims(-1) - self.start
            out = self.consts[2] * ops.sin(self.bessel_roots * x / self.c) / x

            if not self.cutoff:
                return out
            else:
                return out * ((x / self.c) < 1) * (x > 0)

        else:
            raise ValueError(f"Unsupported basis: {self.basis}.")

    def _set_mixed_precision_type_recursive(self, dst_type):
        super()._set_mixed_precision_type_recursive(dst_type)
        self.values = self.values.astype(TMAP[dst_type.__str__()])
        for i in range(len(self.consts)):
            self.consts[i] = self.consts[i].astype(TMAP[dst_type.__str__()])


def soft_one_hot_linspace(x, start, end, number, basis='smooth_finite', cutoff=True):
    r"""
    Projection on a basis of functions. Returns a set of :math:`\{y_i(x)\}_{i=1}^N`,

    .. math::
        y_i(x) = \frac{1}{Z} f_i(x)

    where :math:`x` is the input and :math:`f_i` is the ith basis function.
    :math:`Z` is a constant defined (if possible) such that,

    .. math::
        \langle \sum_{i=1}^N y_i(x)^2 \rangle_x \approx 1

    Note that `bessel` basis cannot be normalized.

    Args:
        x (Tensor): The shape of Tensor is :math:`(...)`.
        start (float): minimum value span by the basis.
        end (float): maximum  value span by the basis.
        number (int): number of basis functions :math:`N`.
        basis (str): {'gaussian', 'cosine', 'smooth_finite', 'fourier', 'bessel'}, the basis family.
            Default: ``'smooth_finite'``.
        cutoff (bool): whether require the :math:`y_i(x)` from the outside domain of (`start`, `end`) to be
            vanished. Default: ``True``.

    Returns:
        Tensor, shape is :math:`(..., N)`.

    Raises:
        ValueError: If `basis` is not in {'gaussian', 'cosine', 'smooth_finite', 'fourier', 'bessel'}.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> from mindchemistry.e3.nn import soft_one_hot_linspace
        >>> from mindspore import ops, Tensor
        >>> x = Tensor(ops.ones((4, 6)))
        >>> outputs = soft_one_hot_linspace(x, -0.5, 1.5, number=4)
        >>> print(outputs.shape)
        (4, 6, 4)

    """
    soft = SoftOneHotLinspace(start, end, number, basis=basis, cutoff=cutoff, dtype=x.dtype)
    return soft(x)
