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
# ==============================================================================
"""embedding
"""
import math

import numpy as np
from mindspore import nn, ops, Tensor, Parameter, float32

from ..e3.o3 import Irreps


def _poly_cutoff(x, factor, p=6.0):
    x = x * factor
    out = 1.0
    out = out - (((p + 1.0) * (p + 2.0) / 2.0) * ops.pow(x, p))
    out = out + (p * (p + 2.0) * ops.pow(x, p + 1.0))
    out = out - ((p * (p + 1.0) / 2) * ops.pow(x, p + 2.0))
    return out * (x < 1.0)


class PolyCutoff(nn.Cell):

    def __init__(self, r_max, p=6):
        super().__init__()
        self.p = float(p)
        self._factor = 1.0 / float(r_max)

    def construct(self, x):
        return _poly_cutoff(x, self._factor, p=self.p)


class MaskPolynomialCutoff(nn.Cell):
    """MaskPolynomialCutoff
    """
    _factor: float
    p: float

    def __init__(self, r_max: float, p: float = 6):
        super().__init__()
        self.p = float(p)
        self._factor = 1.0 / float(r_max)
        self.r_max = Tensor(r_max, dtype=float32)

        self.cutoff = r_max

    def construct(self, distance: Tensor, mask: Tensor = None):
        decay = _poly_cutoff(distance, self._factor, p=self.p)

        mask_lower = distance < self.cutoff
        if mask is not None:
            mask_lower &= mask

        return decay, mask_lower


class BesselBasis(nn.Cell):
    """BesselBasis
    """

    def __init__(self, r_max, num_basis=8, dtype=float32):
        super().__init__()
        self.r_max = r_max
        self.num_basis = num_basis
        self.prefactor = 2.0 / self.r_max
        bessel_weights = Tensor(np.linspace(1., num_basis, num_basis) * math.pi, dtype=dtype)
        self.bessel_weights = Parameter(bessel_weights)

    def construct(self, x):
        numerator = ops.sin(self.bessel_weights * x.unsqueeze(-1) / self.r_max)
        return self.prefactor * (numerator / x.unsqueeze(-1))


class NormBesselBasis(nn.Cell):
    """NormBesselBasis
    """

    def __init__(self, r_max, num_basis=8, norm_num=4000):
        super().__init__()

        self.basis = BesselBasis(r_max=r_max, num_basis=num_basis)
        self.rs = Tensor(np.linspace(0.0, r_max, num=norm_num + 1), float32)[1:]

        self.sqrt = ops.Sqrt()
        self.sq = ops.Square()
        self.div = ops.Div()

        bessel_weights = Tensor(np.linspace(1.0, num_basis, num=num_basis), float32)

        bessel_weights = bessel_weights * Tensor(math.pi, float32)
        edge_length = self.rs
        edge_length_unsqueeze = edge_length.unsqueeze(-1)
        bessel_edge_length = bessel_weights * edge_length_unsqueeze
        if r_max != 0:
            bessel_edge_length = bessel_edge_length / r_max
            prefactor = 2.0 / r_max
        else:
            raise ValueError
        self.sin = ops.Sin()
        numerator = self.sin(bessel_edge_length)
        bs = prefactor * self.div(numerator, edge_length_unsqueeze)

        basis_mean = Tensor(np.mean(bs.asnumpy(), axis=0), float32)
        basis_std = self.sqrt(Tensor(np.mean(self.sq(bs - basis_mean).asnumpy(), 0), float32))
        inv_std = ops.reciprocal(basis_std)

        self.basis_mean = basis_mean
        self.inv_std = inv_std

    def construct(self, edge_length):
        basis_length = self.basis(edge_length)
        return (basis_length - self.basis_mean) * self.inv_std


class RadialEdgeEmbedding(nn.Cell):
    """RadialEdgeEmbedding
    """

    def __init__(self, r_max, num_basis=8, p=6, dtype=float32):
        super().__init__()
        self.num_basis = num_basis
        self.cutoff_p = p
        self.basis = BesselBasis(r_max, num_basis, dtype=dtype)
        self.cutoff = PolyCutoff(r_max, p)

        self.irreps_out = Irreps([(self.basis.num_basis, (0, 1))])

    def construct(self, edge_length):
        edge_length_embedded = self.basis(edge_length) * self.cutoff(edge_length).unsqueeze(-1)
        return edge_length_embedded

    def __repr__(self):
        return f'RadialEdgeEmbedding [num_basis: {self.num_basis}, cutoff_p: ' \
            + f'{self.cutoff_p}] ( -> {self.irreps_out} | {self.basis.num_basis} weights)'
