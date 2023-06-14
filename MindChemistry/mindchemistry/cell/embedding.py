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
import math

import numpy as np

from mindspore import nn, ops, float32, Tensor, Parameter
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


class BesselBasis(nn.Cell):
    def __init__(self, r_max, num_basis=8, dtype=float32):
        super().__init__()
        self.r_max = r_max
        self.num_basis = num_basis
        self.prefactor = 2.0 / self.r_max
        bessel_weights = Tensor(np.linspace(
            1., num_basis, num_basis) * math.pi, dtype=dtype)
        self.bessel_weights = Parameter(bessel_weights)

    def construct(self, x):
        numerator = ops.sin(self.bessel_weights * x.unsqueeze(-1) / self.r_max)
        return self.prefactor * (numerator / x.unsqueeze(-1))


class RadialEdgeEmbedding(nn.Cell):
    def __init__(self, r_max, num_basis=8, p=6, dtype=float32):
        super().__init__()
        self.num_basis = num_basis
        self.cutoff_p = p
        self.basis = BesselBasis(r_max, num_basis, dtype=dtype)
        self.cutoff = PolyCutoff(r_max, p)

        self.irreps_out = Irreps([(self.basis.num_basis, (0, 1))])

    def construct(self, edge_length):
        basis_edge_length = self.basis(edge_length)
        cutoff_edge_length = self.cutoff(edge_length)
        edge_length_embedded = self.basis(
            edge_length) * self.cutoff(edge_length).unsqueeze(-1)
        return edge_length_embedded

    def __repr__(self):
        return f'RadialEdgeEmbedding [num_basis: {self.num_basis }, cutoff_p: ' \
            + f'{self.cutoff_p}] ( -> {self.irreps_out} | {self.basis.num_basis} weights)'
