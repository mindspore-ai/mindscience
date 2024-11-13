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
"""dimenet"""

from math import pi as PI

import numpy as np
import mindspore as ms
import mindspore.mint as mint
from mindspore.common.initializer import initializer

from .glorot_orthogonal import glorot_orthogonal


def swish(x):
    return mint.mul(x, mint.nn.functional.sigmoid(x))


class Envelope(ms.nn.Cell):
    r"""
    Envelope

    Args:
        exponent (int): Exponent of the envelope function.

    Inputs:
        - **x** (Tensor) - The shape of tensor is :math:`(total\_edges, 1)`.

    Outputs:
        - (Tensor) - The shape of tensor is :math:`(total\_edges, 1)`.
    """

    def __init__(self, exponent):
        super().__init__()
        self.p = exponent + 1
        self.a = -(self.p + 1) * (self.p + 2) / 2
        self.b = self.p * (self.p + 2)
        self.c = -self.p * (self.p + 1) / 2

    def construct(self, x):
        x_pow_p0 = mint.pow(x, (self.p - 1))
        x_pow_p1 = mint.mul(x_pow_p0, x)
        x_pow_p2 = mint.mul(x_pow_p1, x)
        out = mint.add(mint.reciprocal(x), x_pow_p0, alpha=self.a)
        out = mint.add(out, x_pow_p1, alpha=self.b)
        out = mint.add(out, x_pow_p2, alpha=self.c)
        return out


class BesselBasisLayer(ms.nn.Cell):
    r"""
    Bessel Basis Layer

    Args:
        num_radial (int): Number of radial basis functions.
        cutoff (float): Cutoff distance. Default=5.0.
        envelope_exponent (int): Exponent of the envelope function. Default=5

    Inputs:
        - **dist** (Tensor) - The shape of tensor is :math:`(total\_edges, 1)`.

    Outputs:
        - (Tensor) - The shape of tensor is :math:`(total\_edges, num\_radial)`.
    """

    def __init__(self, num_radial, cutoff=5.0, envelope_exponent=5):
        super().__init__()
        self.cutoff = cutoff
        self.envelope = Envelope(envelope_exponent)

        self.freq = ms.Parameter(mint.arange(1, num_radial + 1).mul(PI))

    def construct(self, dist):

        dist = mint.div(dist.view(-1, 1), self.cutoff)
        return self.envelope(dist) * mint.sin(self.freq * dist)


class EmbeddingBlock(ms.nn.Cell):
    r"""
    Embedding block

    Args:
        num_radial (int): Number of radial basis functions.
        hidden_channels (int): Hidden channels.
        act (function): Activation function. Default: silu.

    Inputs:
        - **x** (Tensor) - The shape of tensor is :math:`(total\_atoms)`.
        - **rbf** (Tensor) - The shape of tensor is :math:`(total\_edges, num\_radial)`.
        - **i** (Tensor) - The shape of tensor is :math:`(total\_triplets)`.
        - **j** (Tensor) - The shape of tensor is :math:`(total\_triplets)`.

    Outputs:
        - (Tensor) - The shape of tensor is :math:`(total\_triplets, hidden\_channels)`.
    """

    def __init__(self, num_radial, hidden_channels, act=mint.nn.functional.silu):
        super().__init__()
        self.act = act
        # Reason from Dimenet"s author: Atom embeddings: We go up to Pu (94).
        # Use 95 dimensions because of 0-based indexing
        self.emb = mint.nn.Embedding(95, hidden_channels)
        self.lin_rbf = mint.nn.Linear(num_radial, hidden_channels)
        self.lin = mint.nn.Linear(3 * hidden_channels, hidden_channels)
        self.reset_parameters()

    def reset_parameters(self):
        self.emb.weight = ms.Parameter(ms.ops.uniform(
            self.emb.weight.shape,
            minval=ms.Tensor(-np.sqrt(3), dtype=ms.float32),
            maxval=ms.Tensor(np.sqrt(3), dtype=ms.float32)
        ))

    def construct(self, x, rbf, i, j):
        x = self.emb(x)
        rbf = self.act(self.lin_rbf(rbf))
        rbf = mint.cat((mint.index_select(x, 0, i),
                        mint.index_select(x, 0, j), rbf), dim=-1)
        return self.act(self.lin(rbf))


class ResidualLayer(ms.nn.Cell):
    r"""
    Residual layer

    Args:
        hidden_channels (int): Hidden channels.
        act (function): Activation function. Default: swish.
    Inputs:
        - **x** (Tensor) - The shape of tensor is :math:`(*, hidden\_channels)`.
    Outputs:
        - (Tensor) - The shape of tensor is :math:`(*, hidden\_channels)`.
    """

    def __init__(self, hidden_channels, act=swish):
        super().__init__()
        self.act = act
        self.lin1 = mint.nn.Linear(hidden_channels, hidden_channels)
        self.lin2 = mint.nn.Linear(hidden_channels, hidden_channels)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin1.weight = glorot_orthogonal(self.lin1.weight, 2.0)
        self.lin2.weight = glorot_orthogonal(self.lin2.weight, 2.0)
        self.lin1.bias.set_data(initializer(
            "zero", self.lin1.bias.shape, self.lin1.bias.dtype))
        self.lin2.bias.set_data(initializer(
            "zero", self.lin2.bias.shape, self.lin2.bias.dtype))

    def construct(self, x):
        return x + self.act(self.lin2(self.act(self.lin1(x))))
