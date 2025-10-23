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
"""efficient"""

import mindspore as ms
import mindspore.mint as mint
from .he_orthogonal import he_orthogonal_init


class EfficientInteractionDownProjection(ms.nn.Cell):
    r"""
    Down projection in the efficient reformulation.

    Args:
        num_spherical (int): Number of spherical harmonics.
        num_radial (int): Number of radial basis functions.
        emb_size_interm (int): Intermediate embedding size (down-projection size).

    Inputs:
        - **rbf** (Tensor): Radial basis functions. The shape of tensor is :math:`(1, total\_edges, num\_radial)`.
        - **sph** (Tensor): Spherical harmonics. The shape of tensor is :math:`(total\_edges, kmax, num\_spherical)`.
        - **id_ca** (Tensor): The shape of tensor is :math:`(total\_triplets,)`.
        - **id_ragged_idx** (Tensor): The shape of tensor is :math:`(total\_triplets,)`.
        - **kmax** (int): Maximum number of neighbors of the edges.

    Outputs:
        - **sph** (Tensor): The shape of tensor is :math:`(total\_edges, kmax, num\_spherical)`.
    """

    def __init__(
            self,
            num_spherical,
            num_radial,
            emb_size_interm,
    ):
        super().__init__()

        self.num_spherical = num_spherical
        self.num_radial = num_radial
        self.emb_size_interm = emb_size_interm
        self.weight = ms.Parameter(
            mint.zeros(
                (self.num_spherical, self.num_radial, self.emb_size_interm)
            ),
            requires_grad=True,
        )
        self.reset_parameters()

    def reset_parameters(self):
        self.weight = he_orthogonal_init(self.weight)

    def construct(self, rbf, sph, id_ca, id_ragged_idx, kmax):
        """Construct of EfficientInteractionDownProjection."""
        num_edges = rbf.shape[1]

        rbf_w1 = mint.matmul(rbf, self.weight)
        rbf_w1 = mint.permute(rbf_w1, (1, 2, 0))

        sph2 = mint.zeros((num_edges, kmax + 1, self.num_spherical))
        index_sph2 = mint.stack((id_ca, id_ragged_idx), dim=1)
        sph2 = ms.ops.tensor_scatter_update(sph2, index_sph2, sph)

        sph2 = mint.permute(sph2, (0, 2, 1))
        return rbf_w1, sph2


class EfficientInteractionBilinear(ms.nn.Cell):
    r"""
    Efficient reformulation of the bilinear layer and subsequent summation.

    Args:
        emb_size (int): Embedding size.
        emb_size_interm (int): Intermediate embedding size.
        units_out (int): Embedding output size of the bilinear layer.

    Inputs:
        - **rbf_w1** (Tensor): The shape of tensor is :math:`(total\_Edges, emb\_size\_interm, num\_spherical)`.
        - **sph** (Tensor): The shape of tensor is :math:`(total\_Edges, num\_spherical, kmax + 1)`.
        - **m** (Tensor): The shape of tensor is :math:`(total\_triples, emb\_size)`.
        - **id_reduce** (Tensor): The shape of tensor is :math:`(total\_triplets,)`.
        - **id_ragged_idx** (Tensor): The shape of tensor is :math:`(total\_triplets,)`.
        - **kmax** (int): Maximum number of neighbors of the edges.

    Outputs:
        - **m_ca** (Tensor): The shape of tensor is :math:`(total\_Edges, units\_out)`.
    """

    def __init__(
            self,
            emb_size,
            emb_size_interm,
            units_out,
    ):
        super().__init__()
        self.emb_size = emb_size
        self.emb_size_interm = emb_size_interm
        self.units_out = units_out

        self.weight = ms.Parameter(
            mint.zeros(
                (self.emb_size, self.emb_size_interm, self.units_out)
            )
        )
        self.reset_parameters()

    def reset_parameters(self):
        self.weight = he_orthogonal_init(self.weight)

    def construct(self, rbf_w1, sph, m, id_reduce, id_ragged_idx, kmax):
        """Construct of EfficientInteractionBilinear."""
        n_edges = rbf_w1.shape[0]

        m2 = mint.zeros((n_edges, kmax + 1, self.emb_size))
        index_m2 = mint.stack((id_reduce, id_ragged_idx), dim=1)
        m2 = ms.ops.tensor_scatter_update(m2, index_m2, m)

        sum_k = mint.matmul(sph, m2)
        rbf_w1_sum_k = mint.matmul(rbf_w1, sum_k)
        m_ca = mint.matmul(mint.permute(rbf_w1_sum_k, (2, 0, 1)), self.weight)
        m_ca = mint.sum(m_ca, dim=0)
        return m_ca
