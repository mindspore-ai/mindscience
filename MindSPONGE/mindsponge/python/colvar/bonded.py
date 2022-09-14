# Copyright 2021-2022 @ Shenzhen Bay Laboratory &
#                       Peking University &
#                       Huawei Technologies Co., Ltd
#
# This code is a part of MindSPONGE:
# MindSpore Simulation Package tOwards Next Generation molecular modelling.
#
# MindSPONGE is open-source software based on the AI-framework:
# MindSpore (https://www.mindspore.cn/)
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
"""
Collective variables by bonds
"""

import mindspore.numpy as msnp
from mindspore import Tensor
from mindspore import nn
from mindspore.ops import functional as F

from ..function import functions as func
from .colvar import Colvar


class BondedColvar(Colvar):
    r"""Get collective variables by bonds

    """

    def __init__(self,
                 bond_index: int,
                 length_unit: str = None,
                 ):

        super().__init__(
            dim_output=1,
            periodic=False,
            use_pbc=None,
            length_unit=length_unit,
        )

        self.bond_index = bond_index

    def construct(self, bond_vectors: Tensor, bond_distances: Tensor):
        #pylint: disable=arguments-differ
        raise NotImplementedError


class BondedDistances(BondedColvar):
    r"""Get distances by bonds

    """

    def __init__(self,
                 bond_index: int = None,
                 length_unit: str = None,
                 ):
        super().__init__(
            bond_index=bond_index,
            length_unit=length_unit,
        )

    def construct(self, bond_vectors: Tensor, bond_distances: Tensor):
        r"""Compute distance between two atoms.

        Args:
            coordinate (ms.Tensor[float]): coordinate of system with shape (B,A,D)

        Returns:
            distances (ms.Tensor[float]): distance between atoms with shape (B,M,1)

        """

        distances = bond_distances
        if self.bond_index is not None:
            distances = func.gather_values(bond_distances, self.bond_index)

        return distances


class BondedAngles(BondedColvar):
    r"""Get angles by bonds

    """

    def __init__(self, bond_index: int):
        super().__init__(
            bond_index=bond_index,
        )

    def construct(self, bond_vectors: Tensor, bond_distances: Tensor):
        r"""Compute angles formed by three atoms.

        Args:
            coordinate (ms.Tensor[float]): coordinate of system with shape (B,N,D)

        Returns:
            angles (ms.Tensor[float]): angles of atoms with shape (B,n,1)

        """

        # (B,a,2,D) <- gather (B,a,2) from (B,b,D)
        vectors = func.gather_vectors(bond_vectors, self.bond_index)
        # (B,a,2) <- gather (B,a,2) from (B,b)
        distances = func.gather_values(bond_distances, self.bond_index)

        # (B,a) <- (B,a,D)
        vec1vec2 = F.reduce_sum(vectors[:, :, 0, :]*vectors[:, :, 1, :], -1)
        # (B,a) = (B,a) * (B,a)
        dis1dis2 = distances[:, :, 0] * distances[:, :, 1]
        # (B,a)/(B,a)
        costheta = vec1vec2 * msnp.reciprocal(dis1dis2)

        # (B,a)
        return F.acos(costheta)


class BondedTorsions(BondedColvar):
    r"""Get torsion angles by bonds

    """

    def __init__(self, bond_index: int):
        super().__init__(
            bond_index=bond_index,
        )
        self.keep_norm_last_dim = nn.Norm(axis=-1, keep_dims=True)

    def construct(self, bond_vectors: Tensor, bond_distances: Tensor):
        r"""Compute torision angles formed by four atoms.

        Args:
            coordinate (ms.Tensor[float]): coordinate of system with shape (B,A,D)

        Returns:
            angles (ms.Tensor[float]): (B,M,1) angles of atoms

        """

        # (B,a,3,D) <- gather (B,a,3) from (B,b,D)
        vectors = func.gather_vectors(bond_vectors, self.bond_index)

        vec_1 = vectors[:, :, 0, :]
        vec_2 = vectors[:, :, 1, :]
        vec_3 = vectors[:, :, 2, :]

        # (B,d,1) <- (B,M,D)
        v2norm = self.keep_norm_last_dim(vec_2)
        # (B,d,D) = (B,d,D) / (B,d,1)
        norm_vec2 = vec_2 * msnp.reciprocal(v2norm)

        # (B,M,D)
        vec_a = msnp.cross(norm_vec2, vec_1)
        vec_b = msnp.cross(vec_3, norm_vec2)
        cross_ab = msnp.cross(vec_a, vec_b)

        # (B,M)
        sin_phi = F.reduce_sum(cross_ab*norm_vec2, -1)
        cos_phi = F.reduce_sum(vec_a*vec_b, -1)

        # (B,M)
        return F.atan2(-sin_phi, cos_phi)
