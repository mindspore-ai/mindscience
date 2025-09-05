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
Collective variables for fixed atoms
"""

import mindspore as ms
from mindspore import ops, nn
from mindspore import Tensor
from mindspore.ops import functional as F
from mindspore import numpy as msnp

from ..function import functions as func
from ..function import get_ms_array
from .colvar import Colvar


class AtomDistances(Colvar):
    r"""Distances of specific atoms

    Args:
        index (int):        Index of atoms.

        use_pbc (bool):     Whether to use periodic boundary condition. Default: False

        length_unit (str)   Length unit. Default: None

    """
    def __init__(self,
                 index: Tensor,
                 use_pbc: bool = None,
                 length_unit: str = None,
                 ):

        super().__init__(
            dim_output=1,
            periodic=False,
            use_pbc=use_pbc,
            length_unit=length_unit,
        )

        # (B,b,2)
        self.index = get_ms_array(index, ms.int32)
        if self.index.shape[-1] != 2:
            raise ValueError('The last dimension of index in AtomDistances must be 2!')
        self.dim_output = self.index.shape[-2]
        self.identity = ops.Identity()
        self.norm_last_dim = nn.Norm(axis=-1, keep_dims=False)

    def construct(self, coordinate: Tensor, pbc_box: Tensor = None):
        r"""Compute distances.

        Args:
            coordinate (Tensor):    Tensor of shape (B, A, D). Data type is float.
            pbc_box (Tensor):       Tensor of shape (B, D). Data type is float.
                                    Default: None

        Returns:
            distances (Tensor):     Tensor of shape (B, X, 1). Data type is float.

        """

        # (B,b,2)
        index = self.identity(self.index)
        # (B,b,2,D)
        atoms = func.gather_vectors(coordinate, index)

        # (B,b,D)
        vec = self.get_vector(atoms[..., 0, :], atoms[..., 1, :], pbc_box)
        # (B,b)
        return self.norm_last_dim(vec)


class AtomAngles(Colvar):
    r"""Angles of specific atoms

    Args:
        index (int):        Index of atoms.
        use_pbc (bool):     Whether to use periodic boundary condition. Default: False

    """
    def __init__(self,
                 index: Tensor,
                 use_pbc: bool = None,
                 ):

        super().__init__(
            periodic=False,
            use_pbc=use_pbc,
        )

        # (B,a,3)
        self.index = get_ms_array(index, ms.int32)
        if self.index.shape[-1] != 3:
            raise ValueError('The last dimension of index in AtomAngles must be 3!')
        self.dim_output = self.index.shape[-2]
        self.split = ops.Split(-2, 3)
        self.norm_last_dim = nn.Norm(axis=-1, keep_dims=False)

    def construct(self, coordinate: Tensor, pbc_box: Tensor = None):
        r"""Compute angles.

        Args:
            coordinate (Tensor):    Tensor of shape (B, A, D). Data type is float.
            pbc_box (Tensor):       Tensor of shape (B, D). Data type is float.
                                    Default: None

        Returns:
            angles (Tensor):        Tensor of shape (B, X, 1). Data type is float.

        """

        # (B,a,3)
        index = self.identity(self.index)
        # (B,a,3,D)
        atoms = func.gather_vectors(coordinate, index)

        # (B,a,1,D)
        atom0, atom1, atom2 = self.split(atoms)

        vec1 = self.get_vector(atom1, atom0, pbc_box).squeeze(-2)
        vec2 = self.get_vector(atom1, atom2, pbc_box).squeeze(-2)

        # (B,a) <- (B,a,D)
        dis1 = self.norm_last_dim(vec1)
        dis2 = self.norm_last_dim(vec2)

        # (B,a) <- (B,a,D)
        vec1vec2 = F.reduce_sum(vec1*vec2, -1)
        # (B,a) = (B,a) * (B,a)
        dis1dis2 = dis1 * dis2
        # (B,a)/(B,a)
        costheta = vec1vec2 * msnp.reciprocal(dis1dis2)

        # (B,a)
        return F.acos(costheta)


class AtomTorsions(Colvar):
    r"""Torsion (dihedral) angle of specific atoms

    Args:
        index (int):        Index of atoms.
        use_pbc (bool):     Whether to use periodic boundary condition. Default: False

    """
    def __init__(self,
                 index: Tensor,
                 use_pbc: bool = None,
                 ):

        super().__init__(
            periodic=True,
            use_pbc=use_pbc,
        )

        # (B,d,4)
        self.index = get_ms_array(index, ms.int32)
        if self.index.shape[-1] != 4:
            raise ValueError('The last dimension of index in AtomTorsions must be 4!')
        self.dim_output = self.index.shape[-2]
        self.split = ops.Split(-2, 4)
        self.keep_norm_last_dim = nn.Norm(axis=-1, keep_dims=True)

    def construct(self, coordinate: Tensor, pbc_box: Tensor = None):
        r"""Compute torsions.

        Args:
            coordinate (Tensor):    Tensor of shape (B, A, D). Data type is float.
            pbc_box (Tensor):       Tensor of shape (B, D). Data type is float.
                                    Default: None

        Returns:
            torsion (Tensor):       Tensor of shape (B, X, 1). Data type is float.

        """

        # (B,d,4)
        index = self.identity(self.index)
        # (B,d,4,D)
        atoms = func.gather_vectors(coordinate, index)

        # (B,d,1,D)
        atom_a, atom_b, atom_c, atom_d = self.split(atoms)

        # (B,d,1,D)
        vec_1 = self.get_vector(atom_b, atom_a, pbc_box).squeeze(-2)
        vec_2 = self.get_vector(atom_c, atom_b, pbc_box).squeeze(-2)
        vec_3 = self.get_vector(atom_d, atom_c, pbc_box).squeeze(-2)

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
