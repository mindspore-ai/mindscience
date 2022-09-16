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
Collective variables by position
"""

from mindspore import Tensor
from mindspore import nn

from ..function import calc_angle_between_vectors, calc_torsion_for_vectors
from .colvar import Colvar
from .position import Position


class Distance(Colvar):
    r"""Get distances by positions

    Args:

        position0 (Position):   First position,

        position1 (Position):   Second position,

        use_pbc (bool):     Whether to calculate the CV at periodic boundary condition (PBC).
                            If "None" is given, it will be determined at runtime based on
                            whether the "pbc_box" is given or not. Default: None

        length_unit (str):  Length unit for position coordinates.
                            If "None" is given, it will use the global units. Default: None

   """
    def __init__(self,
                 position0: Position,
                 position1: Position,
                 use_pbc: bool = None,
                 length_unit: str = None,
                 ):

        super().__init__(
            dim_output=1,
            periodic=False,
            use_pbc=use_pbc,
            length_unit=length_unit,
        )

        self.position0 = position0
        self.position1 = position1
        self.keep_norm_last_dim = nn.Norm(axis=-1, keep_dims=True)

    def construct(self, coordinate: Tensor, pbc_box: bool = None):
        r"""Compute distance between two atoms.

        Args:
            coordinate (ms.Tensor[B,N,D])

        Returns:
            distance (ms.Tensor[B,n,1]):

        """

        pos0 = self.position0(coordinate)
        pos1 = self.position1(coordinate)

        vec = self.get_vector(pos0, pos1, pbc_box)
        return self.keep_norm_last_dim(vec)


class Angle(Colvar):
    r"""Get angle by positions

    Args:

    """
    def __init__(self,
                 position_a: Position,
                 position_b: Position,
                 position_c: Position,
                 use_pbc: bool = None,
                 ):

        super().__init__(
            dim_output=1,
            periodic=False,
            use_pbc=use_pbc,
        )

        self.position_a = position_a
        self.position_b = position_b
        self.position_c = position_c

    def construct(self, coordinate: Tensor, pbc_box: bool = None):
        r"""Compute distance between two atoms.

        Args:
            coordinate (ms.Tensor[B,N,D])

        Returns:
            distance (ms.Tensor[B,n,1]):

        """

        pos_a = self.position_a(coordinate)
        pos_b = self.position_b(coordinate)
        pos_c = self.position_c(coordinate)

        vec_ba = self.get_vector(pos_b, pos_a, pbc_box)
        vec_bc = self.get_vector(pos_b, pos_c, pbc_box)

        return calc_angle_between_vectors(vec_ba, vec_bc)


class Torsion(Colvar):
    r"""Get torsion by positions

    Args:

    """
    def __init__(self,
                 position_a: Position,
                 position_b: Position,
                 position_c: Position,
                 position_d: Position,
                 use_pbc: bool = None,
                 ):

        super().__init__(
            dim_output=1,
            periodic=True,
            use_pbc=use_pbc,
        )

        self.position_a = position_a
        self.position_b = position_b
        self.position_c = position_c
        self.position_d = position_d

    def construct(self, coordinate: Tensor, pbc_box: bool = None):
        r"""Compute distance between two atoms.

        Args:
            coordinate (ms.Tensor[B,N,D])

        Returns:
            distance (ms.Tensor[B,n,1]):

        """

        pos_a = self.position_a(coordinate)
        pos_b = self.position_b(coordinate)
        pos_c = self.position_c(coordinate)
        pos_d = self.position_d(coordinate)

        vec_ba = self.get_vector(pos_b, pos_a, pbc_box)
        vec_cb = self.get_vector(pos_c, pos_b, pbc_box)
        vec_dc = self.get_vector(pos_d, pos_c, pbc_box)

        return calc_torsion_for_vectors(vec_ba, vec_cb, vec_dc)
