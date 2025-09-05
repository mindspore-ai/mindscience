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
"""Base cell for bais potential"""

import mindspore as ms
from mindspore import Tensor
from mindspore import nn
from mindspore.ops import functional as F

from .bias import Bias
from ...function.units import Units, global_units, Length, Energy
from ...function import functions as func


class SphericalRestrict(Bias):
    r"""
    Basic cell for bias potential.

    .. Math::

        V(R) = k * log(1 + exp((|R - R_0| - r_0) / \sigma))

    Args:
        radius (float):         Radius of sphere (r_0).
        center (Tensor):        Coordinate of the center of sphere (R_0). Default: 0
        force_constant (float): Force constant of the bias potential(k). Default: Energy(500, 'kj/mol')
        depth (float):          Wall depth of the restriction (\sigma). Default: Length(0.01, 'nm')
        length_unit (str):      Length unit for position coordinates. Default: None
        energy_unit (str):      Energy unit. Default: None
        units (Units):          Units of length and energy. Default: global_units
        use_pbc (bool):         Whether to use periodic boundary condition. Default: None

    Returns:
        potential (Tensor), Tensor of shape (B, 1). Data type is float.

    Supported Platforms:
        ``Ascend`` ``GPU``
    """
    def __init__(self,
                 radius: float,
                 center: Tensor = 0,
                 force_constant: float = Energy(500, 'kj/mol'),
                 depth: float = Length(0.01, 'nm'),
                 length_unit: str = None,
                 energy_unit: str = None,
                 units: Units = global_units,
                 use_pbc: bool = None,
                 ):

        super().__init__(
            length_unit=length_unit,
            energy_unit=energy_unit,
            units=units,
            use_pbc=use_pbc,
        )

        self.radius = Tensor(radius, ms.float32)
        self.center = Tensor(center, ms.float32)

        if isinstance(force_constant, Energy):
            force_constant = force_constant(self.units)
        self.force_constant = Tensor(force_constant, ms.float32)

        if isinstance(depth, Length):
            depth = depth(self.units)
        self.depth = Tensor(depth, ms.float32)

        self.norm_last_dim = nn.Norm(axis=-1, keep_dims=False)

    def construct(self,
                  coordinate: Tensor,
                  neighbour_index: Tensor = None,
                  neighbour_mask: Tensor = None,
                  neighbour_coord: Tensor = None,
                  neighbour_distance: Tensor = None,
                  pbc_box: Tensor = None
                  ):
        r"""
        Calculate bias potential.

        Args:
            coordinate (Tensor):            Tensor of shape (B, A, D). Data type is float.
                                            Position coordinate of atoms in system.
            neighbour_index (Tensor):       Tensor of shape (B, A, N). Data type is int.
                                            Index of neighbour atoms. Default: None
            neighbour_mask (Tensor):        Tensor of shape (B, A, N). Data type is bool.
                                            Mask for neighbour atoms. Default: None
            neighbour_coord (Tensor):       Tensor of shape (B, A, N). Data type is bool.
                                            Position coorindates of neighbour atoms.
            neighbour_distance (Tensor):    Tensor of shape (B, A, N). Data type is float.
                                            Distance between neighbours atoms. Default: None
            pbc_box (Tensor):               Tensor of shape (B, D). Data type is float.
                                            Tensor of PBC box. Default: None

        Returns:
            potential (Tensor), Tensor of shape (B, 1). Data type is float.

        Symbols:
            B:  Batchsize, i.e. number of walkers in simulation.
            A:  Number of atoms.
            N:  Maximum number of neighbour atoms.
            D:  Dimension of the simulation system. Usually is 3.
        """

        # (B, A) <- (B, A, D)
        distance = self.norm_last_dim(coordinate - self.center)
        diff = distance - self.radius
        bias = self.force_constant * F.log(1.0 + F.exp(diff/self.depth))

        # (B, 1) <- (B, A)
        return func.keepdim_sum(bias, -1)
