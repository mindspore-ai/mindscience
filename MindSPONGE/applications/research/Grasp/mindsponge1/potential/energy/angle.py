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
"""Angle energy"""

import mindspore as ms
from mindspore import Tensor
from mindspore import Parameter
from mindspore.ops import functional as F

from .energy import EnergyCell
from ...colvar import AtomAngles
from ...function import functions as func
from ...function.units import Units


class AngleEnergy(EnergyCell):
    r"""
    Energy term of bond angles.

    .. Math::

        E_{angle}(\theta_{ijk}) = 1 / 2 \times k_{ijk}^\theta \times (\theta_{ijk} - \theta_{ijk}^0) ^ 2

    Args:
        index (Tensor):             Tensor of shape (B, a, 3). Data type is int.
                                    Atom index of bond angles. Default: None
        force_constant (Tensor):    Tensor of shape (1, a). Data type is float.
                                    The harmonic force constants for angle :math:`(k^{\theta})`. Default: None
        bond_angle (Tensor):        Tensor of shape (1, a). Data type is float.
                                    The equilibrium value of bond angle :math:`({\theta}^0)`. Default: None
        parameters (dict):          Force field parameters. Default: None
        use_pbc (bool):             Whether to use periodic boundary condition. Default: None
        energy_unit (str):          Energy unit. Default: 'kj/mol'
        units (Units):              Units of length and energy. Default: None

    Returns:
        energy (Tensor), Tensor of shape (B, 1). Data type is float.

    Symbols:
        B:  Batchsize, i.e. number of walkers in simulation.
        a:  Number of angles.
        D:  Dimension of the simulation system. Usually is 3.

    Supported Platforms:
        ``Ascend`` ``GPU``
    """

    def __init__(self,
                 index: Tensor = None,
                 force_constant: Tensor = None,
                 bond_angle: Tensor = None,
                 parameters: dict = None,
                 use_pbc: bool = None,
                 energy_unit: str = 'kj/mol',
                 units: Units = None,
                 ):

        super().__init__(
            label='angle_energy',
            output_dim=1,
            use_pbc=use_pbc,
            energy_unit=energy_unit,
            units=units,
        )

        if parameters is not None:
            length_unit = parameters.get('length_unit')
            energy_unit = parameters.get('energy_unit')
            self.units.set_units(length_unit, energy_unit)

            index = parameters.get('index')
            force_constant = parameters.get('force_constant')
            bond_angle = parameters.get('bond_angle')

        # (1,a,3)
        index = Tensor(index, ms.int32)
        if index.shape[-1] != 3:
            raise ValueError('The last dimension of index in AngleEnergy must be 3 but got: ' +
                             str(index.shape[-1]))
        if index.ndim == 2:
            index = F.expand_dims(index, 0)
        if index.ndim != 3:
            raise ValueError('The rank of index must be 2 or 3 but got shape: '+str(index.shape))
        self.index = Parameter(index, name='angle_index', requires_grad=False)

        self.num_angles = index.shape[-2]

        # (1,a)
        force_constant = Tensor(force_constant, ms.float32)
        if force_constant.shape[-1] != self.num_angles:
            raise ValueError('The last shape of force_constant ('+str(force_constant.shape[-1]) +
                             ') must be equal to num_angles ('+str(self.num_angles)+')!')
        if force_constant.ndim == 1:
            force_constant = F.expand_dims(force_constant, 0)
        if force_constant.ndim > 2:
            raise ValueError('The rank of force_constant cannot be larger than 2!')
        self.force_constant = Parameter(force_constant, name='angle_force_constant')

        bond_angle = Tensor(bond_angle, ms.float32)
        if bond_angle.shape[-1] != self.num_angles:
            raise ValueError('The last shape of bond_angle ('+str(bond_angle.shape[-1]) +
                             ') must be equal to num_angles ('+str(self.num_angles)+')!')
        if bond_angle.ndim == 1:
            bond_angle = F.expand_dims(bond_angle, 0)
        if bond_angle.ndim > 2:
            raise ValueError('The rank of bond_angle cannot be larger than 2!')
        self.bond_angle = Parameter(bond_angle, name='bond_angle')

        self.get_angle = AtomAngles(self.index, use_pbc=use_pbc)

    def set_pbc(self, use_pbc=None):
        self.use_pbc = use_pbc
        self.get_angle.set_pbc(use_pbc)
        return self

    def construct(self,
                  coordinate: Tensor,
                  neighbour_index: Tensor = None,
                  neighbour_mask: Tensor = None,
                  neighbour_coord: Tensor = None,
                  neighbour_distance: Tensor = None,
                  inv_neigh_dis: Tensor = None,
                  pbc_box: Tensor = None,
                  ):
        r"""
        Calculate energy term.

        Args:
            coordinate (Tensor):            Tensor of shape (B, A, D). Data type is float.
                                            Position coordinate of atoms in system.
            neighbour_index (Tensor):       Tensor of shape (B, A, N). Data type is int.
                                            Index of neighbour atoms.
            neighbour_mask (Tensor):        Tensor of shape (B, A, N). Data type is bool.
                                            Mask for neighbour index.
            neighbour_coord (Tensor):       Tensor of shape (B, A, N). Data type is bool.
                                            Position coorindates of neighbour atoms.
            neighbour_distance (Tensor):    Tensor of shape (B, A, N). Data type is float.
                                            Distance between neighbours atoms.
            inv_neigh_dis (Tensor):         Tensor of shape (B, A, N). Data type is float.
                                            Reciprocal of distances.
            pbc_box (Tensor):               Tensor of shape (B, D). Data type is float.
                                            Tensor of PBC box. Default: None

        Returns:
            energy (Tensor), Tensor of shape (B, 1). Data type is float.

        Symbols:
            B:  Batchsize, i.e. number of walkers in simulation.
            A:  Number of atoms.
            D:  Dimension of the simulation system. Usually is 3.

        """
        # (B,M)
        theta = self.get_angle(coordinate, pbc_box)
        # (B,M) = (B,M) - (1,M)
        dtheta = theta - self.bond_angle
        dtheta2 = dtheta * dtheta

        # E_angle = 1/2 * k_\theta * (\theta-\theta_0)^2
        # (B,M) = (1,M) * (B,M) * k
        energy = 0.5 * self.force_constant * dtheta2

        # (B,1) <- (B,M)
        return func.keepdim_sum(energy, -1)
