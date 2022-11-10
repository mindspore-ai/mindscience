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
"""Bond energy"""

import mindspore as ms
from mindspore import Tensor
from mindspore import Parameter
from mindspore.ops import functional as F

from .energy import EnergyCell
from ...colvar import AtomDistances
from ...function import functions as func
from ...function.units import Units


class BondEnergy(EnergyCell):
    r"""
    Energy term of bond length.

    .. Math::

        E_{bond}(b_{ij}) = 1 / 2 * k_{ij}^b * (b_{ij} - b_{ij}^0) ^ 2

    Args:
        index (Tensor):             Tensor of shape (B, b, 2). Data type is int.
                                    Atom index of bond.
        force_constant (Tensor):    Tensor of shape (1, b). Data type is float.
                                    The harmonic force constants of bond length (k^b).
        bond_length (Tensor):       Tensor of shape (1, b). Data type is float.
                                    The equilibrium value of bond length (b^0).
        parameters (dict):          Force field parameters. Default: None
        use_pbc (bool):             Whether to use periodic boundary condition.
        length_unit (str):          Length unit for position coordinates. Default: 'nm'
        energy_unit (str):          Energy unit. Default: 'kj/mol'
        units (Units):              Units of length and energy. Default: None

    Returns:
        energy (Tensor), Tensor of shape (B, 1). Data type is float.

    Symbols:
        B:  Batchsize, i.e. number of walkers in simulation.
        b:  Number of bonds.
        D:  Dimension of the simulation system. Usually is 3.

    Supported Platforms:
        ``Ascend`` ``GPU``
    """

    def __init__(self,
                 index: Tensor = None,
                 force_constant: Tensor = None,
                 bond_length: Tensor = None,
                 parameters: dict = None,
                 use_pbc: bool = None,
                 length_unit: str = 'nm',
                 energy_unit: str = 'kj/mol',
                 units: Units = None,
                 ):

        super().__init__(
            label='bond_energy',
            output_dim=1,
            use_pbc=use_pbc,
            length_unit=length_unit,
            energy_unit=energy_unit,
            units=units,
        )

        if parameters is not None:
            length_unit = parameters.get('length_unit')
            energy_unit = parameters.get('energy_unit')
            self.units.set_units(length_unit, energy_unit)

            index = parameters.get('index')
            force_constant = parameters.get('force_constant')
            bond_length = parameters.get('bond_length')

        # (B,b,2)
        index = Tensor(index, ms.int32)
        if index.shape[-1] != 2:
            raise ValueError('The last dimension of index in BondEnergy must be 2 but got: ' +
                             str(index.shape[-1]))
        if index.ndim == 2:
            index = F.expand_dims(index, 0)
        if index.ndim != 3:
            raise ValueError('The rank of index must be 2 or 3 but got shape: '+str(index.shape))
        self.index = Parameter(index, name='bond_index', requires_grad=False)

        # (B,b)
        self.get_bond_length = AtomDistances(self.index, use_pbc=use_pbc, length_unit=self.units)

        # b
        self.num_bonds = index.shape[-2]

        # (B,b)
        force_constant = Tensor(force_constant, ms.float32)
        if force_constant.shape[-1] != self.num_bonds:
            raise ValueError('The last shape of force_constant ('+str(force_constant.shape[-1]) +
                             ') must be equal to num_bonds ('+str(self.num_bonds)+')!')
        if force_constant.ndim == 1:
            force_constant = F.expand_dims(force_constant, 0)
        if force_constant.ndim > 2:
            raise ValueError('The rank of force_constant cannot be larger than 2!')
        self.force_constant = Parameter(force_constant, name='bond_force_constant')

        bond_length = Tensor(bond_length, ms.float32)
        if bond_length.shape[-1] != self.num_bonds:
            raise ValueError('The last shape of bond_length ('+str(bond_length.shape[-1]) +
                             ') must be equal to num_bonds ('+str(self.num_bonds)+')!')
        if bond_length.ndim == 1:
            bond_length = F.expand_dims(bond_length, 0)
        if bond_length.ndim > 2:
            raise ValueError('The rank of bond_length cannot be larger than 2!')
        self.bond_length = Parameter(bond_length, name='bond_length')

    def set_pbc(self, use_pbc=None):
        self.use_pbc = use_pbc
        self.get_bond_length.set_pbc(use_pbc)
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

        # (B,b)
        dis = self.get_bond_length(coordinate, pbc_box) * self.input_unit_scale

        # (B,b) = (B,b) - (B,b)
        diff = dis - self.bond_length
        # (B,b)
        diff2 = F.square(diff)

        # (B,b) = (1,b) * (B,b) * k
        energy = 0.5 * self.force_constant * diff2

        # (B,1) <- (B,b)
        return func.keepdim_sum(energy, -1)
