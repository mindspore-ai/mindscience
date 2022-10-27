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
"""Torsion energy"""

import mindspore as ms
from mindspore import Tensor
from mindspore.ops import functional as F
from mindspore import Parameter

from .energy import EnergyCell
from ...colvar import AtomTorsions
from ...function import functions as func
from ...function.units import Units


class DihedralEnergy(EnergyCell):
    r"""
    Energy term of dihedral (torsion) angles.

    .. Math::

        E_{dihedral}(\omega) = \sum_n 1 / 2 \times V_n \times [1 - cos(n \times \omega - {\gamma}_n)]

    Args:
        index (Tensor):             Tensor of shape (B, d, 4) or (1, d, 4). Data type is int.
                                    Atom index of dihedral angles.
        force_constant (Tensor):    Tensor of shape (B, d) or (1, d). Data type is float.
                                    The harmonic force constants of bond torsional angle (V_n).
        periodicity (Tensor):       Tensor of shape (B, d) or (1, d). Data type is int.
                                    The periodicity of the torsional barrier (n).
        phase (Tensor):             Tensor of shape (B, d) or (1, d). Data type is float.
                                    The phase shift in the torsional function ({\gamma}_n).
        parameters (dict):          Force field parameters. Default: None
        use_pbc (bool):             Whether to use periodic boundary condition. Default: None
        energy_unit (str):          Energy unit. Default: 'kj/mol'
        units (Units):              Units of length and energy. Default: None

    Returns:
        energy (Tensor), Tensor of shape (B, 1). Data type is float.

    Symbols:
        B:  Batchsize, i.e. number of walkers in simulation.
        d:  Number of dihedral angles.
        D:  Dimension of the simulation system. Usually is 3.

    Supported Platforms:
        ``Ascend`` ``GPU``
    """
    def __init__(self,
                 index: Tensor = None,
                 force_constant: Tensor = None,
                 periodicity: Tensor = None,
                 phase: Tensor = None,
                 parameters: dict = None,
                 use_pbc: bool = None,
                 energy_unit: str = 'kj/mol',
                 units: Units = None,
                 ):

        super().__init__(
            label='dihedral_energy',
            output_dim=1,
            use_pbc=use_pbc,
            energy_unit=energy_unit,
            units=units,
        )

        if parameters is not None:
            energy_unit = parameters.get('energy_unit')
            self.units.set_energy_unit(energy_unit)

            index = parameters.get('index')
            force_constant = parameters.get('force_constant')
            periodicity = parameters.get('periodicity')
            phase = parameters.get('phase')

        # (1,d,4)
        index = Tensor(index, ms.int32)
        if index.shape[-1] != 4:
            raise ValueError('The last dimension of index in DihedralEnergy must be 2 but got: ' +
                             str(index.shape[-1]))
        if index.ndim == 2:
            index = F.expand_dims(index, 0)
        if index.ndim != 3:
            raise ValueError(
                'The rank of index must be 2 or 3 but got shape: '+str(index.shape))
        self.index = Parameter(index, name='dihedral_index', requires_grad=False)

        # (1,d)
        self.get_torsion = AtomTorsions(self.index, use_pbc=use_pbc)

        # d
        self.num_torsions = index.shape[-2]

        # (1,d)
        force_constant = Tensor(force_constant, ms.float32)
        if force_constant.shape[-1] != self.num_torsions:
            raise ValueError('The last shape of force_constant ('+str(force_constant.shape[-1]) +
                             ') must be equal to num_torsions ('+str(self.num_torsions)+')!')
        if force_constant.ndim == 1:
            force_constant = F.expand_dims(force_constant, 0)
        if force_constant.ndim > 2:
            raise ValueError('The rank of force_constant cannot be larger than 2!')
        self.force_constant = Parameter(force_constant, name='dihedral_force_constant')

        periodicity = Tensor(periodicity, ms.int32)
        if periodicity.shape[-1] != self.num_torsions:
            raise ValueError('The last shape of periodicity ('+str(periodicity.shape[-1]) +
                             ') must be equal to num_torsions ('+str(self.num_torsions)+')!')
        if periodicity.ndim == 1:
            periodicity = F.expand_dims(periodicity, 0)
        if periodicity.ndim > 2:
            raise ValueError('The rank of periodicity cannot be larger than 2!')
        self.periodicity = Parameter(periodicity, name='periodicity')

        phase = Tensor(phase, ms.float32)
        if phase.shape[-1] != self.num_torsions:
            raise ValueError('The last shape of phase ('+str(phase.shape[-1]) +
                             ') must be equal to num_torsions ('+str(self.num_torsions)+')!')
        if phase.ndim == 1:
            phase = F.expand_dims(phase, 0)
        if phase.ndim > 2:
            raise ValueError('The rank of phase cannot be larger than 2!')
        self.dihedral_phase = Parameter(phase, name='phase')

    def set_pbc(self, use_pbc=None):
        self.use_pbc = use_pbc
        self.get_torsion.set_pbc(use_pbc)
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
        phi = self.get_torsion(coordinate, pbc_box)

        # (B,M) = (1,M) * (B,M)
        nphi = self.periodicity * phi

        # (B,M)
        cosphi = F.cos(nphi - self.dihedral_phase) + 1

        # (B,M) = (1,M) + (B,M)
        energy = 0.5 * self.force_constant * cosphi

        # (B,1) <- (B,M)
        energy = func.keepdim_sum(energy, -1)

        return energy
