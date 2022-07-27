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
"""Electroinc interaction"""

import mindspore as ms
import mindspore.numpy as msnp
from mindspore import Tensor, Parameter
from mindspore import ms_function
from mindspore import ops
from mindspore.nn import Cell
from mindspore.ops import functional as F

from .energy import NonbondEnergy
from ...function import functions as func
from ...function.functions import gather_values
from ...function.units import Units


@ms_function
def coulomb_law(atom_charge: Tensor,
                neighbour_index: Tensor,
                inv_neigh_dis: Tensor,
                ):
    """calculate Coulomb interaction using Coulomb's law"""
    # (B,A,1)
    qi = F.expand_dims(atom_charge, -1)
    # (B,A,N)
    qj = gather_values(atom_charge, neighbour_index)
    # (B,A,N) = (B,A,1) * (B,A,N)
    qiqj = qi * qj

    energy = qiqj * inv_neigh_dis

    # (B,A)
    energy = F.reduce_sum(energy, -1)
    # (B,1)
    energy = func.keepdim_sum(energy, 1) * 0.5

    return energy


class CoulombEnergy(NonbondEnergy):
    r"""Coulomb interaction

    Math:

        E_ele(r_ij) = \sum_ij k_coulomb * q_i * q_j / r_ij

    Args:

        atom_charge (Tensor):   Tensor of shape (B, A). Data type is float.
                                Atom charge.

        parameters (dict):      Force field parameters. Default: None

        cutoff (float):         Cutoff distance. Default: None

        use_pbc (bool):         Whether to use periodic boundary condition. Default: None

        alpha (float):          Alpha for DSF coulomb interaction. Default: 0.25

        length_unit (str):      Length unit for position coordinates. Default: None

        energy_unit (str):      Energy unit. Default: None

        units (Units):          Units of length and energy. Default: None

    """

    def __init__(self,
                 atom_charge: Tensor = None,
                 parameters: dict = None,
                 cutoff: float = None,
                 use_pbc: bool = None,
                 use_pme: bool = False,
                 alpha: float = 0.25,
                 length_unit: str = 'nm',
                 energy_unit: str = 'kj/mol',
                 units: Units = None,
                 ):

        super().__init__(
            label='coulomb_energy',
            output_dim=1,
            cutoff=cutoff,
            use_pbc=use_pbc,
            length_unit=length_unit,
            energy_unit=energy_unit,
            units=units,
        )

        if parameters is not None:
            length_unit = parameters.get('length_unit')
            energy_unit = parameters.get('energy_unit')
            self.units.set_units(length_unit, energy_unit)

        self.atom_charge = self.identity(atom_charge)
        self.coulomb_const = Tensor(self.units.coulomb, ms.float32)

        self.use_pme = use_pme
        if self.use_pme and (not self.use_pbc):
            raise ValueError('PME cannot be used at vacuum')

        if self.use_pbc and self.cutoff is None:
            raise ValueError(
                'Cutoff cannot be None when using periodic boundary condition')

        self.dsf_coulomb = None
        self.pme_coulomb = None
        if self.use_pbc and self.use_pme:
            self.pme_coulomb = ParticleEeshEwaldCoulomb(self.cutoff)
        elif self.cutoff is not None:
            self.dsf_coulomb = DampedShiftedForceCoulomb(self.cutoff, alpha)

    def construct(self,
                  coordinate: Tensor,
                  neighbour_index: Tensor = None,
                  neighbour_mask: Tensor = None,
                  neighbour_coord: Tensor = None,
                  neighbour_distance: Tensor = None,
                  inv_neigh_dis: Tensor = None,
                  pbc_box: Tensor = None,
                  ):
        r"""Calculate energy term.

        Args:
            coordinate (Tensor):            Tensor of shape (B, A, D). Data type is float.
                                            Position coordinate of atoms in system
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
            energy (Tensor):    Tensor of shape (B, 1). Data type is float.

        Symbols:
            B:  Batchsize, i.e. number of walkers in simulation
            A:  Number of atoms.
            D:  Dimension of the simulation system. Usually is 3.

        """

        inv_neigh_dis *= self.inverse_input_scale

        if self.cutoff is None:
            energy = coulomb_law(
                self.atom_charge, neighbour_index, inv_neigh_dis)
        else:
            neighbour_distance *= self.input_unit_scale
            if self.use_pme:
                energy = self.pme_coulomb(self.atom_charge,
                                          coordinate,
                                          neighbour_index,
                                          neighbour_mask,
                                          neighbour_coord,
                                          neighbour_distance,
                                          inv_neigh_dis,
                                          pbc_box
                                          )
            else:
                energy = self.dsf_coulomb(self.atom_charge,
                                          neighbour_index,
                                          neighbour_distance,
                                          inv_neigh_dis
                                          )

        return energy * self.coulomb_const


class DampedShiftedForceCoulomb(Cell):
    r"""Damped shifted force coulomb potential

    Args:

        atom_charge (Tensor):   Tensor of shape (B, A). Data type is float.
                                Atom charge.

        cutoff (float):         Cutoff distance. Default: None

        alpha (float):          Alpha. Default: 0.25

        use_pbc (bool):         Whether to use periodic boundary condition. Default: None

        length_unit (str):      Length unit for position coordinates. Default: None

        energy_unit (str):      Energy unit. Default: None

        units (Units):          Units of length and energy. Default: None

    """

    def __init__(self,
                 cutoff: float,
                 alpha: float = 0.25,
                 ):

        super().__init__()

        self.cutoff = cutoff
        self.alpha = Parameter(Tensor(alpha, ms.float32),
                               name='alpha', requires_grad=False)

        self.erfc = ops.Erfc()
        cutoffsq = self.cutoff * self.cutoff
        erfcc = self.erfc(self.alpha * self.cutoff)
        erfcd = msnp.exp(-self.alpha * self.alpha * cutoffsq)

        self.f_shift = -(erfcc / cutoffsq + 2 / msnp.sqrt(msnp.pi)
                         * self.alpha * erfcd / self.cutoff)
        self.e_shift = erfcc / self.cutoff - self.f_shift * self.cutoff

    def construct(self,
                  atom_charge: Tensor = None,
                  neighbour_index: Tensor = None,
                  neighbour_distance: Tensor = None,
                  inv_neigh_dis: Tensor = None,
                  ):
        r"""Calculate energy term.

        Args:
            coordinate (Tensor):            Tensor of shape (B, A, D). Data type is float.
                                            Position coordinate of atoms in system
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
            energy (Tensor):    Tensor of shape (B, 1). Data type is float.

        Symbols:
            B:  Batchsize, i.e. number of walkers in simulation
            A:  Number of atoms.
            D:  Dimension of the simulation system. Usually is 3.

        """

        # (B,A,1)
        qi = F.expand_dims(atom_charge, -1)
        # (B,A,N)
        qj = gather_values(atom_charge, neighbour_index)
        # (B,A,N) = (B,A,1) * (B,A,N)
        qiqj = qi*qj
        energy = qiqj * inv_neigh_dis * \
            (self.erfc(self.alpha * neighbour_distance) - neighbour_distance *
             self.e_shift - neighbour_distance * neighbour_distance * self.f_shift)
        energy = msnp.where(neighbour_distance < self.cutoff, energy, 0.0)

        # (B,A)
        energy = F.reduce_sum(energy, -1)
        # (B,1)
        energy = func.keepdim_sum(energy, 1) * 0.5

        return energy


class ParticleEeshEwaldCoulomb(Cell):
    r"""Particle mesh ewald algorithm for electronic interaction

    Args:

        atom_charge (Tensor):   Tensor of shape (B, A). Data type is float.
                                Atom charge.

        cutoff (float):         Cutoff distance. Default: None

        use_pbc (bool):         Whether to use periodic boundary condition. Default: None

        length_unit (str):      Length unit for position coordinates. Default: None

        energy_unit (str):      Energy unit. Default: None

        units (Units):          Units of length and energy. Default: None

    """

    def __init__(self,
                 cutoff: float,
                 ):

        super().__init__()

        self.cutoff = cutoff

    def construct(self,
                  coordinate: Tensor,
                  neighbour_index: Tensor = None,
                  neighbour_mask: Tensor = None,
                  neighbour_coord: Tensor = None,
                  neighbour_distance: Tensor = None,
                  inv_neigh_dis: Tensor = None,
                  pbc_box: Tensor = None,
                  ):
        r"""Calculate energy term.

        Args:
            coordinate (Tensor):            Tensor of shape (B, A, D). Data type is float.
                                            Position coordinate of atoms in system
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
            energy (Tensor):    Tensor of shape (B, 1). Data type is float.

        Symbols:
            B:  Batchsize, i.e. number of walkers in simulation
            A:  Number of atoms.
            D:  Dimension of the simulation system. Usually is 3.

        """

        # TODO
        raise NotImplementedError
