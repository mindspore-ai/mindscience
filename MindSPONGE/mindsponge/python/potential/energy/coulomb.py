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
def coulomb_interaction(qi: Tensor, qj: Tensor, inv_dis: Tensor, mask: Tensor = None):
    """calculate Coulomb interaction using Coulomb's law"""

    # (B,A,N) = (B,A,1) * (B,A,N)
    qiqj = qi * qj

    # (B,A,N)
    energy = qiqj * inv_dis

    if mask is not None:
        # (B,A,N) * (B,A,N)
        energy *= mask

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

        self.pme_coulomb = None
        self.dsf_coulomb = None
        if self.use_pme:
            self.pme_coulomb = ParticleEeshEwaldCoulomb(self.cutoff)
        else:
            self.dsf_coulomb = DampedShiftedForceCoulomb(self.cutoff, alpha)

    def set_cutoff(self, cutoff: Tensor):
        """set cutoff distance"""
        if cutoff is None:
            if self.use_pbc:
                raise ValueError('cutoff cannot be none when using periodic boundary condition')
            self.cutoff = None
        else:
            self.cutoff = Tensor(cutoff, ms.float32)
            if self.dsf_coulomb is not None:
                self.dsf_coulomb.set_cutoff(cutoff)
            if self.pme_coulomb is not None:
                self.pme_coulomb.set_cutoff(cutoff)
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

        # (B,A,1)
        qi = F.expand_dims(self.atom_charge, -1)
        # (B,A,N)
        qj = gather_values(self.atom_charge, neighbour_index)

        if self.cutoff is None:
            energy = coulomb_interaction(qi, qj, inv_neigh_dis, neighbour_mask)
        else:
            neighbour_distance *= self.input_unit_scale
            if self.use_pme:
                energy = self.pme_coulomb(
                    qi, qj, neighbour_distance, inv_neigh_dis, neighbour_mask, pbc_box)
            else:
                energy = self.dsf_coulomb(
                    qi, qj, neighbour_distance, inv_neigh_dis, neighbour_mask)

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
                 cutoff: float = None,
                 alpha: float = 0.25,
                 ):

        super().__init__()

        self.alpha = Parameter(Tensor(alpha, ms.float32), name='alpha', requires_grad=False)

        self.erfc = ops.Erfc()
        self.f_shift = None
        self.e_shift = None
        if cutoff is not None:
            self.set_cutoff(cutoff)

    def set_cutoff(self, cutoff: Tensor):
        """set cutoff distance"""
        self.cutoff = Tensor(cutoff, ms.float32)
        cutoff2 = F.square(self.cutoff)
        erfcc = self.erfc(self.alpha * self.cutoff)
        erfcd = msnp.exp(-F.square(self.alpha) * cutoff2)

        self.f_shift = -(erfcc / cutoff2 + 2 / msnp.sqrt(msnp.pi)
                         * self.alpha * erfcd / self.cutoff)
        self.e_shift = erfcc / self.cutoff - self.f_shift * self.cutoff

    def construct(self,
                  qi: Tensor,
                  qj: Tensor,
                  dis: Tensor,
                  inv_dis: Tensor,
                  mask: Tensor = None,
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

        # (B,A,N) = (B,A,1) * (B,A,N)
        qiqj = qi*qj
        energy = qiqj * inv_dis * (self.erfc(self.alpha * dis) -
                                   dis * self.e_shift - F.square(dis) * self.f_shift)

        if mask is None:
            mask = dis < self.cutoff
        else:
            mask = F.logical_and(mask, dis < self.cutoff)

        energy = msnp.where(mask, energy, 0.0)

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

    def set_cutoff(self, cutoff: Tensor):
        """set cutoff distance"""
        self.cutoff = Tensor(cutoff, ms.float32)

    def construct(self,
                  qi: Tensor,
                  qj: Tensor,
                  dis: Tensor,
                  inv_dis: Tensor,
                  mask: Tensor = None,
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
