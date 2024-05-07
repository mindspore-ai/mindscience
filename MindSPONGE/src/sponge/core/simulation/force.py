# Copyright 2021-2023 @ Shenzhen Bay Laboratory &
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
WithEnergyCell
"""

from typing import Tuple
import mindspore as ms
from mindspore import Tensor
from mindspore import ops
from mindspore.nn import Cell

from ...partition import NeighbourList
from ...system import Molecule
from ...potential import ForceCell
from ...sampling.modifier import ForceModifier


class WithForceCell(Cell):
    r"""Cell that wraps the simulation system with the atomic force function.

    Args:
        system (:class:`sponge.system.Molecule`): Simulation system.
        force (`sponge.potential.ForceCell`): Atomic force calculation cell.
        neighbour_list (:class:`sponge.partition.NeighbourList`, optional): Neighbour list.
          Default: ``None``.
        modifier (`sponge.sampling.modifier.ForceModifier`, optional): Force modifier.
          Default: ``None``.

    Inputs:
        - **energy** (Tensor) - Total potential energy of the simulation system.
          Tensor of shape :math:`(B, 1)`.
          Here `B` is batch size, i.e. the number of walkers in simulation. Data type is float.
        - **force** (Tensor) - Data type is float.Force on each atoms of the simulation system.
          Tensor of shape :math:`(B, A, D)`.
          Here :math:`B` is batch size, i.e. the number of walkers in simulation,
          `A` is the number of atoms,
          and :math:`D` is the spatial dimension of the simulation system, which is usually 3.
        - **virial** (Tensor) - Virial tensor of the simulation system.
          Tensor of shape :math:`(B, D)`. Data type is float.

    Outputs:
        - **energy** (Tensor) - with shape of :math:`(B, 1)`. Total potential energy of the simulation system.
          Data type is float.
        - **force** (Tensor) - with shape of :math:`(B, A, D)`. Force on each atoms of the simulation system.
          Data type is float.
        - **virial** (Tensor) - with shape of :math:`(B, D)`. Virial tensor of the simulation system.
          Data type is float.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> # You can find case2.pdb file under MindSPONGE/tutorials/basic/case2.pdb
        >>> from sponge import Protein
        >>> from sponge.potential.forcefield import ForceField
        >>> from sponge.partition import NeighbourList
        >>> from sponge.core.simulation import WithEnergyCell, WithForceCell
        >>> from sponge.sampling import MaskedDriven
        >>> system = Protein(pdb='case2.pdb', rebuild_hydrogen=True)
        >>> energy = ForceField(system, 'AMBER.FF99SB')
        >>> neighbour_list = NeighbourList(system, cutoff=None, cast_fp16=True)
        >>> with_energy = WithEnergyCell(system, energy, neighbour_list=neighbour_list)
        >>> modifier = MaskedDriven(length_unit=with_energy.length_unit,
        ...                         energy_unit=with_energy.energy_unit,
        ...                         mask=system.heavy_atom_mask)
        >>> with_force = WithForceCell(system, neighbour_list=neighbour_list, modifier=modifier)
    """

    def __init__(self,
                 system: Molecule = None,
                 force: ForceCell = None,
                 neighbour_list: NeighbourList = None,
                 modifier: ForceModifier = None,
                 ):

        super().__init__(auto_prefix=False)

        self.system = system
        self.force_function = force

        length_unit = None
        if self.system is not None:
            length_unit = self.system.length_unit

        energy_unit = None
        if self.force_function is not None:
            energy_unit = self.force_function.energy_unit

        self.num_walker = 1
        self.num_atoms = 0

        self.coordinate = None
        self.pbc_box = None
        self.atom_mass = None

        self.pbc_box = None

        self.force_modifier = modifier
        if modifier is None:
            self.force_modifier = ForceModifier(length_unit=length_unit, energy_unit=energy_unit)
        self.modifier_pace = self.force_modifier.update_pace

        self.units = None
        self.length_unit_scale = 1
        self.energy_unit_scale = 1
        self.force_unit_scale = 1

        self.neighbour_list = None
        self.neighbour_index = None
        self.neighbour_mask = None
        self.num_neighbours = 0
        if self.force_function is not None:
            if self.system is None:
                raise ValueError('system cannot be None if force is not None!')

            self.num_walker = self.system.num_walker
            self.num_atoms = self.system.num_atoms

            self.coordinate = self.system.coordinate
            self.pbc_box = self.system.pbc_box
            self.atom_mass = self.system.atom_mass

            self.pbc_box = self.system.pbc_box

            self.units = self.system.units

            self.neighbour_list = neighbour_list
            self.neighbour_index = None
            self.neighbour_mask = None
            self.num_neighbours = 0
            if neighbour_list is not None:
                self.neighbour_list.set_exclude_index(self.force_function.exclude_index)
                self.neighbour_index = self.neighbour_list.neighbours
                self.neighbour_mask = self.neighbour_list.neighbour_mask
                self.num_neighbours = self.neighbour_list.num_neighbours

            self.force_function.set_pbc(self.pbc_box is not None)

            self.length_unit_scale = Tensor(self.units.convert_length_to(
                self.force_function.length_unit), ms.float32)
            self.energy_unit_scale = Tensor(self.units.convert_energy_from(
                self.force_function.energy_unit), ms.float32)
            self.force_unit_scale = self.energy_unit_scale / self.length_unit_scale

            if self.cutoff is not None:
                self.force_function.set_cutoff(self.cutoff)

            for p in self.force_function.trainable_params():
                p.requires_grad = False

        self.identity = ops.Identity()

    @property
    def cutoff(self) -> Tensor:
        r"""Cutoff distance for neighbour list

        Returns:
            Tensor, cutoff
        """
        if self.neighbour_list is None:
            return None
        return self.neighbour_list.cutoff

    @property
    def neighbour_list_pace(self) -> int:
        r"""Update step for neighbour list

        Returns:
            int, step
        """
        if self.neighbour_list is None:
            return 0
        return self.neighbour_list.pace

    @property
    def length_unit(self) -> str:
        r"""Length unit

        Returns:
            str, length unit
        """
        return self.units.length_unit

    @property
    def energy_unit(self) -> str:
        r"""Energy unit

        Returns:
            str, energy unit
        """
        return self.units.energy_unit

    def set_pbc_grad(self, grad_box: bool):
        r"""Set whether to calculate the gradient of PBC box

        Args:
            grad_box (bool):    Whether to calculate the gradient of PBC box.
        """
        self.system.set_pbc_grad(grad_box)
        return self

    def update_modifier(self, step: int):
        r"""Update force modifier

        Args:
            step (int): Simulatio step.
        """
        if self.modifier_pace > 0 and step % self.modifier_pace == 0:
            self.force_modifier.update()
        return self

    def update_neighbour_list(self) -> Tuple[Tensor, Tensor]:
        r"""Update neighbour list

        Args:
            coordinate (Tensor): Position coordinate.
              Tensor of shape :math:`(B, A, D)`.
              Here :math:`B` is the number of walkers in simulation,
              :math:`A` is the number of atoms,
              :math:`D` is the spatial dimension of the simulation system,
              which is usually 3.
              Data type is float.
            pbc_box (Tensor): Size of PBC box.
              Tensor of shape :math:`(B, D)`.
              Data type is float.

        Returns:
            - neigh_idx, Tensor of shape :math:`(B, A, N)`.
              Index of neighbouring atoms of each atoms in system.
              Here :math:`N` is the number of neighbouring atoms. Data type is int.
            - neigh_mask, Tensor of shape :math:`(B, A, N)`.
              Mask for neighbour list `neigh_idx`.
              Data type is bool.
        """
        return self.neighbour_list.update(self.coordinate, self.pbc_box)

    def get_neighbour_list(self) -> Tuple[Tensor, Tensor]:
        r"""Get neighbour list

        Returns:
            - neigh_idx, Tensor of shape :math:`(B, A, N)`.
              Index of neighbouring atoms of each atoms in system.
              Here :math:`B` is the number of walkers in simulation,
              `A` is the number of atoms,
              :math:`N` is the number of neighbouring atoms.
              Data type is int.
            - neigh_mask, Tensor of shape :math:`(B, A, N)`. Mask for neighbour list `neigh_idx`.
              Data type is bool.
        """
        return self.neighbour_list.get_neighbour_list()

    def construct(self,
                  energy: Tensor = 0,
                  force: Tensor = 0,
                  virial: Tensor = None,
                  ) -> Tuple[Tensor, Tensor, Tensor]:

        r"""
        Calculate the energy of system

        Args:
            energy_ad (Tensor): Potential energy for automatic differentiation.
              Tensor of shape :math:`(B, 1)`.
              Here :math:`B` is the number of walkers in simulation.
              Data type is float.
            force_ad (Tensor): Atomic forces from automatic differentiation.
              Tensor of shape :math:`(B, A, D)`.
              Here :math:`A` is the number of atoms,
              :math: `D` is the spatial dimension of the simulation system, which is usually 3.
              Data type is float.

        Returns:
            - energy, Tensor of shape :math:`(B, 1)`. Potential energy of the system.
              Data type is float.
            - force, Tensor of shape :math:`(B, A, D)`. Atomic forces. Data type is float.
            - virial, Tensor of shape :math:`(B, D)`. Virial tensor of the
              system. Data type is float.
        """

        energy_ad = energy
        force_ad = force
        virial_ad = virial

        if self.force_function is not None:
            coordinate, pbc_box = self.system()

            coordinate *= self.length_unit_scale
            if pbc_box is not None:
                pbc_box *= self.length_unit_scale

            neigh_idx = None
            neigh_vec = None
            neigh_dis = None
            neigh_mask = None
            if self.neighbour_list is not None:
                neigh_idx, neigh_vec, neigh_dis, neigh_mask = self.neighbour_list(coordinate, pbc_box)

            energy, force, virial = self.force_function(
                coordinate=coordinate,
                neighbour_index=neigh_idx,
                neighbour_mask=neigh_mask,
                neighbour_vector=neigh_vec,
                neighbour_distance=neigh_dis,
                pbc_box=pbc_box
            )

            energy *= self.energy_unit_scale
            force *= self.force_unit_scale
            if virial is not None:
                virial *= self.energy_unit_scale

        energy, force, virial = self.force_modifier(energy, energy_ad, force, force_ad, virial, virial_ad)

        return energy, force, virial
