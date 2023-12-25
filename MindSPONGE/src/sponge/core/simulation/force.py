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
        system (Molecule):              Simulation system.

        force (PotentialCell):          Atomic force calculation cell.

        cutoff (float):                 Cutoff distance. Default: ``None``.

        neighbour_list (NeighbourList): Neighbour list. Default: ``None``.

        modifier (ForceModifier):       Force modifier. Default: ``None``.

        bias (Bias):                    Bias potential: Default: ``None``.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Note:

        B:  Batchsize, i.e. number of walkers of the simulation.

        A:  Number of the atoms in the simulation system.

        D:  Spatial dimension of the simulation system. Usually is 3.

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
        r"""cutoff distance for neighbour list

        Returns:
            Tensor, cutoff

        """
        if self.neighbour_list is None:
            return None
        return self.neighbour_list.cutoff

    @property
    def neighbour_list_pace(self) -> int:
        r"""update step for neighbour list

        Returns:
            int, step

        """
        if self.neighbour_list is None:
            return 0
        return self.neighbour_list.pace

    @property
    def length_unit(self) -> str:
        r"""length unit

        Returns:
            str, length unit

        """
        return self.units.length_unit

    @property
    def energy_unit(self) -> str:
        r"""energy unit

        Returns:
            str, energy unit

        """
        return self.units.energy_unit

    def set_pbc_grad(self, grad_box: bool):
        r"""set whether to calculate the gradient of PBC box

        Args:
            grad_box (bool):    Whether to calculate the gradient of PBC box.

        """
        self.system.set_pbc_grad(grad_box)
        return self

    def update_modifier(self, step: int):
        r"""update force modifier

        Args:
            step (int): Simulatio step.

        """
        if self.modifier_pace > 0 and step % self.modifier_pace == 0:
            self.force_modifier.update()
        return self

    def update_neighbour_list(self) -> Tuple[Tensor, Tensor]:
        r"""update neighbour list

        Args:
            coordinate (Tensor):    Tensor of shape `(B, A, D)`. Data type is float.
                                    Position coordinate.
            pbc_box (Tensor):       Tensor of shape `(B, D)`. Data type is float.
                                    Size of PBC box.

        Returns:
            neigh_idx (Tensor):     Tensor of shape `(B, A, N)`. Data type is int.
                                    Index of neighbouring atoms of each atoms in system.
            neigh_mask (Tensor):    Tensor of shape `(B, A, N)`. Data type is bool.
                                    Mask for neighbour list `neigh_idx`.
        """
        return self.neighbour_list.update(self.coordinate, self.pbc_box)

    def get_neighbour_list(self) -> Tuple[Tensor, Tensor]:
        r"""get neighbour list

        Returns:
            neigh_idx (Tensor):     Tensor of shape `(B, A, N)`. Data type is int.
                                    Index of neighbouring atoms of each atoms in system.
            neigh_mask (Tensor):    Tensor of shape `(B, A, N)`. Data type is bool.
                                    Mask for neighbour list `neigh_idx`.

        Note:
            B:  Batchsize, i.e. number of walkers of the simulation.
            A:  Number of the atoms in the simulation system.
            N:  Number of the maximum neighbouring atoms.

        """
        return self.neighbour_list.get_neighbour_list()

    def construct(self,
                  energy: Tensor = 0,
                  force: Tensor = 0,
                  virial: Tensor = None,
                  ) -> Tuple[Tensor, Tensor, Tensor]:

        """calculate the energy of system

        Args:
            energy_ad (Tensor): Tensor of shape (B, 1). Data type is float.
                                Potential energy for automatic differentiation.
            force_ad (Tensor):  Tensor of shape (B, A, D). Data type is float.
                                Atomic forces from automatic differentiation.

        Returns:
            energy (Tensor):    Tensor of shape (B, 1). Data type is float.
            force (Tensor):     Tensor of shape (B, A, D). Data type is float.
            virial (Tensor):    Tensor of shape (B, D). Data type is float.

        Note:
            B:  Batchsize, i.e. number of walkers of the simulation.
            A:  Number of the atoms in the simulation system.
            D:  Spatial dimension of the simulation system. Usually is 3.

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
