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

from typing import Union, List, Tuple
import mindspore as ms
import mindspore.numpy as msnp
from mindspore import Tensor
from mindspore import Parameter
from mindspore import ops
from mindspore.ops import functional as F
from mindspore.nn import Cell, CellList

from ...function.units import Units
from ...partition import NeighbourList
from ...system import Molecule
from ...potential import PotentialCell
from ...potential.bias import Bias
from ...sampling.wrapper import EnergyWrapper


class WithEnergyCell(Cell):
    r"""Cell that wraps the simulation system with the potential energy function.

        This Cell calculates the value of the potential energy of the system at the current coordinates and returns it.

    Args:

        system (Molecule):              Simulation system.

        potential (PotentialCell):      Potential energy function cell.

        bias (Union[Bias, List[Bias]]): Bias potential function cell: Default: None

        cutoff (float):                 Cut-off distance for neighbour list. If None is given, it will be assigned
                                        as the cutoff value of the of potential energy.
                                        Defulat: None

        neighbour_list (NeighbourList): Neighbour list. Default: None

        wrapper (EnergyWrapper):        Network to wrap and process potential and bias.
                                        Default: None

    Supported Platforms:

        ``Ascend`` ``GPU``


    Symbols:

        B:  Batchsize, i.e. number of walkers of the simulation.

        A:  Number of the atoms in the simulation system.

        N:  Number of the maximum neighbouring atoms.

        U:  Number of potential energy terms.

        V:  Number of bias potential terms.

    """

    def __init__(self,
                 system: Molecule,
                 potential: PotentialCell,
                 bias: Union[Bias, List[Bias]] = None,
                 cutoff: float = None,
                 neighbour_list: NeighbourList = None,
                 wrapper: EnergyWrapper = None,
                 ):

        super().__init__(auto_prefix=False)

        self.system = system
        self.potential_function = potential

        self.units = Units(self.system.length_unit, self.potential_function.energy_unit)
        self.system.units.set_energy_unit(self.energy_unit)

        self.bias_function = None
        self._num_biases = 0
        self._bias_names = []
        if bias is not None:
            if isinstance(bias, list):
                self._num_biases = len(bias)
                self.bias_function = CellList(bias)
            elif isinstance(bias, Cell):
                self._num_biases = 1
                self.bias_function = CellList([bias])
            else:
                raise TypeError(f'The "bias" must be Cell or list but got: {type(bias)}')

            for i in range(self._num_biases):
                self._bias_names.append(self.bias_function[i].name)

        self.num_walker = self.system.num_walker
        self.num_atoms = self.system.num_atoms

        self.energy_wrapper = wrapper
        if wrapper is None:
            self.energy_wrapper = EnergyWrapper(length_unit=self.length_unit, energy_unit=self.energy_unit)

        self.wrapper_pace = self.energy_wrapper.update_pace

        self.exclude_index = self.potential_function.exclude_index
        self.neighbour_list = neighbour_list
        if neighbour_list is None:
            if cutoff is None and self.potential_function.cutoff is not None:
                cutoff = self.units.length(self.potential_function.cutoff, self.potential_function.length_unit)
            self.neighbour_list = NeighbourList(
                system, cutoff, exclude_index=self.exclude_index, length_unit=self.length_unit)
        else:
            self.neighbour_list.set_exclude_index(self.exclude_index)

        self.neighbour_index = self.neighbour_list.neighbours
        self.neighbour_mask = self.neighbour_list.neighbour_mask
        self.num_neighbours = self.neighbour_list.num_neighbours

        if self.neighbour_list.cutoff is not None:
            if self.potential_function.cutoff is None:
                self.potential_function.set_cutoff(self.neighbour_list.cutoff, self.length_unit)
            elif self.potential_function.cutoff > self.neighbour_list.cutoff:
                raise ValueError(f'The cutoff of the potential function {self.potential_function.cutoff} '
                                 f'cannot be greater than '
                                 f'the cutoff of the neighbour list {self.neighbour_list.cutoff}.')

        self.coordinate = self.system.coordinate
        self.pbc_box = self.system.pbc_box
        self.atom_mass = self.system.atom_mass

        self.pbc_box = self.system.pbc_box

        self.potential_function.set_pbc(self.pbc_box is not None)

        for p in self.potential_function.trainable_params():
            p.requires_grad = False

        self.potential_function_units = self.potential_function.units

        self.input_unit_scale = Tensor(self.units.convert_length_to(
            self.potential_function.length_unit), ms.float32)
        self.output_unit_scale = Tensor(self.units.convert_energy_from(
            self.potential_function.energy_unit), ms.float32)

        self.identity = ops.Identity()

        self._energies = Parameter(msnp.zeros((self.num_walker, self.num_energies),
                                              dtype=ms.float32), name='energies', requires_grad=False)

        bias = msnp.zeros((self.num_walker, 1), dtype=ms.float32)
        if self.bias_function is None:
            self._biases = None
            self._bias = bias
        else:
            self._biases = Parameter(msnp.zeros((self.num_walker, self._num_biases),
                                                dtype=ms.float32), name='biases', requires_grad=False)
            self._bias = Parameter(bias, name='bias', requires_grad=False)

    @property
    def cutoff(self) -> Tensor:
        r"""cutoff distance for neighbour list

        Return:
            Tensor, cutoff

        """
        return self.neighbour_list.cutoff

    @property
    def neighbour_list_pace(self) -> int:
        r"""update step for neighbour list

        Return:
            int, steps

        """
        return self.neighbour_list.pace

    @property
    def length_unit(self) -> str:
        r"""length unit

        Return:
            str, length unit

        """
        return self.units.length_unit

    @property
    def energy_unit(self) -> str:
        r"""energy unit

        Return:
            str, energy unit

        """
        return self.units.energy_unit

    @property
    def num_energies(self) -> int:
        r"""number of energy terms :math:`U`

        Return:
            int, number of energy terms

        """
        return self.potential_function.num_energies

    @property
    def num_biases(self) -> int:
        r"""number of bias potential energies :math:`V`

        Return:
            int, number of bias potential energies

        """
        return self._num_biases

    @property
    def energy_names(self) -> list:
        r"""names of energy terms

        Return:
            list of str, names of energy terms

        """
        return self.potential_function.energy_names

    @property
    def bias_names(self) -> list:
        r"""name of bias potential energies

        Return:
            list of str, the bias potential energies

        """
        return self._bias_names

    @property
    def energies(self) -> Tensor:
        r"""Tensor of potential energy components.

        Return:
            energies(Tensor):   Tensor of shape `(B, U)`. Data type is float.

        """
        return self.identity(self._energies)

    @property
    def biases(self) -> Tensor:
        r"""Tensor of bias potential components.

        Return:
            biases(Tensor): Tensor of shape `(B, V)`. Data type is float.

        """
        if self.bias_function is None:
            return None
        return self.identity(self._biases)

    @property
    def bias(self) -> Tensor:
        r"""Tensor of the total bias potential.

        Return:
            bias(Tensor): Tensor of shape `(B, 1)`. Data type is float.

        """
        return self.identity(self._bias)

    def bias_pace(self, index: int = 0) -> int:
        """return the update freqenucy for bias potential

        Args:
            index (int):    Index of bias potential

        Returns:
            update_pace (int):  Update freqenucy

        """
        return self.bias_function[index].update_pace

    def set_pbc_grad(self, grad_box: bool):
        r"""set whether to calculate the gradient of PBC box

        Args:
            grad_box (bool):    Whether to calculate the gradient of PBC box.

        """
        self.system.set_pbc_grad(grad_box)
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

    def update_bias(self, step: int):
        r"""update bias potential

        Args:
            step (int): Simulatio step.

        """
        if self.bias_function is not None:
            for i in range(self._num_biases):
                if self.bias_pace(i) > 0 and step % self.bias_pace(i) == 0:
                    self.bias_function[i].update(self.coordinate, self.pbc_box)
        return self

    def update_wrapper(self, step: int):
        r"""update energy wrapper

        Args:
            step (int): Simulatio step.

        """
        if self.wrapper_pace > 0 and step % self.wrapper_pace == 0:
            self.energy_wrapper.update()
        return self

    def get_neighbour_list(self) -> Tuple[Tensor, Tensor]:
        r"""get neighbour list

        Returns:
            neigh_idx (Tensor):     Tensor of shape `(B, A, N)`. Data type is int.
                                    Index of neighbouring atoms of each atoms in system.
            neigh_mask (Tensor):    Tensor of shape `(B, A, N)`. Data type is bool.
                                    Mask for neighbour list `neigh_idx`.

        Symbols:
            B:  Batchsize, i.e. number of walkers of the simulation.
            A:  Number of the atoms in the simulation system.
            N:  Number of the maximum neighbouring atoms.

        """
        return self.neighbour_list.get_neighbour_list()

    def calc_energies(self) -> Tensor:
        """calculate the energy terms of the potential energy.

        Return:
            energies (Tensor):  Tensor of shape `(B, U)`. Data type is float.
                                Energy terms.

        Symbols:
            B:  Batchsize, i.e. number of walkers of the simulation.
            U:  Number of potential energy terms.

        """

        coordinate = self.coordinate * self.input_unit_scale
        pbc_box = self.pbc_box
        if pbc_box is not None:
            pbc_box *= self.input_unit_scale

        neigh_idx, neigh_pos, neigh_dis, neigh_mask = self.neighbour_list(coordinate, pbc_box)

        energies = self.potential_function(
            coordinate=coordinate,
            neighbour_index=neigh_idx,
            neighbour_mask=neigh_mask,
            neighbour_coord=neigh_pos,
            neighbour_distance=neigh_dis,
            pbc_box=pbc_box
        ) * self.output_unit_scale

        return energies

    def calc_biases(self) -> Tensor:
        """calculate the bias potential terms.

        Return:
            biases (Tensor):    Tensor of shape `(B, V)`. Data type is float.
                                Energy terms.

        Symbols:
            B:  Batchsize, i.e. number of walkers of the simulation.
            V:  Number of bias potential terms.

        """
        if self.bias_function is None:
            return None

        coordinate = self.coordinate * self.input_unit_scale
        pbc_box = self.pbc_box
        if pbc_box is not None:
            pbc_box *= self.input_unit_scale

        neigh_idx, neigh_pos, neigh_dis, neigh_mask = self.neighbour_list(coordinate, pbc_box)

        biases = ()
        for i in range(self._num_biases):
            bias_ = self.bias_function[i](
                coordinate=coordinate,
                neighbour_index=neigh_idx,
                neighbour_mask=neigh_mask,
                neighbour_coord=neigh_pos,
                neighbour_distance=neigh_dis,
                pbc_box=pbc_box
            )
            biases += (bias_,)

        return msnp.concatenate(biases, axis=-1) * self.output_unit_scale

    def construct(self, *inputs) -> Tensor:
        """calculate the total potential energy (potential energy and bias potential) of the simulation system.

        Return:
            energy (Tensor):    Tensor of shape `(B, 1)`. Data type is float.
                                Total potential energy.

        Symbols:
            B:  Batchsize, i.e. number of walkers of the simulation.

        """
        #pylint: disable=unused-argument
        coordinate, pbc_box = self.system()

        coordinate *= self.input_unit_scale
        if pbc_box is not None:
            pbc_box *= self.input_unit_scale

        neigh_idx, neigh_pos, neigh_dis, neigh_mask = self.neighbour_list(coordinate, pbc_box)

        energies = self.potential_function(
            coordinate=coordinate,
            neighbour_index=neigh_idx,
            neighbour_mask=neigh_mask,
            neighbour_coord=neigh_pos,
            neighbour_distance=neigh_dis,
            pbc_box=pbc_box
        ) * self.output_unit_scale

        energies = F.depend(energies, F.assign(self._energies, energies))

        biases = None
        if self.bias_function is not None:
            biases = ()
            for i in range(self._num_biases):
                bias_ = self.bias_function[i](
                    coordinate=coordinate,
                    neighbour_index=neigh_idx,
                    neighbour_mask=neigh_mask,
                    neighbour_coord=neigh_pos,
                    neighbour_distance=neigh_dis,
                    pbc_box=pbc_box
                )
                biases += (bias_,)

            biases = msnp.concatenate(biases, axis=-1) * self.output_unit_scale
            biases = F.depend(biases, F.assign(self._biases, biases))

        energy, bias = self.energy_wrapper(energies, biases)

        if self.bias_function is not None:
            energy = F.depend(energy, F.assign(self._bias, bias))

        return energy
