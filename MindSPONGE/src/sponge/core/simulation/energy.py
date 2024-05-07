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

from ...function import Units, get_arguments
from ...partition import NeighbourList
from ...system import Molecule
from ...potential import PotentialCell
from ...potential.bias import Bias
from ...sampling.wrapper import EnergyWrapper


class WithEnergyCell(Cell):
    r"""
    Cell that wraps the simulation system with the potential energy function.
    This Cell calculates the value of the potential energy of the system at
    the current coordinates and returns it.

    Args:
        system(:class:`sponge.system.Molecule`): Simulation system.
        potential(:class:`sponge.potential.PotentialCell`): Potential energy function cell.
        bias(Union[`sponge.potential.Bias`, List[`sponge.potential.Bias`]], optional): Bias
            potential function cell. Default: ``None``.
        cutoff(float, optional): Cut-off distance for neighbour list.
            If ``None`` is given, it will be assigned as the cutoff value of the of potential energy.
            Default: ``None``.
        neighbour_list(:class:`sponge.partition.NeighbourList`, optional): Neighbour list.
            Default: ``None``.
        wrapper(`sponge.sampling.wrapper.EnergyWrapper`, optional): Network to wrap and
            process potential and bias. Default: ``None``.
        kwargs(dict): Other arguments.

    Inputs:
        - **\*inputs** (Tuple(Tensor)) - Tuple of input tensors of
          :class:`sponge.core.WithEnergyCell`.

    Outputs:
        - **energy** (Tensor) - with shape of :math:`(B, 1)`. Total potential energy.
          Here `B` is the batch size, i.e. the number of walkers in simulation.
          Data type is float.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> from sponge import WithEnergyCell, RunOneStepCell, Sponge
        >>> from sponge.callback import RunInfo
        >>> from sponge.system import Molecule
        >>> from sponge.potential.forcefield import ForceField
        >>> from sponge.optimizer import Updater
        >>> system = Molecule(template='water.tip3p.yaml')
        >>> potential = ForceField(system, parameters='SPCE')
        >>> optimizer = Updater(system, controller=None, time_step=1e-3)
        >>> sim = WithEnergyCell(system, potential)
        >>> one_step = RunOneStepCell(energy=sim, optimizer=optimizer)
        >>> md = Sponge(one_step)
        >>> run_info = RunInfo(800)
        >>> md.run(2000, callbacks=[run_info])
        >>> # Output example:
        >>> # [MindSPONGE] Started simulation at 2024-04-29 01:02:10
        >>> # [MindSPONGE] Compilation Time: 0.66s
        >>> # [MindSPONGE] Step: 0, E_pot: 1.4293396, E_kin: 0.0, E_tot: 1.
        >>> # 4293396, Temperature: 0.0, Time: 662.63ms
        >>> # [MindSPONGE] Step: 800, E_pot: 1.4293396, E_kin: 0.0, E_tot: 1.
        >>> # 4293396, Temperature: 0.0, Time: 13.77ms
        >>> # [MindSPONGE] Step: 1600, E_pot: 1.4293396, E_kin: 0.0, E_tot: 1.
        >>> # 4293396, Temperature: 0.0, Time: 14.82ms
        >>> # [MindSPONGE] Finished simulation at 2024-04-29 01:02:39
        >>> # [MindSPONGE] Simulation time: 29.03 seconds.
    """

    def __init__(self,
                 system: Molecule,
                 potential: PotentialCell,
                 bias: Union[Bias, List[Bias]] = None,
                 cutoff: float = None,
                 neighbour_list: NeighbourList = None,
                 wrapper: EnergyWrapper = None,
                 **kwargs
                 ):

        super().__init__(auto_prefix=False)
        self._kwargs = get_arguments(locals(), kwargs)

        self.system = system
        self.potential_function = potential

        self.units = Units(self.system.length_unit, self.potential_function.energy_unit)
        self.system.units.set_energy_unit(self.energy_unit)

        self.bias_function: List[Bias] = None
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
            self.energy_wrapper = EnergyWrapper(length_unit=self.length_unit,
                                                energy_unit=self.energy_unit)

        self.wrapper_pace = self.energy_wrapper.update_pace

        self.exclude_index = self.potential_function.exclude_index
        self.neighbour_list = neighbour_list
        if neighbour_list is None:
            if cutoff is None and self.potential_function.cutoff is not None:
                cutoff = self.potential_function.cutoff
            self.neighbour_list = NeighbourList(system, cutoff, exclude_index=self.exclude_index)
        else:
            self.neighbour_list.set_exclude_index(self.exclude_index)

        self.neighbour_index = self.neighbour_list.neighbours
        self.neighbour_mask = self.neighbour_list.neighbour_mask
        self.num_neighbours = self.neighbour_list.num_neighbours

        if self.neighbour_list.cutoff is not None:
            if self.potential_function.cutoff is None:
                self.potential_function.set_cutoff(self.neighbour_list.cutoff, self.length_unit)
            else:
                pot_cutoff = self.units.length(self.potential_function.cutoff,
                                               self.potential_function.length_unit)
                nl_cutoff = self.neighbour_list.cutoff
                if self.potential_function.cutoff > self.neighbour_list.cutoff:
                    raise ValueError(f'The cutoff of the potential function '
                                     f'({pot_cutoff} {self.length_unit}) '
                                     f'cannot be greater than '
                                     f'the cutoff of the neighbour list '
                                     f'({nl_cutoff} {self.length_unit}).')

        self.coordinate = self.system.coordinate
        self.pbc_box = self.system.pbc_box
        self.atom_mass = self.system.atom_mass

        self.pbc_box = self.system.pbc_box

        self.potential_function.set_pbc(self.pbc_box is not None)

        for p in self.potential_function.trainable_params():
            p.requires_grad = False

        self.potential_function_units = self.potential_function.units

        self.length_unit_scale = Tensor(self.units.convert_length_to(
            self.potential_function.length_unit), ms.float32)

        self.identity = ops.Identity()

        self._energies = Parameter(msnp.zeros((self.num_walker,
                                               self.num_energies),
                                              dtype=ms.float32),
                                   name='energies',
                                   requires_grad=False)

        bias = msnp.zeros((self.num_walker, 1), dtype=ms.float32)
        if self.bias_function is None:
            self._biases = None
            self._bias = bias
        else:
            self._biases = Parameter(msnp.zeros((self.num_walker, self._num_biases),
                                                dtype=ms.float32),
                                     name='biases',
                                     requires_grad=False)
            self._bias = Parameter(bias, name='bias', requires_grad=False)

    @property
    def cutoff(self) -> Tensor:
        r"""
        Cutoff distance for neighbour list.

        Returns:
            Tensor, cutoff distance.
        """
        return self.neighbour_list.cutoff

    @property
    def neighbour_list_pace(self) -> int:
        r"""
        Update step for neighbour list.

        Returns:
            int, update steps.
        """
        return self.neighbour_list.pace

    @property
    def length_unit(self) -> str:
        r"""
        Length unit.

        Returns:
            str, length unit.
        """
        return self.units.length_unit

    @property
    def energy_unit(self) -> str:
        r"""
        Energy unit.

        Returns:
            str, energy unit.
        """
        return self.units.energy_unit

    @property
    def num_energies(self) -> int:
        r"""
        Number of energy terms :math:`U`.

        Returns:
            int, number of energy terms.
        """
        return self.potential_function.num_energies

    @property
    def num_biases(self) -> int:
        r"""
        Number of bias potential energies :math:`V`.

        Returns:
            int, number of bias potential energies.
        """
        return self._num_biases

    @property
    def energy_names(self) -> list:
        r"""
        Names of energy terms.

        Returns:
            list[str], names of energy terms.
        """
        return self.potential_function.energy_names

    @property
    def bias_names(self) -> list:
        r"""
        Name of bias potential energies.

        Returns:
            list[str], the bias potential energies.
        """
        return self._bias_names

    @property
    def energies(self) -> Tensor:
        r"""
        Tensor of potential energy components.

        Returns:
            Tensor, Tensor of shape `(B, U)`. Data type is float.
        """
        return self.identity(self._energies)

    @property
    def biases(self) -> Tensor:
        r"""
        Tensor of bias potential components.

        Returns:
            Tensor, Tensor of shape :math:`(B, V)`. Data type is float.
        """
        if self.bias_function is None:
            return None
        return self.identity(self._biases)

    @property
    def bias(self) -> Tensor:
        r"""
        Tensor of the total bias potential.

        Returns:
            Tensor, Tensor of shape :math:`(B, 1)`. Data type is float.
        """
        return self.identity(self._bias)

    def bias_pace(self, index: int = 0) -> int:
        """
        Return the update freqenucy for bias potential.

        Args:
            index(int): Index of bias potential. Default: ``0``.

        Returns:
            int, update freqenucy.
        """
        return self.bias_function[index].update_pace

    def set_pbc_grad(self, grad_box: bool):
        r"""
        Set whether to calculate the gradient of PBC box.

        Args:
            grad_box(bool): Whether to calculate the gradient of PBC box.
        """
        self.system.set_pbc_grad(grad_box)
        return self

    def update_neighbour_list(self) -> Tuple[Tensor, Tensor]:
        r"""
        Update neighbour list.

        Returns:
            - neigh_idx, Tensor. Tensor of shape :math:`(B, A, N)`. Data type is int.
              Index of neighbouring atoms of each atoms in system.
            - neigh_mask, Tensor. Tensor of shape :math:`(B, A, N)`. Data type is bool.
              Mask for neighbour list `neigh_idx`.
        """
        return self.neighbour_list.update(self.coordinate, self.pbc_box)

    def update_bias(self, step: int):
        r"""
        Update bias potential.

        Args:
            step(int):  Current simulation step. If it can be divided by
              update frequency, update the bias potential.
        """
        if self.bias_function is not None:
            for i in range(self._num_biases):
                if self.bias_pace(i) > 0 and step % self.bias_pace(i) == 0:
                    self.bias_function[i].update(self.coordinate, self.pbc_box)
        return self

    def update_wrapper(self, step: int):
        r"""
        Update energy wrapper.

        Args:
            step(int):  Current simulation step. If it can be divided by
              update frequency, update the energy wrapper.
        """
        if self.wrapper_pace > 0 and step % self.wrapper_pace == 0:
            self.energy_wrapper.update()
        return self

    def get_neighbour_list(self) -> Tuple[Tensor, Tensor]:
        r"""
        Get neighbour list.

        Returns:
            - neigh_idx, Tensor. Tensor of shape :math:`(B, A, N)`. Data type is int.
              Index of neighbouring atoms of each atoms in system.
            - neigh_mask, Tensor. Tensor of shape :math:`(B, A, N)`. Data type is bool.
              Mask for neighbour list `neigh_idx`.
        """
        return self.neighbour_list.get_neighbour_list()

    def calc_energies(self) -> Tensor:
        """
        Calculate the energy terms of the potential energy.

        Returns:
            Tensor, Tensor of shape :math:`(B, U)`. Data type is float. Energy terms.
        """

        neigh_idx, neigh_vec, neigh_dis, neigh_mask = self.neighbour_list(self.coordinate, self.pbc_box)

        coordinate = self.coordinate * self.length_unit_scale
        pbc_box = self.pbc_box
        if pbc_box is not None:
            pbc_box *= self.length_unit_scale
        neigh_vec *= self.length_unit_scale
        neigh_dis *= self.length_unit_scale

        energies = self.potential_function(
            coordinate=coordinate,
            neighbour_index=neigh_idx,
            neighbour_mask=neigh_mask,
            neighbour_vector=neigh_vec,
            neighbour_distance=neigh_dis,
            pbc_box=pbc_box
        )

        return energies

    def calc_biases(self) -> Tensor:
        """
        Calculate the bias potential terms.

        Returns:
            Tensor, Tensor of shape :math:`(B, V)`. Data type is float. Bias potential terms.
        """
        if self.bias_function is None:
            return None

        neigh_idx, neigh_vec, neigh_dis, neigh_mask = self.neighbour_list(self.coordinate, self.pbc_box)

        coordinate = self.coordinate * self.length_unit_scale
        pbc_box = self.pbc_box
        if pbc_box is not None:
            pbc_box *= self.length_unit_scale
        neigh_vec *= self.length_unit_scale
        neigh_dis *= self.length_unit_scale

        biases = ()
        for i in range(self._num_biases):
            bias_ = self.bias_function[i](
                coordinate=coordinate,
                neighbour_index=neigh_idx,
                neighbour_mask=neigh_mask,
                neighbour_vector=neigh_vec,
                neighbour_distance=neigh_dis,
                pbc_box=pbc_box
            )
            biases += (bias_,)

        return msnp.concatenate(biases, axis=-1)

    def construct(self, *inputs) -> Tensor:
        r"""
        Calculate the total potential energy (potential energy and bias
        potential) of the simulation system.

        Returns:
            energy (Tensor): Total potential energy.
              Tensor of shape :math:`(B, 1)`.
              Here :math:`B` is the batch size, i.e. the number of walkers in simulation.
              Data type is float.
        """
        #pylint: disable=unused-argument
        coordinate, pbc_box = self.system()

        neigh_idx, neigh_vec, neigh_dis, neigh_mask = self.neighbour_list(coordinate, pbc_box)

        coordinate *= self.length_unit_scale
        if pbc_box is not None:
            pbc_box *= self.length_unit_scale

        if neigh_idx is not None:
            neigh_vec *= self.length_unit_scale
            neigh_dis *= self.length_unit_scale

        energies = self.potential_function(
            coordinate=coordinate,
            neighbour_index=neigh_idx,
            neighbour_mask=neigh_mask,
            neighbour_vector=neigh_vec,
            neighbour_distance=neigh_dis,
            pbc_box=pbc_box
        )

        energies = F.depend(energies, F.assign(self._energies, energies))

        biases = None
        if self.bias_function is not None:
            biases = ()
            for i in range(self._num_biases):
                bias_ = self.bias_function[i](
                    coordinate=coordinate,
                    neighbour_index=neigh_idx,
                    neighbour_mask=neigh_mask,
                    neighbour_vector=neigh_vec,
                    neighbour_distance=neigh_dis,
                    pbc_box=pbc_box
                )
                biases += (bias_,)

            biases = msnp.concatenate(biases, axis=-1)
            biases = F.depend(biases, F.assign(self._biases, biases))

        energy, bias = self.energy_wrapper(energies, biases)

        if self.bias_function is not None:
            energy = F.depend(energy, F.assign(self._bias, bias))

        return energy
