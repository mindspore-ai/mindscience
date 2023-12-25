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
"""Force filed"""
import os
from typing import Union, List
import numpy as np
from numpy import ndarray
import mindspore as ms
from mindspore import Tensor
from mindspore import ops
from mindspore.nn import CellList

from .energy import EnergyCell, get_energy_cell
from .potential import PotentialCell
from ..data.forcefield import get_forcefield
from ..system import Molecule
from ..function import get_arguments
from ..function import Units, Length


THIS_PATH = os.path.abspath(__file__)
BUILTIN_FF_PATH = THIS_PATH.replace('potential/forcefield.py', 'data/forcefield/')


class ForceFieldBase(PotentialCell):
    r"""
    Base class for the potential energy of classical force filed. It is a subclass of `PotentialCell`.

    A `ForceFieldBase` object contains multiple `EnergyCell` objects. The last dimension of its output Tensor
    is equal to the number of `EnergyCell` objects it contains.

    Args:
        Energy (Union[EnergyCell, List[EnergyCell]]):
                                List of `EnergyCell` objects. Default: ``None``.

        cutoff (Union[float, Length, Tensor]):  Cutoff distance. Default: ``None``.

        exclude_index (Union[Tensor, ndarray, List[int]]):
                                Array of indexes of atoms that should be excluded from neighbour list.
                                The shape of the tensor is `(B, A, Ex)`. The data type is int.
                                Default: ``None``.

        length_unit (str):      Length unit. If None is given, it will be assigned with the global length unit.
                                Default: ``None``.

        energy_unit (str):      Energy unit. If None is given, it will be assigned with the global energy unit.
                                Default: ``None``.

        use_pbc (bool):         Whether to use periodic boundary condition.

    Returns:
        energy (Tensor), Tensor of shape `(B, E)`. Data type is float.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Note:
        B:  Batchsize, i.e. number of walkers in simulation.
        E:  Number of energy terms.

    """

    def __init__(self,
                 energy: Union[EnergyCell, List[EnergyCell]] = None,
                 cutoff: Union[float, Length, Tensor] = None,
                 exclude_index: Union[Tensor, ndarray, List[int]] = None,
                 length_unit: str = None,
                 energy_unit: str = None,
                 use_pbc: bool = None,
                 name: str = 'potential',
                 **kwargs,
                 ):

        super().__init__(
            length_unit=length_unit,
            energy_unit=energy_unit,
            use_pbc=use_pbc,
            name=name,
        )
        self._kwargs = get_arguments(locals(), kwargs)

        if isinstance(cutoff, Length):
            cutoff = cutoff(self.units)

        if cutoff is not None:
            self.cutoff = Tensor(cutoff, ms.float32)

        self._exclude_index = self._check_exclude_index(exclude_index)

        self._num_energies = 0
        self._energy_index = {}
        self.energies: List[EnergyCell] = None
        self.output_unit_scale = 1
        self.set_energies(energy)

        self.energy_scale = 1

        self.concat = ops.Concat(-1)

    def set_energy_scale(self, scale: Tensor):
        """set energy scale"""
        scale = Tensor(scale, ms.float32)
        if scale.ndim != 1 and scale.ndim != 0:
            raise ValueError('The rank of energy scale must be 0 or 1.')
        if scale.shape[-1] != self._num_energies and scale.shape[-1] != 1:
            raise ValueError(f'The dimension of energy scale must be equal to '
                             f'the dimension of energy {self._num_energies} or 1, '
                             f'but got: {scale.shape[-1]}')
        self.energy_scale = scale
        return self

    def set_energies(self, energy: EnergyCell):
        """set energy"""
        if energy is None:
            self.output_unit_scale = 1
            return self
        if isinstance(energy, EnergyCell):
            self._num_energies = 1
            self.energies = CellList([energy])
        elif isinstance(energy, list):
            self._num_energies = len(energy)
            self.energies = CellList(energy)
        else:
            raise TypeError(f'The type of energy must be EnergyCell or list but got: {type(energy)}')

        self._energy_names = []
        self._energy_index = {}
        index = 0
        if self.energies is not None:
            for i in range(self._num_energies):
                name = self.energies[i].name
                self._energy_index[name] = index
                self._energy_names.append(name)
                index += 1

        self.set_unit_scale()
        return self

    def append_energy(self, energy: EnergyCell):
        """append energy terms"""
        if not isinstance(energy, EnergyCell):
            raise TypeError(f'The type of energy must be EnergyCell or list but got: {type(energy)}')

        self.energies.append(energy)
        self._energy_names.append(energy.name)
        self._energy_index[energy.name] = self._num_energies
        self._num_energies += 1
        self.set_unit_scale()

        return self

    def set_unit_scale(self) -> Tensor:
        """set unit scale"""
        if self.energies is None:
            self.output_unit_scale = 1
            return self
        output_unit_scale = []
        for i in range(self._num_energies):
            self.energies[i].set_input_unit(self.units)
            scale = self.energies[i].convert_energy_to(self.units)
            output_unit_scale.append(scale)
        self.output_unit_scale = Tensor(output_unit_scale, ms.float32)
        return self

    def set_units(self, length_unit: str = None, energy_unit: str = None, units: Units = None):
        """set units"""
        super().set_units(length_unit=length_unit, energy_unit=energy_unit, units=units)
        self.set_unit_scale()
        return self

    def set_pbc(self, use_pbc: bool = None):
        """set whether to use periodic boundary condition."""
        for i in range(self._num_energies):
            self.energies[i].set_pbc(use_pbc)
        return self

    def set_cutoff(self, cutoff: float = None, unit: str = None):
        """set cutoff distance"""
        super().set_cutoff(cutoff, unit)
        for i in range(self._num_energies):
            self.energies[i].set_cutoff(self.cutoff, self.length_unit)
        return self

    def construct(self,
                  coordinate: Tensor,
                  neighbour_index: Tensor = None,
                  neighbour_mask: Tensor = None,
                  neighbour_vector: Tensor = None,
                  neighbour_distance: Tensor = None,
                  pbc_box: Tensor = None
                  ):
        r"""Calculate potential energy.

        Args:
            coordinate (Tensor):            Tensor of shape (B, A, D). Data type is float.
                                            Position coordinate of atoms in system.
            neighbour_index (Tensor):       Tensor of shape (B, A, N). Data type is int.
                                            Index of neighbour atoms. Default: ``None``.
            neighbour_mask (Tensor):        Tensor of shape (B, A, N). Data type is bool.
                                            Mask for neighbour atoms. Default: ``None``.
            neighbour_vector (Tensor):       Tensor of shape (B, A, N, D). Data type is bool.
                                            Vectors from central atom to neighbouring atoms.
            neighbour_distance (Tensor):    Tensor of shape (B, A, N). Data type is float.
                                            Distance between neighbours atoms. Default: ``None``.
            pbc_box (Tensor):               Tensor of shape (B, D). Data type is float.
                                            Tensor of PBC box. Default: ``None``.

        Returns:
            potential (Tensor): Tensor of shape (B, E). Data type is float.

        Note:
            B:  Batchsize, i.e. number of walkers in simulation.
            A:  Number of atoms.
            N:  Maximum number of neighbour atoms.
            D:  Spatial dimension of the simulation system. Usually is 3.
            E:  Number of energy terms.

        """

        energies = ()
        for i in range(self._num_energies):
            energy = self.energies[i](
                coordinate=coordinate,
                neighbour_index=neighbour_index,
                neighbour_mask=neighbour_mask,
                neighbour_vector=neighbour_vector,
                neighbour_distance=neighbour_distance,
                pbc_box=pbc_box
            )
            energies += (energy.astype(ms.float32),)

        potential = self.concat(energies) * self.energy_scale * self.output_unit_scale

        return potential


class ForceField(ForceFieldBase):
    r"""
    Potential energy of classical force field. It is a subclass of `ForceFieldBase`.

    The `ForceField` class use force field parameter files to build the potential energy.

    Args:
        system (Molecule):      Simulation system.

        parameters (Union[dict, str, List[Union[dict, str]]]):
                                Force field parameters. It can be a `dict` of force field parameters,
                                a `str` of filename of a force field file in MindSPONGE YAML format,
                                or a `list` or `tuple` containing multiple `dict` or `str`.
                                If a filename is given, it will first look for a file with the same name
                                in the current directory. If the file does not exist, it will search
                                in MindSPONGE's built-in force field.
                                If multiple sets of parameters are given and the same atom type
                                is present in different parameters, then the atom type in the parameter
                                at the back of the array will replace the one at the front.

        cutoff (float):         Cutoff distance. Default: ``None``.

        rebuild_system (bool):  Whether to rebuild the atom types and bond connection of the system
                                based on the template in parameters.
                                Default: ``True``.

        length_unit (str):      Length unit. If None is given, it will be assigned with the global length unit.
                                Default: ``None``.

        energy_unit (str):      Energy unit. If None is given, it will be assigned with the global energy unit.
                                Default: ``None``.

    Returns:
        energy (Tensor), Tensor of shape `(B, E)`. Data type is float.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Note:
        B:  Batchsize, i.e. number of walkers in simulation.
        E:  Number of energy terms.

    """

    def __init__(self,
                 system: Molecule,
                 parameters: Union[dict, str, List[Union[dict, str]]],
                 cutoff: float = None,
                 rebuild_system: bool = True,
                 length_unit: str = None,
                 energy_unit: str = None,
                 name: str = 'potential',
                 **kwargs,
                 ):

        super().__init__(
            cutoff=cutoff,
            exclude_index=None,
            length_unit=length_unit,
            energy_unit=energy_unit,
            name=name,
        )
        self._kwargs = get_arguments(locals(), kwargs)

        # Generate Forcefield Parameters
        parameters, template = get_forcefield(parameters)

        if rebuild_system:
            if template is None:
                print('[WARNING] No template in parameters! Cannot rebuild the system!')
            else:
                for res in system.residue:
                    res.build_atom_type(template.get(res.name))
                    res.build_atom_charge(template.get(res.name))
                system.build_system()

        self._exclude_index = self.get_exclude_index(system)

        energy = []
        for energy_name, energy_params in parameters.items():
            if energy_name != 'coulomb_energy':
                energy_ = get_energy_cell(cls_name=energy_name,
                                          system=system,
                                          parameters=energy_params,
                                          exclude_index=self._exclude_index,
                                          )
            else:
                energy_ = get_energy_cell(cls_name=energy_name,
                                          system=system,
                                          parameters=energy_params,
                                          exclude_index=self._exclude_index,
                                          **kwargs
                                          )

            if energy_ is not None:
                energy.append(energy_)

        self.set_energies(energy)
        self.set_cutoff(self.cutoff, self.length_unit)

    @staticmethod
    def get_exclude_index(system: Molecule) -> Tensor:
        """
        Get the exclude atoms index.

        Args:
            bonds (ndarray):        Array of bonds.
            angles (ndarray):       Array of angles.
            dihedrals (ndarray):    Array of dihedrals.
            improper (ndarray):     Array of improper.

        Returns:
            np.ndarray, the index of exclude atoms.
        """

        if system.bonds is None:
            return None

        num_atoms = system.num_atoms
        bonds = system.bonds[0].asnumpy()
        angles = None if system.angles is None else system.angles.asnumpy()
        dihedrals = None if system.dihedrals is None else system.dihedrals.asnumpy()
        improper = None if system.improper_dihedrals is None else system.improper_dihedrals.asnumpy()

        excludes_ = []
        for i in range(num_atoms):
            bond_excludes = bonds[np.where(
                np.isin(bonds, i).sum(axis=1))[0]].flatten()
            this_excludes = bond_excludes

            if angles is not None:
                angle_excludes = angles[
                    np.where(np.isin(angles, i).sum(axis=1))[0]
                ].flatten()
                this_excludes = np.append(this_excludes, angle_excludes)

            if dihedrals is not None:
                dihedral_excludes = dihedrals[
                    np.where(np.isin(dihedrals, i).sum(axis=1))[0]
                ].flatten()
                this_excludes = np.append(this_excludes, dihedral_excludes)
            if improper is not None:
                idihedral_excludes = improper[
                    np.where(np.isin(improper, i).sum(axis=1))[0]
                ].flatten()
                this_excludes = np.append(this_excludes, idihedral_excludes)

            this_excludes = np.unique(this_excludes)
            excludes_.append(this_excludes[np.where(
                this_excludes != i)[0]].tolist())
        padding_length = 0
        for i in range(num_atoms):
            padding_length = max(padding_length, len(excludes_[i]))
        excludes = np.empty((num_atoms, padding_length))
        for i in range(num_atoms):
            excludes[i] = np.pad(
                np.array(excludes_[i]),
                (0, padding_length - len(excludes_[i])),
                mode="constant",
                constant_values=num_atoms,
            )
        return Tensor(excludes[None, :], ms.int32)
