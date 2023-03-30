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
import copy
from typing import Union, List
import numpy as np
from numpy import ndarray
import mindspore as ms
import mindspore.numpy as msnp
from mindspore import Tensor
from mindspore import ops
from mindspore.nn import CellList

from .energy import EnergyCell, BondEnergy, AngleEnergy, DihedralEnergy, NonbondPairwiseEnergy
from .energy import CoulombEnergy, LennardJonesEnergy
from .potential import PotentialCell
from ..data.parameters import ForceFieldParameters
from ..data.forcefield import get_forcefield
from ..system import Molecule
from ..function.units import Units


THIS_PATH = os.path.abspath(__file__)
BUILTIN_FF_PATH = THIS_PATH.replace('potential/forcefield.py', 'data/forcefield/')


class ForceFieldBase(PotentialCell):
    r"""Base class for the potential energy of classical force filed. It is a subclass of `PotentialCell`.

        A `ForceFieldBase` object contains multiple `EnergyCell` objects. The last dimension of its output Tensor
        is equal to the number of `EnergyCell` objects it contains.

    Args:

        Energy (Union[EnergyCell, List[EnergyCell]]):
                                List of `EnergyCell` objects. Default: None

        cutoff (float):         Cutoff distance. Default: None

        exclude_index (Tensor): Tensor of shape `(B, A, Ex)`. Data type is int
                                The indexes of atoms that should be excluded from neighbour list.
                                Default: None

        length_unit (str):      Length unit. If None is given, it will be assigned with the global length unit.
                                Default: None

        energy_unit (str):      Energy unit. If None is given, it will be assigned with the global energy unit.
                                Default: None

        use_pbc (bool):         Whether to use periodic boundary condition.

    Returns:

        energy (Tensor):    Tensor of shape `(B, E)`. Data type is float.

    Supported Platforms:

        ``Ascend`` ``GPU``

    Symbols:

        B:  Batchsize, i.e. number of walkers in simulation.
        E:  Number of energy terms.

    """

    def __init__(self,
                 energy: Union[EnergyCell, List[EnergyCell]] = None,
                 cutoff: Union[float, Tensor, ndarray] = None,
                 exclude_index: Union[Tensor, ndarray, List[int]] = None,
                 length_unit: str = None,
                 energy_unit: str = None,
                 use_pbc: bool = None,
                 ):

        super().__init__(
            length_unit=length_unit,
            energy_unit=energy_unit,
            use_pbc=use_pbc,
        )

        if cutoff is not None:
            self.cutoff = Tensor(cutoff, ms.float32)

        self._exclude_index = self._check_exclude_index(exclude_index)

        self._num_energies = 0
        self._energy_index = {}
        self.energies = None
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
        if units is not None:
            self.units.set_units(units=units)
        else:
            if length_unit is not None:
                self.units.set_length_unit(length_unit)
            if energy_unit is not None:
                self.units.set_energy_unit(energy_unit)

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
                  neighbour_coord: Tensor = None,
                  neighbour_distance: Tensor = None,
                  pbc_box: Tensor = None
                  ):
        r"""Calculate potential energy.

        Args:
            coordinate (Tensor):            Tensor of shape (B, A, D). Data type is float.
                                            Position coordinate of atoms in system.
            neighbour_index (Tensor):       Tensor of shape (B, A, N). Data type is int.
                                            Index of neighbour atoms. Default: None
            neighbour_mask (Tensor):        Tensor of shape (B, A, N). Data type is bool.
                                            Mask for neighbour atoms. Default: None
            neighbour_coord (Tensor):       Tensor of shape (B, A, N, D). Data type is bool.
                                            Position coorindates of neighbour atoms.
            neighbour_distance (Tensor):    Tensor of shape (B, A, N). Data type is float.
                                            Distance between neighbours atoms. Default: None
            pbc_box (Tensor):               Tensor of shape (B, D). Data type is float.
                                            Tensor of PBC box. Default: None

        Returns:
            potential (Tensor): Tensor of shape (B, E). Data type is float.

        Symbols:
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
                neighbour_coord=neighbour_coord,
                neighbour_distance=neighbour_distance,
                pbc_box=pbc_box
            )
            energies += (energy,)

        potential = self.concat(energies) * self.energy_scale * self.output_unit_scale

        return potential


class ForceField(ForceFieldBase):
    r"""Potential energy of classical force field. It is a subclass of `ForceFieldBase`.

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

        cutoff (float):         Cutoff distance. Default: None

        rebuild_system (bool):  Whether to rebuild the atom types and bond connection of the system
                                based on the template in parameters.
                                Default: True

        length_unit (str):      Length unit. If None is given, it will be assigned with the global length unit.
                                Default: None

        energy_unit (str):      Energy unit. If None is given, it will be assigned with the global energy unit.
                                Default: None

    Returns:

        energy (Tensor):    Tensor of shape `(B, E)`. Data type is float.

    Supported Platforms:

        ``Ascend`` ``GPU``

    Symbols:

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
                 ):

        super().__init__(
            cutoff=cutoff,
            exclude_index=None,
            length_unit=length_unit,
            energy_unit=energy_unit,
        )

        use_pbc = system.use_pbc

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

        ff_params = ForceFieldParameters(
            system.atom_type, copy.deepcopy(parameters), atom_names=system.atom_name,
            atom_charges=self.identity(system.atom_charge).asnumpy())

        if isinstance(system.bond, np.ndarray):
            system_params = ff_params(system.bond)
        if isinstance(system.bond, Tensor):
            system_params = ff_params(system.bond.asnumpy())

        energy = []

        # Bond energy
        if system_params.bond_params is not None:
            bond_index = system_params.bond_params['bond_index']
            bond_force_constant = system_params.bond_params['force_constant']
            bond_length = system_params.bond_params['bond_length']

            bond_params: dict = parameters.get('bond_energy')
            length_unit = bond_params.get('length_unit')
            energy_unit = bond_params.get('energy_unit')
            bond_energy = BondEnergy(bond_index, force_constant=bond_force_constant,
                                     bond_length=bond_length, use_pbc=use_pbc,
                                     length_unit=length_unit, energy_unit=energy_unit)
            energy.append(bond_energy)

        # Angle energy
        if system_params.angle_params is not None:
            angle_index = system_params.angle_params['angle_index']
            angle_force_constant = system_params.angle_params['force_constant']
            bond_angle = system_params.angle_params['bond_angle']

            angle_params: dict = parameters.get('angle_energy')
            energy_unit = angle_params.get('energy_unit')
            angle_energy = AngleEnergy(angle_index, force_constant=angle_force_constant,
                                       bond_angle=bond_angle, use_pbc=use_pbc, energy_unit=energy_unit)
            energy.append(angle_energy)

        # Dihedral energy
        if system_params.dihedral_params is not None:
            dihedral_index = Tensor(system_params.dihedral_params['dihedral_index'][None, :], ms.int32)
            dihe_force_constant = Tensor(system_params.dihedral_params['force_constant'][None, :], ms.float32)
            periodicity = Tensor(system_params.dihedral_params['periodicity'][None, :], ms.int32)
            phase = Tensor(system_params.dihedral_params['phase'][None, :], ms.float32)

            # improper Parameters
            improper_index = Tensor(system_params.improper_params['improper_index'][None, :], ms.int32)

            # Appending dihedral parameters and improper dihedral parameters.
            dihedral_index = msnp.append(dihedral_index, improper_index, axis=1)
            dihe_force_constant = msnp.append(dihe_force_constant, Tensor(
                system_params.improper_params['force_constant'][None, :], ms.float32), axis=1)
            periodicity = msnp.append(periodicity, Tensor(
                system_params.improper_params['periodicity'][None, :], ms.int32), axis=1)
            phase = msnp.append(phase, Tensor(
                system_params.improper_params['phase'][None, :], ms.float32), axis=1)

            dihedral_params: dict = parameters.get('dihedral_energy')
            energy_unit = dihedral_params.get('energy_unit')
            dihedral_energy = DihedralEnergy(dihedral_index, force_constant=dihe_force_constant,
                                             periodicity=periodicity, phase=phase, use_pbc=use_pbc,
                                             energy_unit=energy_unit)
            energy.append(dihedral_energy)

        # Exclude Parameters
        self._exclude_index = Tensor(system_params.excludes[None, :], ms.int32)

        # Electronic energy
        if system.atom_charge is not None:
            coulomb_params: dict = parameters.get('coulomb_energy')
            length_unit = coulomb_params.get('length_unit')
            energy_unit = coulomb_params.get('energy_unit')
            ele_energy = CoulombEnergy(atom_charge=system.atom_charge, pbc_box=system.pbc_box,
                                       exclude_index=self._exclude_index,
                                       length_unit=length_unit, energy_unit=energy_unit)
            energy.append(ele_energy)

        # VDW energy
        epsilon = None
        sigma = None
        if system_params.vdw_param is not None:
            epsilon = system_params.vdw_param['epsilon']
            sigma = system_params.vdw_param['sigma']
            mean_c6 = system_params.vdw_param['mean_c6']

            vdw_params: dict = parameters.get('vdw_energy')
            length_unit = vdw_params.get('length_unit')
            energy_unit = vdw_params.get('energy_unit')
            vdw_energy = LennardJonesEnergy(epsilon=epsilon, sigma=sigma, mean_c6=mean_c6, use_pbc=use_pbc,
                                            length_unit=length_unit, energy_unit=energy_unit)
            energy.append(vdw_energy)

        # Non-bonded pairwise energy
        if system_params.pair_params is not None and system_params.pair_params is not None:
            pair_index = Tensor(ff_params.pair_index[None, :], ms.int32)
            qiqj = system_params.pair_params['qiqj']
            epsilon_ij = system_params.pair_params['epsilon_ij']
            sigma_ij = system_params.pair_params['sigma_ij']
            r_scale = system_params.pair_params['r_scale']
            r6_scale = system_params.pair_params['r6_scale']
            r12_scale = system_params.pair_params['r12_scale']

            pair_params: dict = parameters.get('nb_pair_energy')
            length_unit = pair_params.get('length_unit')
            energy_unit = pair_params.get('energy_unit')
            pair_energy = NonbondPairwiseEnergy(pair_index, qiqj=qiqj, epsilon_ij=epsilon_ij, sigma_ij=sigma_ij,
                                                r_scale=r_scale, r6_scale=r6_scale, r12_scale=r12_scale,
                                                length_unit=length_unit, energy_unit=energy_unit, use_pbc=use_pbc)
            energy.append(pair_energy)

        self.set_energies(energy)
        self.set_cutoff(self.cutoff, self.length_unit)
