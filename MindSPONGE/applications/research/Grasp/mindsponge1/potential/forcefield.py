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
"""Force filed"""
import os
import copy
from typing import Union
import numpy as np
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
    r"""
    Basic cell for force filed.

    Args:
        energy (Union[EnergyCell, list]):    Energy terms. The type of energy parameter can be list or EnergyCell.
                                             Default: None.
        cutoff (float):                      Cutoff distance. Default: None.
        exclude_index (Tensor):              Tensor of shape (B, A, Ex). Data type is int.
                                             The indexes of atoms that should be excluded from neighbour list.
                                             Default: None.
        length_unit (str):                   Length unit for position coordinate. Default: None.
        energy_unit (str):                   Energy unit. Default: None.
        units (Units):                       Units of length and energy. Default: None.
        use_pbc (bool, optional):            Whether to use periodic boundary condition.
                                             If this is "None", that means do not use periodic boundary condition.
                                             Default: None.

    Returns:
        potential (Tensor), Tensor of shape (B, 1). Data type is float.

    Supported Platforms:
        ``Ascend`` ``GPU``
    """

    def __init__(self,
                 energy: Union[EnergyCell, list] = None,
                 cutoff: float = None,
                 exclude_index: Tensor = None,
                 length_unit: str = None,
                 energy_unit: str = None,
                 units: Units = None,
                 use_pbc: bool = None,
                 ):

        super().__init__(
            cutoff=cutoff,
            exclude_index=exclude_index,
            length_unit=length_unit,
            energy_unit=energy_unit,
            units=units,
            use_pbc=use_pbc,
        )

        self.num_energy = 0
        self.energy_cell = self.set_energy_cell(energy)

        self.energy_scale = 1

        self.output_unit_scale = self.set_unit_scale()

        self.concat = ops.Concat(-1)

    def set_energy_scale(self, scale: Tensor):
        """
        Set energy scale.

        Args:
            scale (Tensor):                 Tensor of shape(B, 1). The scale parameter is used to set energy scale.
        """
        scale = Tensor(scale, ms.float32)
        if scale.ndim != 1 and scale.ndim != 0:
            raise ValueError('The rank of energy scale must be 0 or 1.')
        if scale.shape[-1] != self.output_dim and scale.shape[-1] != 1:
            raise ValueError('The dimension of energy scale must be equal to the dimension of energy ' +
                             str(self.output_dim)+' or 1, but got: '+str(scale.shape[-1]))
        self.energy_scale = scale
        return self

    def set_energy_cell(self, energy: EnergyCell) -> CellList:
        """
        Set energy.

        Args:
            energy (Union[EnergyCell, list]):    Energy terms. The type of energy parameter can be list or EnergyCell.
                                                 Default: None.

        Returns:
            CellList.
        """
        if energy is None:
            return None
        if isinstance(energy, EnergyCell):
            self.num_energy = 1
            energy = CellList([energy])
        elif isinstance(energy, list):
            self.num_energy = len(energy)
            energy = CellList(energy)
        else:
            raise TypeError(
                'The type of energy must be EnergyCell or list but got: '+str(type(energy)))

        self.output_dim = 0
        if energy is not None:
            for i in range(self.num_energy):
                self.output_dim += energy[i].output_dim
        return energy

    def set_unit_scale(self) -> Tensor:
        """
        set unit scale.

        Returns:
            Tensor, output unit scale.
        """
        if self.energy_cell is None:
            return 1
        output_unit_scale = ()
        for i in range(self.num_energy):
            self.energy_cell[i].set_input_unit(self.units)
            dim = self.energy_cell[i].output_dim
            scale = np.ones((dim,), np.float32) * \
                self.energy_cell[i].convert_energy_to(self.units)
            output_unit_scale += (scale,)
        output_unit_scale = np.concatenate(output_unit_scale, axis=-1)
        return Tensor(output_unit_scale, ms.float32)

    def set_units(self, length_unit: str = None, energy_unit: str = None, units: Units = None):
        """
        Set units.

        Args:
            length_unit (str):              Length unit for position coordinate. Default: None.
            energy_unit (str):              Energy unit. Default: None.
            units (Units):                  Units of length and energy. Default: None.
        """
        if units is not None:
            self.units.set_units(units=units)
        else:
            if length_unit is not None:
                self.units.set_length_unit(length_unit)
            if energy_unit is not None:
                self.units.set_energy_unit(energy_unit)

        self.output_unit_scale = self.set_unit_scale()

        return self

    def set_pbc(self, use_pbc: bool = None):
        """
        Set whether to use periodic boundary condition.

        Args:
            use_pbc (bool, optional):       Whether to use periodic boundary condition.
                                            If this is "None", that means do not use periodic boundary condition.
                                            Default: None.
        """
        for i in range(self.num_energy):
            self.energy_cell[i].set_pbc(use_pbc)
        return self

    def set_cutoff(self, cutoff: Tensor = None):
        """
        Set cutoff distance.

        Args:
            cutoff (Tensor):                 Cutoff distance. Default: None.
        """
        self.cutoff = None
        if cutoff is not None:
            self.cutoff = Tensor(cutoff, ms.float32)
        for i in range(self.num_energy):
            self.energy_cell[i].set_cutoff(self.cutoff)
        return self

    def construct(self,
                  coordinate: Tensor,
                  neighbour_index: Tensor = None,
                  neighbour_mask: Tensor = None,
                  neighbour_coord: Tensor = None,
                  neighbour_distance: Tensor = None,
                  pbc_box: Tensor = None
                  ):
        r"""
        Calculate potential energy.

        Args:
            coordinate (Tensor):           Tensor of shape (B, A, D). Data type is float.
                                            Position coordinate of atoms in system.
            neighbour_index (Tensor):       Tensor of shape (B, A, N). Data type is int.
                                            Index of neighbour atoms. Default: None
            neighbour_mask (Tensor):        Tensor of shape (B, A, N). Data type is bool.
                                            Mask for neighbour atoms. Default: None
            neighbour_coord (Tensor):       Tensor of shape (B, A, N, D). Data type is bool.
                                            Position coorindates of neighbour atoms.
            neighbour_distance (Tensor):   Tensor of shape (B, A, N). Data type is float.
                                            Distance between neighbours atoms. Default: None
            pbc_box (Tensor):               Tensor of shape (B, D). Data type is float.
                                            Tensor of PBC box. Default: None

        Returns:
            potential (Tensor), Tensor of shape (B, 1). Data type is float.

        Symbols:
            B:  Batchsize, i.e. number of walkers in simulation.
            A:  Number of atoms.
            N:  Maximum number of neighbour atoms.
            D:  Dimension of the simulation system. Usually is 3.
        """

        inv_neigh_dis = 0
        inv_neigh_dis = msnp.reciprocal(neighbour_distance)
        if neighbour_mask is not None:
            inv_neigh_dis = msnp.where(neighbour_mask, inv_neigh_dis, 0)

        potential = ()
        for i in range(self.num_energy):
            ene = self.energy_cell[i](
                coordinate=coordinate,
                neighbour_index=neighbour_index,
                neighbour_mask=neighbour_mask,
                neighbour_coord=neighbour_coord,
                neighbour_distance=neighbour_distance,
                inv_neigh_dis=inv_neigh_dis,
                pbc_box=pbc_box
            )
            potential += (ene,)

        potential = self.concat(potential) * self.energy_scale * self.output_unit_scale

        return potential


class ForceField(ForceFieldBase):
    r"""
    Potential of classical force field.

    Args:
        system (Molecule):               Simulation system.
        parameters (Union[dict, str]):   Force field parameters.
        cutoff (float):                  Cutoff distance. Default: None.
        length_unit (str):               Length unit for position coordinate. Default: None.
        energy_unit (str):               Energy unit. Default: None.
        units (Units):                   Units of length and energy. Default: None.

    Supported Platforms:
        ``Ascend`` ``GPU``
    """

    def __init__(self,
                 system: Molecule,
                 parameters: Union[dict, str],
                 cutoff: float = None,
                 length_unit: str = None,
                 energy_unit: str = None,
                 units: Units = None,
                 ):

        super().__init__(
            cutoff=cutoff,
            exclude_index=None,
            length_unit=length_unit,
            energy_unit=energy_unit,
            units=units,
        )

        use_pbc = system.use_pbc

        # Generate Forcefield Parameters
        parameters, template = get_forcefield(parameters)
        for residue in system.residue:
            residue.build_atom_type(template.get(residue.name))
            residue.build_atom_charge(template.get(residue.name))

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

        # Electronic energy
        if system.atom_charge is not None:
            coulomb_params: dict = parameters.get('coulomb_energy')
            length_unit = coulomb_params.get('length_unit')
            energy_unit = coulomb_params.get('energy_unit')
            ele_energy = CoulombEnergy(atom_charge=system.atom_charge, use_pbc=use_pbc,
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

        # Exclude Parameters
        self._exclude_index = Tensor(system_params.excludes[None, :], ms.int32)
        self.energy_cell = self.set_energy_cell(energy)
        self.output_unit_scale = self.set_unit_scale()
