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
from ..data.forcefield import get_forcefield_parameters
from ..system import Molecule
from ..function.units import Units


THIS_PATH = os.path.abspath(__file__)
BUILTIN_FF_PATH = THIS_PATH.replace('potential/forcefield.py', 'data/forcefield/')


class ForceFieldBase(PotentialCell):
    r"""Basic cell for force filed

    Args:

        Energy (EnergyCell or list):    Energy terms. Default: None

        cutoff (float):                 Cutoff distance. Default: None

        exclude_index (Tensor):         Tensor of shape (B, A, Ex). Data type is int
                                        The indexes of atoms that should be excluded from neighbour list.
                                        Default: None

        length_unit (str):              Length unit for position coordinate. Default: None

        energy_unit (str):              Energy unit. Default: None

        units (Units):                  Units of length and energy. Default: None

        use_pbc (bool):                 Whether to use periodic boundary condition.

    """

    def __init__(self,
                 energy: EnergyCell = None,
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
        self.energy_network = self.set_energy_network(energy)

        self.energy_scale = 1

        self.output_unit_scale = self.set_unit_scale()

        self.concat = ops.Concat(-1)

    def set_energy_scale(self, scale: Tensor):
        """set energy scale"""
        scale = Tensor(scale, ms.float32)
        if scale.ndim != 1 and scale.ndim != 0:
            raise ValueError('The rank of energy scale must be 0 or 1.')
        if scale.shape[-1] != self.output_dim and scale.shape[-1] != 1:
            raise ValueError('The dimension of energy scale must be equal to the dimension of energy ' +
                             str(self.output_dim)+' or 1, but got: '+str(scale.shape[-1]))
        self.energy_scale = scale
        return self

    def set_energy_network(self, energy: EnergyCell) -> CellList:
        """set energy"""
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
        """set unit scale"""
        if self.energy_network is None:
            return 1
        output_unit_scale = ()
        for i in range(self.num_energy):
            self.energy_network[i].set_input_unit(self.units)
            dim = self.energy_network[i].output_dim
            scale = np.ones((dim,), np.float32) * \
                self.energy_network[i].convert_energy_to(self.units)
            output_unit_scale += (scale,)
        output_unit_scale = np.concatenate(output_unit_scale, axis=-1)
        return Tensor(output_unit_scale, ms.float32)

    def set_units(self, length_unit: str = None, energy_unit: str = None, units: Units = None):
        """set units"""
        if units is not None:
            self.units.set_units(units=units)
        else:
            if length_unit is not None:
                self.units.set_length_unit(length_unit)
            if energy_unit is not None:
                self.units.set_energy_unit(energy_unit)

        self.output_unit_scale = self.set_unit_scale()

        return self

    def set_pbc(self, use_pbc: Tensor = None):
        """set whether to use periodic boundary condition."""
        for i in range(self.num_energy):
            self.energy_network[i].set_pbc(use_pbc)
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
            potential (Tensor): Tensor of shape (B, 1). Data type is float.

        Symbols:
            B:  Batchsize, i.e. number of walkers in simulation
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
            ene = self.energy_network[i](
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
    r"""Potential of classical force field

    Args:

        system (Molecule):  Simulation system.

        cutoff (float):     Cutoff distance. Default: None

        length_unit (str):  Length unit for position coordinate. Default: None

        energy_unit (str):  Energy unit. Default: None

        units (Units):      Units of length and energy. Default: None

        use_pbc (bool):     Whether to use periodic boundary condition.

    """

    def __init__(self,
                 system: Molecule,
                 parameters: dict,
                 cutoff: float = None,
                 length_unit: str = None,
                 energy_unit: str = None,
                 units: Units = None,
                 use_pbc: bool = None,
                 ):

        super().__init__(
            cutoff=cutoff,
            exclude_index=None,
            length_unit=length_unit,
            energy_unit=energy_unit,
            units=units,
            use_pbc=use_pbc,
        )
        # Check Parameters
        self.check_params(system)

        # Generate Forcefield Parameters
        ff_dict = get_forcefield_parameters(parameters)
        for residue in system.residue:
            residue.build_atom_type(ff_dict['template'])
            residue.build_atom_charge(ff_dict['template'])

        system.build_system()

        ff_params = ForceFieldParameters(
            system.atom_type, copy.deepcopy(ff_dict), atom_names=system.atom_name[0],
            atom_charges=self.identity(system.atom_charge).asnumpy()[None, :])

        if isinstance(system.bond, np.ndarray):
            force_params = ff_params(system.bond[0])
        if isinstance(system.bond, Tensor):
            force_params = ff_params(system.bond[0].asnumpy())

        energy = []

        # Bond energy
        if system.bond is not None:
            bond_index = Tensor(system.bond[0], ms.int32)
            bond_force_constant = Tensor(force_params.bond_params[:, 2][None, :], ms.float32)
            bond_length = Tensor(force_params.bond_params[:, 3][None, :], ms.float32)

            length_unit = ff_dict['parameters']['bond_energy']['length_unit']
            energy_unit = ff_dict['parameters']['bond_energy']['energy_unit']
            bond_energy = BondEnergy(bond_index, force_constant=bond_force_constant,
                                     bond_length=bond_length, use_pbc=use_pbc,
                                     length_unit=length_unit, energy_unit=energy_unit)
            energy.append(bond_energy)

        # Angle energy
        if force_params.angles is not None:
            angle_index = Tensor(force_params.angles[None, :], ms.int32)
            angle_force_constant = Tensor(force_params.angle_params[:, 3][None, :], ms.float32)
            bond_angle = Tensor(force_params.angle_params[:, 4][None, :], ms.float32)

            energy_unit = ff_dict['parameters']['angle_energy']['energy_unit']
            angle_energy = AngleEnergy(angle_index, force_constant=angle_force_constant,
                                       bond_angle=bond_angle, use_pbc=use_pbc, energy_unit=energy_unit)
            energy.append(angle_energy)

        # Dihedral energy
        if force_params.dihedral_params is not None:
            dihedral_index = Tensor(
                force_params.dihedral_params[:, [0, 1, 2, 3]][None, :], ms.int32)
            dihe_force_constant = Tensor(force_params.dihedral_params[:, 5][None, :], ms.float32)
            periodicity = Tensor(force_params.dihedral_params[:, 4][None, :], ms.int32)
            phase = Tensor(force_params.dihedral_params[:, 6][None, :], ms.float32)

            # Idihedral Parameters
            idihedral_index = Tensor(
                force_params.improper_dihedral_params[:, [0, 1, 2, 3]][None, :], ms.int32)

            # Appending dihedral parameters and improper dihedral parameters.
            dihedral_index = msnp.append(
                dihedral_index, idihedral_index, axis=1)
            dihe_force_constant = msnp.append(dihe_force_constant, Tensor(
                force_params.improper_dihedral_params[:, 5][None, :], ms.float32), axis=1)
            periodicity = msnp.append(periodicity, Tensor(
                force_params.improper_dihedral_params[:, 4][None, :], ms.int32), axis=1)
            phase = msnp.append(phase, Tensor(
                force_params.improper_dihedral_params[:, 6][None, :], ms.float32), axis=1)

            energy_unit = ff_dict['parameters']['dihedral_energy']['energy_unit']
            dihedral_energy = DihedralEnergy(dihedral_index, force_constant=dihe_force_constant,
                                             periodicity=periodicity, phase=phase, use_pbc=use_pbc,
                                             energy_unit=energy_unit)
            energy.append(dihedral_energy)

        # Electronic energy
        if system.atom_charge is not None:
            length_unit = ff_dict['parameters']['coulomb_energy']['length_unit']
            energy_unit = ff_dict['parameters']['coulomb_energy']['energy_unit']
            ele_energy = CoulombEnergy(atom_charge=system.atom_charge, use_pbc=use_pbc,
                                       length_unit=length_unit, energy_unit=energy_unit)
            energy.append(ele_energy)

        # VDW energy
        epsilon = None
        sigma = None
        if force_params.vdw_param is not None:
            epsilon = Tensor(force_params.vdw_param[:, 0][None, :], ms.float32)
            sigma = Tensor(force_params.vdw_param[:, 1][None, :], ms.float32)

            length_unit = ff_dict['parameters']['vdw_energy']['length_unit']
            energy_unit = ff_dict['parameters']['vdw_energy']['energy_unit']
            vdw_energy = LennardJonesEnergy(epsilon=epsilon, sigma=sigma, use_pbc=use_pbc,
                                            length_unit=length_unit, energy_unit=energy_unit)
            energy.append(vdw_energy)

        # Non-bonded pairwise energy
        if ff_params.pair_index is not None:
            pair_index = Tensor(ff_params.pair_index[None, :], ms.int32)
            qiqj = Tensor(force_params.pair_params[:, 0][None, :], ms.float32)
            epsilon_ij = Tensor(force_params.pair_params[:, 1][None, :], ms.float32)
            sigma_ij = Tensor(force_params.pair_params[:, 2][None, :], ms.float32)

            r_scale = ff_dict['parameters']['nb_pair_energy']['r_scale']
            r6_scale = ff_dict['parameters']['nb_pair_energy']['r6_scale']
            r12_scale = ff_dict['parameters']['nb_pair_energy']['r12_scale']
            length_unit = ff_dict['parameters']['nb_pair_energy']['length_unit']
            energy_unit = ff_dict['parameters']['nb_pair_energy']['energy_unit']
            pair_energy = NonbondPairwiseEnergy(pair_index, qiqj=qiqj, epsilon_ij=epsilon_ij, sigma_ij=sigma_ij,
                                                r_scale=r_scale, r6_scale=r6_scale, r12_scale=r12_scale,
                                                length_unit=length_unit, energy_unit=energy_unit, use_pbc=use_pbc)
            energy.append(pair_energy)

        # Exclude Parameters
        self._exclude_index = Tensor(force_params.excludes[None, :], ms.int32)

        self.energy_network = self.set_energy_network(energy)
        self.output_unit_scale = self.set_unit_scale()

    def check_params(self, system: Molecule):
        """Check if the input parameters for force field is legal.
        """
        # Dimension Checking
        if system.atom_mass.shape[0] != 1:
            raise ValueError('The first dimension of atom mass should be 1.')

        if system.atom_type.shape[0] != 1:
            raise ValueError('The first dimension of atom type should be 1.')

        if system.atom_name.shape[0] != 1:
            raise ValueError('The first dimension of atom name should be 1.')

        if system.bond.shape[0] != 1:
            raise ValueError('The first dimension of bond should be 1.')

        # Type Checking
        if not isinstance(system.atom_name, np.ndarray):
            raise ValueError(
                'The data type of atom name should be numpy.ndarray.')

        if not isinstance(system.atom_type, np.ndarray):
            raise ValueError(
                'The data type of atom type should be numpy.ndarray.')

        if not isinstance(system.atom_mass, np.ndarray) and not isinstance(system.atom_mass, Tensor):
            raise ValueError(
                'The data type of atom mass should be numpy.ndarray or Tensor.')

        if not isinstance(system.bond, np.ndarray) and not isinstance(system.bond, Tensor):
            raise ValueError(
                'The data type of bond should be numpy.ndarray or Tensor.')
