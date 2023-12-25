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
"""Angle energy"""

from typing import Union, List
from operator import itemgetter
import numpy as np
from numpy import ndarray

import mindspore as ms
from mindspore import Tensor
from mindspore import Parameter
from mindspore.ops import functional as F

from .energy import EnergyCell, _energy_register
from ...colvar import Angle
from ...system import Molecule
from ...data import get_bonded_types
from ...function import functions as func
from ...function import get_ms_array, get_arguments


@_energy_register('angle_energy')
class AngleEnergy(EnergyCell):
    r"""Energy term of bond angles

    Math:

    .. math::

        E_{angle}(\theta_{ijk}) = \frac{1}{2} k_{ijk}^{\theta} (\theta_{ijk} - \theta_{ijk}^0) ^ 2

    Args:
        index (Union[Tensor, ndarray, List[int]]):
                            Array of the indices of the atoms forming the bond angles.
                            The shape of array is `(B, a, 3)`, and the data type is int.

        force_constant (Union[Tensor, ndarray, List[float]]):
                            Array of the harmonic force constant :math:`k^\theta` for the bond angles.
                            The shape of array is `(1, a)`, and the data type is float.

        bond_angle (Union[Tensor, ndarray, List[float]]):
                            Array of the equilibrium value :math:`\theta^0` for the bond angles.
                            The shape of array is `(1, a)`, and the data type is float.

        parameters (dict):  Force field parameters. Default: ``None``.

        use_pbc (bool):     Whether to use periodic boundary condition.

        length_unit (str):  Length unit. If None is given, it will be assigned with the global length unit.
                            Default: 'nm'

        energy_unit (str):  Energy unit. If None is given, it will be assigned with the global energy unit.
                            Default: 'kj/mol'

        name (str):         Name of the energy. Default: 'angle'

    Supported Platforms:
        ``Ascend`` ``GPU``

    Note:

        B:  Batchsize, i.e. number of walkers in simulation

        a:  Number of angles.

        D:  Spatial dimension of the simulation system. Usually is 3.

    """

    def __init__(self,
                 system: Molecule = None,
                 parameters: dict = None,
                 index: Union[Tensor, ndarray, List[int]] = None,
                 force_constant: Union[Tensor, ndarray, List[float]] = None,
                 bond_angle: Union[Tensor, ndarray, List[float]] = None,
                 use_pbc: bool = None,
                 length_unit: str = 'nm',
                 energy_unit: str = 'kj/mol',
                 name: str = 'angle_energy',
                 **kwargs,
                 ):

        super().__init__(
            name=name,
            use_pbc=use_pbc,
            length_unit=length_unit,
            energy_unit=energy_unit,
        )
        self._kwargs = get_arguments(locals(), kwargs)
        if 'exclude_index' in self._kwargs.keys():
            self._kwargs.pop('exclude_index')

        if parameters is not None:
            if system is None:
                raise ValueError('`system` cannot be None when using `parameters`!')
            length_unit = parameters.get('length_unit')
            energy_unit = parameters.get('energy_unit')
            self.units.set_units(length_unit, energy_unit)
            self._use_pbc = system.use_pbc
            index, force_constant, bond_angle = self.get_parameters(system, parameters)

        # (1,a,3)
        index = get_ms_array(index, ms.int32)
        if index.shape[-1] != 3:
            raise ValueError(f'The last dimension of index in AngleEnergy must be 3 but got: {index.shape[-1]}')
        if index.ndim == 2:
            index = F.expand_dims(index, 0)
        if index.ndim != 3:
            raise ValueError(f'The rank of index must be 2 or 3 but got shape: {index.shape}')
        self.index = Parameter(index, name='angle_index', requires_grad=False)

        # (a,)
        self.get_angles = Angle(atoms=self.index, use_pbc=use_pbc, batched=True)
        # a
        self.num_angles = self.get_angles.shape[-1]

        # (1,a)
        force_constant = get_ms_array(force_constant, ms.float32)
        if force_constant.shape[-1] != self.num_angles:
            raise ValueError(f'The last shape of force_constant ({force_constant.shape[-1]}) must be equal to '
                             f'the num_angles ({self.num_angles})!')
        if force_constant.ndim == 1:
            force_constant = F.expand_dims(force_constant, 0)
        if force_constant.ndim > 2:
            raise ValueError('The rank of force_constant cannot be larger than 2!')
        self.force_constant = Parameter(force_constant, name='angle_force_constant')

        bond_angle = get_ms_array(bond_angle, ms.float32)
        if bond_angle.shape[-1] != self.num_angles:
            raise ValueError(f'The last shape of bond_angle {bond_angle.shape[-1]}) must be equal to '
                             f'the num_angles ({self.num_angles})!')
        if bond_angle.ndim == 1:
            bond_angle = F.expand_dims(bond_angle, 0)
        if bond_angle.ndim > 2:
            raise ValueError('The rank of bond_angle cannot be larger than 2!')
        self.bond_angle = Parameter(bond_angle, name='bond_angle')

    @staticmethod
    def check_system(system: Molecule) -> bool:
        """Check if the system needs to calculate this energy term"""
        return system.angles is not None

    @staticmethod
    def get_parameters(system: Molecule, parameters: dict):
        """
        Get the force field bond parameters.

        Args:
            system (Molecule): Simulation system.
            parameters (dict): parameters.

        Returns:
            index (ndarray)
            force_constant (ndarray)
            bond_length (ndarray)

        """
        atom_type = system.atom_type[0]
        index = system.angles.asnumpy()

        angle_atoms = np.take(atom_type, index, -1)

        k_index = parameters['parameter_names']["pattern"].index('force_constant')
        t_index = parameters['parameter_names']["pattern"].index('bond_angle')

        angle_params: dict = parameters['parameters']
        params = {}
        for k, v in angle_params.items():
            [a, b, c] = k.split('-')
            if a != c:
                params[c + '-' + b + '-' + a] = v
        angle_params.update(params)

        angle_type = get_bonded_types(angle_atoms)
        type_list: list = angle_type.reshape(-1).tolist()

        if len(type_list) == 1:
            bond_angle = [angle_params[type_list[0]][t_index]]
            force_constant = [angle_params[type_list[0]][k_index]]
        else:
            bond_angle = []
            force_constant = []
            for params in itemgetter(*type_list)(angle_params):
                bond_angle.append(params[t_index])
                force_constant.append(params[k_index])

        force_constant = np.array(force_constant, np.float32).reshape(angle_type.shape)
        bond_angle = np.array(bond_angle, np.float32).reshape(angle_type.shape) / 180 * np.pi

        return index, force_constant, bond_angle

    def set_pbc(self, use_pbc: bool):
        self._use_pbc = use_pbc
        self.get_angles.set_pbc(use_pbc)
        return self

    def construct(self,
                  coordinate: Tensor,
                  neighbour_index: Tensor = None,
                  neighbour_mask: Tensor = None,
                  neighbour_vector: Tensor = None,
                  neighbour_distance: Tensor = None,
                  pbc_box: Tensor = None
                  ):
        r"""Calculate energy term.

        Args:
            coordinate (Tensor):            Tensor of shape (B, A, D). Data type is float.
                                            Position coordinate of atoms in system
            neighbour_index (Tensor):       Tensor of shape (B, A, N). Data type is int.
                                            Index of neighbour atoms.
            neighbour_mask (Tensor):        Tensor of shape (B, A, N). Data type is bool.
                                            Mask for neighbour index.
            neighbour_vector (Tensor):       Tensor of shape (B, A, N). Data type is bool.
                                            Vectors from central atom to neighbouring atoms.
            neighbour_distance (Tensor):    Tensor of shape (B, A, N). Data type is float.
                                            Distance between neighbours atoms.
            inv_neigh_dis (Tensor):         Tensor of shape (B, A, N). Data type is float.
                                            Reciprocal of distances.
            pbc_box (Tensor):               Tensor of shape (B, D). Data type is float.
                                            Tensor of PBC box. Default: ``None``.

        Returns:
            energy (Tensor):    Tensor of shape (B, 1). Data type is float.

        Note:
            B:  Batchsize, i.e. number of walkers in simulation
            A:  Number of atoms.
            D:  Spatial dimension of the simulation system. Usually is 3.

        """
        # (B,M)
        theta = self.get_angles(coordinate, pbc_box)
        # (B,M) = (B,M) - (1,M)
        dtheta = theta - self.bond_angle
        dtheta2 = dtheta * dtheta

        # E_angle = 1/2 * k_\theta * (\theta-\theta_0)^2
        # (B,M) = (1,M) * (B,M) * k
        energy = 0.5 * self.force_constant * dtheta2

        # (B,1) <- (B,M)
        return func.keepdims_sum(energy, -1)
