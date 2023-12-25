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
"""Torsion energy"""

from typing import Union, List
import re
from operator import itemgetter
import numpy as np
from numpy import ndarray

from mindspore import Tensor

from .energy import _energy_register
from .dihedral import DihedralEnergy
from ...data import get_improper_types
from ...system import Molecule
from ...function import get_arguments


@_energy_register('improper_energy')
class ImproperEnergy(DihedralEnergy):
    r"""Energy term of improper dihedral (torsion) angles.

    Math:

    .. math::

        E_{dihedral}(\omega) = \sum_n \frac{1}{2} V_n [1 - \cos{(n \omega - \gamma_n)}]

    Args:
        index (Union[Tensor, ndarray, List[int]]):
                            Array of the indices of the atoms forming the dihedral angles.
                            The shape of array is `(B, d, 4)`, and the data type is int.

        force_constant (Union[Tensor, ndarray, List[float]]):
                            Array of the harmonic force constant :math:`V_n` for the dihedral angles.
                            The shape of array is `(B, d)`, and the data type is float.

        periodicity (Union[Tensor, ndarray, List[float]]):
                            Array of the periodicity :math:`n` for the dihedral angles.
                            The shape of array is `(B, d)`, and the data type is int.

        phase (Union[Tensor, ndarray, List[float]]):
                            Array of the phase shift :math:`\gamma_n` for the dihedral angles.
                            The shape of array is `(B, d)`, and the data type is float.

        parameters (dict):  Force field parameters. Default: ``None``.

        use_pbc (bool):     Whether to use periodic boundary condition.

        length_unit (str):  Length unit. If None is given, it will be assigned with the global length unit.
                            Default: 'nm'

        energy_unit (str):  Energy unit. If None is given, it will be assigned with the global energy unit.
                            Default: 'kj/mol'

        name (str):         Name of the energy. Default: 'dihedral'

    Supported Platforms:
        ``Ascend`` ``GPU``

    Note:

        B:  Batchsize, i.e. number of walkers in simulation

        d:  Number of dihedral angles.

        D:  Spatial dimension of the simulation system. Usually is 3.

    """
    def __init__(self,
                 system: Molecule = None,
                 parameters: dict = None,
                 index: Union[Tensor, ndarray, List[int]] = None,
                 force_constant: Union[Tensor, ndarray, List[float]] = None,
                 periodicity: Union[Tensor, ndarray, List[float]] = None,
                 phase: Union[Tensor, ndarray, List[float]] = None,
                 use_pbc: bool = None,
                 length_unit: str = 'nm',
                 energy_unit: str = 'kj/mol',
                 name: str = 'improper_energy',
                 **kwargs,
                 ):

        super().__init__(
            system=system,
            parameters=parameters,
            index=index,
            force_constant=force_constant,
            periodicity=periodicity,
            phase=phase,
            name=name,
            use_pbc=use_pbc,
            length_unit=length_unit,
            energy_unit=energy_unit,
        )
        self._kwargs = get_arguments(locals(), kwargs)
        if 'exclude_index' in self._kwargs.keys():
            self._kwargs.pop('exclude_index')

    @staticmethod
    def check_system(system: Molecule) -> bool:
        """Check if the system needs to calculate this energy term"""
        return system.improper_dihedrals is not None

    @staticmethod
    def get_parameters(system: Molecule, parameters: dict):
        """
        Get the force field dihedral parameters.

        Args:
            dihedrals_in (ndarray): Array of input dihedrals.
            atom_type (ndarray):    Array of the types of atoms.

        Returns:
            dict, params.
        """

        atom_type = np.append(system.atom_type[0], np.array(["X"], dtype=np.str_))

        index = system.improper_dihedrals.asnumpy()
        third_id = system.improper_axis_atoms.asnumpy()

        improper_atoms = np.take(atom_type, index, -1)

        k_index = parameters['parameter_names']["pattern"][0].index('force_constant')
        phi_index = parameters['parameter_names']["pattern"][0].index('phase')
        t_index = parameters['parameter_names']["pattern"][0].index('periodicity')

        improper_params: dict = parameters['parameters']

        key_types_ndarray = np.array([specific_name.split('-') for specific_name in improper_params.keys()], np.str_)
        types_sorted_args = np.argsort((key_types_ndarray == '?').sum(axis=-1))
        sorted_key_types = key_types_ndarray[types_sorted_args]
        transformed_key_types = ['-'.join(specific_name).replace('?', '.+').replace('*', '\\*') for specific_name in
                                 sorted_key_types]

        improper_types, orders = get_improper_types(improper_atoms)
        type_list = improper_types[0].reshape(-1)

        not_defined_mask = np.zeros(type_list.shape).astype(np.int32)
        for i, _ in enumerate(type_list):
            for key_type in transformed_key_types:
                for j, itypes in enumerate(improper_types):
                    if re.match('^'+key_type+'$', itypes[i]):
                        this_improper = index[i][np.array(list(orders[j]))]
                        if this_improper[2] != third_id[i]:
                            continue
                        index[i] = this_improper
                        not_defined_mask[i] = 1
                        type_list[i] = key_type.replace('.+', '?').replace('\\', '')
                        break
                else:
                    continue
                break

        type_list = type_list[np.where(not_defined_mask > 0)[0]]

        force_constant = []
        phase = []
        periodicity = []
        improper_index = []
        improper = index[np.where(not_defined_mask > 0)[0]]
        for i, params in enumerate(itemgetter(*type_list)(improper_params)):
            for _, lastd_params in enumerate(params):
                improper_index.append(improper[i])
                force_constant.append(lastd_params[k_index])
                phase.append(lastd_params[phi_index])
                periodicity.append(lastd_params[t_index])

        index = np.array(improper_index, np.int32)
        force_constant = np.array(force_constant, np.float32)
        periodicity = np.array(periodicity, np.float32)
        phase = np.array(phase, np.float32) / 180 * np.pi

        return index, force_constant, periodicity, phase
