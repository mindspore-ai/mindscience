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
"""Lennard-Jones potential"""

from typing import Union, List, Tuple
from operator import itemgetter
import numpy as np
from numpy import ndarray

import mindspore as ms
import mindspore.numpy as msnp
from mindspore import Tensor, Parameter
from mindspore.ops import functional as F

from .energy import NonbondEnergy, _energy_register
from ...system import Molecule
from ... import function as func
from ...function.functions import gather_value, get_ms_array, get_arguments


@_energy_register('lj_energy')
class LennardJonesEnergy(NonbondEnergy):
    r"""Lennard-Jones potential

    Math:

    .. math::

        E_{lj}(r_{ij}) = 4 \epsilon_{ij} \left (\frac{\sigma_{ij}^{12}}{r_{ij}^{12}}
                                              - \frac{\sigma_{ij}^{6}}{r_{ij}^{6}} \right)

        \epsilon_{ij} = \sqrt{\epsilon_i \epsilon_j}

        \sigma_{ij} = \frac{1}{2} (\sigma_i + \sigma_j)

    Args:

        epsilon (Union[Tensor, ndarray, List[float]]):
                            Parameter :math:`\epsilon` for LJ potential.
                            The shape of array is `(B, A)`, and the data type is float.

        sigma (Union[Tensor, ndarray, List[float]]):
                            Parameter :math:`\sigma` in LJ potential.
                            The shape of array is `(B, A)`, and the data type is float.

        mean_c6 (Union[Tensor, ndarray, List[float]]):
                            Average dispersion :math:`\langle C_6 \rangle` of the system
                            used for long range correction of dispersion interaction.
                            The shape of array is `(B, 1)`, and the data type is float.
                            Default: 0

        parameters (dict):  Force field parameters. Default: ``None``.

        cutoff (float):     Cutoff distance. Default: ``None``.

        use_pbc (bool):     Whether to use periodic boundary condition. Default: ``None``.

        length_unit (str):  Length unit. If None is given, it will be assigned with the global length unit.
                            Default: 'nm'

        energy_unit (str):  Energy unit. If None is given, it will be assigned with the global energy unit.
                            Default: 'kj/mol'

        name (str):         Name of the energy. Default: 'vdw'

    Supported Platforms:
        ``Ascend`` ``GPU``

    Note:

        B:  Batchsize, i.e. number of walkers in simulation

        A:  Number of atoms.

        N:  Maximum number of neighbour atoms.

        D:  Spatial dimension of the simulation system. Usually is 3.

    """

    def __init__(self,
                 system: Molecule = None,
                 parameters: dict = None,
                 epsilon: Union[Tensor, ndarray, List[float]] = None,
                 sigma: Union[Tensor, ndarray, List[float]] = None,
                 mean_c6: Union[Tensor, ndarray, List[float]] = 0,
                 cutoff: float = None,
                 use_pbc: bool = None,
                 length_unit: str = 'nm',
                 energy_unit: str = 'kj/mol',
                 name: str = 'lj_energy',
                 **kwargs,
                 ):

        super().__init__(
            name=name,
            cutoff=cutoff,
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

            epsilon, sigma, mean_c6 = self.get_parameters(system, parameters)

        sigma = get_ms_array(sigma, ms.float32)
        epsilon = get_ms_array(epsilon, ms.float32)

        if sigma.shape[-1] != epsilon.shape[-1]:
            raise ValueError(f'the last dimension of sigma {sigma.shape[-1]} must be equal to '
                             f'the last dimension of epsilon ({epsilon.shape[-1]})!')

        self.num_atoms = sigma.shape[-1]

        if sigma.ndim == 1:
            sigma = F.expand_dims(sigma, 0)
        if sigma.ndim > 2:
            raise ValueError('The rank of sigma cannot be larger than 2!')
        self.sigma = Parameter(sigma, name='sigma')

        if epsilon.ndim == 1:
            epsilon = F.expand_dims(epsilon, 0)
        if epsilon.ndim > 2:
            raise ValueError('The rank of epsilon cannot be larger than 2!')
        self.epsilon = Parameter(epsilon, name='epsilon')

        self.mean_c6 = None
        if mean_c6 is not None:
            mean_c6 = get_ms_array(mean_c6, ms.float32)
            if mean_c6.ndim == 0:
                mean_c6 = mean_c6.reshape(1, 1)
            elif mean_c6.ndim == 1:
                mean_c6 = F.expand_dims(mean_c6, 0)
            elif mean_c6.ndim > 2:
                raise ValueError('The rank of mean_c6 cannot be larger than 2!')
            self.mean_c6 = Parameter(get_ms_array(mean_c6, ms.float32), name='average_dispersion', requires_grad=False)

        self.disp_corr = self._calc_disp_corr()

    @staticmethod
    def get_parameters(system: Molecule, parameters: dict) -> Tuple[ndarray]:
        r"""get the force field parameters for the system

        ['H','HO','HS','HC','H1','H2','H3','HP','HA','H4',
         'H5','HZ','O','O2','OH','OS','OP','C*','CI','C5',
         'C4','CT','CX','C','N','N3','S','SH','P','MG',
         'C0','F','Cl','Br','I','2C','3C','C8','CO']

        Args:
            atom_type (ndarray):    Array of atoms.

        Returns:
            dict, parameters.
        """

        atom_type = system.atom_type[0]

        sigma_index = parameters['parameter_names']["pattern"].index('sigma')
        eps_index = parameters['parameter_names']["pattern"].index('epsilon')

        vdw_params = parameters['parameters']
        type_list: list = atom_type.reshape(-1).tolist()
        sigma = []
        epsilon = []
        for params in itemgetter(*type_list)(vdw_params):
            sigma.append(params[sigma_index])
            epsilon.append(params[eps_index])

        if atom_type.ndim == 2 and atom_type.shape[0] > 1:
            #TODO
            type_list: list = atom_type[0].tolist()

        type_set = list(set(type_list))
        count = np.array([type_list.count(i) for i in type_set], np.int32)

        sigma_set = []
        eps_set = []
        for params in itemgetter(*type_set)(vdw_params):
            sigma_set.append(params[sigma_index])
            eps_set.append(params[eps_index])

        sigma_set = np.array(sigma_set)
        eps_set = np.array(eps_set)
        c6_set = 4 * eps_set * np.power(sigma_set, 6)
        param_count = count.reshape(1, -1) * count.reshape(-1, 1) - np.diag(count)
        mean_c6 = np.sum(c6_set * param_count) / param_count.sum()

        epsilon = np.array(epsilon, np.float32).reshape(atom_type.shape)
        sigma = np.array(sigma, np.float32).reshape(atom_type.shape)
        mean_c6 = mean_c6.astype(np.float32)

        return epsilon, sigma, mean_c6

    def set_cutoff(self, cutoff: float, unit: str = None):
        """set cutoff distance"""
        super().set_cutoff(cutoff, unit)
        self.disp_corr = self._calc_disp_corr()
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

        inv_neigh_dis = msnp.reciprocal(neighbour_distance * self.input_unit_scale)
        if neighbour_mask is not None:
            inv_neigh_dis = msnp.where(neighbour_mask, inv_neigh_dis, inv_neigh_dis)

        epsilon = self.identity(self.epsilon)
        sigma = self.identity(self.sigma)

        # \epsilon_i: (B,A,1)
        eps_i = F.expand_dims(epsilon, -1)
        # \epsilon_j: (B,A,N)
        eps_j = gather_value(epsilon, neighbour_index)
        # \epsilon_{ij}: (B,A,N) = (B,A,1) * (B,A,N)
        eps_ij = F.sqrt(eps_i * eps_j)

        # \sigma_i: (B,A,1)
        sigma_i = F.expand_dims(sigma, -1)
        # \sigma_j: (B,A,N)
        sigma_j = gather_value(sigma, neighbour_index)
        # \sigma_{ij}: (B,A,N) = (B,A,1) * (B,A,N)
        sigma_ij = (sigma_i + sigma_j) * 0.5

        # (\sigma_{ij} / r_{ij}) ^ 6
        r0_6 = F.pows(sigma_ij * inv_neigh_dis, 6)

        # 4 * \epsilon_{ij} * (\sigma_{ij} / r_{ij}) ^ 6
        ene_bcoeff = 4 * eps_ij * r0_6
        # 4 * \epsilon_{ij} * (\sigma_{ij} / r_{ij}) ^ 12
        ene_acoeff = ene_bcoeff * r0_6

        # (B,A,N)
        energy = ene_acoeff - ene_bcoeff

        # (B,A)
        energy = F.reduce_sum(energy, -1)
        # (B,1)
        energy = func.keepdims_sum(energy, -1) * 0.5

        if self.cutoff is not None and pbc_box is not None:
            # (B,1) <- (B,D)
            volume = func.keepdims_prod(pbc_box * self.input_unit_scale, -1)
            # E_corr = -2 / 3 * pi * N * \rho * <C_6> * r_c^-3
            #        = -2 / 3 * pi * N * (N / V) * <C_6> * r_c^-3
            #        = -2 / 3 * pi * N^2 * <C_6> / V
            #        = k_corr * <C_6> / V
            ene_corr = self.disp_corr * self.mean_c6 * msnp.reciprocal(volume)
            energy += ene_corr

        return energy

    def _calc_disp_corr(self) -> Tensor:
        """calculate the long range correct factor for dispersion"""
        if self.cutoff is None:
            return 0
        return -2.0 / 3.0 * msnp.pi * self.num_atoms**2 / msnp.power(self.cutoff, 3)
