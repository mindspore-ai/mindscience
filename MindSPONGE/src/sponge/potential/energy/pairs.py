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
"""Non-bonded pairwise energy"""

from typing import Union, List
from numpy import ndarray

import mindspore as ms
import mindspore.numpy as msnp
from mindspore import Tensor
from mindspore import Parameter
from mindspore import ops
from mindspore.ops import functional as F

from .energy import EnergyCell
from ...colvar import Distance
from ...function.functions import get_integer, get_ms_array, get_arguments, keepdims_sum


class NonbondPairwiseEnergy(EnergyCell):
    r"""Energy of non-bonded atom paris.

    Math:

    .. math::

        E_{pairs}(r_{ij}) = A_{ij}^p E_1(r_{ij}) + B_{ij}^p E_6(r_{ij}) + C_{ij}^p E_{12}(r_{ij})
                          = A_{ij}^p k_{coulomb} \frac{q_i q_j}{r_{ij}} -
                            B_{ij}^p \cdot 4 \epsilon_{ij} \frac{\sigma_{ij}^6}{r_{ij}^6}  +
                            C_{ij}^p \cdot 4 \epsilon_{ij} \frac{\sigma_{ij}^{12}}{r_{ij}^{12}}

    Args:

        index (Union[Tensor, ndarray, List[int]]):
                            Array of the indices of the non-bonded pair atoms.
                            The shape of Tensor is `(B, p, 2)`, and the data type is int.

        qiqj (Union[Tensor, ndarray, List[float]]):
                            Array of the products of the charges of non-bonded atom pairs.
                            The shape of Tensor is `(B, p)`, and the data type is float.

        epsilon_ij (Union[Tensor, ndarray, List[float]]):
                            Array of the parameter :math:`\epsilon` of the non-bonded atom pairs.
                            The shape of Tensor is `(B, p)`, and the data type is float.

        sigma_ij (Union[Tensor, ndarray, List[float]]):
                            Array of the parameter :math:`\sigma` of the non-bonded atom pairs.
                            The shape of Tensor is `(B, p)`, and the data type is float.

        r_scale (Union[Tensor, ndarray, List[float]]):
                            Array of the scaling constant :math:`A^p` for :math:`r^{-1}` terms in non-bond interaction.
                            The shape of Tensor is `(1, p)`, and the data type is float.

        r6_scale (Union[Tensor, ndarray, List[float]]):
                            Array of the scaling constant :math:`B^p` for :math:`r^{-6}` terms in non-bond interaction.
                            The shape of Tensor is `(1, p)`, and the data type is float.

        r12_scale (Union[Tensor, ndarray, List[float]]):
                            Array of the scaling constant :math:`C^p` for :math:`r^{-12}` terms in non-bond interaction.
                            The shape of Tensor is `(1, p)`, and the data type is float.

        use_pbc (bool):     Whether to use periodic boundary condition.

        length_unit (str):  Length unit. If None is given, it will be assigned with the global length unit.
                            Default: 'nm'

        energy_unit (str):  Energy unit. If None is given, it will be assigned with the global energy unit.
                            Default: 'kj/mol'

        name (str):         Name of the energy. Default: 'nb_pairs'

    Supported Platforms:

        ``Ascend`` ``GPU``

    Symbols:

        B:  Batchsize, i.e. number of walkers in simulation

        p:  Number of non-bonded atom pairs.

        D:  Spatial dimension of the simulation system. Usually is 3.

    """

    def __init__(self,
                 index: Union[Tensor, ndarray, List[int]] = None,
                 qiqj: Union[Tensor, ndarray, List[float]] = None,
                 epsilon_ij: Union[Tensor, ndarray, List[float]] = None,
                 sigma_ij: Union[Tensor, ndarray, List[float]] = None,
                 r_scale: Union[Tensor, ndarray, List[float]] = None,
                 r6_scale: Union[Tensor, ndarray, List[float]] = None,
                 r12_scale: Union[Tensor, ndarray, List[float]] = None,
                 parameters: dict = None,
                 cutoff: float = None,
                 use_pbc: bool = None,
                 length_unit: str = 'nm',
                 energy_unit: str = 'kj/mol',
                 name: str = 'nb_pairs',
                 **kwargs
                 ):

        super().__init__(
            name=name,
            use_pbc=use_pbc,
            length_unit=length_unit,
            energy_unit=energy_unit,
        )
        self._kwargs = get_arguments(locals(), kwargs)

        if parameters is not None:
            length_unit = parameters.get('length_unit')
            energy_unit = parameters.get('energy_unit')
            self.units.set_units(length_unit, energy_unit)

            index = parameters.get('index')
            qiqj = parameters.get('qiqj')
            epsilon_ij = parameters.get('epsilon_ij')
            sigma_ij = parameters.get('sigma_ij')
            r_scale = parameters.get('r_scale')
            r6_scale = parameters.get('r6_scale')
            r12_scale = parameters.get('r12_scale')

        # (1,p,2)
        index = get_ms_array(index, ms.int32)
        if index.shape[-1] != 2:
            raise ValueError(f'The last dimension of index in NonbondPairwiseEnergy must be 2, '
                             f'but got: {index.shape[-1]}')
        if index.ndim == 2:
            index = F.expand_dims(index, 0)
        if index.ndim != 3:
            raise ValueError(f'The rank of index must be 2 or 3 but got shape: {index.shape}')
        self.index = Parameter(index, name='pairs_index', requires_grad=False)

        self.num_pairs = index.shape[-2]

        qiqj = get_ms_array(qiqj, ms.float32)
        if qiqj.shape[-1] != self.num_pairs:
            raise ValueError(f'The last dimension of qiqj ({qiqj.shape[-1]}) must be equal to '
                             f'the number of non-bonded atom pairs({self.num_pairs})!')
        if qiqj.ndim == 1:
            qiqj = F.expand_dims(qiqj, 0)
        if qiqj.ndim > 2:
            raise ValueError('The rank of qiqj cannot be larger than 2!')
        self.qiqj = Parameter(qiqj, name='qiqj', requires_grad=False)

        epsilon_ij = get_ms_array(epsilon_ij, ms.float32)
        if epsilon_ij.shape[-1] != self.num_pairs:
            raise ValueError(f'The last dimension of epsilon_ij ({epsilon_ij.shape[-1]}) must be equal to '
                             f'the number of non-bonded atom pairs ({self.num_pairs})!')
        if epsilon_ij.ndim == 1:
            epsilon_ij = F.expand_dims(epsilon_ij, 0)
        if epsilon_ij.ndim > 2:
            raise ValueError('The rank of epsilon_ij cannot be larger than 2!')
        self.epsilon_ij = Parameter(epsilon_ij, name='epsilon_ij', requires_grad=False)

        sigma_ij = get_ms_array(sigma_ij, ms.float32)
        if sigma_ij.shape[-1] != self.num_pairs:
            raise ValueError(f'The last dimension of sigma_ij ({sigma_ij.shape[-1]}) must be equal to '
                             f'the number of non-bonded atom pairs({self.num_pairs})!')
        if sigma_ij.ndim == 1:
            sigma_ij = F.expand_dims(sigma_ij, 0)
        if sigma_ij.ndim > 2:
            raise ValueError('The rank of sigma_ij cannot be larger than 2!')
        self.sigma_ij = Parameter(sigma_ij, name='sigma_ij', requires_grad=False)

        r_scale = get_ms_array(r_scale, ms.float32)
        if r_scale.ndim == 0:
            r_scale = r_scale.reshape(1, 1)
        elif r_scale.ndim == 1:
            r_scale = F.expand_dims(r_scale, 0)
        elif r_scale.ndim > 2:
            raise ValueError('The rank of r_scale cannot be larger than 2!')
        if r_scale.shape[-1] != self.num_pairs and r_scale.shape[-1] != 1:
            raise ValueError(f'The last dimension of r_scale ({r_scale.shape[-1]}) must be equal to 1 or '
                             f'the number of non-bonded atom pairs({self.num_pairs})!')
        self.r_scale = Parameter(r_scale, name='r_scale_factor')

        r6_scale = get_ms_array(r6_scale, ms.float32)
        if r6_scale.ndim == 0:
            r6_scale = r6_scale.reshape(1, 1)
        elif r6_scale.ndim == 1:
            r6_scale = F.expand_dims(r6_scale, 0)
        elif r6_scale.ndim > 2:
            raise ValueError('The rank of r6_scale cannot be larger than 2!')
        if r6_scale.shape[-1] != self.num_pairs and r6_scale.shape[-1] != 1:
            raise ValueError(f'The last dimension of r6_scale ({r6_scale.shape[-1]}) must be equal to 1 or '
                             f'the number of non-bonded atom pairs({self.num_pairs})!')
        self.r6_scale = Parameter(r6_scale, name='r6_scale_factor')

        r12_scale = get_ms_array(r12_scale, ms.float32)
        if r12_scale.ndim == 0:
            r12_scale = r12_scale.reshape(1, 1)
        elif r12_scale.ndim == 1:
            r12_scale = F.expand_dims(r12_scale, 0)
        elif r12_scale.ndim > 2:
            raise ValueError('The rank of r12_scale cannot be larger than 2!')
        if r12_scale.shape[-1] != self.num_pairs and r12_scale.shape[-1] != 1:
            raise ValueError(f'The last dimension of r12_scale ({r12_scale.shape[-1]}) must be equal to 1 or '
                             f'the number of non-bonded atom pairs({self.num_pairs})!')
        self.r12_scale = Parameter(r12_scale, name='r12_scale_factor')

        self.cutoff = None
        if cutoff is not None:
            self.cutoff = get_integer(cutoff)

        self.get_pairs_distance = Distance(atoms=self.index, use_pbc=use_pbc, batched=True)

        self.coulomb_const = self.units.coulomb

        self.concat = ops.Concat(-1)

    def set_pbc(self, use_pbc=None):
        """set whether to use periodic boundary condition."""
        self._use_pbc = use_pbc
        self.get_pairs_distance.set_pbc(use_pbc)
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
                                            Tensor of PBC box. Default: None

        Returns:
            energy (Tensor):    Tensor of shape (B, 1). Data type is float.

        Symbols:
            B:  Batchsize, i.e. number of walkers in simulation
            A:  Number of atoms.
            p:  Number of non-bonded atom pairs.
            D:  Spatial dimension of the simulation system. Usually is 3.

        """

        # (B, p)
        distance = self.get_pairs_distance(coordinate, pbc_box) * self.input_unit_scale
        inv_dis = msnp.reciprocal(distance)

        # (B, p) = (1, p) * (B, p) * (1, p)
        # A * k * qi * qj / r
        energy_r = self.coulomb_const * self.qiqj * inv_dis * self.r_scale

        # \sigma_ij / r_ij
        sigma_over_rij = self.sigma_ij * inv_dis
        # (\sigma_ij / r_ij) ^ 6
        sigma_over_rij_6 = F.pows(sigma_over_rij, 6)

        ene_r6 = 4 * self.epsilon_ij * sigma_over_rij_6
        # -B * 4 * \epsilon * (\sigma_ij / r_ij) ^ 6
        energy_r6 = -ene_r6 * self.r6_scale
        # C * 4 * \epsilon * (\sigma_ij / r_ij) ^ 12
        energy_r12 = ene_r6 * sigma_over_rij_6 * self.r12_scale

        # (B,1) <- (B,p)
        energy_r = keepdims_sum(energy_r, -1)
        energy_r6 = keepdims_sum(energy_r6, -1)
        energy_r12 = keepdims_sum(energy_r12, -1)

        # (B, 1)
        energy = energy_r + energy_r6 + energy_r12

        return energy
