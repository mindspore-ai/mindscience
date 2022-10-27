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
"""Lennard-Jones potential"""

import mindspore as ms
import mindspore.numpy as msnp
from mindspore import Tensor, Parameter
from mindspore.ops import functional as F

from .energy import NonbondEnergy
from ...function import functions as func
from ...function.functions import gather_values
from ...function.units import Units


class LennardJonesEnergy(NonbondEnergy):
    r"""
    Lennard-Jones potential

    .. Math::

        E_{lj}(r_{ij}) = 4 * \epsilon_{ij} * [(\sigma_{ij} / r_{ij}) ^ {12} - (\sigma_{ij} / r_{ij}) ^ 6]

        \epsilon_{ij} = \sqrt(\epsilon_i * \epsilon_j)

        \sigma_{ij} = 1 / 2 * (\sigma_i + \sigma_j)

    ...

    Args:
        epsilon (Tensor):   Tensor of shape (B, A). Data type is float.
                            Parameter \epsilon for LJ potential. Default: None
        sigma (Tensor):     Tensor of shape (B, A). Data type is float.
                            Parameter \sigma in LJ potential. Default: None
        mean_c6 (Tensor):   Tensor of shape (B, A). Data type is float.
                            Average dispersion (<C6>) of the system used for
                            long range correction of dispersion interaction. Default: 0
        parameters (dict):  Force field parameters. Default: None
        cutoff (float):     Cutoff distance. Default: None
        use_pbc (bool):     Whether to use periodic boundary condition. Default: None
        length_unit (str):  Length unit for position coordinates. Default: 'nm'
        energy_unit (str):  Energy unit. Default: 'kj/mol'
        units (Units):      Units of length and energy. Default: None

    Returns:
        energy (Tensor), Tensor of shape (B, 1). Data type is float.

    Symbols:
        B:  Batchsize, i.e. number of walkers in simulation.
        A:  Number of atoms.
        N:  Maximum number of neighbour atoms.
        D:  Dimension of the simulation system. Usually is 3.

    Supported Platforms:
        ``Ascend`` ``GPU``
    """

    def __init__(self,
                 epsilon: Tensor = None,
                 sigma: Tensor = None,
                 mean_c6: Tensor = 0,
                 parameters: dict = None,
                 cutoff: float = None,
                 use_pbc: bool = None,
                 length_unit: str = 'nm',
                 energy_unit: str = 'kj/mol',
                 units: Units = None,
                 ):

        super().__init__(
            label='vdw_energy',
            output_dim=1,
            cutoff=cutoff,
            use_pbc=use_pbc,
            length_unit=length_unit,
            energy_unit=energy_unit,
            units=units,
        )

        if parameters is not None:
            length_unit = parameters.get('length_unit')
            energy_unit = parameters.get('energy_unit')
            self.units.set_units(length_unit, energy_unit)

            epsilon = parameters.get('epsilon')
            sigma = parameters.get('sigma')
            mean_c6 = parameters.get('mean_c6')

        sigma = Tensor(sigma, ms.float32)
        epsilon = Tensor(epsilon, ms.float32)

        if sigma.shape[-1] != epsilon.shape[-1]:
            raise ValueError('the last dimension of sigma'+str(sigma.shape[-1]) +
                             'must be equal to the last dimension of epsilon ('+str(epsilon.shape[-1])+')!')

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
            mean_c6 = Tensor(mean_c6, ms.float32)
            if mean_c6.ndim == 0:
                mean_c6 = mean_c6.reshape(1, 1)
            elif mean_c6.ndim == 1:
                mean_c6 = F.expand_dims(mean_c6, 0)
            elif mean_c6.ndim > 2:
                raise ValueError('The rank of mean_c6 cannot be larger than 2!')
            self.mean_c6 = Parameter(Tensor(mean_c6, ms.float32), name='average_dispersion', requires_grad=False)

        self.disp_corr = self._calc_disp_corr()

    def set_cutoff(self, cutoff: float):
        """
        Set cutoff distance.

        Args:
            cutoff (float):     Cutoff distance. Default: None.
        """
        super().set_cutoff(cutoff)
        self.disp_corr = self._calc_disp_corr()
        return self

    def _calc_disp_corr(self) -> Tensor:
        """
        calculate the long range correct factor for dispersion

        Returns:
            Tensor, the long range correct factor for dispersion.
        """

        if self.cutoff is None:
            return 0
        return -2.0 / 3.0 * msnp.pi * self.num_atoms**2 / msnp.power(self.cutoff, 3)

    def construct(self,
                  coordinate: Tensor,
                  neighbour_index: Tensor = None,
                  neighbour_mask: Tensor = None,
                  neighbour_coord: Tensor = None,
                  neighbour_distance: Tensor = None,
                  inv_neigh_dis: Tensor = None,
                  pbc_box: Tensor = None,
                  ):
        r"""
        Calculate energy term

        Args:
            coordinate (Tensor):            Tensor of shape (B, A, D). Data type is float.
                                            Position coordinate of atoms in system.
            neighbour_index (Tensor):       Tensor of shape (B, A, N). Data type is int.
                                            Index of neighbour atoms.
            neighbour_mask (Tensor):        Tensor of shape (B, A, N). Data type is bool.
                                            Mask for neighbour index.
            neighbour_coord (Tensor):       Tensor of shape (B, A, N). Data type is bool.
                                            Position coorindates of neighbour atoms.
            neighbour_distance (Tensor):    Tensor of shape (B, A, N). Data type is float.
                                            Distance between neighbours atoms.
            inv_neigh_dis (Tensor):         Tensor of shape (B, A, N). Data type is float.
                                            Reciprocal of distances.
            pbc_box (Tensor):               Tensor of shape (B, D). Data type is float.
                                            Tensor of PBC box. Default: None

        Returns:
            energy (Tensor), Tensor of shape (B, 1). Data type is float.

        Symbols:
            B:  Batchsize, i.e. number of walkers in simulation.
            A:  Number of atoms.
            D:  Dimension of the simulation system. Usually is 3.

        """

        inv_neigh_dis *= self.inverse_input_scale

        epsilon = self.identity(self.epsilon)
        sigma = self.identity(self.sigma)

        # (B,A,1)
        eps_i = F.expand_dims(epsilon, -1)
        # (B,A,N)
        eps_j = gather_values(epsilon, neighbour_index)
        # (B,A,N) = (B,A,1) * (B,A,N)
        eps_ij = F.sqrt(eps_i * eps_j)

        # (B,A,1)
        sigma_i = F.expand_dims(sigma, -1)
        # (B,A,N)
        sigma_j = gather_values(sigma, neighbour_index)
        # (B,A,N) = (B,A,1) * (B,A,N)
        sigma_ij = (sigma_i + sigma_j) * 0.5

        # \sigma_ij / r_ij
        sigma_over_rij = sigma_ij * inv_neigh_dis
        # (\sigma_ij / r_ij) ^ 6
        sigma_over_rij_6 = F.pows(sigma_over_rij, 6)

        # 4 * \epsilon * (\sigma_ij / r_ij) ^ 6
        ene_bcoeff = 4 * eps_ij * sigma_over_rij_6
        # 4 * \epsilon * (\sigma_ij / r_ij) ^ 12
        ene_acoeff = ene_bcoeff * sigma_over_rij_6

        # (B,A,N)
        energy = ene_acoeff - ene_bcoeff

        # (B,A)
        energy = F.reduce_sum(energy, -1)
        # (B,1)
        energy = func.keepdim_sum(energy, -1) * 0.5

        if self.cutoff is not None and pbc_box is not None:
            # (B,1) <- (B,D)
            volume = func.keepdim_prod(pbc_box, -1)
            # E_corr = -2 / 3 * pi * N * \rho * <C_6> * r_c^-3
            #        = -2 / 3 * pi * N * (N / V) * <C_6> * r_c^-3
            #        = -2 / 3 * pi * N^2 * <C_6> / V
            #        = k_corr * <C_6> / V
            ene_corr = self.disp_corr * self.mean_c6 * msnp.reciprocal(volume)
            energy += ene_corr

        return energy
