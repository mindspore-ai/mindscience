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
BT
"""

from typing import Union, List
import mindspore as ms
import mindspore.numpy as msnp
from mindspore import Tensor, Parameter
from mindspore.nn import Cell, CellList
from mindspore.ops import functional as F

from .energy import WithEnergyCell
from ...partition import NeighbourList
from ...system import Molecule
from ...potential import PotentialCell
from ...potential.bias import Bias
from ...sampling.wrapper import EnergyWrapper
from ...function import get_ms_array, keepdims_mean


class BT(WithEnergyCell):
    r"""Wrapper Cell for BT, which is a subclass of `WithEnergyCell`

    Args:

        system (Molecule): Simulation system.

        potential (PotentialCell): Potential energy function cell.

        kernel (Union[Bias, List[Bias]]): Kernel function cell.

        temperature (float): Simulation temperature.

        bias (Union[Bias, List[Bias]]): Bias potential function cell. Default: None

        cutoff (float): Cut-off distance for neighbour list. If None is given, it will be assigned
            as the cutoff value of the of potential energy. Defulat: None

        neighbour_list (NeighbourList): Neighbour list. Default: None

        wrapper (EnergyWrapper): Network to wrap and process potential and bias. Default: None

    Supported Platforms:

        ``Ascend`` ``GPU``

    Symbols:

        B:  Batchsize, i.e. number of walkers of the simulation.

        A:  Number of the atoms in the simulation system.

        N:  Number of the maximum neighbouring atoms.

        U:  Number of potential energy terms.

        V:  Number of bias potential terms.

    """

    def __init__(self,
                 system: Molecule,
                 potential: PotentialCell,
                 kernel: Union[Bias, List[Bias]],
                 temperature: float,
                 bias: Union[Bias, List[Bias]],
                 cutoff: float = None,
                 neighbour_list: NeighbourList = None,
                 wrapper: EnergyWrapper = None,
                 ):

        super().__init__(
            system=system,
            potential=potential,
            bias=bias,
            cutoff=cutoff,
            neighbour_list=neighbour_list,
            wrapper=wrapper,
        )

        self._num_kernels = 0
        self._kernel_names = []
        self.kernel_function: List[Bias] = None
        if isinstance(kernel, list):
            self._num_kernels = len(kernel)
            self.kernel_function = CellList(kernel)
        elif isinstance(kernel, Cell):
            self._num_kernels = 1
            self.kernel_function = CellList([kernel])
        else:
            raise TypeError(f'kernel must be Cell or list but got: {type(kernel)}')

        for i in range(self._num_kernels):
            self._kernel_names.append(self.kernel_function[i].name)

        self.temperature = get_ms_array(temperature)
        self.boltzmann = self.units.boltzmann
        # k_B T
        self.kbt = self.boltzmann * self.temperature
        # \beta = \frac{1}{k_B T}
        self.beta = msnp.reciprocal(self.kbt)

        self._kernels = Parameter(msnp.zeros((self.num_walker, self._num_kernels), dtype=ms.float32),
                                  name='kernels', requires_grad=False)
        self._kernel = Parameter(msnp.zeros((self.num_walker, 1), dtype=ms.float32),
                                 name='kernel', requires_grad=False)

    def construct(self, *inputs) -> Tensor:
        """calculate the total potential energy (potential energy and bias potential) of the simulation system.

        Return:
            beta_energy (Tensor):   Tensor of shape `(B, 1)`. Data type is float.
                                    :math:`\beta E(R)`.

        Symbols:
            B:  Batchsize, i.e. number of walkers of the simulation.

        """

        #pylint: disable=unused-argument
        coordinate, pbc_box = self.system()

        neigh_idx, neigh_vec, neigh_dis, neigh_mask = self.neighbour_list(coordinate, pbc_box)

        coordinate *= self.length_unit_scale
        neigh_vec *= self.length_unit_scale
        neigh_dis *= self.length_unit_scale
        if pbc_box is not None:
            pbc_box *= self.length_unit_scale

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

        # (B, 1)
        kernel = ()
        for i in range(self._num_kernels):
            kernel_ = self.kernel_function[i](
                coordinate=coordinate,
                neighbour_index=neigh_idx,
                neighbour_mask=neigh_mask,
                neighbour_vector=neigh_vec,
                neighbour_distance=neigh_dis,
                pbc_box=pbc_box
            )
            kernel += (kernel_,)
        kernel = msnp.concatenate(kernel, axis=-1)

        # (B, 1) + (B, 1)
        vfe = kernel + self.beta * energy

        # (B, 1) <- (1, 1)
        return keepdims_mean(vfe, 0)
