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
"""
Simulation Cell
"""

import mindspore as ms
import mindspore.numpy as msnp
from mindspore import Tensor
from mindspore import Parameter
from mindspore import context
from mindspore import ops, nn
from mindspore.ops import functional as F
from mindspore.nn import Cell, CellList

from ...partition import NeighbourList
from ...system import Molecule
from ...potential import PotentialCell
from ...potential.bias import Bias
from ...function.functions import gather_vectors
from ...function.operations import GetVector
from ..wrapper import EnergyWrapper, get_energy_wrapper


class SimulationCell(Cell):
    r"""
    Core cell for simulation.

    Args:
        system (Molecule):              Simulation system.
        potential (PotentialCell):      Potential energy.
        cutoff (float):                 Cutoff distance. Default: None
        neighbour_list (NeighbourList): Neighbour list. Default: None
        wrapper (EnergyWrapper):        Network to wrap and process potential and bias.
                                        Default: 'sum'
        bias (Bias):                    Bias potential: Default: None

    Supported Platforms:
        ``Ascend`` ``GPU``
    """

    def __init__(self,
                 system: Molecule,
                 potential: PotentialCell,
                 cutoff: float = None,
                 neighbour_list: NeighbourList = None,
                 wrapper: EnergyWrapper = 'sum',
                 bias: Bias = None,
                 ):

        super().__init__(auto_prefix=False)

        self.system = system
        self.potential = potential

        self.bias_network = None
        self.num_bias = 0
        if bias is not None:
            if isinstance(bias, list):
                self.num_bias = len(bias)
                self.bias_network = CellList(bias)
            elif isinstance(bias, Cell):
                self.num_bias = 1
                self.bias_network = CellList([bias])
            else:
                raise TypeError('The "bias" must be Cell or list but got: '+str(type(bias)))

        self.num_walker = self.system.num_walker
        self.num_atoms = self.system.num_atoms

        self.dim_potential = self.potential.output_dim
        self.dim_bias = 0
        if self.bias_network is not None:
            self.dim_bias = len(self.bias_network)

        self.energy_wrapper = get_energy_wrapper(
            wrapper,
            num_walker=self.num_walker,
            dim_potential=self.dim_potential,
            dim_bias=self.dim_bias,
        )

        self.exclude_index = self.potential.exclude_index
        self.neighbour_list = neighbour_list
        if neighbour_list is None:
            self.neighbour_list = NeighbourList(
                system, cutoff, exclude_index=self.exclude_index)
        else:
            self.neighbour_list.set_exclude_index(self.exclude_index)

        self.neighbour_index = self.neighbour_list.neighbours
        self.neighbour_mask = self.neighbour_list.neighbour_mask

        self.no_mask = False
        if context.get_context("mode") == context.PYNATIVE_MODE and self.neighbour_list.no_mask:
            self.no_mask = True

        self.num_neighbours = self.neighbour_list.num_neighbours

        self.cutoff = self.neighbour_list.cutoff
        if self.cutoff is not None:
            self.potential.set_cutoff(self.cutoff)
        self.nl_update_steps = self.neighbour_list.update_steps

        self.coordinate = self.system.coordinate
        self.pbc_box = self.system.pbc_box
        self.atom_mass = self.system.atom_mass

        self.pbc_box = self.system.pbc_box
        use_pbc = self.pbc_box is not None

        self.potential.set_pbc(use_pbc)

        for p in self.potential.trainable_params():
            p.requires_grad = False

        self.units = self.system.units

        self.potential_units = self.potential.units

        self.input_unit_scale = Tensor(self.units.convert_length_to(
            self.potential.length_unit), ms.float32)
        self.output_unit_scale = Tensor(self.units.convert_energy_from(
            self.potential.energy_unit), ms.float32)

        self.get_vector = GetVector(use_pbc)

        mask_fill = self.units.length(10, 'nm')
        self.mask_fill = Tensor(mask_fill, ms.float32)

        self.identity = ops.Identity()

        self.bias = None
        if self.bias_network is not None:
            self.bias = Parameter(msnp.zeros((self.num_walker, self.num_bias), dtype=ms.float32),
                                  name='bias_potential', requires_grad=False)

        self.norm_last_dim = nn.Norm(axis=-1, keep_dims=False)

        self.norm_last_dim = nn.Norm(axis=-1, keep_dims=False)

    @property
    def length_unit(self):
        return self.units.length_unit

    @property
    def energy_unit(self):
        return self.units.energy_unit

    def set_pbc_grad(self, grad_box: bool):
        """
        set whether to calculate the gradient of PBC box.

        Args:
            grad_box (bool):    Whether to calculate the gradient of PBC box.
        """
        self.system.set_pbc_grad(grad_box)
        return self

    def update_neighbour_list(self):
        """update neighbour list."""
        coordinate, pbc_box = self.system()
        return self.neighbour_list(coordinate, pbc_box)

    def get_neighbour_list(self):
        """
        get neighbour list.

        Returns:
            - neighbour_index (Tensor).
            - neighbour_mask (Tensor).
        """
        neighbour_index, neighbour_mask = self.neighbour_list.get_neighbour_list()
        return neighbour_index, neighbour_mask

    def construct(self, *inputs):
        """
        calculate the energy of system.

        Returns:
            - energy (Tensor).
            - force (Tensor).
        """
        #pylint: disable=unused-argument
        coordinate, pbc_box = self.system()

        coordinate *= self.input_unit_scale
        if pbc_box is not None:
            pbc_box *= self.input_unit_scale

        neighbour_index, neighbour_mask = self.get_neighbour_list()

        # (B,A,1,D) <- (B,A,D):
        atoms = F.expand_dims(coordinate, -2)
        # (B,A,N,D) <- (B,A,D):
        neighbour_coord = gather_vectors(coordinate, neighbour_index)
        neighbour_vector = self.get_vector(atoms, neighbour_coord, pbc_box)

        # Add a non-zero value to the neighbour_vector whose mask value is False
        # to prevent them from becoming zero values after Norm operation,
        # which could lead to auto-differentiation errors
        if neighbour_mask is not None:
            # (B,A,N):
            mask_fill = msnp.where(neighbour_mask, 0, self.mask_fill)
            # (B,A,N,D) = (B,A,N,D) + (B,A,N,1)
            neighbour_vector += F.expand_dims(mask_fill, -1)

        # (B,A,N) = (B,A,N,D):
        neighbour_distance = self.norm_last_dim(neighbour_vector)

        if self.cutoff is not None:
            distance_mask = neighbour_distance < self.cutoff
            if neighbour_mask is None:
                neighbour_mask = distance_mask
            else:
                neighbour_mask = F.logical_and(distance_mask, neighbour_mask)

        potential = self.potential(
            coordinate=coordinate,
            neighbour_index=neighbour_index,
            neighbour_mask=neighbour_mask,
            neighbour_coord=neighbour_coord,
            neighbour_distance=neighbour_distance,
            pbc_box=pbc_box
        ) * self.output_unit_scale

        bias = None
        if self.bias_network is not None:
            bias = ()
            for i in range(self.num_bias):
                bias_ = self.bias_network[i](
                    coordinate=coordinate,
                    neighbour_index=neighbour_index,
                    neighbour_mask=neighbour_mask,
                    neighbour_coord=neighbour_coord,
                    neighbour_distance=neighbour_distance,
                    pbc_box=pbc_box
                    )
                bias += (bias_,)

            bias = msnp.concatenate(bias, axis=-1) * self.output_unit_scale
            F.depend(potential, F.assign(self.bias, bias))

        return self.energy_wrapper(potential, bias)
