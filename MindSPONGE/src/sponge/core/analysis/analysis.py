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
Analyse Cell
"""

from typing import Tuple
import mindspore as ms
from mindspore import ops
from mindspore.nn import Cell
from mindspore.common import Tensor

from ...system import Molecule
from ...potential import PotentialCell
from ...partition import NeighbourList


class AnalysisCell(Cell):
    r"""Cell for analysis

    Args:
        system (Molecule):              Simulation system.

        potential (PotentialCell):      Potential energy.

        neighbour_list (NeighbourList): Neighbour list. Default: ``None``.

    Supported Platforms:
        ``Ascend`` ``GPU``

    """
    def __init__(self,
                 system: Molecule,
                 potential: PotentialCell,
                 neighbour_list: NeighbourList = None,
                 ):

        super().__init__(auto_prefix=False)

        self.system = system
        self.potential = potential
        self.pbc_box = self.system.pbc_box

        self.neighbour_list = neighbour_list
        if neighbour_list is None:
            self.neighbour_list = NeighbourList(system)

        self.system_units = self.system.units
        self.potential_units = self.potential.units

        self.units = self.system.units

        self.length_unit_scale = Tensor(self.units.convert_length_to(
            self.potential.length_unit), ms.float32)
        self.energy_unit_scale = Tensor(self.units.convert_energy_to(
            self.potential.energy_unit), ms.float32)
        self.force_unit_scale = self.energy_unit_scale / self.length_unit_scale

        self.grad = ops.GradOperation()

    def construct(self,
                  coordinate: Tensor = None,
                  pbc_box: Tensor = None,
                  energy: Tensor = 0,
                  force: Tensor = 0,
                  potentials: Tensor = 0,
                  total_bias: Tensor = 0,
                  biases: Tensor = 0,
                  ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """

        Args:
            coordinate (Tensor):    Tensor of shape (B, A, D). Data type is float.
                                    Position coordinate of atoms in system. Default: ``None``.
            pbc_box (Tensor):       Tensor of shape (B, D). Data type is float.
                                    Tensor of PBC box. Default: ``None``.
            energy (Tensor):        Tensor of shape (B, 1). Data type is float.
                                    Total potential energy of the simulation system. Default: 0
            force (Tensor):         Tensor of shape (B, A, D). Data type is float.
                                    Force on each atoms of the simulation system. Default: 0
            potentials (Tensor):    Tensor of shape (B, U). Data type is float.
                                    Original potential energies from force field. Default: 0
            total_bias (Tensor):    Tensor of shape (B, 1). Data type is float.
                                    Total bias energy for reweighting. Default: 0
            biases (Tensor):        Tensor of shape (B, V). Data type is float
                                    Original bias potential energies from bias functions. Default: 0

        Returns:
            coordinate (Tensor):    Tensor of shape (B, A, D). Data type is float.
                                    Position coordinate of atoms in system.
            pbc_box (Tensor):       Tensor of shape (B, D). Data type is float.
                                    Tensor of PBC box.
            energy (Tensor):        Tensor of shape (B, 1). Data type is float.
                                    Total potential energy of the simulation system.
            force (Tensor):         Tensor of shape (B, A, D). Data type is float.
                                    Force on each atoms of the simulation system.
            potentials (Tensor):    Tensor of shape (B, U). Data type is float.
                                    Original potential energies from force field.
            total_bias (Tensor):    Tensor of shape (B, 1). Data type is float.
                                    Total bias energy for reweighting.
            biases (Tensor):        Tensor of shape (B, V). Data type is float
                                    Original bias potential energies from bias functions.

        Note:
            B:  Batchsize, i.e. number of walkers in simulation.
            A:  Number of atoms of the simulation system.
            D:  Dimension of the space of the simulation system. Usually is 3.
            U:  Number of potential energies.
            V:  Number of bias potential energies.
        """

        if coordinate is None:
            coordinate, pbc_box = self.system()

        coordinate *= self.length_unit_scale
        if pbc_box is None:
            pbc_box = 0
        else:
            pbc_box *= self.length_unit_scale


        if energy is None:
            energy = 0
        else:
            energy *= self.energy_unit_scale

        if force is None:
            force = 0
        else:
            force *= self.force_unit_scale

        if potentials is None:
            potentials = 0
        else:
            potentials *= self.energy_unit_scale

        if total_bias is None:
            total_bias = 0
        else:
            total_bias *= self.energy_unit_scale

        if biases is None:
            biases = 0
        else:
            biases *= self.energy_unit_scale

        return coordinate, pbc_box, energy, force, potentials, total_bias, biases
