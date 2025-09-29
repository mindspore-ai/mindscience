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
Analyse Cell
"""

import mindspore as ms
from mindspore import ops
from mindspore.nn import Cell
from mindspore.common import Tensor

from ...system import Molecule
from ...potential import PotentialCell
from ...partition import NeighbourList


class AnalyseCell(Cell):
    r"""
    Core cell for analysis.

    Args:
        system (Molecule):              Simulation system.
        potential (PotentialCell):      Potential energy.
        neighbour_list (NeighbourList): Neighbour list. Default: None
        calc_energy (bool):             Whether to calculate the energy. Default: False
        calc_forces (bool):             Whether to calculate the forces. Default: False

    Outputs:
        - energy.
        - forces.
        - coordinates.
        - pbc_box.

    Supported Platforms:
        ``Ascend`` ``GPU``
    """
    def __init__(self,
                 system: Molecule,
                 potential: PotentialCell,
                 neighbour_list: NeighbourList = None,
                 calc_energy: bool = False,
                 calc_forces: bool = False,
                 ):

        super().__init__(auto_prefix=False)

        self.system = system
        self.potential = potential
        self.pbc_box = self.system.pbc_box

        self.neighbour_list = neighbour_list
        if neighbour_list is None:
            self.neighbour_list = NeighbourList(system)

        self.calc_energy = calc_energy
        self.calc_forces = calc_forces

        self.system_units = self.system.units
        self.potential_units = self.potential.units

        self.units = self.system.units

        self.input_unit_scale = Tensor(self.units.convert_length_to(
            self.potential.length_unit()), ms.float32)
        self.output_unit_scale = Tensor(self.units.convert_energy_from(
            self.potential.energy_unit()), ms.float32)

        self.grad = ops.GradOperation()

    def construct(self, coordinates=None, pbc_box=None):
        """analyse the system."""
        if coordinates is None:
            coordinates, pbc_box = self.system()

        coordinates *= self.input_unit_scale
        if self.pbc_box is not None:
            pbc_box *= self.input_unit_scale

        energy = None
        if self.calc_energy:
            energy = self.potential(coordinates, pbc_box)

        forces = None
        if self.calc_forces:
            forces = -self.grad(self.potential)(coordinates, pbc_box)

        return energy, forces, coordinates, pbc_box
