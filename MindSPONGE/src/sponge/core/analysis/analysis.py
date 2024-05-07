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
    r"""
    Cell for analysis.

    Args:
        system (:class:`sponge.system.Molecule`): Simulation system.
        potential (:class:`sponge.potential.PotentialCell`): Potential energy.
        neighbour_list (:class:`sponge.partition.NeighbourList`, optional): Neighbour list.
          Default: ``None``.

    Inputs:
        - **coordinate** (Tensor) - Coordinate. Tensor of shape :math:`(B, A, D)`.
          Data type is float.
          Here :math:`B` is the number of walkers in simulation,
          :math:`A` is the number of atoms, and
          :math:`D` is the spatial dimension of the simulation system, which is usually 3.
        - **pbc_box** (Tensor) - Periodic boundary condition box.
          Tensor of shape :math:`(B, D)`. Data type is float.
        - **energy** (Tensor) - Energy. Tensor of shape :math:`(B, 1)`. Data type is float.
        - **force** (Tensor) - Force. Tensor of shape :math:`(B, A, D)`. Data type is float.
        - **potentials** (Tensor, optional) - Original potential energies from force field.
          Tensor of shape :math:`(B, U)`.
          Here :math:`U` is the number of potential energies.
          Data type is float.  Default: ``0``.
        - **total_bias** (Tensor, optional) - Total bias energy for reweighting.
          Tensor of shape :math:`(B, 1)`. Data type is float. Default: ``0``.
        - **biases** (Tensor, optional) - Original bias potential energies from bias functions.
          Tensor of shape :math:`(B, V)`.
          Here `V` is the number of bias potential energies. Data type is float. Default: ``0``.

    Outputs:
        - **coordinate** (Tensor) - with shape of :math:`(B, A, D)`. Coordinate.
          Data type is float.
        - **pbc_box** (Tensor) - with shape of :math:`(B, D)`, PBC box. Data type is float.
        - **energy** (Tensor) - with shape of :math:`(B, 1)`,
          Total potential energy of the simulation system.
          Data type is float.
        - **force** (Tensor) - with shape of :math:`(B, A, D)`.
          Force on each atom of the simulation system.
          Data type is float.
        - **potentials** (Tensor) - with shape of :math:`(B, U)`. Original potential energies from force field.
          Data type is float.
        - **total_bias** (Tensor) - with shape of :math:`(B, 1)`. Total bias energy for reweighting.
          Data type is float.
        - **biases** (Tensor) - with shape of :math:`(B, V)`. Bias potential energies from bias functions.
          Data type is float.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> from sponge.system import Molecule
        >>> from sponge.potential.forcefield import ForceField
        >>> from sponge.core.sponge import Sponge
        >>> from sponge.core.analysis import AnalysisCell
        >>> system = Molecule(template='water.tip3p.yaml')
        >>> potential = ForceField(system, parameters='SPCE')
        >>> analysis = AnalysisCell(system, potential)
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
            coordinate (Tensor): Position coordinate of atoms in system.
              Tensor of shape :math:`(B, A, D)` .
              Here :math:`B` is the number of walkers in simulation,
              :math:`A` is the number of atoms,
              :math:`D` is the spatial dimension of the simulation system,
              which is usually 3.
              Data type is float.
              Default: ``None``.
            pbc_box (Tensor): Tensor of PBC box.
              Tensor of shape :math:`(B, D)`. Data type is float.
              Default: ``None``.
            energy (Tensor): Total potential energy of the simulation system.
              Tensor of shape :math:`(B, 1)`.
              Data type is float.
              Default: ``0``.
            force (Tensor): Force on each atoms of the simulation system.
              Tensor of shape :math:`(B, A, D)`.
              Data type is float.
              Default: ``0``.
            potentials (Tensor): Original potential energies from force field.
              Tensor of shape :math:`(B, U)`.
              Here :math:`U` is the number of potential energies. Data type is float.
              Default: ``0``.
            total_bias (Tensor): Total bias energy for reweighting.
              Tensor of shape :math:`(B, 1)`. Data type is float.
              Default: ``0``.
            biases (Tensor): Original bias potential energies from bias functions.
              Tensor of shape :math:`(B, V)`.
              Here :math:`V` is the number of bias potential energies.
              Data type is float. Default: ``0``.

        Returns:
            - coordinate, Tensor of shape :math:`(B, A, D)`. Position
              coordinate of atoms in system.
              Data type is float.
            - pbc_box, Tensor of shape :math:`(B, D)`. PBC box.
              Data type is float.
            - energy, Tensor of shape :math:`(B, 1)`. Total potential energy
              of the simulation system. Data type is float.
            - force, Tensor of shape :math:`(B, A, D)`. Force on each atoms
              of the simulation system. Data type is float.
            - potentials, Tensor of shape :math:`(B, U)`. Original potential
              energies from force field. Data type is float.
            - total_bias, Tensor of shape :math:`(B, 1)`. Total bias energy
              for reweighting. Data type is float.
            - biases, Tensor of shape :math:`(B, V)`. Original bias potential
              energies from bias functions. Data type is float.
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
