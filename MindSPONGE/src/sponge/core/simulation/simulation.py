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
Simulation Cell
"""

from ...partition import NeighbourList
from ...system import Molecule
from ...potential import PotentialCell
from ...potential.bias import Bias
from ...sampling.wrapper import EnergyWrapper
from .energy import WithEnergyCell


class SimulationCell(WithEnergyCell):
    r"""Cell for simulation, equivalent to `WithEnergyCell`.
        NOTE: This Cell will be removed a future release, Please use `WithEnergyCell` instead.

    Args:

        system (Molecule):              Simulation system.

        potential (PotentialCell):      Potential energy function cell.

        bias (Union[Bias, List[Bias]]): Bias potential function cell: Default: None

        cutoff (float):                 Cut-off distance for neighbour list. If None is given, it will be assigned
                                        as the cutoff value of the of potential energy.
                                        Defulat: None

        neighbour_list (NeighbourList): Neighbour list. Default: None

        wrapper (EnergyWrapper):        Network to wrap and process potential and bias.
                                        Default: None

    Supported Platforms:

        ``Ascend`` ``GPU``

    """

    def __init__(self,
                 system: Molecule,
                 potential: PotentialCell,
                 bias: Bias = None,
                 cutoff: float = None,
                 neighbour_list: NeighbourList = None,
                 wrapper: EnergyWrapper = None,
                 ):

        super().__init__(
            system=system,
            potential=potential,
            cutoff=cutoff,
            neighbour_list=neighbour_list,
            wrapper=wrapper,
            bias=bias,
        )

        print('[WARNING] `SimulationCell` will be removed a future release. '
              'Please use "WithEnergyCell" instead.')
