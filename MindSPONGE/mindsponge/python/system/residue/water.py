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
Water
"""

from .residue import Residue
from ...function.functions import get_integer


class Water(Residue):
    r"""TODO: Water molecule

    Args:


    """

    def __init__(self,
                 num_points: int = 3,
                 template: dict = 'water_3p.yaml',
                 atom_name: list = None,
                 start_index: int = 0,
                 ):

        super().__init__(
            atom_name=atom_name,
            start_index=start_index,
            name='WAT',
            template=template,
        )

        self.num_points = get_integer(num_points)
        if self.num_points == 3:
            atom_name = ['O', 'H1', 'H2']
            atom_type = ['OW', 'HW', 'HW']
            atom_mass = [16.0, 1.008, 1.008]
            atom_charge = [-0.834, 0.417, 0.417]
            atomic_number = [8, 1, 1]
            bond = [[0, 1], [0, 2]]
        elif self.num_points == 4:
            atom_name = ['O', 'H1', 'H2', 'EP']
            atom_type = ['OW', 'HW', 'HW', 'EP']
            atom_mass = [16.0, 1.008, 1.008, 0]
            atom_charge = [0, 0.53, 0.52, -1.04]
            atomic_number = [8, 1, 1, 0]
            bond = [[0, 1], [0, 2], [1, 2]]
        elif self.num_points == 5:
            atom_name = ['O', 'H1', 'H2', 'EP1', 'EP2']
            atom_type = ['OW', 'HW', 'HW', 'EP', 'EP']
            atom_mass = [16.0, 1.008, 1.008, 0, 0]
            atom_charge = [0, 0.241, 0.241, -0.241, -0.241]
            atomic_number = [8, 1, 1, 0, 0]
            bond = [[0, 1], [0, 2], [1, 2]]
        else:
            raise ValueError(
                'The points of water model must be 3, 4 or 5 but got: '+str(self.num_points))
