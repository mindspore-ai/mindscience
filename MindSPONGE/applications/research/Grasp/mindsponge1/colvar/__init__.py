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
"""Collective variables"""

from .colvar import Colvar
from .base import Distance, Angle, Torsion
from .position import Atom, Position
from .atoms import AtomDistances, AtomAngles, AtomTorsions
from .bonded import BondedColvar, BondedDistances, BondedTorsions, BondedAngles
from .index import IndexColvar, IndexVectors, IndexDistances

__all__ = ['Colvar', 'Distance', 'Angle', 'Torsion', 'Atom', 'Position',
           'AtomDistances', 'AtomAngles', 'AtomTorsions', 'BondedColvar',
           'BondedDistances', 'BondedTorsions', 'BondedAngles', 'IndexColvar',
           'IndexVectors', 'IndexDistances']
