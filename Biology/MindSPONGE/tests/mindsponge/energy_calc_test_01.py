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
Test single point energy.
Running method:
$ python3 -m pytest
"""

import os
import sys
sys.path.insert(0, '../..')
from mindspore import nn, context
from sponge import Protein, ForceField, Sponge, WithEnergyCell, set_global_units
from sponge.partition import NeighbourList

EPSILON = 1e-03
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
set_global_units('A', 'kcal/mol')


def test_case2_energy():
    pdb = CURRENT_DIR + '/data/case1_addH.pdb'
    pro = Protein(pdb, rebuild_hydrogen=False)
    energy = ForceField(pro, parameters=['AMBER.FF14SB'])
    opt = nn.Adam(pro.trainable_params(), 1e-03)
    neighbours = NeighbourList(pro, cutoff=None, cast_fp16=True)
    sim = WithEnergyCell(pro, energy, neighbour_list=neighbours)
    md = Sponge(sim, optimizer=opt)
    total_energy = md.calc_energy().asnumpy()[0][0]
    assert abs((total_energy - 1.2002E+03) / total_energy) < EPSILON
