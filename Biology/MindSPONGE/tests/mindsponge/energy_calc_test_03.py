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

EPSILON = 3e-02
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
set_global_units('A', 'kcal/mol')


def test_case1_energy():
    pdb = CURRENT_DIR + '/data/case1_addH.pdb'
    pro = Protein(pdb, rebuild_hydrogen=False)
    energy = ForceField(pro, parameters=['AMBER.FF19SB'])
    opt = nn.Adam(pro.trainable_params(), 1e-03)
    neighbours = NeighbourList(pro, cutoff=None)
    sim = WithEnergyCell(pro, energy, neighbour_list=neighbours)
    md = Sponge(sim, optimizer=opt)
    energies = md.calc_energies().asnumpy()[0]
    assert abs((energies[0] - 55.3429) / energies[0]) < EPSILON
    assert abs((energies[1] - 250.7942) / energies[1]) < EPSILON
    assert abs((energies[[2, 3]].sum() - 36.2521) / energies[[2,3]].sum()) < EPSILON
    assert abs((energies[4] + 1.2243) / energies[4]) < EPSILON
    assert abs((energies[5] + 164.6996) / energies[4]) < EPSILON
    assert abs((energies[6] - 968.3315) / energies[5]) < EPSILON
    assert abs((energies[7] - 28.0361 - 10.9454) / energies[6]) < EPSILON
    assert abs((energies.sum() - 1.1838E+03) / energies.sum()) < EPSILON


def test_case2_energy():
    pdb = CURRENT_DIR + '/data/case2_addH.pdb'
    pro = Protein(pdb, rebuild_hydrogen=False)
    energy = ForceField(pro, parameters=['AMBER.FF19SB'])
    opt = nn.Adam(pro.trainable_params(), 1e-03)
    neighbours = NeighbourList(pro, cutoff=None)
    sim = WithEnergyCell(pro, energy, neighbour_list=neighbours)
    md = Sponge(sim, optimizer=opt)
    energies = md.calc_energies().asnumpy()[0]
    assert abs((energies[0] - 4954.9515) / energies[0]) < EPSILON
    assert abs((energies[1] - 27317.6385) / energies[1]) < EPSILON
    assert abs((energies[[2,3]].sum() - 2406.9262) / energies[[2,3]].sum()) < EPSILON
    assert abs((energies[4] - 143.8608) / energies[4]) < EPSILON
    assert abs((energies[5] + 10840.3869) / energies[5]) < EPSILON
    assert abs((energies[6] + 111.1352) / energies[6]) < EPSILON
    assert abs((energies[7] - 2387.1636 - 6504.1785) / energies[7]) < EPSILON
    assert abs((energies.sum() - 3.2763E+04) / energies.sum()) < EPSILON
