# %% [markdown]
# Copyright 2021-2023 @ Shenzhen Bay Laboratory & Peking University & Huawei Technologies Co., Ltd
#
# This code is a part of MindSPONGE: MindSpore Simulation Package tOwards Next Generation molecular modelling.
#
# MindSPONGE is open-source software based on the AI-framework: MindSpore (https://www.mindspore.cn/)
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
#
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#
# See the License for the specific language governing permissions and limitations under the License.
#
# MindSPONGE tutorial protein 02: Add solvent and Periodic boundary conditions.
"""
MindSPONGE basic tutorial p02
"""

import argparse
import sys
sys.path.append('../../../src')
from mindspore import context, nn
from sponge import ForceField
from sponge.system import get_molecule
from sponge import set_global_units, WithEnergyCell
from sponge.core import Sponge
from sponge.callback import WriteH5MD, RunInfo

parser = argparse.ArgumentParser()
parser.add_argument("-e", help="Set the backend.", default="GPU")
parser.add_argument("-m", help="Set the compile mode.", default="0")
parser.add_argument("-g", help="Choose to use graph kernel compilation or not.", default="0")
parser.add_argument("-id", help="Set the backend index.", default="0")
args = parser.parse_args()

if args.g == '0':
    enable_graph_kernel = False
else:
    enable_graph_kernel = True


if args.m == '0':
    context.set_context(mode=context.GRAPH_MODE, device_target=args.e, device_id=int(args.id),
                        enable_graph_kernel=enable_graph_kernel)
else:
    context.set_context(mode=context.PYNATIVE_MODE, device_target=args.e, device_id=int(args.id))

set_global_units('nm', 'kj/mol')

pdb_name = './case1.pdb'
out_pdb = 'case1_sol.pdb'

mol = get_molecule(pdb_name, template=['protein0.yaml'], rebuild_hydrogen=True)
mol.fill_water(edge=0.4, pdb_out=out_pdb, template='water.spce.yaml')

if args.e == 'Ascend':
    energy = ForceField(mol, parameters=['AMBER.FF14SB', 'SPCE'], use_pme=True)
else:
    energy = ForceField(mol, parameters=['AMBER.FF14SB', 'SPCE'], use_pme=False)

min_opt = nn.Adam(mol.trainable_params(), 1e-3)
sim = WithEnergyCell(mol, energy)
md = Sponge(sim, optimizer=min_opt)
run_info = RunInfo(10)
cb_h5md = WriteH5MD(mol, 'p02.h5md', save_freq=10, save_last_pdb='p02.pdb', write_image=False)
md.run(800, callbacks=[run_info, cb_h5md])

print('The final pbc box size is: {}'.format(mol.pbc_box.asnumpy()))
