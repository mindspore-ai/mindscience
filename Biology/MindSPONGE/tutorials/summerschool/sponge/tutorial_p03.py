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
# MindSPONGE tutorial protein 03: Sampling Process: Minimization - NVT - NPT - Product.
"""
MindSPONGE basic tutorial p03
"""

import argparse
import sys
sys.path.append('../../../src')
from mindspore import context
from mindspore import Tensor
from sponge import ForceField
from sponge.system import get_molecule
from sponge import set_global_units, WithEnergyCell
from sponge.core import Sponge
from sponge.callback import WriteH5MD, RunInfo
from sponge.function import VelocityGenerator
from sponge import UpdaterMD

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

mol = get_molecule('p02.pdb', template=['protein0.yaml', 'water.spce.yaml'])
mol.set_pbc_box(pbc_box=Tensor([2.1184928, 2.285693, 1.549593]))

if args.e == 'Ascend':
    energy = ForceField(mol, parameters=['AMBER.FF14SB', 'SPCE'], use_pme=True)
else:
    energy = ForceField(mol, parameters=['AMBER.FF14SB', 'SPCE'], use_pme=False)

sim = WithEnergyCell(mol, energy)

temp = 300
vgen = VelocityGenerator(temp)
velocity = vgen(mol.shape, mol.atom_mass)
opt = UpdaterMD(system=mol,
                time_step=1e-3,
                velocity=velocity,
                integrator='velocity_verlet',
                temperature=300,
                thermostat='langevin',)
md = Sponge(sim, optimizer=opt)
cb_h5md = WriteH5MD(mol, 'p03_1.h5md', save_freq=10, write_image=False)
run_info = RunInfo(10)
md.run(1000, callbacks=[run_info, cb_h5md])

# NPT(control the pressure)

temp = 300
vgen = VelocityGenerator(temp)
velocity = vgen(mol.shape, mol.atom_mass)
opt = UpdaterMD(system=mol,
                time_step=1e-3,
                velocity=velocity,
                integrator='velocity_verlet',
                temperature=300,
                thermostat='langevin',
                pressure=1,
                barostat='berendsen',)
md.change_optimizer(opt)
cb_h5md = WriteH5MD(mol, 'p03_2.h5md', save_freq=10, write_image=False)
md.run(1000, callbacks=[run_info, cb_h5md])

# Product simulation(10000 steps for example)

temp = 300
vgen = VelocityGenerator(temp)
velocity = vgen(mol.shape, mol.atom_mass)
opt = UpdaterMD(system=mol,
                time_step=1e-3,
                velocity=velocity,
                integrator='velocity_verlet',
                temperature=300,
                thermostat='langevin',
                pressure=1,
                barostat='berendsen',)
md.change_optimizer(opt)
cb_h5md = WriteH5MD(mol, 'p03_3.h5md', save_freq=10, write_image=False)
md.run(10000, callbacks=[run_info, cb_h5md])
