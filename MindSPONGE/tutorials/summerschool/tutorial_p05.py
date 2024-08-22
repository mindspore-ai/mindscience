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
"""MindSPONGE tutorial protein 05"""

import os
import argparse
import sys
from mindspore import context
sys.path.insert(0, '../../src') # pylint: disable=C0413
from sponge import Sponge
from sponge import ForceField
from sponge import set_global_units, WithEnergyCell
from sponge import UpdaterMD
from sponge.optimizer import SteepestDescent
from sponge.control import VelocityVerlet
from sponge.callback import WriteH5MD, RunInfo
from sponge.control import Langevin
from sponge.function import VelocityGenerator
from sponge.colvar import Torsion
from sponge.sampling import Metadynamics
from sponge.function import PI
from sponge.system import Protein

os.environ['GLOG_v'] = '4'
os.environ['MS_JIT_MODULES'] = 'sponge'
parser = argparse.ArgumentParser()
parser.add_argument("-e", help="Set the backend.", default="Ascend")
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

pdb_name = '../pdb/case1.pdb'
out_pdb = 'case1_sol.pdb'
mol = Protein(pdb_name, template=['protein0.yaml'], rebuild_hydrogen=True)
mol.fill_water(edge=0.4, pdb_out=out_pdb, template='water.spce.yaml')

if args.e == 'Ascend':
    energy = ForceField(mol, parameters=['AMBER.FF14SB', 'SPCE'], use_pme=True)
else:
    energy = ForceField(mol, parameters=['AMBER.FF14SB', 'SPCE'], use_pme=True)

min_opt = SteepestDescent(mol.trainable_params(), 1e-6)
sim = WithEnergyCell(mol, energy)
md = Sponge(sim, optimizer=min_opt)
run_info = RunInfo(10)
cb_h5md = WriteH5MD(mol, 'p05_1.h5md', save_freq=10, save_last_pdb='p05.pdb', write_image=False)
md.run(500, callbacks=[run_info, cb_h5md])

# Step 3: NVT

temp = 300
vgen = VelocityGenerator(temp)
velocity = vgen(mol.shape, mol.atom_mass)
opt = UpdaterMD(system=mol,
                time_step=1e-3,
                velocity=velocity,
                integrator='velocity_verlet',
                temperature=300,
                thermostat='langevin',)
md.change_optimizer(opt)
cb_h5md = WriteH5MD(mol, 'p05_2.h5md', save_freq=10, write_image=False)
md.run(1000, callbacks=[run_info, cb_h5md])

# Step 4: NPT

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
cb_h5md = WriteH5MD(mol, 'p05_3.h5md', save_freq=10, write_image=False)
md.run(1000, callbacks=[run_info, cb_h5md])

# Step 5: define CVs

phi = Torsion([3, 12, 13, 21])
psi = Torsion([12, 13, 21, 36])

# Step 6: Normal MD

sim = WithEnergyCell(mol, energy)
vgen = VelocityGenerator(300)
velocity = vgen(mol.shape, mol.atom_mass)
opt = UpdaterMD(
    system=mol,
    integrator=VelocityVerlet(mol),
    thermostat=Langevin(mol, 300),
    time_step=1e-3,
    velocity=velocity,
    pressure=1,
    barostat='berendsen',
)
md = Sponge(sim, optimizer=opt, metrics={'phi': phi, 'psi': psi})
cb_h5md = WriteH5MD(mol, 'p05_a.h5md', save_freq=100, write_image=False)
md.run(10000, callbacks=[run_info, cb_h5md])

# Step 7: MetaD settings

metad = Metadynamics(
    colvar=[phi, psi],
    update_pace=10,
    height=2.5,
    sigma=0.05,
    grid_min=-PI,
    grid_max=PI,
    grid_bin=360,
    temperature=300,
    bias_factor=100,
)

# Step 8: Enhanced Sampling using MetaD

sim = WithEnergyCell(mol, energy, bias=metad)
vgen = VelocityGenerator(300)
velocity = vgen(mol.shape, mol.atom_mass)
opt = UpdaterMD(
    system=mol,
    integrator=VelocityVerlet(mol),
    thermostat=Langevin(mol, 300),
    time_step=1e-3,
    velocity=velocity,
    pressure=1,
    barostat='berendsen',
)
md = Sponge(sim, optimizer=opt, metrics={'phi': phi, 'psi': psi})
cb_h5md = WriteH5MD(mol, 'p05_b.h5md', save_freq=100, write_image=False)
md.run(10000, callbacks=[run_info, cb_h5md])
