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
Homework-0822

    1.Build a PBC system for case1.pdb, run a normal product simulation of 10 ps, plot the phi-psi distribution.
        Hints: mainly base on tutorial_p03, and add CVs output for phi and psi.

    2.Use the same system, run a MetaD simulation of 10 ps, plot the phi-psi distribution.
        Hints: add MetaD modules, and modify the WithEnergyCell as tutorial_p04 done.

    3.compare the results, and analysis the effort of MetaD.

    Upload attachments contains:
        homework.py
        homework.ipynb
        results(.jpg or .png): plot the phi-psi distribution results of (a) normal MD and (b) MetaD.
"""
import os
import math
import h5py
import matplotlib.pyplot as plt
from mindspore import context, nn
from sponge.system import get_molecule
from sponge import Sponge
from sponge import ForceField
from sponge import set_global_units, WithEnergyCell
from sponge import UpdaterMD
from sponge.control import VelocityVerlet
from sponge.callback import WriteH5MD, RunInfo
from sponge.control import Langevin
from sponge.colvar import Torsion
from sponge.function import VelocityGenerator
from sponge.function import PI
from sponge.sampling import Metadynamics

os.environ['GLOG_v'] = '4'
os.environ['MS_JIT_MODULES'] = 'sponge'

context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
set_global_units('nm', 'kj/mol')

pdb_name = '../case1.pdb'
out_pdb = '../case1_sol.pdb'
mol = get_molecule(pdb_name, template=['protein0.yaml'], rebuild_hydrogen=True)
mol.fill_water(edge=0.4, pdb_out=out_pdb, template='water.spce.yaml')

energy = ForceField(mol, parameters=['AMBER.FF14SB', 'SPCE'], use_pme=False)

phi = Torsion([3, 12, 13, 21])
psi = Torsion([12, 13, 21, 36])

min_opt = nn.Adam(mol.trainable_params(), 1e-3)
sim = WithEnergyCell(mol, energy)
md = Sponge(sim, optimizer=min_opt, metrics={'phi': phi, 'psi': psi})
run_info = RunInfo(200)
cb_h5md = WriteH5MD(mol, './homework.h5md', save_freq=10, save_last_pdb='./homework.pdb', write_image=False)
md.run(1000, callbacks=[run_info, cb_h5md])

temp = 300
vgen = VelocityGenerator(temp)
velocity = vgen(mol.shape, mol.atom_mass)
opt = UpdaterMD(system=mol,
                time_step=1e-3,
                velocity=velocity,
                integrator='velocity_verlet',
                temperature=300,
                thermostat='langevin',)
md = Sponge(sim, optimizer=opt, metrics={'phi': phi, 'psi': psi})
cb_h5md = WriteH5MD(mol, './homework.h5md', save_freq=10, write_image=False)
md.run(1000, callbacks=[run_info, cb_h5md])

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
cb_h5md = WriteH5MD(mol, './homework.h5md', save_freq=10, write_image=False)
md.run(1000, callbacks=[run_info, cb_h5md])

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
cb_h5md = WriteH5MD(mol, './output.h5md', save_freq=10, write_image=False)
md.run(10000, callbacks=[run_info, cb_h5md])

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

sim = WithEnergyCell(mol, energy, bias=metad)
vgen = VelocityGenerator(300)
velocity = vgen(mol.shape, mol.atom_mass)
opt = UpdaterMD(
    mol,
    integrator=VelocityVerlet(mol),
    thermostat=Langevin(mol, 300),
    time_step=1e-3,
    velocity=velocity
)
md = Sponge(sim, optimizer=opt, metrics={'phi': phi, 'psi': psi})
cb_h5md = WriteH5MD(mol, './output_with_metad.h5md', save_freq=10, write_image=False)
md.run(10000, callbacks=[run_info, cb_h5md])

with h5py.File('./output.h5md', 'r') as file:
    dataset_path = 'observables/trajectory/'
    phi_md = file[dataset_path]['phi/value'][:]
    psi_md = file[dataset_path]['psi/value'][:]

# 处理数据使其分布在 0 - pi
for i in range(len(phi_md)):
    phi_md = (phi_md % math.pi + math.pi) % math.pi
    psi_md = (psi_md % math.pi + math.pi) % math.pi

with h5py.File('./output_with_metad.h5md') as file:
    dataset_path = 'observables/trajectory/'
    phi_metad = file[dataset_path]['phi/value'][:]
    psi_metad = file[dataset_path]['psi/value'][:]

# 处理数据使其分布在 0 - pi
for i in range(len(phi_md)):
    phi_metad = (phi_metad % math.pi + math.pi) % math.pi
    psi_metad = (psi_metad % math.pi + math.pi) % math.pi
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

axs[0].scatter(phi_md, psi_md, color='blue', label='MD Data')
axs[0].set_xlabel('Phi (radian)')
axs[0].set_ylabel('Psi (radian)')
axs[0].set_title('MD Scatter Plot')
axs[0].legend()

axs[1].scatter(phi_metad, psi_metad, color='orange', label='MetaD Data')
axs[1].set_xlabel('Phi (radian)')
axs[1].set_ylabel('Psi (radian)')
axs[1].set_title('MetaD Scatter Plot')
axs[1].legend()

plt.tight_layout()
plt.savefig('./result.png')
plt.show()
