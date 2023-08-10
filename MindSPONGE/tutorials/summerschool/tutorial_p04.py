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
# MindSPONGE tutorial protein 04: Hybrid enhanced sampling MetaITS.
"""
MindSPONGE basic tutorial p04
"""

# %%
import argparse
import sys
from mindspore import context
from sponge import Sponge
from sponge import ForceField
from sponge import set_global_units, WithEnergyCell
from sponge import Protein
from sponge import UpdaterMD
from sponge.optimizer import SteepestDescent
from sponge.control import VelocityVerlet
from sponge.callback import WriteH5MD, RunInfo
from sponge.control import Langevin
from sponge.function import VelocityGenerator
from sponge.colvar import Torsion
from sponge.partition import NeighbourList
from sponge.sampling import Metadynamics
from sponge.function import PI

sys.path.append('../../src')

parser = argparse.ArgumentParser()
parser.add_argument("-e", help="Set the backend.", default="GPU")
parser.add_argument("-m", help="Set the compile mode.", default="0")
parser.add_argument("-id", help="Set the backend index.", default="0")
args = parser.parse_args()

if args.m == '0':
    context.set_context(mode=context.GRAPH_MODE, device_target=args.e, device_id=int(args.id))
else:
    context.set_context(mode=context.PYNATIVE_MODE, device_target=args.e, device_id=int(args.id))

set_global_units('nm', 'kj/mol')

system = Protein(pdb='./case1_addH.pdb', rebuild_hydrogen=False)
energy = ForceField(system, 'AMBER.FF14SB')

phi = Torsion([3, 12, 13, 21])
psi = Torsion([12, 13, 21, 36])

neighbours = NeighbourList(system, cutoff=None, cast_fp16=True)
min_opt = SteepestDescent(system.trainable_params(), 1e-7)
md = Sponge(system, energy, min_opt, metrics={'phi': phi, 'psi': psi})
run_info = RunInfo(10)
cb_h5md = WriteH5MD(system, 'p04_1.h5md', save_freq=10)
md.run(500, callbacks=[run_info, cb_h5md])

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

sim = WithEnergyCell(system, energy, bias=metad, neighbour_list=neighbours)
vgen = VelocityGenerator(300)
velocity = vgen(system.shape, system.atom_mass)
opt = UpdaterMD(
    system,
    integrator=VelocityVerlet(system),
    thermostat=Langevin(system, 300),
    time_step=1e-3,
    velocity=velocity
)
md = Sponge(sim, optimizer=opt, metrics={'phi': phi, 'psi': psi})
cb_h5md = WriteH5MD(system, 'p04_2.h5md', save_freq=10)
md.run(1000, callbacks=[run_info, cb_h5md])
