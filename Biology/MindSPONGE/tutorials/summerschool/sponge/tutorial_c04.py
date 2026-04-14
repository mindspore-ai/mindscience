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
MindSPONGE basic tutorial c04
"""

import argparse
import sys
sys.path.append('../../../src')
from mindspore import context, nn
from sponge import ForceField, set_global_units, WithEnergyCell, UpdaterMD
from sponge.system import get_molecule
from sponge.control import SETTLE, LeapFrog, Langevin
from sponge.core import Sponge
from sponge.callback import WriteH5MD, RunInfo
from sponge.function import VelocityGenerator

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


if __name__ == '__main__':

    set_global_units('nm', 'kj/mol')

    if args.m == '0':
        context.set_context(mode=context.GRAPH_MODE, device_target=args.e, device_id=int(args.id),
                            enable_graph_kernel=enable_graph_kernel)
    else:
        context.set_context(mode=context.PYNATIVE_MODE, device_target=args.e, device_id=int(args.id))

    pdb_name = './case1.pdb'
    out_pdb = './case1_add_water.pdb'

    mol = get_molecule(pdb_name, template=['protein0.yaml'], rebuild_hydrogen=True)
    mol.fill_water(edge=0.4, template='water.tip3p.yaml', pdb_out=out_pdb)

    energy = ForceField(mol, parameters=['AMBER.FF99SB', 'TIP3P'], use_pme=True)

    min_opt = nn.Adam(mol.trainable_params(), 1e-03)
    sim = WithEnergyCell(mol, energy)
    md = Sponge(sim, optimizer=min_opt)

    run_info = RunInfo(10)
    md.run(2000, callbacks=[run_info])

    temp = 300
    vgen = VelocityGenerator(temp)
    velocity = vgen(mol.shape, mol.atom_mass)

    updater = UpdaterMD(
        mol,
        time_step=1e-3,
        velocity=velocity,
        integrator=LeapFrog(mol),
        thermostat=Langevin(mol, temperature=300),
        constraint=SETTLE(mol)
    )

    md.change_optimizer(updater)

    cb_h5md = WriteH5MD(mol, 'tutorial_c04.h5md', save_freq=10, save_last_pdb='tutorial_c04.pdb', write_image=False)

    run_info = RunInfo(10)
    md.run(3000, callbacks=[run_info, cb_h5md])
