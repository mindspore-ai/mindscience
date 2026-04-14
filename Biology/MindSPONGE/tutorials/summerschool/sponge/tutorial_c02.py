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
MindSPONGE basic tutorial c02: Run NVT molecular dynamics of the water system.
"""

import argparse
from mindspore import context
from mindspore.nn import Adam

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


if __name__ == "__main__":

    import sys
    sys.path.append('../../../src')

    from sponge import Sponge
    from sponge import Molecule
    from sponge import ForceField
    from sponge import UpdaterMD
    from sponge import WithEnergyCell, set_global_units
    from sponge.potential import SphericalRestrict
    from sponge.function import VelocityGenerator
    from sponge.callback import WriteH5MD, RunInfo

    set_global_units('nm', 'kj/mol')

    if args.m == '0':
        context.set_context(mode=context.GRAPH_MODE, device_target=args.e, device_id=int(args.id),
                            enable_graph_kernel=enable_graph_kernel)
    else:
        context.set_context(mode=context.PYNATIVE_MODE, device_target=args.e, device_id=int(args.id))

    system = Molecule(template='water.tip3p.yaml')

    system.reduplicate([0.3, 0, 0])
    system.reduplicate([0, 0.3, 0])
    system.reduplicate([0, 0, 0.3])
    system.reduplicate([-0.6, 0, 0])
    system.reduplicate([0, -0.6, 0])
    system.reduplicate([0, 0, -0.6])

    potential = ForceField(system, parameters='TIP3P')

    opt = Adam(system.trainable_params(), 1e-3)
    mini = Sponge(system, potential, opt)

    run_info = RunInfo(10)
    mini.run(500, callbacks=[run_info])

    temp = 300
    vgen = VelocityGenerator(temp)
    velocity = vgen(system.shape, system.atom_mass)

    updater = UpdaterMD(
        system=system,
        time_step=1e-3,
        velocity=velocity,
        integrator='velocity_verlet',
        temperature=300,
        thermostat='langevin',
        constraint='all-bonds'
    )

    sim = WithEnergyCell(system, potential, bias=SphericalRestrict(radius=1.0, center=[0, 0, 0]))
    md = Sponge(sim, optimizer=updater)

    cb_h5md = WriteH5MD(system, 'tutorial_c02.h5md', save_freq=20, write_velocity=True, write_force=True)

    md.run(2000, callbacks=[run_info, cb_h5md])
