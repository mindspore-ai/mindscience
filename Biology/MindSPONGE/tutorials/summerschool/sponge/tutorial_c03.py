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
MindSPONGE basic tutorial c03: Run NPT molecular dynamics of the water system.
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

    from sponge import Sponge, Molecule, ForceField, UpdaterMD, WithEnergyCell
    from sponge.function import VelocityGenerator
    from sponge.callback import WriteH5MD, RunInfo

    if args.m == '0':
        context.set_context(mode=context.GRAPH_MODE, device_target=args.e, device_id=int(args.id),
                            enable_graph_kernel=enable_graph_kernel)
    else:
        context.set_context(mode=context.PYNATIVE_MODE, device_target=args.e, device_id=int(args.id))

    system = Molecule(template='water.tip3p.yaml')
    system.set_pbc_box([0.4, 0.4, 0.4])
    system.repeat_box([5, 5, 5])

    potential = ForceField(system, parameters=['TIP3P'], use_pme=True)

    opt = Adam(system.trainable_params(), 1e-3)

    sim = WithEnergyCell(system, potential)
    mini = Sponge(sim, optimizer=opt)

    run_info = RunInfo(10)
    mini.run(500, callbacks=[run_info])

    temp = 300
    vgen = VelocityGenerator(temp)
    velocity = vgen(system.shape, system.atom_mass)

    nvt = UpdaterMD(
        system=system,
        time_step=1e-3,
        velocity=velocity,
        integrator='velocity_verlet',
        temperature=300,
        thermostat='langevin',
    )
    md = mini.change_optimizer(nvt)

    cb_h5md = WriteH5MD(system, 'tutorial_c03_nvt.h5md', save_freq=20, write_velocity=True, write_force=True)
    md.run(2000, callbacks=[run_info, cb_h5md])

    npt = UpdaterMD(
        system=system,
        time_step=1e-3,
        velocity=velocity,
        integrator='velocity_verlet',
        temperature=300,
        pressure=1,
        thermostat='langevin',
    )
    md.change_optimizer(npt)

    cb_h5md = WriteH5MD(system, 'tutorial_c03_npt.h5md', save_freq=20, write_velocity=True, write_force=True)
    md.run(2000, callbacks=[run_info, cb_h5md])

    print('The final pbc box is: {}'.format(system.pbc_box.asnumpy()))
