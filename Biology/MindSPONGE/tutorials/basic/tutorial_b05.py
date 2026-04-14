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
MindSPONGE basic tutorial 05: MD simulation with periodic boundary condition.
"""

from mindspore import context, set_seed, nn

if __name__ == "__main__":
    set_seed(0)

    import sys
    sys.path.append('../../src')

    from sponge import Sponge
    from sponge import Molecule
    from sponge import ForceField
    from sponge import UpdaterMD
    from sponge import WithEnergyCell
    from sponge.function import VelocityGenerator
    from sponge.callback import WriteH5MD, RunInfo
    from sponge import set_global_units
    from sponge.partition import NeighbourList

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

    set_global_units('nm', 'kj/mol')

    system = Molecule(template='water.tip3p.yaml')
    system.set_pbc_box([0.4, 0.4, 0.4])
    system.repeat_box([5, 5, 5])

    potential = ForceField(system, parameters='TIP3P')

    opt = nn.Adam(system.trainable_params(), 1e-03)

    neighbours = NeighbourList(system, cast_fp16=True)
    sim = WithEnergyCell(system, potential, neighbour_list=neighbours)
    mini = Sponge(sim, optimizer=opt)

    run_info = RunInfo(10)
    mini.run(100, callbacks=[run_info])

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
        pressure=1,
        barostat='berendsen',
    )

    md = Sponge(sim, optimizer=updater)

    cb_h5md = WriteH5MD(system, 'tutorial_b05.h5md', save_freq=10)

    run_info = RunInfo(10)
    md.run(3000, callbacks=[run_info, cb_h5md])
