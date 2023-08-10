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
MindSPONGE basic tutorial 04: MD simulation with bias potential
"""

from mindspore import context
from mindspore.nn import Adam

if __name__ == "__main__":

    import sys
    sys.path.append('../../src')

    from sponge import Sponge
    from sponge import Molecule
    from sponge import ForceField
    from sponge import UpdaterMD
    from sponge import WithEnergyCell
    from sponge.potential import SphericalRestrict
    from sponge.function import VelocityGenerator
    from sponge.callback import WriteH5MD, RunInfo

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

    system = Molecule(template='water.spce.yaml')

    system.reduplicate([0.3, 0, 0])
    system.reduplicate([0, 0.3, 0])
    new_sys = system.copy([0, 0, -0.3])
    system.reduplicate([0, 0, 0.3])
    system.append(new_sys)

    potential = ForceField(system, parameters='SPCE')

    opt = Adam(system.trainable_params(), 1e-3)

    mini = Sponge(system, potential, opt)

    run_info = RunInfo(10)
    mini.run(1000, callbacks=[run_info])

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
    )

    sim = WithEnergyCell(system, potential, bias=SphericalRestrict(radius=1.5, center=[0, 0, 0]))
    md = Sponge(sim, optimizer=updater)

    cb_h5md = WriteH5MD(system, 'tutorial_b04.h5md', save_freq=10, write_velocity=True, write_force=True)

    md.run(2000, callbacks=[run_info, cb_h5md])
