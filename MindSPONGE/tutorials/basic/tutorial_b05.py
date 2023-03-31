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

from mindspore import context

if __name__ == "__main__":

    import sys
    sys.path.append('..')

    from mindsponge import Sponge
    from mindsponge import Molecule
    from mindsponge import ForceField
    from mindsponge import UpdaterMD
    from mindsponge import WithEnergyCell
    from mindsponge.control import VelocityVerlet
    from mindsponge.control import Langevin, BerendsenBarostat
    from mindsponge.function import VelocityGenerator
    from mindsponge.callback import WriteH5MD, RunInfo
    from mindsponge.optimizer import SteepestDescent
    from mindsponge import set_global_units

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

    set_global_units('nm', 'kj/mol')

    system = Molecule(template='water.spce.yaml')
    system.set_pbc_box([0.32, 0.32, 0.32])

    system.repeat_box([10, 10, 10])

    potential = ForceField(system, parameters='SPCE')

    opt = SteepestDescent(system.trainable_params(), 1e-6)

    sim = WithEnergyCell(system, potential, cutoff=1.0)
    mini = Sponge(sim, optimizer=opt)

    run_info = RunInfo(10)
    mini.run(100, callbacks=[run_info])

    temp = 300
    vgen = VelocityGenerator(temp)
    velocity = vgen(system.coordinate.shape, system.atom_mass)

    updater = UpdaterMD(
        system,
        integrator=VelocityVerlet(system),
        thermostat=Langevin(system, temp),
        barostat=BerendsenBarostat(system, 1),
        velocity=velocity,
        time_step=1e-3,
    )

    md = Sponge(sim, optimizer=updater)

    cb_h5md = WriteH5MD(system, 'tutorial_b05.h5md', save_freq=50)

    md.run(10000, callbacks=[run_info, cb_h5md])
