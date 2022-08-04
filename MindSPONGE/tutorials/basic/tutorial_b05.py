# Copyright 2021-2022 @ Shenzhen Bay Laboratory &
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
    from mindsponge import DynamicUpdater
    from mindsponge import SimulationCell
    from mindsponge.control import VelocityVerlet, Langevin, BerendsenBarostat
    from mindsponge.function import VelocityGenerator
    from mindsponge.callback import WriteH5MD, RunInfo
    from mindsponge.optimizer import SteepestDescent

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

    system = Molecule(template='water.spce.yaml')
    system.set_pbc_box([0.4, 0.4, 0.4])

    system.repeat_box([10, 10, 10])

    potential = ForceField(system, parameters='SPCE')

    opt = SteepestDescent(system.trainable_params(), 1e-6)

    sim = SimulationCell(system, potential, cutoff=1.0)
    md = Sponge(sim, optimizer=opt)

    run_info = RunInfo(10)
    md.run(100, callbacks=[run_info])

    temp = 300
    vgen = VelocityGenerator(temp)
    velocity = vgen(system.coordinate.shape, system.atom_mass)

    opt = DynamicUpdater(
        system,
        integrator=VelocityVerlet(system),
        thermostat=Langevin(system, temp),
        barostat=BerendsenBarostat(system, 1),
        velocity=velocity,
        time_step=5e-4,
    )
    md = Sponge(sim, optimizer=opt)

    md.change_optimizer(opt)

    cb_h5md = WriteH5MD(system, 'tutorial_b05.h5md', save_freq=10)

    md.run(1000, callbacks=[run_info, cb_h5md])
