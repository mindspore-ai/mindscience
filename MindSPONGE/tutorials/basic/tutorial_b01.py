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
MindSPONGE basic tutorial 01: Create a simple simulation system manually
"""

import numpy as np
from mindspore import context

if __name__ == "__main__":

    import sys
    sys.path.append('..')

    from mindsponge import Sponge
    from mindsponge import Molecule
    from mindsponge import ForceFieldBase
    from mindsponge import DynamicUpdater

    from mindsponge.potential import BondEnergy, AngleEnergy
    from mindsponge.callback import WriteH5MD, RunInfo
    from mindsponge.function import VelocityGenerator
    from mindsponge.control import LeapFrog, BerendsenThermostat

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

    system = Molecule(
        atoms=['O', 'H', 'H'],
        coordinate=[[0, 0, 0], [0.1, 0, 0], [-0.0333, 0.0943, 0]],
        bond=[[[0, 1], [0, 2]]],
    )

    bond_energy = BondEnergy(
        index=system.bond,
        force_constant=[[345000, 345000]],
        bond_length=[[0.1, 0.1]],
    )

    angle_energy = AngleEnergy(
        index=[[1, 0, 2]],
        force_constant=[[383]],
        bond_angle=[[109.47 / 180 * np.pi]],
    )

    potential = ForceFieldBase(energy=[bond_energy, angle_energy])

    vgen = VelocityGenerator(300)
    velocity = vgen(system.coordinate.shape, system.atom_mass)

    opt = DynamicUpdater(
        system,
        integrator=LeapFrog(system),
        thermostat=BerendsenThermostat(system, 300),
        time_step=1e-3,
        velocity=velocity
    )

    md = Sponge(system, potential, opt)

    run_info = RunInfo(10)
    cb_h5md = WriteH5MD(system, 'tutorial_b01.h5md', save_freq=10, write_velocity=True, write_force=True)

    md.run(1000, callbacks=[run_info, cb_h5md])
