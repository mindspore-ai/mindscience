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
MindSPONGE advanced tutorial 02: Bias potential and metadynamics (MetaD)
"""

from mindspore import context

if __name__ == "__main__":

    import sys
    sys.path.insert(0, '../../src')

    from sponge import Sponge
    from sponge import ForceField
    from sponge import set_global_units
    from sponge import Protein
    from sponge import WithEnergyCell
    from sponge import UpdaterMD
    from sponge.optimizer import SteepestDescent
    from sponge.control import VelocityVerlet
    from sponge.callback import WriteH5MD, RunInfo
    from sponge.control import Langevin
    from sponge.sampling import Metadynamics
    from sponge.function import VelocityGenerator
    from sponge.colvar import Torsion
    from sponge.function import PI

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

    set_global_units('nm', 'kj/mol')

    system = Protein(pdb='alad.pdb')

    potential = ForceField(system, 'AMBER.FF14SB')

    min_opt = SteepestDescent(system.trainable_params(), 1e-7)

    phi = Torsion([4, 6, 8, 14])
    psi = Torsion([6, 8, 14, 16])

    mini = Sponge(system, potential, min_opt, metrics={'phi': phi, 'psi': psi})

    run_info = RunInfo(10)
    mini.run(100, callbacks=[run_info])

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

    sim = WithEnergyCell(system, potential, bias=metad)

    vgen = VelocityGenerator(300)
    velocity = vgen(system.shape, system.atom_mass)

    opt = UpdaterMD(
        system,
        velocity=velocity,
        time_step=1e-3,
        integrator=VelocityVerlet(system),
        thermostat=Langevin(system, 300),
    )

    md = Sponge(sim, optimizer=opt, metrics={'phi': phi, 'psi': psi})

    cb_h5md = WriteH5MD(system, 'tutorial_a02.h5md', save_freq=10)

    md.run(1000, callbacks=[run_info, cb_h5md])
