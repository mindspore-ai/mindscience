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
MindSPONGE basic tutorial 06: Minimization and MD simulation of protein molecule.
"""

from mindspore import context

if __name__ == "__main__":

    import sys
    sys.path.append('..')

    from mindsponge import Sponge
    from mindsponge import ForceField
    from mindsponge import set_global_units
    from mindsponge import Protein
    from mindsponge import UpdaterMD
    from mindsponge.optimizer import SteepestDescent
    from mindsponge.control import VelocityVerlet
    from mindsponge.callback import WriteH5MD, RunInfo
    from mindsponge.control import Langevin
    from mindsponge.function import VelocityGenerator

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

    set_global_units('nm', 'kj/mol')

    PDB_NAME = 'case2.pdb'
    system = Protein(pdb=PDB_NAME, rebuild_hydrogen=True)

    energy = ForceField(system, 'AMBER.FF14SB')

    min_opt = SteepestDescent(system.trainable_params(), 1e-7)

    md = Sponge(system, energy, min_opt)

    run_info = RunInfo(10)
    md.run(500, callbacks=[run_info])

    vgen = VelocityGenerator(300)
    velocity = vgen(system.coordinate.shape, system.atom_mass)

    updater = UpdaterMD(
        system,
        integrator=VelocityVerlet(system),
        thermostat=Langevin(system, 300),
        time_step=1e-3,
        velocity=velocity
    )

    cb_h5md = WriteH5MD(system, 'tutorial_b06.h5md', save_freq=10)

    md.change_optimizer(updater)
    md.run(2000, callbacks=[run_info, cb_h5md])
