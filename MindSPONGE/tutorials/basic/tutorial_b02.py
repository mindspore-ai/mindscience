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
MindSPONGE basic tutorial 02: Create a simple simulation system using template and parameters file.
"""

from mindspore import context

if __name__ == "__main__":

    import sys
    sys.path.append('../../src')

    from sponge import Sponge
    from sponge import Molecule
    from sponge import ForceField
    from sponge import UpdaterMD
    from sponge.callback import WriteH5MD, RunInfo

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU", enable_graph_kernel=False)

    system = Molecule(template='water.tip3p.yaml')

    potential = ForceField(system, parameters='TIP3P')

    updater = UpdaterMD(
        system=system,
        time_step=1e-3,
        integrator='velocity_verlet',
        temperature=300,
        thermostat='langevin',
    )

    md = Sponge(system, potential, updater)

    run_info = RunInfo(10)
    cb_h5md = WriteH5MD(system, 'tutorial_b02.h5md', save_freq=10, write_velocity=True, write_force=True)

    md.run(1000, callbacks=[run_info, cb_h5md])
