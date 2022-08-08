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
MindSPONGE basic tutorial 03: Edit system and minimization.
"""

from mindspore import context
from mindspore.nn import Adam

if __name__ == "__main__":

    import sys
    sys.path.append('..')

    from mindsponge import Sponge
    from mindsponge import Molecule
    from mindsponge import ForceField

    from mindsponge.callback import WriteH5MD, RunInfo

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

    system = Molecule(template='water.spce.yaml')

    system.reduplicate([0.3, 0, 0])
    system.reduplicate([0, 0.3, 0])
    new_sys = system.copy([0, 0, -0.3])
    system.reduplicate([0, 0, 0.3])
    system.append(new_sys)

    potential = ForceField(system, parameters='SPCE')

    opt = Adam(system.trainable_params(), 1e-3)

    md = Sponge(system, potential, opt)

    run_info = RunInfo(10)
    cb_h5md = WriteH5MD(system, 'tutorial_b03.h5md', save_freq=10)

    md.run(1000, callbacks=[run_info, cb_h5md])
