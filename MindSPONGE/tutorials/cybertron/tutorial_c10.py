# Copyright 2021-2022 @ Shenzhen Bay Laboratory &
#                       Peking University &
#                       Huawei Technologies Co., Ltd
#
# This code is a part of Cybertron package.
#
# The Cybertron is open-source software based on the AI-framework:
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
Cybertron tutorial 10: Run MD simulation in with CybertronFF as potential
"""

import sys
import time
import numpy as np
from mindspore import load_checkpoint
from mindspore import context

if __name__ == '__main__':

    sys.path.append('..')

    from cybertron.model import MolCT
    from cybertron.readout import AtomwiseReadout
    from cybertron.cybertron import CybertronFF

    from mindsponge import Molecule
    from mindsponge import Sponge
    from mindsponge import set_global_units
    from mindsponge.callback import RunInfo, WriteH5MD
    from mindsponge.control import LeapFrog
    from mindsponge.control import Langevin
    from mindsponge.optimizer import DynamicUpdater

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

    set_global_units('A', 'kcal/mol')

    atom_types = np.array([[6, 1, 1, 1, 6, 8, 6, 8, 6, 8, 6, 8, 6, 8, 25]], np.int32)
    coordinate = np.array([
        [0.782936, -0.21384, 1.940403],
        [0.90026, -1.258313, 2.084498],
        [1.793443, 0.267702, 1.791434],
        [0.161631, 0.247471, 2.702921],
        [-1.775807, 0.660242, 0.992526],
        [-2.573144, 0.82639, 1.806692],
        [-0.793238, 0.551875, -1.559148],
        [-0.922246, 0.719072, -2.702972],
        [1.526357, -0.229486, -0.35567],
        [2.624975, -0.473657, -0.641924],
        [-0.786405, -1.533853, -0.007962],
        [-1.266142, -2.537492, 0.254628],
        [0.394547, 1.910025, 0.468161],
        [0.747036, 3.027445, 0.458565],
        [-0.163356, 0.241977, 0.175396],
    ])
    system = Molecule(atomic_number=atom_types, coordinate=coordinate)

    mod = MolCT(
        dim_feature=128,
        num_atom_types=100,
        n_interaction=3,
        n_heads=8,
        max_cycles=1,
        cutoff=10,
        fixed_cycles=True,
        length_unit='A',
    )

    readout = AtomwiseReadout(
        model=mod,
        dim_output=1,
        activation=mod.activation,
        scale=0.5,
    )

    potential = CybertronFF(
        model=mod,
        readout=readout,
        atom_types=atom_types,
        length_unit='A',
        energy_unit='kcal/mol',
    )

    param_file = 'checkpoint_c10.ckpt'
    load_checkpoint(param_file, net=potential)

    opt = DynamicUpdater(
        system,
        integrator=LeapFrog(system),
        thermostat=Langevin(system, 300),
        time_step=1e-4,
    )

    md = Sponge(system, potential, opt)
    print(md.energy())

    cb_h5md = WriteH5MD(system, 'Tutorial_C10.h5md', save_freq=10)
    cb_sim = RunInfo(10)

    beg_time = time.time()
    md.run(1000, callbacks=[cb_sim, cb_h5md])
    end_time = time.time()
    used_time = end_time - beg_time
    m, s = divmod(used_time, 60)
    h, m = divmod(m, 60)

    print("Run Time: %02d:%02d:%02d" % (h, m, s))
