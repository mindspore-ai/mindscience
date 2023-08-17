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
MindSPONGE basic tutorial 08: Minimize the protein energy via only move hydrogen atoms.
"""

from mindspore import context
from mindspore import nn

if __name__ == "__main__":

    import sys
    sys.path.append('../../src')

    from sponge import Sponge
    from sponge import ForceField
    from sponge import set_global_units
    from sponge import Protein
    from sponge.optimizer import SteepestDescent
    from sponge.callback import WriteH5MD, RunInfo
    from sponge.core import WithEnergyCell, WithForceCell, RunOneStepCell
    from sponge.sampling import MaskedDriven
    from sponge.partition import NeighbourList

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

    set_global_units('nm', 'kj/mol')

    # Define a normal force field
    PDB_NAME = 'case2.pdb'
    system = Protein(pdb=PDB_NAME, rebuild_hydrogen=True)
    energy = ForceField(system, 'AMBER.FF99SB')

    # Define a optimizer with dynamic learning rate
    steps = 500
    initial_lr = 1e-07
    decay_rate = 1.001
    decay_steps = 1
    max_shift = 1e-08
    dynamic_lr = nn.ExponentialDecayLR(initial_lr, decay_rate, decay_steps, is_stair=True)
    min_opt = SteepestDescent(system.trainable_params(), dynamic_lr, max_shift=None)

    # Define a neighbour list for both energy cell and force cell
    neighbours = NeighbourList(system, cutoff=1.0, cast_fp16=True)

    with_energy = WithEnergyCell(system, energy, neighbour_list=neighbours)

    # Define a force cell with mask
    modifier = MaskedDriven(length_unit=with_energy.length_unit,
                            energy_unit=with_energy.energy_unit,
                            mask=system.heavy_atom_mask)
    with_force = WithForceCell(system, neighbour_list=neighbours, modifier=modifier)

    one_step = RunOneStepCell(energy=with_energy, force=with_force, optimizer=min_opt)
    md = Sponge(network=one_step)

    run_info = RunInfo(10)
    cb_h5md = WriteH5MD(system, 'tutorial_b08.h5md', save_freq=10, save_last_pdb='last_08.pdb')
    md.run(steps, callbacks=[run_info, cb_h5md])
