# Copyright 2021 The AIMM Group at Shenzhen Bay Laboratory & Peking University & Huawei Technologies Co., Ltd
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

"""Protein relax pipeline
1. Usage:
$ python3 protein_relax.py -i examples/protein/case2.pdb -o examples/protein/case2-optimized.pdb
"""

import argparse
from mindspore import context, Tensor, nn
from mindspore import numpy as msnp
import mindspore as ms

from sponge import Sponge
from sponge import set_global_units
from sponge import Protein
from sponge import ForceField
from sponge.callback import RunInfo
from sponge.core import WithEnergyCell, WithForceCell, RunOneStepCell
from sponge.optimizer import SteepestDescent
from sponge.sampling import MaskedDriven
from sponge.partition import NeighbourList
from sponge.potential.bias import OscillatorBias
from sponge.system.modelling.pdb_generator import gen_pdb

from mindsponge.common.utils import get_pdb_info
from mindsponge.pipeline.structure_violations import get_structural_violations

parser = argparse.ArgumentParser()
parser.add_argument("-i", help="Set the input pdb file path.")
parser.add_argument("-o", help="Set the output pdb file path.")
parser.add_argument(
    "-addh", help="Set to 1 if need to add H atoms, default to be 1..", default=1
)
args = parser.parse_args()
pdb_name = args.i
save_pdb_name = args.o
addh = args.addh
if context.get_context("device_target") == "Ascend":
    context.set_context(mode=context.GRAPH_MODE, enable_graph_kernel=True,
                        graph_kernel_flags="--enable_cluster_ops=ReduceSum --reduce_fuse_depth=10")
else:
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU", device_id=1, enable_graph_kernel=True)


def get_violation_loss(system):
    """ Package the violation loss calculation module. """
    gen_pdb(
        system.coordinate.asnumpy(),
        system.atom_name[0],
        system.init_resname,
        system.init_resid,
        pdb_name=save_pdb_name,
    )
    features = get_pdb_info(save_pdb_name)
    atom14_atom_exists_t = Tensor(features.get("atom14_gt_exists")).astype(ms.float32)
    residue_index_t = Tensor(features.get("residue_index")).astype(ms.float32)
    residx_atom14_to_atom37_t = Tensor(features.get("residx_atom14_to_atom37")).astype(ms.int32)
    atom14_positions_t = Tensor(features.get("atom14_gt_positions")).astype(ms.float32)
    aatype_t = Tensor(features.get("aatype")).astype(ms.int32)
    violations = get_structural_violations(atom14_atom_exists_t, residue_index_t, aatype_t,
                                           residx_atom14_to_atom37_t, atom14_positions_t)
    return violations


def optimize_strategy(system, gds, loops, ads, adm, nonh_mask, mode=1):
    """ The optimize strategy including 3 modes.
    Args:
        system(Molecule): The given Molecule object.
        gds(int): Optimize steps while using Gradient Descent.
        loops(int): The number of loops to use different optimizers.
        ads(int): The optimize steps of using Adam.
        adm(int): The repeat number of using Adam in each loop.
        nonh_mask(bool): The mask of Hydrogen atoms. For atom whose atomic number > 1 would be labeled as 1.
        mode(int): The optimize mode, for now only mode = 1, 2, 3 are supported.
            mode == 1: Use the hybrid optimize strategy which includes total energy and bonded energy.
            mode == 2: Use the total energy only.
            mode == 3: Use the bonded energy only.
    """
    energy = ForceField(system, "AMBER.FF14SB")
    learning_rate = 1e-07
    factor = 1.003
    dynamic_lr = nn.ExponentialDecayLR(learning_rate, factor, 1, is_stair=True)
    opt = SteepestDescent(system.trainable_params(), dynamic_lr, max_shift=1.0)
    neighbours = NeighbourList(system, cutoff=None, cast_fp16=True)
    with_energy = WithEnergyCell(system, energy, neighbour_list=neighbours)
    modifier = MaskedDriven(length_unit=with_energy.length_unit,
                            energy_unit=with_energy.energy_unit,
                            mask=nonh_mask)
    with_force = WithForceCell(system, neighbour_list=neighbours, modifier=modifier)
    one_step = RunOneStepCell(energy=with_energy, force=with_force, optimizer=opt)

    for i, param in enumerate(opt.trainable_params()):
        print(i, param.name, param.shape)

    md = Sponge(network=one_step)
    run_info = RunInfo(1)
    md.run(gds, callbacks=[run_info])

    if msnp.isnan(md.calc_energy().sum()):
        return 0

    for _ in range(loops):
        k_coe = 10
        harmonic_energy = OscillatorBias(1 * system.coordinate, k_coe, nonh_mask)
        learning_rate = 5e-02

        if mode in (1, 2):
            energy.set_energy_scale([1, 1, 1, 1, 1, 1])
            simulation_network = WithEnergyCell(system, energy, bias=[harmonic_energy], neighbour_list=neighbours)

            for _ in range(adm):
                opt = nn.Adam(system.trainable_params(), learning_rate=learning_rate)
                one_step = RunOneStepCell(energy=simulation_network, optimizer=opt)
                for i, param in enumerate(opt.trainable_params()):
                    print(i, param.name, param.shape)
                md = Sponge(network=one_step)
                print(md.calc_energy())
                run_info = RunInfo(1)
                md.run(ads, callbacks=[run_info])
                if msnp.isnan(md.calc_energy().sum()):
                    return 0

        if mode in (1, 3):
            energy.set_energy_scale([1, 1, 1, 0, 0, 0])
            simulation_network = WithEnergyCell(system, energy, bias=[harmonic_energy], neighbour_list=neighbours)

            for _ in range(adm):
                opt = nn.Adam(system.trainable_params(), learning_rate=learning_rate)
                one_step = RunOneStepCell(energy=simulation_network, optimizer=opt)
                for i, param in enumerate(opt.trainable_params()):
                    print(i, param.name, param.shape)
                md = Sponge(network=one_step)
                print(md.calc_energy())
                run_info = RunInfo(1)
                md.run(ads, callbacks=[run_info])
                if msnp.isnan(md.calc_energy().sum()):
                    return 0

    return system


def main():
    seed = 2333
    ms.set_seed(seed)
    set_global_units("A", "kcal/mol")
    system = Protein(pdb=pdb_name, rebuild_hydrogen=True)
    nonh_mask = system.heavy_atom_mask

    gds, loops, ads, adm = 100, 3, 200, 2
    system = optimize_strategy(system, gds, loops, ads, adm, nonh_mask, mode=1)

    try:
        violations = get_violation_loss(system)
        violation_loss = violations[-1]
        print("The first try violation loss value is: {}".format(violation_loss))

    except AttributeError:
        import traceback
        traceback.print_exc()

    while system == 0:
        system = Protein(pdb=pdb_name)
        gds = int(0.5 * gds)
        ads = int(0.8 * ads)
        system = optimize_strategy(system, gds, loops, ads, adm, nonh_mask, mode=1)
        try:
            violations = get_violation_loss(system)
            violation_loss = violations[-1]
            print("The first try violation loss value is: {}".format(violation_loss))
        except AttributeError:
            continue

    if violation_loss > 0:
        gds = 200
        system = Protein(pdb=pdb_name)
        loops, ads, adm = 6, 200, 1
        system = optimize_strategy(system, gds, loops, ads, adm, nonh_mask, mode=2)

        violations = get_violation_loss(system)
        violation_loss = violations[-1]
        print("The second try violation loss value is: {}".format(violation_loss))

    if violation_loss > 0:
        gds = 200
        system = Protein(pdb=pdb_name)
        loops, ads, adm = 6, 200, 1
        system = optimize_strategy(system, gds, loops, ads, adm, nonh_mask, mode=3)

        violations = get_violation_loss(system)
        violation_loss = violations[-1]
        print("The third try violation loss value is: {}".format(violation_loss))

    if violation_loss > 0:
        gds = 100
        system = Protein(pdb=pdb_name)
        loops, ads, adm = 8, 100, 1
        system = optimize_strategy(system, gds, loops, ads, adm, nonh_mask, mode=3)

        violations = get_violation_loss(system)
        violation_loss = violations[-1]
        print("The forth try violation loss value is: {}".format(violation_loss))

    if violation_loss > 0:
        system = Protein(pdb=pdb_name)
        gds, loops, ads, adm = 30, 2, 150, 2
        system = optimize_strategy(system, gds, loops, ads, adm, nonh_mask, mode=1)

        violations = get_violation_loss(system)
        violation_loss = violations[-1]
        print("The final try violation loss value is: {}".format(violation_loss))


main()
