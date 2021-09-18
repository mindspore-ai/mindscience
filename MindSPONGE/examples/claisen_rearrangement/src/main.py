# Copyright 2021 Huawei Technologies Co., Ltd
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
'''main'''
import argparse
import time

from mindspore import context
from mindspore import Tensor
from mindspore import load_checkpoint

from simulation_cybertron import SimulationCybertron
from mindsponge.md.cybertron.mdnn import Mdnn, TransCrdToCV
from mindsponge.md.cybertron.models import MolCT
from mindsponge.md.cybertron.readouts import AtomwiseReadout
from mindsponge.md.cybertron.cybertron import Cybertron

parser = argparse.ArgumentParser(description='SPONGE Controller')
parser.add_argument('--i', type=str, default='md.in', help='Input file')
parser.add_argument('--amber_parm', type=str, default='cba.prmtop', help='Paramter file in AMBER type')
parser.add_argument('--c', type=str, default='cba_its_mw0_trans.rst7', help='Initial coordinates file')
parser.add_argument('--r', type=str, default=None, help='')
parser.add_argument('--x', type=str, default="mdcrd", help='')
parser.add_argument('--o', type=str, default="mdout.txt", help='Output file')
parser.add_argument('--box', type=str, default="mdbox.txt", help='')
parser.add_argument('--device_id', type=int, default=1, help='GPU device id')
parser.add_argument('--u', type=bool, default=False, help='If use mdnn to update the atom charge')
parser.add_argument('--checkpoint', type=str, default="", help='Checkpoint file')
parser.add_argument('--datfile', type=str, default="crd_record.dat", help='Store the evolution path in a dat file.')
parser.add_argument('--initial_coordinates_file', default=None, type=str, help='Initial rst7 pos file.')
parser.add_argument('--meta', type=bool, default=0, help='Set to 1 if MetaDynamics is used.')
parser.add_argument('--with_box', type=bool, default=1, help='Set to be 1 if periodic map is needed.')
parser.add_argument('--np_iter', type=bool, default=0, help='Set to be 1 if you want to use msnp.')

if __name__ == '__main__':

    args_opt = parser.parse_args()
    args_opt.initial_coordinates_file = args_opt.c

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU", device_id=args_opt.device_id, save_graphs=False)
    atom_types = Tensor([6, 1, 6, 1, 6, 1, 1, 6, 1, 6, 1, 6, 1, 1, 8])
    num_atom = atom_types.size
    mod = MolCT(
        min_rbf_dis=0.1,
        max_rbf_dis=10,
        num_rbf=128,
        rbf_sigma=0.2,
        n_interactions=3,
        dim_feature=128,
        n_heads=8,
        max_cycles=1,
        use_time_embedding=True,
        fixed_cycles=True,
        self_dis=0.1,
        unit_length='A',
        use_feed_forward=False,
    )
    scales = 3.0

    readout = AtomwiseReadout(n_in=mod.dim_feature, n_interactions=mod.n_interactions, activation=mod.activation,
                              n_out=1, mol_scale=scales, unit_energy='kcal/mol')
    net = Cybertron(mod, atom_types=atom_types, full_connect=True, readout=readout, unit_dis='A',
                    unit_energy='kcal/mol')

    param_file = 'cba_kcal_mol_A_MolCT-best.ckpt'
    load_checkpoint(param_file, net=net)

    simulation = SimulationCybertron(args_opt, network=net)
    if args_opt.u and args_opt.checkpoint:
        net = Mdnn()
        load_checkpoint(args_opt.checkpoint)
        transcrd = TransCrdToCV(simulation)

    start = time.time()
    compiler_time = 0
    save_path = args_opt.o
    simulation.main_initial()
    for steps in range(simulation.md_info.step_limit):
        print_step = steps % simulation.ntwx
        if steps == simulation.md_info.step_limit - 1:
            print_step = 0
        temperature, total_potential_energy, sigma_of_bond_ene, sigma_of_angle_ene, sigma_of_dihedral_ene, \
        nb14_lj_energy_sum, nb14_cf_energy_sum, LJ_energy_sum, ee_ene, _ = simulation(Tensor(steps), Tensor(print_step))

        if steps == 0:
            compiler_time = time.time()
        if steps % simulation.ntwx == 0 or steps == simulation.md_info.step_limit - 1:
            simulation.main_print(steps, temperature, total_potential_energy, sigma_of_bond_ene, sigma_of_angle_ene,
                                  Tensor(0), Tensor(0), nb14_cf_energy_sum, LJ_energy_sum, ee_ene)

        if args_opt.u and args_opt.checkpoint and steps % (4 * simulation.ntwx) == 0:
            print("Update charge!")
            inputs = transcrd(Tensor(simulation.crd), Tensor(simulation.last_crd))
            t_charge = net(inputs)
            simulation.charge = transcrd.updatecharge(t_charge)

    end = time.time()
    print("Main time(s):", end - start)
    print("Main time(s) without compiler:", end - compiler_time)
    simulation.main_destroy()
