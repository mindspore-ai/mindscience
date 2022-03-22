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
from npt_numpy import NPT
import mindspore.common.dtype as mstype
import mindspore.context as context
from mindspore import Tensor

parser = argparse.ArgumentParser(description='SPONGE Controller')
parser.add_argument('--i', type=str, default="./data/covidmdin2.txt", help='Input .in file')
parser.add_argument('--amber_parm', type=str, default=None, help='Paramter file in AMBER type')
parser.add_argument('--c', type=str, default=None, help='Initial coordinates file')
parser.add_argument('--r', type=str, default="restrt.rst7", help='')
parser.add_argument('--x', type=str, default="mdcrd", help='')
parser.add_argument('--o', type=str, default="mdout", help='Output file')
parser.add_argument('--box', type=str, default="mdbox", help='')
parser.add_argument('--device_id', type=int, default=0, help='GPU device id')
parser.add_argument('--u', type=bool, default=False, help='If use mdnn to update the atom charge')
parser.add_argument('--checkpoint', type=str, default="", help='Checkpoint file')
args_opt = parser.parse_args()

context.set_context(mode=context.GRAPH_MODE, device_target="GPU", enable_graph_kernel=True)
context.set_context(graph_kernel_flags="--enable_cluster_ops=TensorScatterAdd,UnSortedSegmentSum,GatherNd,Gather \
                    --enable_parallel_fusion=true --disable_expand_ops=StandardNormal \
                    --enable_recompute_fusion=false")

if __name__ == "__main__":
    simulation = NPT(args_opt)

    start = time.time()
    compiler_time = 0
    save_path = args_opt.o
    simulation.main_initial()
    for steps in range(1, simulation.md_info.step_limit + 1):
        print_step = 1 if steps % simulation.ntwx == 0 or steps == 1 or steps == simulation.md_info.step_limit else 0
        update_step = 0
        if simulation.mode > 0:
            update_step = 1 if (steps != 1 and steps % simulation.update_interval == 0) else 0

        temperature, total_potential_energy, sigma_of_bond_ene, sigma_of_angle_ene, sigma_of_dihedral_ene, \
        nb14_lj_energy_sum, nb14_cf_energy_sum, LJ_energy_sum, ee_ene, res, pressure, res1, res2, res3, res4, res5, \
        test_uint_crd = simulation(Tensor(steps), Tensor(print_step), Tensor(update_step, mstype.int32))

        if steps == 1:
            compiler_time = time.time()
        if steps == 1 or steps % simulation.ntwx == 0 or steps == simulation.md_info.step_limit - 1:
            simulation.main_print(steps, temperature, total_potential_energy, sigma_of_bond_ene, sigma_of_angle_ene,
                                  sigma_of_dihedral_ene, nb14_lj_energy_sum, nb14_cf_energy_sum, LJ_energy_sum, ee_ene)
            print("pressure:", pressure.asnumpy())
        if steps % simulation.md_info.output.write_restart_file_interval == 0 \
                or steps == simulation.md_info.step_limit - 1:
            simulation.export_restart_file()

    end = time.time()
    print("Main time(s):", end - start)
    print("Main time(s) without compiler:", end - compiler_time)
    simulation.main_destroy()
