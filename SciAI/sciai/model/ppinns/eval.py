# Copyright 2023 Huawei Technologies Co., Ltd
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
"""ppinns eval"""
import matplotlib.pyplot as plt
import mindspore as ms
import numpy as np
from mpi4py import MPI
from sciai.context import init_project
from sciai.utils import to_tensor, amp2datatype, print_log
from sciai.utils.python_utils import print_time

from src.finesolver import FineSolver
from src.process import prepare


@print_time("eval")
def main(args):
    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    init_project(args=args, device_id=comm_rank)
    comm_size = comm.Get_size()
    num = comm_size - 1
    chunks = np.linspace(args.t_range[0], args.t_range[1], num + 1)
    dtype = amp2datatype(args.amp_level)

    fine_solver = FineSolver(comm, comm_rank, chunks, args) if comm_rank > 0 else None

    if fine_solver:
        if dtype == ms.float16:
            fine_solver.net.to_float(ms.float16)
        ckpt_file = args.load_ckpt_path

        param_dict = ms.load_checkpoint(ckpt_file)
        ms.load_param_into_net(fine_solver.net, param_dict)
        t_test = to_tensor(fine_solver.t_test, dtype=ms.float32)
        x_bc_0 = to_tensor(fine_solver.x_bc_0, dtype=ms.float32)
        u_pred_t = fine_solver.net(t_test, x_bc_0)[0]
        mse_val = ms.ops.mse_loss(u_pred_t, t_test)
        if comm_rank == comm_size - 1:
            print_log("MSE: ", mse_val)

        plt.plot(fine_solver.t_test, u_pred_t.asnumpy(), color='r', linestyle="--", linewidth=2)
        plt.title(f"fine_solver_{fine_solver.comm_rank}")

        u_t = fine_solver.t_test + np.sin(0.5 * np.pi * fine_solver.t_test)
        plt.plot(fine_solver.t_test, u_t, color='b', linestyle="-", linewidth=1)
        plt.savefig(f'{fine_solver.fig_dir}/evaluation_{fine_solver.comm_rank}.png')


if __name__ == "__main__":
    args_ = prepare()
    main(*args_)
