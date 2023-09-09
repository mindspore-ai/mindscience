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
"""ppinns train"""
import time

import numpy as np
from mpi4py import MPI

from sciai.context import init_project
from sciai.utils import print_log
from sciai.utils.python_utils import print_time

from src.coarsesolver import CoarseSolver
from src.finesolver import FineSolver
from src.process import prepare


def init(args, comm, comm_rank, num):
    """init communication"""
    chunks = np.linspace(args.t_range[0], args.t_range[1], num + 1)
    coarse_solver = CoarseSolver(comm, chunks, args) if comm_rank == 0 else None
    fine_solver = FineSolver(comm, comm_rank, chunks, args) if comm_rank > 0 else None
    if comm_rank == 0:
        # prepare the coarse solver for the first loop
        coarse_solver.init_ug(args.t_range)
        coarse_solver.send_u(0, offset=20)
    else:
        fine_solver.receive_u(offset=20)
    return coarse_solver, fine_solver


def train(coarse_solver, comm, comm_rank, fine_solver, num):
    """train process"""
    niter = 1
    error = 1
    while niter <= num and error > 1.0e-2:
        # fine solver
        if comm_rank >= niter:
            fine_solver.solve(niter)
            fine_solver.send_f(offset=50)
        elif comm_rank == 0:
            coarse_solver.receive_f(niter, offset=50)

        comm.Barrier()
        if comm_rank == 0:
            error = coarse_solver.update(niter)
            coarse_solver.send_u(niter, offset=80)
        elif comm_rank > niter:
            fine_solver.receive_u(offset=80)

        error = comm.bcast(error, root=0)
        niter += 1
    return niter


@print_time("train")
def main(args):
    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    init_project(args=args, device_id=comm_rank)
    comm_size = comm.Get_size()
    num = comm_size - 1

    # thread [0]            --> coarse solver
    # thread [1, ..., num]  --> fine solver
    coarse_solver, fine_solver = init(args, comm, comm_rank, num)

    comm.Barrier()
    if comm_rank == 0:
        start_time = time.perf_counter()

    # iteration
    niter = train(coarse_solver, comm, comm_rank, fine_solver, num)

    if comm_rank == 0:
        if niter < num:
            print_log('Converges!')
        else:
            print_log('Exceed the maximum iteration number!')
        stop_time = time.perf_counter()
        print_log('Duration time is ' + str(stop_time - start_time) + ' seconds')


if __name__ == "__main__":
    args_ = prepare()
    init_project(args=args_[0])
    main(*args_)
