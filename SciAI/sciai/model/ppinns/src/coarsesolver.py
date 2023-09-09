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
"""coarse solver"""
import numpy as np

from sciai.utils import print_log


class CoarseSolver:
    """class coarse solver"""
    def __init__(self, comm, chunks, args):
        self._u0 = np.zeros((1, 1))
        self._t0 = np.zeros((1, 1))

        self.chunks = chunks
        self.comm = comm
        self.comm_size = comm.Get_size()
        self.num = self.comm_size - 1
        self.nt_c = args.nt_coarse
        self.ne_c = int((self.nt_c - 1) / self.num + 1)
        self.x_c_list = []
        self._init_x_c_list()
        self.u_list = [self._u0] + [np.zeros((self._t0.shape[0], 1)) for _ in range(self.num)]
        self.g_list = [np.zeros((self._t0.shape[0], 1)) for _ in range(self.num + 1)]
        self.f_list = [np.zeros((self._t0.shape[0], 1)) for _ in range(self.num + 1)]

        self.save_output = args.save_output
        self.save_data_path = args.save_data_path

    @staticmethod
    def fdm(t_range, nc, u0):
        """fdm"""
        dt = (t_range[1] - t_range[0]) / (nc - 1)
        u = np.zeros((nc, 1))
        u[0] = u0

        for i in range(1, nc):
            u[i] = dt + u[i - 1]

        return u

    def init_ug(self, t_range_c):
        u_coarse_0 = self.fdm(t_range_c, self.nt_c, 0)
        bc_id = np.linspace(0, self.nt_c - 1, self.num + 1)
        bc_id = bc_id.astype(int)
        for i in range(1, self.num):
            self.u_list[i] = u_coarse_0[bc_id[i]:(bc_id[i] + 1)]
            self.g_list[i] = u_coarse_0[bc_id[i]:(bc_id[i] + 1)]

    def get_x_c_list(self, n):
        return self.x_c_list[n]

    def send_u(self, idx, offset):
        for i in range(idx, self.num):
            self.comm.send(self.u_list[i], dest=i + 1, tag=offset + i + 1)

    def receive_f(self, idx, offset):
        for i in range(idx, self.num + 1):
            self.f_list[i] = self.comm.recv(source=i, tag=i + offset)

    def update(self, niter):
        """update the coarse solution"""
        error = 0.0
        g_new = self.g_list[niter]
        for n in range(niter, self.num):
            print_log('Coarse solver for chunk#: ' + str(n))
            error += np.linalg.norm(g_new - self.g_list[n]) / (np.linalg.norm(g_new) + 1.0e-9)
            self.u_list[n] = g_new + self.f_list[n] - self.g_list[n]
            self.g_list[n] = g_new
            x = self.get_x_c_list(n)
            u_bc_0 = self.u_list[n]

            u_pred_c = self.fdm(x, self.ne_c, u_bc_0)
            g_new = u_pred_c[-1:]

            if self.save_output:
                filename = f'{self.save_data_path}/u_c_loop_{niter + 1}_chunk_{n}'
                np.savetxt(filename, u_pred_c, fmt='%e')

        ret = error / (self.num - niter + 1.0e-9)
        print_log('Iteration: ' + str(niter) + ', Error: ' + str(ret))
        return ret

    def _init_x_c_list(self):
        # training dataset for coarse solver in each subdomain
        for i in range(self.comm_size - 1):
            t_range_e = [self.chunks[i], self.chunks[i + 1]]
            self.x_c_list.append(t_range_e)
