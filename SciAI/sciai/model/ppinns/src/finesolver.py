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
"""finesolver"""
import os

import matplotlib.pyplot as plt
import mindspore as ms
from mindspore import nn
import numpy as np
from sciai.common import TrainCellWithCallBack, lbfgs_train
from sciai.utils import print_log, to_tensor

from .dataset import Dataset
from .model import Net, MyWithLossCell
from .net import FNN, ODENN


class FineSolver:
    """Fine Solver"""
    def __init__(self, comm, comm_rank, chunks, args):
        self.comm = comm
        self.comm_rank = comm_rank
        self.comm_size = comm.Get_size()
        self.num = self.comm_size - 1
        self.ne = int((args.nt_fine - 1) / self.num + 1)
        self.ckpt_dir = f'{args.save_ckpt_path}/fine_solver_{self.comm_rank}_{args.amp_level}'
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)

        self.t_range = [chunks[self.comm_rank - 1], chunks[self.comm_rank]]
        self.t_test = np.linspace(self.t_range[0], self.t_range[1], self.ne).reshape((-1, 1))
        self._init_data(args.n_train)

        self._init_net(args)

        self.epochs = args.epochs
        self.lbfgs_epochs = args.lbfgs_epochs
        self.use_lbfgs = args.lbfgs
        self.save_fig = args.save_fig
        self.fig_dir = args.figures_path
        self.save_output = args.save_output
        self.save_data_path = args.save_data_path

        self.u_initial = None
        self.f_bc = None

    def solve(self, niter):
        """solve"""
        print_log('Fine solver for chunk#:' + str(self.comm_rank))
        loss_c = 1e-3
        x, x_bc_0, u_initial, t_test = to_tensor((self.x, self.x_bc_0, self.u_initial, self.t_test), dtype=ms.float32)
        for _ in range(self.epochs):
            loss_ = self.train_cell(x, x_bc_0, u_initial)
            if loss_ < loss_c:
                break
        ms.save_checkpoint(self.net, self.ckpt_dir + f'/result_iter_{niter}.ckpt')

        if self.use_lbfgs:
            self.loss_cell.to_float(ms.float32)
            lbfgs_train(self.loss_cell, (x, x_bc_0, u_initial), self.lbfgs_epochs)

        u_pred_t = self.net(t_test, x_bc_0)[0]
        u_test = np.reshape(u_pred_t.asnumpy(), (-1, 1))

        if self.save_output:
            filename = f'{self.save_data_path}/u_fine_loop_' + str(niter) + '_chunk_' + str(self.comm_rank)
            np.savetxt(filename, u_test, fmt='%e')

        self.f_bc = u_test[-1:]

        if self.save_fig:
            self._plot_prediction(self.t_test, u_pred_t, niter)

    def receive_u(self, offset):
        """receive u"""
        self.u_initial = self.comm.recv(source=0, tag=offset + self.comm_rank)

    def send_f(self, offset):
        """send f"""
        self.comm.send(self.f_bc, dest=0, tag=offset + self.comm_rank)

    def _init_data(self, n_train):
        """init data"""
        data = Dataset(self.t_range, self.ne, n_train, 1)
        self.x, self.x_bc_0, self.u_bc_0, self.x_min, self.x_max = data.build_data()

    def _init_net(self, args):
        """init net"""
        # size of the DNN
        layers = args.layers
        ckpt_interval = 2000 if args.save_ckpt else 0
        ckpt_dir = args.save_ckpt_path + f"/fine_solver_{self.comm_rank}_{args.amp_level}"

        # physics-informed neural networks for fine solver in each subdomain
        self.fnn = FNN(layers, self.x_min, self.x_max)
        self.odenn = ODENN(self.fnn)
        self.net = Net(self.fnn, self.odenn)
        if args.load_ckpt:
            ms.load_checkpoint(args.load_ckpt_path, self.net)
        self.criterion = nn.MSELoss()
        self.optimizer = nn.Adam(self.net.trainable_params())
        self.loss_cell = MyWithLossCell(self.net, self.criterion)
        self.train_cell = TrainCellWithCallBack(self.loss_cell, self.optimizer,
                                                loss_interval=args.print_interval, time_interval=args.print_interval,
                                                ckpt_interval=ckpt_interval, ckpt_dir=ckpt_dir,
                                                amp_level=args.amp_level, model_name=args.model_name)

    def _plot_prediction(self, t, u_pred_t, niter):
        """plot prediction"""
        if not os.path.exists(self.fig_dir):
            os.makedirs(self.fig_dir)

        u_t = t + np.sin(0.5 * np.pi * t)
        plt.plot(t, u_pred_t.asnumpy(), color='r', linestyle="--", linewidth=2)
        plt.plot(t, u_t, color='b', linestyle="-", linewidth=1)
        plt.title("fine_solver_%d" % self.comm_rank)
        plt.savefig(f'{self.fig_dir}/ucontour_' + str(niter) + '_chunk_' + str(self.comm_rank) + '.png')
