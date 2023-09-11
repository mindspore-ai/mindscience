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

"""problem definition"""
from abc import abstractmethod

import numpy as np
import mindspore as ms
from mindspore import nn

from sciai.common import TrainCellWithCallBack
from sciai.utils import to_tensor, print_log, lazy_property, amp2datatype, calc_ckpt_name


class Problem:
    """problem definition"""

    def __init__(self, args):
        self.args = args
        self.data_type = None
        self.dim = None
        self.net = None

    @abstractmethod
    @lazy_property
    def net_class(self):
        pass

    @abstractmethod
    @lazy_property
    def xi(self):
        pass

    def build_param(self):
        self.data_type = amp2datatype(self.args.amp_level)
        self.dim = self.args.layers[0] - 1  # number of dimensions
        self.net = self.net_class(self.args.terminal_time, self.args.batch_size,
                                  self.args.num_snapshots, self.dim, self.args.layers, self.data_type)

    def solve(self):
        """to solve"""
        self.net.to_float(self.data_type)
        if self.args.load_ckpt:
            ms.load_checkpoint(self.args.load_ckpt_path, self.net)
        total_epochs = 0
        for single_lr, single_epoch in zip(self.args.lr, self.args.epochs):
            self.train(epochs=single_epoch, learning_rate=single_lr)
            if self.args.save_ckpt:
                total_epochs += single_epoch
                ms.save_checkpoint(self.net, f"{self.args.save_ckpt_path}/{calc_ckpt_name(self.args)}")
        self.test()

    def train(self, epochs, learning_rate):
        """to train"""
        optimizer = nn.Adam(params=self.net.trainable_params(), learning_rate=learning_rate)
        ckpt_interval = 5000 if self.args.save_ckpt else 0
        train_cell = TrainCellWithCallBack(self.net, optimizer, time_interval=500,
                                           loss_interval=500, ckpt_interval=ckpt_interval,
                                           ckpt_dir=self.args.save_ckpt_path, loss_names="loss", grad_first=True,
                                           model_name=self.args.model_name)
        for it in range(epochs):
            t_batch, w_batch = self.fetch_minibatch()  # M x (N+1) x 1, M x (N+1) x D
            t_batch, w_batch, xi = to_tensor((t_batch, w_batch, self.xi), self.data_type)
            _, _, _, y0_pred = train_cell(t_batch, w_batch, xi)

            if it % 5000 == 0:
                print_log('\b, Y0: %.3f, Learning Rate: %.3e' % (y0_pred, learning_rate))

    def predict(self, t_star, w_star):
        t_star, w_star, xi = to_tensor((t_star, w_star, self.xi), self.data_type)
        _, x_star, y_star, _ = self.net(t_star, w_star, xi)
        return x_star.asnumpy(), y_star.asnumpy()

    def fetch_minibatch(self):
        dt = np.zeros((self.args.batch_size, self.args.num_snapshots + 1, 1))  # M x (N+1) x 1
        dw = np.zeros((self.args.batch_size, self.args.num_snapshots + 1, self.dim))  # M x (N+1) x D
        dt_ = self.args.terminal_time / self.args.num_snapshots
        dt[:, 1:, :] = dt_
        dw[:, 1:, :] = np.sqrt(dt_) * np.random.normal(size=(self.args.batch_size, self.args.num_snapshots, self.dim))
        t = np.cumsum(dt, axis=1)  # M x (N+1) x 1
        w = np.cumsum(dw, axis=1)  # M x (N+1) x D
        return t, w

    @abstractmethod
    def test(self):
        pass
