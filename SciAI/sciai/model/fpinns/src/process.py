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

"""Data processing"""
import argparse
import os

import yaml
import numpy as np
from matplotlib import pyplot as plt
import mindspore as ms
from mindspore import nn

from sciai.common import TrainCellWithCallBack
from sciai.utils import parse_arg, print_log, to_tensor
from .dataset import random_points
from .net import Net, MyWithLossCell, FNNWithTransform, MyWithLossCellFPDE
from .problem import Problem


def prepare(problem=None):
    abs_dir = os.path.abspath(os.path.dirname(__file__))
    with open(f"{abs_dir}/../config.yaml") as f:
        config_dict = yaml.safe_load(f)
    if problem is not None:
        config_dict["problem"] = problem
    args_, problem_ = generate_args(config_dict)
    return args_, problem_


def generate_args(config):
    """to generate args from config"""
    common_config = {k: v for k, v in config.items() if not isinstance(v, dict)}
    problem_name = find_problem(config)
    problem_config = config.get(problem_name)
    problem_config.update(common_config)
    args = parse_arg(problem_config)

    problem: Problem = {
        "diffusion_1d": Diffusion1D,
        "fractional_diffusion_1d": FractionalDiffusion1D
    }.get(problem_name, Diffusion1D)(args)
    return args, problem


def find_problem(config):
    parser = argparse.ArgumentParser(description=config.get("case"))
    parser.add_argument(f'--problem', type=str, default=config.get("problem", "diffusion_1d"))
    args = parser.parse_known_args()
    return args[0].problem


class Diffusion1D(Problem):
    """diffusion 1d problem's structure"""

    def __init__(self, args):  # pylint: disable=W0235
        super().__init__(args)

    def setup_networks(self, args):
        net = Net(args.layers)

        return net

    def setup_train_cell(self, args, net):
        ckpt_interval = 2000 if args.save_ckpt else 0
        criterion = nn.MSELoss()
        optimizer = nn.Adam(net.trainable_params(), learning_rate=args.lr)
        loss_cell = MyWithLossCell(net, criterion)
        train_cell = TrainCellWithCallBack(loss_cell, optimizer, time_interval=args.print_interval,
                                           loss_interval=args.print_interval, ckpt_interval=ckpt_interval,
                                           ckpt_dir=args.save_ckpt_path, amp_level=args.amp_level,
                                           model_name=args.model_name)
        return train_cell

    def train(self, train_cell):
        x_train, t_train, _ = self.generate_data(self.num_domain)
        x_initial, t_initial, _ = self.generate_data(self.num_initial)

        x_train = to_tensor(np.vstack((x_initial, x_train)))
        t_train = to_tensor(np.vstack((t_initial, t_train)))
        y_train = to_tensor(self.func(x_train, t_train))

        x_bc_np, t_bc_np = self._generate_boundary_data(self.num_boundary, self.t_range)
        t_bc, x_bc, y_bc = to_tensor((t_bc_np, x_bc_np, self.func(x_bc_np, t_bc_np)), dtype=self.dtype)

        for i in range(self.epochs):
            loss_ = train_cell(t_train, t_bc, x_train, x_bc, y_bc, y_train)
            if i % 1000 == 0:
                self.loss.append(loss_)
                self.steps.append(i)

    def predict(self, network, *inputs):
        x_test, t_test = inputs
        y_pred = network(t_test, t_test, x_test, x_test)[0]
        return y_pred

    def func(self, x, t):
        return np.sin(np.pi * x) * np.exp(-t)

    def plot_result(self, x_test, t_test, y_test, y_res):
        print_log('plotting...')
        plt.figure(2)
        plt.plot(y_res.asnumpy(), color='r', linestyle="--", linewidth=2)
        plt.plot(y_test, color='b', linestyle="-", linewidth=1)
        plt.savefig(f'{self.figures_path}/predict_result_ms.png')

    def generate_data(self, num):
        x_diam = self.x_range[1] - self.x_range[0]
        t_diam = self.t_range[1] - self.t_range[0]
        x = random_points(num, x_diam, self.x_range[0])
        t = random_points(num, t_diam, self.t_range[0])
        t = np.random.permutation(t)
        y = self.func(x, t)
        return x, t, y

    def _generate_boundary_data(self, num, t_range):
        t_diam = t_range[1] - t_range[0]
        x = self._generate_boundary_x(num)
        t = random_points(num, t_diam, t_range[0])
        t = np.random.permutation(t)
        return x, t

    def _generate_boundary_x(self, num):
        return np.random.choice([-1, 1], num)[:, None].astype(self.dtype_np)


class FractionalDiffusion1D(Problem):
    """Fractional diffusion 1d problem's structure"""

    def __init__(self, args):
        super().__init__(args)
        self.resolution = 52
        self.alpha = 1.8

        self.nt = int(round(args.num_domain / (self.resolution - 2))) + 1
        self.dt = (args.t_range[1] - args.t_range[0]) / (self.nt - 1)
        self.num_bcs = [0, 0]

        self._w_init = self._get_init_weight()
        self.x_train, self.t_train, self.y_train = self.generate_data(self.num_domain)
        self.int_mat = self._generate_matrix()

    def setup_networks(self, args):
        fnn = FNNWithTransform(args.layers)
        return fnn

    def setup_train_cell(self, args, net):
        criterion = nn.MSELoss()
        loss_cell = MyWithLossCellFPDE(net, criterion, self.num_bcs, alpha=self.alpha, dtype=self.dtype)
        if self.dtype == ms.float16:
            loss_cell.to_float(ms.float16)
        optimizer = nn.Adam(net.trainable_params(), learning_rate=args.lr)
        ckpt_interval = 2000 if args.save_ckpt else 0
        train_cell = TrainCellWithCallBack(loss_cell, optimizer, time_interval=args.print_interval,
                                           loss_interval=args.print_interval, ckpt_interval=ckpt_interval,
                                           ckpt_dir=args.save_ckpt_path, amp_level=args.amp_level)
        return train_cell

    def func(self, x, t):
        return np.exp(-t) * x ** 3 * (1 - x) ** 3

    def train(self, train_cell):
        x_train_ms, t_train_ms = to_tensor((self.x_train, self.t_train), dtype=self.dtype)
        int_mat_ms = to_tensor(self.int_mat, dtype=self.dtype)

        for i in range(self.epochs):
            loss_ = train_cell(x_train_ms, t_train_ms, int_mat_ms)

            if i % 100 == 0:
                self.loss.append(loss_)
                self.steps.append(i)

    def predict(self, network, *inputs):
        x_test, t_test = inputs
        y_pred = network(x_test, t_test)
        return y_pred

    def generate_data(self, num):
        self.nt = int(round(num / (self.resolution - 2))) + 1
        self.dt = (self.t_range[1] - self.t_range[0]) / (self.nt - 1)

        x = self._get_x_static()
        x_bc = self._get_x_bc(x)
        x = x[self.resolution + 2 * self.nt - 2:, :]
        x_train = np.vstack((x_bc, x))
        y_train = self.func(x_train[:, 0], x_train[:, 1])[:, None]
        return x_train[:, 0:1], x_train[:, 1:2], y_train

    def plot_result(self, x_test, t_test, y_test, y_res):
        print_log('plotting...')
        ax = plt.figure(2).add_subplot(projection='3d')
        ax.scatter(
            x_test.asnumpy(),
            t_test.asnumpy(),
            y_res.asnumpy()
        )
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$t$")
        ax.set_zlabel("$y$")
        plt.savefig(f'{self.figures_path}/predict_result_ms.png')

    def _get_x_bc(self, train_all):
        x_all = train_all[:, :-1]
        t_all = train_all[:, -1:]
        is_bc = np.array([np.any(np.isclose(x_all[i], self.x_range), axis=-1) for i in range(len(x_all))])
        is_ic = np.isclose(t_all, self.t_range[0]).flatten()
        x_bc = train_all[is_bc]
        x_ic = train_all[is_ic]
        self.num_bcs = [len(x_bc), len(x_ic)]
        return np.vstack([x_bc, x_ic])

    def _get_x_static(self):
        """to get static x"""
        x = np.linspace(self.x_range[0], self.x_range[1], num=self.resolution, dtype=self.dtype_np)[:, None]
        x = np.roll(x, 1)[:, 0]
        d = np.empty((self.resolution * self.nt, 2), dtype=self.dtype_np)
        d[0:self.resolution, 0] = x
        d[0:self.resolution, 1] = self.t_range[0]
        beg = self.resolution
        for i in range(1, self.nt):
            d[beg:beg + 2, 0] = x[:2]
            d[beg:beg + 2, 1] = self.t_range[0] + i * self.dt
            beg += 2
        for i in range(1, self.nt):
            d[beg:beg + self.resolution - 2, 0] = x[2:]
            d[beg:beg + self.resolution - 2, 1] = self.t_range[0] + i * self.dt
            beg += self.resolution - 2
        return d

    def _generate_matrix(self):
        int_matrix = self._get_matrix_core()
        num_bc = sum(self.num_bcs)
        int_mat = to_tensor(np.pad(int_matrix, ((num_bc, 0), (num_bc, 0))), dtype=self.dtype)
        return int_mat

    def _get_matrix_core(self):
        """to get matrix core"""
        n = (self.resolution - 2) * (self.nt - 1)
        int_mat = np.zeros((self.resolution, self.resolution), dtype=self.dtype_np)
        h = (self.x_range[1] - self.x_range[0]) / (self.resolution - 1)
        for i in range(1, self.resolution - 1):
            int_mat[i, 1:i + 2] = np.flipud(self._get_weight(i))
            int_mat[i, i - 1:-1] += self._get_weight(self.resolution - 1 - i)
        int_mat_one = h ** (- self.alpha) * int_mat

        int_mat = np.zeros((n, n), dtype=self.dtype_np)
        beg = 0
        for _ in range(self.nt - 1):
            int_mat[beg:beg + self.resolution - 2, beg:beg + self.resolution - 2] = int_mat_one[1:-1, 1:-1]
            beg += self.resolution - 2
        return int_mat

    def _get_weight(self, n):
        return self._w_init[:n + 1]

    def _get_init_weight(self):
        w = [1]
        for i in range(1, self.resolution):
            w.append(w[-1] * (i - 1 - self.alpha) / i)
        return np.array(w, dtype=self.dtype_np)
