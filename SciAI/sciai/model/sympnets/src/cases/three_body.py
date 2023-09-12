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
"""Three body"""
import matplotlib.pyplot as plt
import numpy as np

from sciai.utils import amp2datatype
from .problem import Problem
from ..data import Data
from ..nn.hnn import HNN
from ..stormer_verlet import StormerVerlet


class TBData(Data):
    """Data for learning the three body system."""

    def __init__(self, dtype, h, train_traj_num, test_traj_num, train_num, test_num, add_h=False):
        super(TBData, self).__init__(dtype)
        self.solver = StormerVerlet(None, self.dh, iterations=1, order=6, n=100)
        self.h = h
        self.train_traj_num = train_traj_num
        self.test_traj_num = test_traj_num
        self.train_num = train_num
        self.test_num = test_num
        self.add_h = add_h
        x0 = self.random_config(self.train_traj_num + self.test_traj_num)
        x_train, y_train = self.__generate_flow(x0[:self.train_traj_num], self.h, self.train_num)
        x_test, y_test = self.__generate_flow(x0[self.train_traj_num:], self.h, self.test_num)
        self.x_train = x_train.reshape([self.train_num * self.train_traj_num, -1])
        self.y_train = y_train.reshape([self.train_num * self.train_traj_num, -1])
        self.x_test = x_test.reshape([self.test_num * self.test_traj_num, -1])
        self.y_test = y_test.reshape([self.test_num * self.test_traj_num, -1])

    @property
    def dim(self):
        return 12

    def dh(self, p, q):
        """dh"""
        p1, p2, p3 = p[..., :2], p[..., 2:4], p[..., 4:6]
        q1, q2, q3 = q[..., :2], q[..., 2:4], q[..., 4:6]
        dhdp1, dhdp2, dhdp3 = p1, p2, p3
        dhdq1 = (q1 - q2) / np.sum((q1 - q2) ** 2, axis=-1, keepdims=True) ** 1.5 + \
                (q1 - q3) / np.sum((q1 - q3) ** 2, axis=-1, keepdims=True) ** 1.5
        dhdq2 = (q2 - q3) / np.sum((q2 - q3) ** 2, axis=-1, keepdims=True) ** 1.5 + \
                (q2 - q1) / np.sum((q2 - q1) ** 2, axis=-1, keepdims=True) ** 1.5
        dhdq3 = (q3 - q1) / np.sum((q3 - q1) ** 2, axis=-1, keepdims=True) ** 1.5 + \
                (q3 - q2) / np.sum((q3 - q2) ** 2, axis=-1, keepdims=True) ** 1.5
        dhdp = np.hstack([dhdp1, dhdp2, dhdp3])
        dhdq = np.hstack([dhdq1, dhdq2, dhdq3])
        return dhdp, dhdq

    def rotate2d(self, p, theta):
        """rotate 2d"""
        c, s = np.cos(theta), np.sin(theta)
        r = np.array([[c, -s], [s, c]])
        r = np.transpose(r)
        return p.dot(r)

    def random_config(self, n, nu=2e-1, min_radius=0.9, max_radius=1.2):
        """random config"""
        q1 = 2 * np.random.rand(n, 2) - 1
        r = np.random.rand(n) * (max_radius - min_radius) + min_radius

        ratio = r / np.sqrt(np.sum((q1 ** 2), axis=1))
        q1 *= np.tile(np.expand_dims(ratio, 1), (1, 2))
        q2 = self.rotate2d(q1, theta=2 * np.pi / 3)
        q3 = self.rotate2d(q2, theta=2 * np.pi / 3)

        # # velocity that yields a circular orbit
        v1 = self.rotate2d(q1, theta=np.pi / 2)
        v1 = v1 / np.tile(np.expand_dims(r ** 1.5, axis=1), (1, 2))
        v1 = v1 * np.sqrt(np.sin(np.pi / 3) / (2 * np.cos(np.pi / 6) ** 2))  # scale factor to get circular trajectories
        v2 = self.rotate2d(v1, theta=2 * np.pi / 3)
        v3 = self.rotate2d(v2, theta=2 * np.pi / 3)

        # make the circular orbits slightly chaotic
        v1 *= 1 + nu * (2 * np.random.rand(2) - 1)
        v2 *= 1 + nu * (2 * np.random.rand(2) - 1)
        v3 *= 1 + nu * (2 * np.random.rand(2) - 1)

        q = np.zeros([n, 6])
        p = np.zeros([n, 6])

        q[:, :2] = q1
        q[:, 2:4] = q2
        q[:, 4:] = q3
        p[:, :2] = v1
        p[:, 2:4] = v2
        p[:, 4:] = v3

        return np.hstack([p, q])

    def __generate_flow(self, x0, h, num):
        x_ = self.solver.flow(np.array(x0), h, num)
        x, y = x_[:, :-1], x_[:, 1:]
        if self.add_h:
            x = np.concatenate([x, self.h * np.ones([x.shape[0], x.shape[1], 1])], axis=2)
        return x, y


class ThreeBody(Problem):
    """Three body"""

    def plot(self, data, net, figure_path):
        h_true = data.h / 10
        test_num_true = (data.test_num - 1) * 10
        if isinstance(net, HNN):
            flow_true = data.solver.flow(data.x_test_np[0][:-1], h_true, test_num_true)
            flow_pred = net.predict(data.x_test[0][:-1], data.h, data.test_num - 1)
        else:
            flow_true = data.solver.flow(data.x_test_np[0], h_true, test_num_true)
            flow_pred = net.predict(data.x_test[0], data.test_num - 1)
        plt.plot(flow_true[:, 6], flow_true[:, 7], color='b', label='Ground truth')
        plt.plot(flow_true[:, 8], flow_true[:, 9], color='b')
        plt.plot(flow_true[:, 10], flow_true[:, 11], color='b')
        plt.scatter(flow_pred[:, 6], flow_pred[:, 7], color='r', label='Predicted solution')
        plt.scatter(flow_pred[:, 8], flow_pred[:, 9], color='r')
        plt.scatter(flow_pred[:, 10], flow_pred[:, 11], color='r')
        plt.legend(loc='upper left')
        plt.savefig(f'{figure_path}/three_body.pdf')

    def init_data(self, args):
        # data
        h = 0.5
        train_num = 10
        test_num = 10
        train_traj_num = 4000
        test_traj_num = 1000
        # net
        dtype = amp2datatype(args.amp_level)
        net_type = args.net_type
        add_h = net_type == 'HNN'
        criterion = None if net_type == 'HNN' else 'MSE'
        data = TBData(dtype, h, train_traj_num, test_traj_num, train_num, test_num, add_h)
        return criterion, data
