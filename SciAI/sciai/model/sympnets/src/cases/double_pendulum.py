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
"""double pendulum"""
import matplotlib.pyplot as plt
import numpy as np

from sciai.utils import amp2datatype
from .problem import Problem
from ..data import Data
from ..nn.hnn import HNN
from ..stormer_verlet import StormerVerlet


class DBData(Data):
    """Data for learning the double pendulum system with the Hamiltonian  H(p1,p2,q1,q2)
    = (m2l2^2p1^2 + (m1+m2)l_1^2p2^2 - 2m2l1l2p1p2cos(q1-q2))/(2m2l1^2l2^2(m1+m2sin^2(q1-q2)))
    -(m1+m2)gl1cosq1 - m2gl2cosq2.
    """

    def __init__(self, dtype, x0, h, train_num, test_num, add_h=False):
        super(DBData, self).__init__(dtype)
        self.solver = StormerVerlet(None, self.dh, iterations=10, order=6, n=5)
        self.x0 = x0
        self.h = h
        self.train_num = train_num
        self.test_num = test_num
        self.add_h = add_h
        self.x_train, self.y_train = self.__generate_flow(self.x0, self.h, self.train_num)
        self.x_test, self.y_test = self.__generate_flow(self.y_train[-1], self.h, self.test_num)

    @property
    def dim(self):
        return 4

    def dh(self, p, q):
        """dh"""
        p1 = p[..., 0]
        p2 = p[..., 1]
        q1 = q[..., 0]
        q2 = q[..., 1]
        h1 = p1 * p2 * np.sin(q1 - q2) / (1 + np.sin(q1 - q2) ** 2)
        h2 = (p1 ** 2 + 2 * p2 ** 2 - 2 * p1 * p2 * np.cos(q1 - q2)) / 2 / (1 + np.sin(q1 - q2) ** 2) ** 2
        dhdp1 = (p1 - p2 * np.cos(q1 - q2)) / (1 + np.sin(q1 - q2) ** 2)
        dhdp2 = (-p1 * np.cos(q1 - q2) + 2 * p2) / (1 + np.sin(q1 - q2) ** 2)
        dhdq1 = 2 * np.sin(q1) + h1 - h2 * np.sin(2 * (q1 - q2))
        dhdq2 = np.sin(q2) - h1 + h2 * np.sin(2 * (q1 - q2))
        dhdp = np.hstack([dhdp1, dhdp2])
        dhdq = np.hstack([dhdq1, dhdq2])
        return dhdp, dhdq

    def __generate_flow(self, x0, h, num):
        x_ = self.solver.flow(np.array(x0), h, num)
        x, y = x_[:-1], x_[1:]
        if self.add_h:
            x = np.hstack([x, self.h * np.ones([x.shape[0], 1])])
        return x, y


class DoublePendulum(Problem):
    """Double pendulum"""

    def plot(self, data, net, figure_path):
        t_test = np.arange(0, data.h * data.test_num, data.h)
        if isinstance(net, HNN):
            flow_true = data.solver.flow(data.x_test_np[0][:-1], data.h, data.test_num - 1)
            flow_pred = net.predict(data.x_test[0][:-1], data.h, data.test_num)
        else:
            flow_true = data.solver.flow(data.x_test_np[0], data.h, data.test_num - 1)
            flow_pred = net.predict(data.x_test[0], data.test_num)

        plt.figure(figsize=[6 * 2, 4.8 * 1])
        plt.subplot(121)
        plt.plot(t_test, flow_true[:, 2], color='b', label='Ground truth', zorder=0)
        plt.scatter(t_test, flow_pred[:, 2], color='r', label='Predicted solution', zorder=1)
        plt.ylim([-1.5, 2])
        plt.title('Pendulum 1')
        plt.legend(loc='upper left')
        plt.subplot(122)
        plt.plot(t_test, flow_true[:, 3], color='b', label='Ground truth', zorder=0)
        plt.scatter(t_test, flow_pred[:, 3], color='r', label='Predicted solution', zorder=1)
        plt.ylim([-1.5, 2])
        plt.title('Pendulum 2')
        plt.legend(loc='upper left')
        plt.savefig(f'{figure_path}/double_pendulum.pdf')

    def init_data(self, args):
        # data
        x0 = [0, 0, np.pi * 3 / 7, np.pi * 3 / 8]
        h = 0.75
        train_num = 200
        test_num = 100
        dtype = amp2datatype(args.amp_level)
        net_type = args.net_type
        add_h = net_type == 'HNN'
        criterion = None if net_type == 'HNN' else 'MSE'

        data = DBData(dtype, x0, h, train_num, test_num, add_h)
        return criterion, data
