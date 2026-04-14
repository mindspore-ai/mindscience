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
"""pendulum"""
import matplotlib.pyplot as plt
import numpy as np

from sciai.utils import amp2datatype
from .problem import Problem
from ..data import Data
from ..nn.hnn import HNN
from ..stormer_verlet import StormerVerlet


class PDData(Data):
    """Data for learning the pendulum system with the Hamiltonian H(p,q)=(1/2)p^2−cos(q)."""

    def __init__(self, dtype, x0, h, train_num, test_num, add_h=False):
        super(PDData, self).__init__(dtype)
        self.solver = StormerVerlet(None, self.dh, iterations=1, order=6, n=10)
        self.x0 = x0
        self.h = h
        self.train_num = train_num
        self.test_num = test_num
        self.add_h = add_h
        self.x_train, self.y_train = self.__generate_flow(self.x0, self.h, self.train_num)
        self.x_test, self.y_test = self.__generate_flow(self.y_train[-1], self.h, self.test_num)

    @property
    def dim(self):
        return 2

    def dh(self, p, q):
        """dh"""
        return p, np.sin(q)

    def __generate_flow(self, x0, h, num):
        x_ = self.solver.flow(np.array(x0), h, num)
        x, y = x_[:-1], x_[1:]
        if self.add_h:
            x = np.hstack([x, self.h * np.ones([x.shape[0], 1])])
        return x, y


class Pendulum(Problem):
    """Pendulum"""

    def plot(self, data, net, figure_path):
        steps = 1000
        if isinstance(net, HNN):
            flow_true = data.solver.flow(data.x_test_np[0][:-1], data.h, steps)
            flow_pred = net.predict(data.x_test[0][:-1], data.h, steps)
        else:
            flow_true = data.solver.flow(data.x_test_np[0], data.h, steps)
            flow_pred = net.predict(data.x_test[0], steps)

        plt.plot(flow_true[:, 0], flow_true[:, 1], color='b', label='Ground truth', zorder=0)
        plt.plot(flow_pred[:, 0], flow_pred[:, 1], color='r', label='Predicted flow', zorder=1)
        plt.scatter(data.x_train_np[:, 0], data.x_train_np[:, 1], color='b', label='Learned data', zorder=2)
        plt.legend()
        plt.savefig(f'{figure_path}/pendulum.pdf')

    def init_data(self, args):
        # data
        x0 = [0, 1]
        h = 0.1
        train_num = 40
        test_num = 100
        # net
        net_type = args.net_type
        dtype = amp2datatype(args.amp_level)

        add_h = net_type == 'HNN'
        criterion = None if net_type == 'HNN' else 'MSE'
        data = PDData(dtype, x0, h, train_num, test_num, add_h)
        return criterion, data
