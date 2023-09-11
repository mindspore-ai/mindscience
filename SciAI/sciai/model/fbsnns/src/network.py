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

"""fbsnns network"""
from abc import abstractmethod

from mindspore import ops, nn
from sciai.architecture import MLP, MSE
from sciai.common.initializer import XavierTruncNormal


class NeuralNetwork(nn.Cell):
    """Neural network"""
    def __init__(self, layers):
        super().__init__()
        self.mlp = MLP(layers, weight_init=XavierTruncNormal(), bias_init="zeros",
                       activation=ops.Sin())

    def construct(self, t, x):
        """Network forward pass"""
        h = ops.concat([t, x], 1)
        h = self.mlp(h)
        return h


class NetU(nn.Cell):
    """Neural network with Grad"""
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.grad = ops.GradOperation(get_all=True)
        self.net_u = self.grad(self.net)

    def construct(self, t, x):  # M x 1, M x D
        """Network forward pass"""
        u = self.net(t, x)  # M x 1
        _, u_x = self.net_u(t, x)
        return u, u_x


class FBSNN(nn.Cell):
    """Forward-Backward Stochastic Neural Network"""
    def __init__(self, terminal, m, n, dim, layers, data_type):
        super().__init__()
        self.terminal = terminal  # terminal time

        self.m = m  # number of trajectories
        self.n = n  # number of time snapshots
        self.dim = dim  # number of dimensions

        # initialize NN
        self.neural_net = NeuralNetwork(layers)
        self.net_u = NetU(self.neural_net)
        self.reduce_sum = ops.ReduceSum(keep_dims=True)
        self.square = ops.Square()
        self.stack = ops.Stack(axis=1)
        self.mse = MSE()
        self.reduce_sum = ops.ReduceSum(keep_dims=True)

        self.d_g = ops.grad(self.g)
        self.data_type = data_type

    def construct(self, t, w, xi):  # M x (N+1) x 1, M x (N+1) x D, 1 x D
        """Network forward pass"""
        loss = 0
        x_list = []
        y_list = []

        t0 = t[:, 0, :]
        w0 = w[:, 0, :]
        x0 = ops.tile(xi, (self.m, 1))  # M x D
        y0, z0 = self.net_u(t0, x0)  # M x 1, M x D

        x_list.append(x0)
        y_list.append(y0)
        x1, y1, z1 = x0, y0, z0
        for n in range(self.n):
            t1 = t[:, n + 1, :]
            w1 = w[:, n + 1, :]
            x1 = x0 + self.mu() * (t1 - t0) + ops.squeeze(
                ops.matmul(self.sigma(t0, x0, y0), ops.expand_dims(w1 - w0, -1)), -1)
            y1_tilde = y0 + self.phi(t0, x0, y0, z0) * (t1 - t0) + self.reduce_sum(
                z0 * ops.squeeze(ops.matmul(self.sigma(t0, x0, y0), ops.expand_dims(w1 - w0, -1))), 1)
            y1, z1 = self.net_u(t1, x1)
            loss += self.mse(y1 - y1_tilde)
            t0, w0, x0, y0, z0 = t1, w1, x1, y1, z1
            x_list.append(x0)
            y_list.append(y0)

        loss += self.mse(y1 - self.g(x1))
        loss += self.mse(z1 - self.d_g(x1))
        x_pred = ops.stack(x_list, axis=1)
        y_pred = ops.stack(y_list, axis=1)
        y0_pred = y_pred[0, 0, 0]

        return loss, x_pred, y_pred, y0_pred

    @abstractmethod
    def phi(self, t, x, y, z):  # M x 1, M x D, M x 1, M x D
        pass  # M x1

    @abstractmethod
    def g(self, x):  # M x D
        pass  # M x 1

    def mu(self):  # M x 1, M x D, M x 1, M x D
        return ops.zeros((self.m, self.dim), self.data_type)  # M x D

    @abstractmethod
    def sigma(self, t, x, y):  # M x 1, M x D, M x 1
        pass


class AllenCahn(FBSNN):
    def phi(self, t, x, y, z):  # M x 1, M x D, M x 1, M x D
        return - y + y ** 3  # M x 1

    def g(self, x):
        return 1.0 / (2.0 + 0.4 * self.reduce_sum(x ** 2, 1))

    def sigma(self, t, x, y):  # M x 1, M x D, M x 1
        return ops.matrix_diag(ops.ones((self.m, self.dim), self.data_type))  # M x D x D


class BlackScholesBarenblatt(FBSNN):
    def phi(self, t, x, y, z):  # M x 1, M x D, M x 1, M x D
        return 0.05 * (y - self.reduce_sum(x * z, 1))  # M x 1

    def g(self, x):  # M x D
        return self.reduce_sum(x ** 2, 1)  # M x 1

    def sigma(self, t, x, y):  # M x 1, M x D, M x 1
        return 0.4 * ops.matrix_diag(x)  # M x D x D
