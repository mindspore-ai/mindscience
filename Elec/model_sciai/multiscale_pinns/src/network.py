# -*- coding: utf-8 -*-
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
"""
Network definitions
"""
import numpy as np
import mindspore as ms
from mindspore import nn, ops
from mindspore.common.initializer import XavierNormal

from sciai.architecture import MLP, MSE
from sciai.operators import grad


class NetNN(nn.Cell):
    """NN network"""
    def __init__(self, layers, sigma):
        super().__init__()
        self.mlp = MLP(layers, weight_init=XavierNormal(), bias_init="zeros", activation="tanh")
        self.sigma = sigma

    def construct(self, t, x):
        """Network forward pass"""
        out = ops.concat([t, x], 1)
        out = self.mlp(out)
        return out


class NetFF(nn.Cell):
    """FF network"""
    def __init__(self, layers, sigma):
        super().__init__()
        self.w = ms.Tensor(np.random.normal(size=(2, layers[0] // 2)) * sigma, dtype=ms.float32)
        self.mlp = MLP(layers, weight_init=XavierNormal(), bias_init="zeros", activation="tanh")

    # Evaluates the forward pass
    def construct(self, t, x):
        """Network forward pass"""
        out = ops.concat([t, x], 1)
        # Fourier feature encoding
        out = ops.concat([ops.sin(ops.matmul(out, self.w)), ops.cos(ops.matmul(out, self.w))], 1)
        out = self.mlp(out)
        return out


class NetSTFF(nn.Cell):
    """STFF network"""
    def __init__(self, layers, sigma):
        super().__init__()
        self.w_t = ms.Tensor(np.random.normal(size=(1, layers[0] // 2)), dtype=ms.float32)
        self.w_x = ms.Tensor(np.random.normal(size=(1, layers[0] // 2)) * sigma, dtype=ms.float32)
        self.mlp = MLP(layers[:-1], weight_init=XavierNormal(), bias_init="zeros", activation="tanh")
        self.last_layer = nn.Dense(layers[-2], layers[-1], weight_init=XavierNormal(), bias_init="zeros")

    # Evaluates the forward pass
    def construct(self, t, x):
        """Network forward pass"""
        # Temporal Fourier feature encoding
        h_t = ops.concat([ops.sin(ops.matmul(t, self.w_t)), ops.cos(ops.matmul(t, self.w_t))], 1)  # H1  (N ,50))
        # Spatial Fourier feature encoding
        h_x = ops.concat([ops.sin(ops.matmul(x, self.w_x)), ops.cos(ops.matmul(x, self.w_x))], 1)
        h_t = self.mlp(h_t)
        h_x = self.mlp(h_x)
        # Merge the outputs via point-wise multiplication
        h = ops.mul(h_t, h_x)
        h = self.last_layer(h)
        return h


class Operator(nn.Cell):
    """Operations"""
    def __init__(self, net, k, sigma_t, sigma_x):
        super().__init__()
        self.net = net
        self.u_grad = grad(self.net, output_index=0)
        self.u_x_grad = grad(self.u_grad, output_index=1)
        self.k, self.sigma_t, self.sigma_x = k, sigma_t, sigma_x

    def construct(self, t, x):
        """Network forward pass"""
        u_t, u_x = self.u_grad(t, x)
        _, u_xx = self.u_x_grad(t, x)
        u_t, u_x, u_xx = u_t / self.sigma_t, u_x / self.sigma_x, u_xx / self.sigma_x
        residual = u_t - self.k * u_xx
        return residual


class Heat1D(nn.Cell):
    """1D heat loss"""
    def __init__(self, k, res_sampler, net_u):
        super().__init__()
        x, _ = res_sampler.sample(np.int32(1e5))
        self.mu_x, self.sigma_x = x.mean(0), x.std(0)
        sigma_t, sigma_x = self.sigma_x[0], self.sigma_x[1]
        # Forward pass for u
        self.net_u = net_u
        # Define differential operator for Forward pass for residual
        self.net_r = Operator(self.net_u, k, sigma_t, sigma_x)
        self.mse = MSE()

    def construct(self, *args):
        """Network forward pass"""
        t_ics, x_ics, u_ics, t_bc1, x_bc1, t_bc2, x_bc2, t_r, x_r = args
        # Evaluate predictions
        u_ics_pred = self.net_u(t_ics, x_ics)
        u_bc1_pred = self.net_u(t_bc1, x_bc1)
        u_bc2_pred = self.net_u(t_bc2, x_bc2)

        r_pred = self.net_r(t_r, x_r)

        # Boundary loss and Initial loss
        loss_ic = self.mse(u_ics - u_ics_pred)
        loss_bc1 = self.mse(u_bc1_pred)
        loss_bc2 = self.mse(u_bc2_pred)

        loss_bcs = loss_bc1 + loss_bc2
        loss_ics = loss_ic

        # Residual loss
        loss_res = self.mse(r_pred)

        # Total loss
        return loss_bcs, loss_ics, loss_res

    def fetch_minibatch(self, sampler, n):
        x, y = sampler.sample(n)
        x = (x - self.mu_x) / self.sigma_x
        return x, y

    def predict_u(self, x_star, dtype):
        """Evaluate predictions at test points"""
        x_star = (x_star - self.mu_x) / self.sigma_x
        t_u = ms.Tensor(x_star[:, 0:1], dtype)
        x_u = ms.Tensor(x_star[:, 1:2], dtype)
        u_star = self.net_u(t_u, x_u)
        return u_star.asnumpy()
