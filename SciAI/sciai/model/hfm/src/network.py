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

"""HFM network"""
import numpy as np
import mindspore as ms
from mindspore import nn, ops
from sciai.architecture import MSE
from sciai.operators import grad


class HFM(nn.Cell):
    """
    HFM

    notational conventions
    _ms: mindspore input/output data and points used to regress the equations
    _pred: output of neural network
    _eqns: points used to regress the equations
    _data: input-output data
    _star: predictions
    """

    def __init__(self, t_data, x_data, y_data, layers, pec, rey):
        super().__init__()
        # flow properties
        self.pec = pec
        self.rey = rey
        # physics "uninformed" neural networks
        self.net_cuvp = NeuralNet(t_data, x_data, y_data, layers=layers)

        self.net_grad_c = grad(self.net_cuvp, 0, (0, 1, 2))
        self.net_grad_u = grad(self.net_cuvp, 1, (0, 1, 2))
        self.net_grad_v = grad(self.net_cuvp, 2, (0, 1, 2))
        self.net_grad_p = grad(self.net_cuvp, 3, (0, 1, 2))

        self.net_grad_cxx = grad(self.net_grad_c, 1, 1)
        self.net_grad_cyy = grad(self.net_grad_c, 2, 2)
        self.net_grad_uxx = grad(self.net_grad_u, 1, 1)
        self.net_grad_uyy = grad(self.net_grad_u, 2, 2)
        self.net_grad_vxx = grad(self.net_grad_v, 1, 1)
        self.net_grad_vyy = grad(self.net_grad_v, 2, 2)

        self.mse = MSE()

    def construct(self, *inputs):
        """Network forward pass"""
        t_data_ms, x_data_ms, y_data_ms, c_data_ms, t_eqns_ms, x_eqns_ms, y_eqns_ms = inputs

        c_data_pred, u_data_pred, v_data_pred, p_data_pred = self.net_cuvp(t_data_ms, x_data_ms, y_data_ms)
        _, u_eqns_pred, v_eqns_pred, _ = self.net_cuvp(t_eqns_ms, x_eqns_ms, y_eqns_ms)

        c_t, c_x, c_y = self.net_grad_c(t_eqns_ms, x_eqns_ms, y_eqns_ms)
        u_t, u_x, u_y = self.net_grad_u(t_eqns_ms, x_eqns_ms, y_eqns_ms)
        v_t, v_x, v_y = self.net_grad_v(t_eqns_ms, x_eqns_ms, y_eqns_ms)
        _, p_x, p_y = self.net_grad_p(t_eqns_ms, x_eqns_ms, y_eqns_ms)

        c_xx = self.net_grad_cxx(t_eqns_ms, x_eqns_ms, y_eqns_ms)
        c_yy = self.net_grad_cyy(t_eqns_ms, x_eqns_ms, y_eqns_ms)
        u_xx = self.net_grad_uxx(t_eqns_ms, x_eqns_ms, y_eqns_ms)
        u_yy = self.net_grad_uyy(t_eqns_ms, x_eqns_ms, y_eqns_ms)
        v_xx = self.net_grad_vxx(t_eqns_ms, x_eqns_ms, y_eqns_ms)
        v_yy = self.net_grad_vyy(t_eqns_ms, x_eqns_ms, y_eqns_ms)

        e1_eqns_pred = c_t + (u_eqns_pred * c_x + v_eqns_pred * c_y) - (1.0 / self.pec) * (c_xx + c_yy)
        e2_eqns_pred = u_t + (u_eqns_pred * u_x + v_eqns_pred * u_y) + p_x - (1.0 / self.rey) * (u_xx + u_yy)
        e3_eqns_pred = v_t + (u_eqns_pred * v_x + v_eqns_pred * v_y) + p_y - (1.0 / self.rey) * (v_xx + v_yy)
        e4_eqns_pred = u_x + v_y

        loss = self.mse(c_data_pred - c_data_ms) + self.mse(e1_eqns_pred) + \
               self.mse(e2_eqns_pred) + self.mse(e3_eqns_pred) + self.mse(e4_eqns_pred)

        return loss, c_data_pred, u_data_pred, v_data_pred, p_data_pred


class MyWithLossCell(nn.Cell):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def construct(self, *inputs):
        """Network forward pass"""
        loss = self.net(*inputs)[0]
        return loss


class NeuralNet(nn.Cell):
    """Neural net"""
    def __init__(self, *inputs, layers):
        super().__init__()
        self.layers = layers
        self.num_layers = len(self.layers)

        if not inputs:
            in_dim = self.layers[0]
            self.x_mean = ops.zeros((1, in_dim))
            self.x_std = ops.ones((1, in_dim), dtype=ms.float32)
        else:
            x = ops.concat(inputs, 1)
            self.x_mean = ops.mean(x, 0, keep_dims=True)
            self.x_std = ops.sqrt(ops.mean(ops.square(x), axis=0, keep_dims=True)
                                  - ops.square(ops.mean(x, axis=0, keep_dims=True)))

        self.weights, self.biases, self.gammas = self.nn_init(layers)
        self.sigmoid = ops.Sigmoid()

    def construct(self, *inputs):
        """Network forward pass"""
        h = (ops.concat(inputs, 1) - self.x_mean) / self.x_std
        for l in range(0, self.num_layers - 1):
            w = self.weights[l]
            b = self.biases[l]
            g = self.gammas[l]
            # weight normalization
            v = w / w.norm(dim=0, keepdim=True)
            # matrix multiplication
            h = ops.matmul(h, v)
            # add bias
            h = g * h + b
            # activation
            if l < self.num_layers - 2:
                h = h * self.sigmoid(h)
        c, u, v, p = ops.split(h, axis=1, split_size_or_sections=1)
        return c, u, v, p

    def nn_init(self, layers):
        """initialize nn"""
        weights = []
        biases = []
        gammas = []

        for l in range(0, self.num_layers - 1):
            in_dim = layers[l]
            out_dim = layers[l + 1]

            w_init = np.random.normal(size=(in_dim, out_dim)).astype(float)
            b_init = np.zeros([1, out_dim]).astype(float)
            g_init = np.ones([1, out_dim]).astype(float)

            w = ms.Parameter(ms.Tensor(w_init, dtype=ms.float32), name="W" + str(l))
            b = ms.Parameter(ms.Tensor(b_init, dtype=ms.float32), name="b" + str(l))
            g = ms.Parameter(ms.Tensor(g_init, dtype=ms.float32), name="g" + str(l))

            weights.append(w)
            biases.append(b)
            gammas.append(g)

        return ms.ParameterTuple(weights), ms.ParameterTuple(biases), ms.ParameterTuple(gammas)
