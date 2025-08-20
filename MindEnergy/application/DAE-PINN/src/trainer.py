# Copyright 2025 Huawei Technologies Co., Ltd
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
# pylint: disable-all
import numpy as np
from mindspore import ops, Tensor, load_param_into_net, load_checkpoint


def power_net_dae(yn, pred, h, IRK_weights):
    T = 1.0
    # parameters
    m_1, m_2, d, d_d, b = .052, .0531, .05, .005, 10.
    v_1, v_2, p_g, p_l, q_l = 1.02, 1.05, -2.0, 3.0, .1
    # pinn
    w1, w2, d2, d3, v3 = pred

    xi_w1 = w1[..., :-1]
    xi_w2 = w2[..., :-1]
    xi_d2 = d2[..., :-1]
    xi_d3 = d3[..., :-1]
    zeta_v3 = v3[..., :-1]

    f_1 = b * v_1 * v_2 * ops.sin(xi_d2) + b * \
        v_2 * zeta_v3 * ops.sin(xi_d2 - xi_d3) + p_g
    f_2 = b * v_1 * zeta_v3 * \
        ops.sin(xi_d3) + b * v_2 * zeta_v3 * ops.sin(xi_d3 - xi_d2) + p_l

    # compute dynamic residuals
    F0 = T * (1 / m_1) * (- d * xi_w1 + f_1 + f_2)
    F1 = T * (1 / m_2) * (- d * xi_w2 - f_1)
    F2 = T * (xi_w2 - xi_w1)

    F3 = T * (- xi_w1 - (1 / d_d) * f_2)

    f0 = yn[..., 0:1] - (w1 - h*F0.mm(IRK_weights.T))
    f1 = yn[..., 1:2] - (w2 - h*F1.mm(IRK_weights.T))
    f2 = yn[..., 2:3] - (d2 - h*F2.mm(IRK_weights.T))
    f3 = yn[..., 3:4] - (d3 - h*F3.mm(IRK_weights.T))

    # compute algebrtaic residuals
    G = 2 * b * (v3 ** 2) - b * v3 * v_1 * ops.cos(d3) - \
        b * v3 * v_2 * ops.cos(d3 - d2) + q_l
    g = - (T/v3) * G

    return [f0, f1, f2, f3], [g]


class DaeTrainer:
    def __init__(self, net, IRK_weights, IRK_times, h=0.1, dyn_weight=64.0, alg_weight=1.0):
        self.net = net
        self.IRK_weights = IRK_weights
        self.IRK_times = IRK_times
        self.h = Tensor([h])
        self.loss_weights = [dyn_weight, alg_weight]

    def get_pred(self, x):
        return self.net(x)

    def sum_loss(self, loss_list):
        for k, weight in enumerate(self.loss_weights):
            loss_list[k] *= weight
        return sum(loss_list)

    def get_loss(self, inputs):
        """ compute losses """
        loss_list = []
        pred = self.get_pred(inputs)
        f, g = power_net_dae(inputs, pred, self.h, self.IRK_weights)
        # losses from dynamic equations
        losses_dyn = [ops.mse_loss(fi, Tensor(0)) for fi in f]
        loss_list.append(sum(losses_dyn))
        # losses from algebraic/perturbed/stiff equations
        losses_alg = [ops.mse_loss(gi, Tensor(0)) for gi in g]
        loss_list.append(sum(losses_alg))
        return self.sum_loss(loss_list)

    def predict(self, inputs, model_restore_path=None):
        if model_restore_path is not None:
            param_dict = load_checkpoint(model_restore_path)
            load_param_into_net(self.net, param_dict)
        self.net.set_train(False)
        inputs = Tensor(inputs)
        vel1, vel2, ang2, ang3, v3 = self.net(inputs)
        y_pred = np.vstack((vel1.numpy(), vel2.numpy(),
                           ang2.numpy(), ang3.numpy(), v3.numpy()))
        return y_pred

    def integrate(self, x0, N=1, dyn_state_dim=4, model_restore_path=None):
        yn = x0
        soln = []
        for _ in range(N):
            y_pred_n = self.predict(yn.reshape(
                1, -1), model_restore_path=model_restore_path)
            soln.append(y_pred_n)
            yn = y_pred_n[:dyn_state_dim, -1]
        return np.hstack(soln)
