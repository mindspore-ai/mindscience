
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

"""Network architectures for Label-free DNN Surrogate"""
import math

import numpy as np
import mindspore as ms
from mindspore import nn, ops, Tensor
from mindspore.common.initializer import HeUniform
from sciai.architecture import MLP, MSE, Swish
from sciai.operators import grad


class UVPNet(nn.Cell):
    """UVP net"""
    def __init__(self, layers, sigma, mu, r_inlet, x_start, d_p, l, x_end):
        super().__init__()
        self.sigma, self.mu, self.r_inlet, self.x_start, self.d_p, self.l, self.x_end \
            = sigma, mu, r_inlet, x_start, d_p, l, x_end
        he_uniform = HeUniform(negative_slope=math.sqrt(5))
        swish = Swish()
        self.net_u = MLP(layers, weight_init=he_uniform, bias_init="zeros", activation=swish)
        self.net_v = MLP(layers, weight_init=he_uniform, bias_init="zeros", activation=swish)
        self.net_p = MLP(layers, weight_init=he_uniform, bias_init="zeros", activation=swish)
        self.const = Tensor([1 / math.sqrt(2 * np.pi * sigma ** 2)], dtype=ms.float32)

    def construct(self, x, y, scale):
        """Network forward pass"""
        net_in = ops.concat((x, y, scale), axis=1)
        u = self.net_u(net_in)
        v = self.net_v(net_in)
        p = self.net_p(net_in)
        u = u.view(len(u), -1)
        v = v.view(len(v), -1)
        p = p.view(len(p), -1)

        # analytical symmetric boundary
        r = scale * self.const * ops.exp(-(x - self.mu) ** 2 / (2 * self.sigma ** 2))
        h = self.r_inlet - r

        u_hard = u * (h ** 2 - y ** 2)
        v_hard = (h ** 2 - y ** 2) * v
        p_hard = (self.x_start - x) * 0 + self.d_p * (self.x_end - x) / self.l + 0 * y + (self.x_start - x) * (
            self.x_end - x) * p
        return u_hard, v_hard, p_hard


class WithLossCell(nn.Cell):
    """Loss net"""
    def __init__(self, uvp_net, nu, rho):
        super().__init__()
        self.nu, self.rho = nu, rho
        self.grad = ops.GradOperation(get_all=True, sens_param=True)
        self.uvp_net = uvp_net
        self.u_grad, self.v_grad, self.p_grad = (grad(self.uvp_net, output_index=i) for i in range(3))
        self.u_x_grad, self.u_y_grad = (grad(self.u_grad, output_index=i) for i in range(2))
        self.v_x_grad, self.v_y_grad = (grad(self.v_grad, output_index=i) for i in range(2))
        self.loss_func = MSE()

    def construct(self, x, y, scale):
        """Network forward pass"""
        u_hard, v_hard, _ = self.uvp_net(x, y, scale)
        p_x, p_y, _ = self.p_grad(x, y, scale)

        u_x, u_y, _ = self.u_grad(x, y, scale)
        u_xx, _, _ = self.u_x_grad(x, y, scale)
        _, u_yy, _ = self.u_y_grad(x, y, scale)
        loss_1 = u_hard * u_x + v_hard * u_y - self.nu * (u_xx + u_yy) + 1 / self.rho * p_x

        v_x, v_y, _ = self.v_grad(x, y, scale)
        v_xx, _, _ = self.v_x_grad(x, y, scale)
        _, v_yy, _ = self.v_y_grad(x, y, scale)
        loss_2 = u_hard * v_x + v_hard * v_y - self.nu * (v_xx + v_yy) + 1 / self.rho * p_y

        loss_3 = u_x + v_y

        loss = self.loss_func(loss_1) + self.loss_func(loss_2) + self.loss_func(loss_3)
        return loss
