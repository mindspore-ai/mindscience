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
"""Network definitions"""
import mindspore as ms
from mindspore import nn, ops

from sciai.architecture import MLP, MSE
from sciai.common.initializer import XavierTruncNormal
from sciai.operators import grad


class VPNSFnet(nn.Cell):
    """VPNSF network"""
    def __init__(self, xb, yb, ub, vb, x, y, layers):
        super().__init__()
        # remove the second bracket
        xb_list = ops.concat([xb, yb], 1)
        x_list = ops.concat([x, y], 1)

        self.xb_list = xb_list
        self.x_list = x_list

        self.xb = xb
        self.yb = yb
        self.x = x
        self.y = y

        self.ub = ub
        self.vb = vb

        self.layers = layers
        self.alpha = ms.Tensor(1, dtype=ms.float32)
        self.grad = ops.GradOperation(get_all=True)

        # Initialize NN
        self.neural_net = Net(layers, xb_list.min(0), xb_list.max(0))
        self.net_f_ns = NetFNS(self.neural_net)
        self.mse = MSE()

    def construct(self, *inputs):
        """Network forward pass"""
        xb, yb, ub, vb, x, y = inputs
        u_boundary_pred, v_boundary_pred, _ = self.neural_net(xb, yb)
        _, _, _, f_u_pred, f_v_pred, f_e_pred = self.net_f_ns(x, y)

        # set loss function
        loss = self.alpha * self.mse(ub - u_boundary_pred) + self.alpha * self.mse(vb - v_boundary_pred) + \
               self.mse(f_u_pred) + self.mse(f_v_pred) + self.mse(f_e_pred)
        return loss


class NetFNS(nn.Cell):
    """FNS network"""
    def __init__(self, neural_net):
        super().__init__()
        self.neural_net = neural_net
        self.net_grad_u = grad(self.neural_net, 0, (0, 1))  # du/dx, du/dy
        self.net_grad_v = grad(self.neural_net, 1, (0, 1))  # dv/dx, dv/dy
        self.net_grad_p = grad(self.neural_net, 2, (0, 1))  # dp/dx, dp/dy
        self.ops_ux2 = grad(self.net_grad_u, 0, (0, 1))  # du2/dx2, du2/dxdy
        self.ops_uy2 = grad(self.net_grad_u, 1, (0, 1))  # du2/dydx, du2/dy2
        self.ops_vx2 = grad(self.net_grad_v, 0, (0, 1))  # dv2/dx2, dv2/dxdy
        self.ops_vy2 = grad(self.net_grad_v, 1, (0, 1))  # dv2/dydx, dv2/dy2

    def construct(self, x, y):
        """Network forward pass"""
        u, v, p = self.neural_net(x, y)

        u_x, u_y = self.net_grad_u(x, y)
        u_xx, _ = self.ops_ux2(x, y)
        _, u_yy = self.ops_uy2(x, y)

        v_x, v_y = self.net_grad_v(x, y)
        v_xx, _ = self.ops_vx2(x, y)
        _, v_yy = self.ops_vy2(x, y)

        p_x, p_y = self.net_grad_p(x, y)

        f_u = (u * u_x + v * u_y) + p_x - (1.0 / 40) * (u_xx + u_yy)
        f_v = (u * v_x + v * v_y) + p_y - (1.0 / 40) * (v_xx + v_yy)
        f_e = u_x + v_y

        return u, v, p, f_u, f_v, f_e


class Net(nn.Cell):
    """MLP network"""
    def __init__(self, layers, lowb, upb):
        super().__init__()
        self.lowb, self.upb = lowb, upb
        self.mlp = MLP(layers=layers, weight_init=XavierTruncNormal(), bias_init="zeros",
                       activation="tanh")

    def construct(self, x, y):
        """Network forward pass"""
        out = ops.concat([x, y], 1)
        out = 2.0 * (out - self.lowb) / (self.upb - self.lowb) - 1.0
        out = self.mlp(out)
        return ops.split(out, split_size_or_sections=1, axis=1)
