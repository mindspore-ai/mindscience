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
"""Linear advection sphere definition"""
import random

from mindspore import ops, nn


class NetGrad(nn.Cell):
    """Gradient network"""
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.grad = ops.GradOperation(get_all=True)

    def construct(self, inp1, inp2, inp3):
        """Network forward pass"""
        return self.grad(self.net)(inp1, inp2, inp3)


class Net(nn.Cell):
    """Backbone network"""
    def __init__(self, model, alpha, u00, u):
        super().__init__()
        self.model = model
        self.net_grad = NetGrad(self.model)
        self.alpha = alpha
        self.u00 = u00
        self.u = u

    def construct(self, pdes, inits):
        """Network forward pass"""
        # PDE points and associated self-adaptation weights
        t_pde, x_pde, y_pde = pdes[:, :1], pdes[:, 1:2], pdes[:, 2:3]

        # Initial value points and associated self-adaptation weights
        t_init, x_init, y_init, h_init = inits[:, :1], inits[:, 1:2], inits[:, 2:3], inits[:, 3:4]
        # Outer gradient for tuning network parameters
        dhdt, dhdx, dhdy = self.net_grad(t_pde, x_pde, y_pde)
        # Solve the linear advection equation
        u = self.u00 * (ops.cos(y_pde) * ops.cos(self.alpha) + ops.sin(y_pde) * ops.cos(x_pde) *
                        ops.sin(self.alpha)) / self.u
        v = -self.u00 * ops.sin(x_pde) * ops.sin(self.alpha) / self.u
        eqn = dhdt + u / ops.cos(y_pde) * dhdx + v * dhdy

        # Define the PDE loss
        pde_loss = ops.reduce_mean(ops.square(eqn))

        # Define the IVP loss
        h_init_pred = self.model(t_init, x_init, y_init)
        ic_loss = ops.reduce_mean(ops.square(h_init - h_init_pred))
        return pde_loss, ic_loss


class TrainOneStepCell(nn.Cell):
    """One-step training process"""
    def __init__(self, network, optimizer):
        super().__init__()
        self.network = network
        self.optimizer = optimizer
        self.weights = self.optimizer.parameters
        self.grad = ops.GradOperation(get_by_list=True, sens_param=True)

    def construct(self, pdes, inits):
        """Network forward pass"""
        pde_loss, ic_loss = self.network(pdes, inits)
        ones_pde_loss = ops.ones_like(pde_loss)
        ones_ic_loss = ops.ones_like(ic_loss)
        zeros_pde_loss = ops.zeros_like(pde_loss)
        zeros_ic_loss = ops.zeros_like(ic_loss)
        grad_pde = self.grad(self.network, self.weights)(pdes, inits, (ones_pde_loss, zeros_ic_loss))
        grad_ivp = self.grad(self.network, self.weights)(pdes, inits, (zeros_pde_loss, ones_ic_loss))
        grad_pde_f = ops.concat([ops.reshape(p, (-1,)) for p in grad_pde], axis=0)
        grad_ivp_f = ops.concat([ops.reshape(p, (-1,)) for p in grad_ivp], axis=0)

        # Project conflicting gradients
        mul_res = ops.mul(grad_ivp_f, grad_pde_f)
        if ops.reduce_sum(mul_res) < 0:
            dice = random.choice([0, 1])
            if dice < 0.5:
                proj = ops.reduce_sum(mul_res) / ops.reduce_sum(ops.square(grad_ivp_f))
                grad_pde = tuple(g_pde - proj * g_ivp for g_pde, g_ivp in zip(grad_pde, grad_ivp))
            else:
                proj = ops.reduce_sum(mul_res) / ops.reduce_sum(ops.square(grad_pde_f))
                grad_ivp = tuple(g_ivp - proj * g_pde for g_pde, g_ivp in zip(grad_pde, grad_ivp))

        grads = tuple(g_pde + g_ivp for g_pde, g_ivp in zip(grad_pde, grad_ivp))
        self.optimizer(grads)
        return pde_loss, ic_loss
