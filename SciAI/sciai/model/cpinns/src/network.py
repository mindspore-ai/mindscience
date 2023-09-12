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

"""Network architecture for cpinns"""
import mindspore as ms
from mindspore import nn, ops
from sciai.architecture import MLPAAF, MSE, Normalize
from sciai.common.initializer import XavierTruncNormal
from sciai.operators import grad
from sciai.utils.ms_utils import to_tensor


class Net(nn.Cell):
    """Net"""
    def __init__(self, layers, lb):
        super().__init__()
        self.mlp = MLPAAF(layers=layers, weight_init=XavierTruncNormal(), bias_init="zeros", a_value=0.05,
                          scale=20, share_type="global")
        self.normalize = Normalize(ms.Tensor(lb), ms.Tensor(1.0))

    def construct(self, x, t):
        """Network forward pass"""
        h = ops.concat([x, t], 1)
        h = self.normalize(h)
        h = self.mlp(h)
        return h


class GradNet(nn.Cell):
    """GradNet"""
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.net_grad = grad(self.net, input_index=(0, 1))
        self.net_grad_xx = grad(grad(self.net, input_index=0), input_index=0)

    def construct(self, x, t):
        """Network forward pass"""
        u = self.net(x, t)
        u_x, u_t = self.net_grad(x, t)
        u_xx = self.net_grad_xx(x, t)
        return u, u_x, u_t, u_xx


class NetF(nn.Cell):
    """NetF"""
    def __init__(self, nu, *neural_nets):
        super().__init__()
        self.nu = nu
        net_u1, net_u2, net_u3, net_u4 = neural_nets
        self.grad_net1 = GradNet(net_u1)
        self.grad_net2 = GradNet(net_u2)
        self.grad_net3 = GradNet(net_u3)
        self.grad_net4 = GradNet(net_u4)

    def construct(self, *args):
        """Network forward pass"""
        x1, t1, x2, t2, x3, t3, x4, t4, xi1, ti1, xi2, ti2, xi3, ti3 = args
        u1, u1_x, u1_t, u1_xx = self.grad_net1(x1, t1)
        u1i1, u1i1_x, _, _ = self.grad_net1(xi1, ti1)

        u2, u2_x, u2_t, u2_xx = self.grad_net2(x2, t2)
        u2i1, _, _, u2i1_xx = self.grad_net2(xi1, ti1)
        u2i2, u2i2_x, _, _ = self.grad_net2(xi2, ti2)

        u3, u3_x, u3_t, u3_xx = self.grad_net3(x3, t3)
        u3i2, _, _, u3i2_xx = self.grad_net3(xi2, ti2)
        u3i3, u3i3_x, _, _ = self.grad_net3(xi3, ti3)

        u4, u4_x, u4_t, u4_xx = self.grad_net4(x4, t4)
        u4i3, _, _, u4i3_xx = self.grad_net4(xi3, ti3)

        uavgi1 = (u1i1 + u2i1) / 2
        uavgi2 = (u2i2 + u3i2) / 2
        uavgi3 = (u3i3 + u4i3) / 2

        f1 = u1_t + u1 * u1_x - self.nu * u1_xx
        f2 = u2_t + u2 * u2_x - self.nu * u2_xx
        f3 = u3_t + u3 * u3_x - self.nu * u3_xx
        f4 = u4_t + u4 * u4_x - self.nu * u4_xx

        fi1 = u1i1 ** 2 / 2 - self.nu * u1i1_x - (u2i1 ** 2 / 2 - self.nu * u2i1_xx)
        fi2 = u2i2 ** 2 / 2 - self.nu * u2i2_x - (u3i2 ** 2 / 2 - self.nu * u3i2_xx)
        fi3 = u3i3 ** 2 / 2 - self.nu * u3i3_x - (u4i3 ** 2 / 2 - self.nu * u4i3_xx)

        return f1, f2, f3, f4, fi1, fi2, fi3, uavgi1, uavgi2, uavgi3, u1i1, u2i1, u2i2, u3i2, u3i3, u4i3


class PINN(nn.Cell):
    """PINN network"""
    def __init__(self, nn_layers_total, nu, x_interface, dtype):
        super().__init__()
        self.dtype = dtype
        self.net1 = Net(nn_layers_total[0], x_interface[0])
        self.net2 = Net(nn_layers_total[1], x_interface[1])
        self.net3 = Net(nn_layers_total[2], x_interface[2])
        self.net4 = Net(nn_layers_total[3], x_interface[3])

        # Forward Differentiation Network
        self.net_f = NetF(nu, self.net1, self.net2, self.net3, self.net4)
        self.mse = MSE()

    def construct(self, *inputs):
        """Network forward pass"""
        u1, u2, u3, u4, x_u1, t_u1, x_u2, t_u2, x_u3, t_u3, x_u4, t_u4 = inputs[:12]

        f1_pred, f2_pred, f3_pred, f4_pred, fi1_pred, fi2_pred, fi3_pred, uavgi1_pred, uavgi2_pred, \
        uavgi3_pred, u1i1_pred, u2i1_pred, u2i2_pred, u3i2_pred, u3i3_pred, u4i3_pred = self.net_f(*inputs[12:])

        u1_pred = self.net1(x_u1, t_u1)
        u2_pred = self.net2(x_u2, t_u2)
        u3_pred = self.net3(x_u3, t_u3)
        u4_pred = self.net4(x_u4, t_u4)

        loss1 = 20 * self.mse(u1 - u1_pred) \
                + self.mse(f1_pred) \
                + 20 * self.mse(fi1_pred) \
                + 20 * self.mse(u1i1_pred - uavgi1_pred)

        loss2 = 20 * self.mse(u2 - u2_pred) \
                + self.mse(f2_pred) \
                + 20 * (self.mse(fi1_pred) + self.mse(fi2_pred)) \
                + 20 * self.mse(u2i1_pred - uavgi1_pred) \
                + 20 * self.mse(u2i2_pred - uavgi2_pred)

        loss3 = 20 * self.mse(u3 - u3_pred) \
                + self.mse(f3_pred) \
                + 20 * (self.mse(fi2_pred) + self.mse(fi3_pred)) \
                + 20 * self.mse(u3i2_pred - uavgi2_pred) \
                + 20 * self.mse(u3i3_pred - uavgi3_pred)

        loss4 = 20 * (self.mse(u4 - u4_pred)) \
                + self.mse(f4_pred) \
                + 20 * self.mse(fi3_pred) \
                + 20 * self.mse(u4i3_pred - uavgi3_pred)

        return loss1, loss2, loss3, loss4

    def predict(self, x1_star, x2_star, x3_star, x4_star):
        x1, t1 = to_tensor((x1_star[:, 0:1], x1_star[:, 1:2]), self.dtype)
        x2, t2 = to_tensor((x2_star[:, 0:1], x2_star[:, 1:2]), self.dtype)
        x3, t3 = to_tensor((x3_star[:, 0:1], x3_star[:, 1:2]), self.dtype)
        x4, t4 = to_tensor((x4_star[:, 0:1], x4_star[:, 1:2]), self.dtype)
        u1_star = self.net1(x1, t1)
        u2_star = self.net2(x2, t2)
        u3_star = self.net3(x3, t3)
        u4_star = self.net4(x4, t4)
        return u1_star.asnumpy(), u2_star.asnumpy(), u3_star.asnumpy(), u4_star.asnumpy()

    def save(self, folder):
        ms.save_checkpoint(self.net1, f"{folder}/net1.ckpt")
        ms.save_checkpoint(self.net2, f"{folder}/net2.ckpt")
        ms.save_checkpoint(self.net3, f"{folder}/net3.ckpt")
        ms.save_checkpoint(self.net4, f"{folder}/net4.ckpt")
