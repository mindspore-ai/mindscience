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
"""Network architectures for XPINNs"""
import mindspore as ms
from mindspore import nn, ops
from sciai.architecture import MLPAAF, MSE
from sciai.operators.derivatives import grad
from sciai.utils import to_tensor


class Net(nn.Cell):
    """
    The DNN part of XPINN.
    Args:
        layers (list): a list that represents the number of neurons in each layer.
        act (Cell or ops.func): the activation function for the Net.
    """

    def __init__(self, layers, act):
        super().__init__()
        self.mlp = MLPAAF(layers, weight_init="XavierNormal", bias_init="zeros",
                          activation=act, a_value=0.05, scale=20, share_type="layer_wise")

    def construct(self, x, y):
        """Network forward pass"""
        return self.mlp(ops.concat([x, y], 1))


class SecondGradNet(nn.Cell):
    """
    The second order differential network for the input net.
    Args:
        net (Cell): the input network to which you want to differentiate.
    """

    def __init__(self, net):
        super().__init__()
        self.net = net
        self.d_net = grad(self.net, output_index=0, input_index=(0, 1))
        self.net_xx = grad(self.d_net, output_index=0, input_index=0)
        self.net_yy = grad(self.d_net, output_index=1, input_index=1)

    def construct(self, x, y):
        """Network forward pass"""
        u = self.net(x, y)
        u_xx = self.net_xx(x, y)
        u_yy = self.net_yy(x, y)
        return u, u_xx, u_yy


class NetF(nn.Cell):
    """
    The PDE part of XPINN.
    Args:
        net_u1 (Cell): the DNN network for subdomain 1.
        net_u2 (Cell): the DNN network for subdomain 2.
        net_u3 (Cell): the DNN network for subdomain 3.
    """

    def __init__(self, net_u1, net_u2, net_u3):
        super(NetF, self).__init__()
        self.grad_net1 = SecondGradNet(net_u1)
        self.grad_net2 = SecondGradNet(net_u2)
        self.grad_net3 = SecondGradNet(net_u3)

    def construct(self, *inputs):
        """Network forward pass"""
        x1, y1, x2, y2, x3, y3, xi1, yi1, xi2, yi2 = inputs
        _, u1_xx, u1_yy = self.grad_net1(x1, y1)  # Sub-Net1
        _, u2_xx, u2_yy = self.grad_net2(x2, y2)  # Sub-Net2
        _, u3_xx, u3_yy = self.grad_net3(x3, y3)  # Sub-Net3
        u1i1, u1i1_xx, u1i1_yy = self.grad_net1(xi1, yi1)  # Sub-Net1, Interface 1
        u2i1, u2i1_xx, u2i1_yy = self.grad_net2(xi1, yi1)  # Sub-Net2, Interface 1
        u1i2, u1i2_xx, u1i2_yy = self.grad_net1(xi2, yi2)  # Sub-Net1, Interface 2
        u3i2, u3i2_xx, u3i2_yy = self.grad_net3(xi2, yi2)  # Sub-Net3, Interface 2
        # Average value (Required for enforcing the average solution along the interface)
        uavgi1 = (u1i1 + u2i1) / 2
        uavgi2 = (u1i2 + u3i2) / 2
        # Residuals
        f1 = u1_xx + u1_yy - (ops.exp(x1) + ops.exp(y1))
        f2 = u2_xx + u2_yy - (ops.exp(x2) + ops.exp(y2))
        f3 = u3_xx + u3_yy - (ops.exp(x3) + ops.exp(y3))
        # Residual continuity conditions on the interfaces
        fi1 = (u1i1_xx + u1i1_yy - (ops.exp(xi1) + ops.exp(yi1))) - (u2i1_xx + u2i1_yy - (ops.exp(xi1) + ops.exp(yi1)))
        fi2 = (u1i2_xx + u1i2_yy - (ops.exp(xi2) + ops.exp(yi2))) - (u3i2_xx + u3i2_yy - (ops.exp(xi2) + ops.exp(yi2)))
        return f1, f2, f3, fi1, fi2, uavgi1, uavgi2, u1i1, u1i2, u2i1, u3i2


class XPINN(nn.Cell):
    """
    X-PINN
    Args:
        layers1 (Cell): the layer list for DNN network of subdomain 1.
        layers2 (Cell): the layer list for DNN network of subdomain 2.
        layers3 (Cell): the layer list for DNN network of subdomain 3.
    """

    def __init__(self, layers1, layers2, layers3):
        super().__init__()
        self.net_u1 = Net(layers1, nn.Tanh())
        self.net_u2 = Net(layers2, ops.sin)
        self.net_u3 = Net(layers3, ops.cos)
        self.net_f = NetF(self.net_u1, self.net_u2, self.net_u3)
        self.mse = MSE()

    def construct(self, *inputs):
        """Network forward pass"""
        ub, x_ub, y_ub, x_f1, y_f1, x_f2, y_f2, x_f3, y_f3, x_fi1, y_fi1, x_fi2, y_fi2 = inputs
        ub1_pred = self.net_u1(x_ub, y_ub)
        f1_pred, f2_pred, f3_pred, fi1_pred, fi2_pred, uavgi1_pred, uavgi2_pred, u1i1_pred, u1i2_pred, u2i1_pred, \
        u3i2_pred = self.net_f(x_f1, y_f1, x_f2, y_f2, x_f3, y_f3, x_fi1, y_fi1, x_fi2, y_fi2)

        loss1 = 20 * self.mse(ub - ub1_pred) \
                + self.mse(f1_pred) \
                + self.mse(fi1_pred) \
                + self.mse(fi2_pred) \
                + 20 * self.mse(u1i1_pred - uavgi1_pred) \
                + 20 * self.mse(u1i2_pred - uavgi2_pred)
        loss2 = self.mse(f2_pred) \
                + self.mse(fi1_pred) \
                + 20 * self.mse(u2i1_pred - uavgi1_pred)
        loss3 = self.mse(f3_pred) \
                + self.mse(fi2_pred) \
                + 20 * self.mse(u3i2_pred - uavgi2_pred)
        return loss1, loss2, loss3

    def predict(self, x_star1, x_star2, x_star3):
        x_ub, y_ub = to_tensor((x_star1[:, 0:1], x_star1[:, 1:2]), ms.float32)
        x_f2, y_f2 = to_tensor((x_star2[:, 0:1], x_star2[:, 1:2]), ms.float32)
        x_f3, y_f3 = to_tensor((x_star3[:, 0:1], x_star3[:, 1:2]), ms.float32)
        u1_star = self.net_u1(x_ub, y_ub)
        u2_star = self.net_u2(x_f2, y_f2)
        u3_star = self.net_u3(x_f3, y_f3)
        return u1_star.asnumpy(), u2_star.asnumpy(), u3_star.asnumpy()

    def save(self, folder):
        ms.save_checkpoint(self.net_u1, f"{folder}/net1.ckpt")
        ms.save_checkpoint(self.net_u2, f"{folder}/net2.ckpt")
        ms.save_checkpoint(self.net_u3, f"{folder}/net3.ckpt")
