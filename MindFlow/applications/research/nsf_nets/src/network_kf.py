# Copyright 2024 Huawei Technologies Co., Ltd
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
# pylint: disable=C0103
import mindspore as ms
from mindspore import nn, ops
from mindspore.common.api import jit
from mindspore.amp import DynamicLossScaler
import numpy as np
from src.derivatives import SecondOrderGrad, Grad


# the deep neural network
class neural_net(nn.Cell):
    """
    neural_net
    """
    def __init__(self, layers, msfloat_type, lb, ub):
        super(neural_net, self).__init__()

        # normalize the scope
        self.lb = lb
        self.ub = ub

        # network parameters
        self.depth = len(layers) - 1

        # set up activation
        self.activation = nn.Tanh()

        layer_list = []
        for i in range(self.depth - 1):
            layer_list.append(nn.Dense(layers[i], layers[i+1]).to_float(msfloat_type))
            layer_list.append(self.activation.to_float(msfloat_type))

        layer_list.append(nn.Dense(layers[-2], layers[-1]).to_float(msfloat_type))

        # deploy layers
        self.layers = nn.SequentialCell(layer_list).to_float(msfloat_type)

    @jit
    def construct(self, x):
        H = 2.0 * (x - self.lb) / (self.ub - self.lb) - 1.0
        out = self.layers(H)
        return out

class VPNSFNets:
    """
    VPNSFNets
    """
    # Initialize the class
    def __init__(self, xb, yb, ub, vb, x_f, y_f, layers, use_npu,
                 msfloat_type, npfloat_type, load_params, second_path):

        self.use_npu = use_npu

        self.msfloat_type = msfloat_type
        self.npfloat_type = npfloat_type

        self.Xb_l = ops.concat([ms.Tensor(np.array(xb, self.npfloat_type)),
                                ms.Tensor(np.array(yb, self.npfloat_type))], 1)
        self.X_l = ops.concat([ms.Tensor(np.array(x_f, self.npfloat_type)),
                               ms.Tensor(np.array(y_f, self.npfloat_type))], 1)

        self.Xmin = ms.Tensor(np.array(self.Xb_l.min(0), self.npfloat_type))
        self.Xmax = ms.Tensor(np.array(self.Xb_l.max(0), self.npfloat_type))


        self.layers = layers

        self.x_f = ms.Tensor(np.array(x_f, self.npfloat_type))
        self.y_f = ms.Tensor(np.array(y_f, self.npfloat_type))

        self.xb = ms.Tensor(np.array(xb, self.npfloat_type))
        self.yb = ms.Tensor(np.array(yb, self.npfloat_type))
        self.ub = ms.Tensor(np.array(ub, self.npfloat_type))
        self.vb = ms.Tensor(np.array(vb, self.npfloat_type))

        # Initialize NNs---deep neural networks
        self.dnn = neural_net(layers, self.msfloat_type, self.Xmin, self.Xmax)
        if load_params:
            params_dict = ms.load_checkpoint(f'model/{second_path}/model.ckpt')
            ms.load_param_into_net(self.dnn, params_dict)

        # The function of Auto-differentiation
        self.grad = Grad(self.dnn)
        self.hessian_u_xx = SecondOrderGrad(self.dnn, 0, 0, output_idx=0)
        self.hessian_u_yy = SecondOrderGrad(self.dnn, 1, 1, output_idx=0)

        self.hessian_v_xx = SecondOrderGrad(self.dnn, 0, 0, output_idx=1)
        self.hessian_v_yy = SecondOrderGrad(self.dnn, 1, 1, output_idx=1)

        self.loss_scaler = DynamicLossScaler(1024, 2, 100)

    def net_u(self, x, y):
        H = ms.ops.concat([x, y], 1)
        Y = self.dnn(H)
        return Y

    def net_f(self, x_f, y_f):
        """
        net_f
        """
        U = self.net_u(x_f, y_f)
        u = U[:, 0:1]
        v = U[:, 1:2]
        p = U[:, 2:3]

        data = ms.ops.concat([x_f, y_f], 1)

        u_x = self.grad(data, 0, 0, U)
        u_y = self.grad(data, 1, 0, U)
        u_xx = self.hessian_u_xx(data)
        u_yy = self.hessian_u_yy(data)

        v_x = self.grad(data, 0, 1, U)
        v_y = self.grad(data, 1, 1, U)
        v_xx = self.hessian_v_xx(data)
        v_yy = self.hessian_v_yy(data)

        p_x = self.grad(data, 0, 2, U)
        p_y = self.grad(data, 1, 2, U)

        f_u = (u * u_x + v * u_y) + p_x - (1.0/40) * (u_xx + u_yy)
        f_v = (u * v_x + v * v_y) + p_y - (1.0/40) * (v_xx + v_yy)
        f_e = u_x + v_y

        return u, v, p, f_u, f_v, f_e


    def loss_fn(self, xb, yb, x_f, y_f, ub, vb):
        """
        loss_fn
        """
        Ub = self.net_u(xb, yb)
        ub_pred = Ub[:, 0:1]
        vb_pred = Ub[:, 1:2]


        _, _, _, f_u_pred, f_v_pred, f_e_pred = self.net_f(x_f, y_f)

        loss_ub = ops.reduce_mean(ops.square(ub_pred - ub))
        loss_vb = ops.reduce_mean(ops.square(vb_pred - vb))

        alpha = ms.Tensor(1., dtype=self.msfloat_type)

        loss_f_u = ops.reduce_mean(ops.square(f_u_pred - ms.ops.zeros_like(f_u_pred)))
        loss_f_v = ops.reduce_mean(ops.square(f_v_pred - ms.ops.zeros_like(f_v_pred)))
        loss_f_e = ops.reduce_mean(ops.square(f_e_pred - ms.ops.zeros_like(f_e_pred)))

        loss_b = loss_ub + loss_vb
        loss_f = loss_f_u + loss_f_v + loss_f_e

        loss = alpha * loss_b + loss_f
        if self.use_npu:
            loss = self.loss_scaler.scale(loss)
        return loss, loss_b, loss_f
