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
    def __init__(self, xb, yb, zb, tb, xi, yi, zi, ti, ub, vb, wb, ui, vi, wi, x_f, y_f, z_f, t_f, layers, Re, \
        Xmin, Xmax, use_npu, msfloat_type, npfloat_type, load_params, second_path):

        self.use_npu = use_npu

        self.msfloat_type = msfloat_type
        self.npfloat_type = npfloat_type

        self.Xmin = ms.Tensor(np.array(Xmin, self.npfloat_type))
        self.Xmax = ms.Tensor(np.array(Xmax, self.npfloat_type))

        # The size of network
        self.layers = layers
        self.Re = Re

        # Training data
        self.x_f = ms.Tensor(np.array(x_f, self.npfloat_type))
        self.y_f = ms.Tensor(np.array(y_f, self.npfloat_type))
        self.z_f = ms.Tensor(np.array(z_f, self.npfloat_type))
        self.t_f = ms.Tensor(np.array(t_f, self.npfloat_type))

        self.xb = ms.Tensor(np.array(xb, self.npfloat_type))
        self.yb = ms.Tensor(np.array(yb, self.npfloat_type))
        self.zb = ms.Tensor(np.array(zb, self.npfloat_type))
        self.tb = ms.Tensor(np.array(tb, self.npfloat_type))

        self.xi = ms.Tensor(np.array(xi, self.npfloat_type))
        self.yi = ms.Tensor(np.array(yi, self.npfloat_type))
        self.zi = ms.Tensor(np.array(zi, self.npfloat_type))
        self.ti = ms.Tensor(np.array(ti, self.npfloat_type))

        self.ub = ms.Tensor(np.array(ub, self.npfloat_type))
        self.vb = ms.Tensor(np.array(vb, self.npfloat_type))
        self.wb = ms.Tensor(np.array(wb, self.npfloat_type))

        self.ui = ms.Tensor(np.array(ui, self.npfloat_type))
        self.vi = ms.Tensor(np.array(vi, self.npfloat_type))
        self.wi = ms.Tensor(np.array(wi, self.npfloat_type))

        # Initialize NNs---deep neural networks
        self.dnn = neural_net(layers, self.msfloat_type, self.Xmin, self.Xmax)
        if load_params:
            params_dict = ms.load_checkpoint(f'model/{second_path}/model.ckpt')
            ms.load_param_into_net(self.dnn, params_dict)

        self.grad = Grad(self.dnn)
        self.hessian_u_xx = SecondOrderGrad(self.dnn, 0, 0, output_idx=0)
        self.hessian_u_yy = SecondOrderGrad(self.dnn, 1, 1, output_idx=0)
        self.hessian_u_zz = SecondOrderGrad(self.dnn, 2, 2, output_idx=0)

        self.hessian_v_xx = SecondOrderGrad(self.dnn, 0, 0, output_idx=1)
        self.hessian_v_yy = SecondOrderGrad(self.dnn, 1, 1, output_idx=1)
        self.hessian_v_zz = SecondOrderGrad(self.dnn, 2, 2, output_idx=1)

        self.hessian_w_xx = SecondOrderGrad(self.dnn, 0, 0, output_idx=2)
        self.hessian_w_yy = SecondOrderGrad(self.dnn, 1, 1, output_idx=2)
        self.hessian_w_zz = SecondOrderGrad(self.dnn, 2, 2, output_idx=2)

        self.loss_scaler = DynamicLossScaler(1024, 2, 100)

    def net_u(self, x, y, z, t):
        H = ms.ops.concat([x, y, z, t], 1)
        Y = self.dnn(H)
        return Y

    def net_f(self, x_f, y_f, z_f, t_f):
        """ The minspore autograd version of calculating residual """
        U = self.net_u(x_f, y_f, z_f, t_f)
        u = U[:, 0:1]
        v = U[:, 1:2]
        w = U[:, 2:3]
        p = U[:, 3:4]

        data = ms.ops.concat([x_f, y_f, z_f, t_f], 1)

        u_x = self.grad(data, 0, 0, U)
        u_y = self.grad(data, 1, 0, U)
        u_z = self.grad(data, 2, 0, U)
        u_t = self.grad(data, 3, 0, U)
        u_xx = self.hessian_u_xx(data)
        u_yy = self.hessian_u_yy(data)
        u_zz = self.hessian_u_zz(data)

        v_x = self.grad(data, 0, 1, U)
        v_y = self.grad(data, 1, 1, U)
        v_z = self.grad(data, 2, 1, U)
        v_t = self.grad(data, 3, 1, U)
        v_xx = self.hessian_v_xx(data)
        v_yy = self.hessian_v_yy(data)
        v_zz = self.hessian_v_zz(data)

        w_x = self.grad(data, 0, 2, U)
        w_y = self.grad(data, 1, 2, U)
        w_z = self.grad(data, 2, 2, U)
        w_t = self.grad(data, 3, 2, U)
        w_xx = self.hessian_w_xx(data)
        w_yy = self.hessian_w_yy(data)
        w_zz = self.hessian_w_zz(data)

        p_x = self.grad(data, 0, 3, U)
        p_y = self.grad(data, 1, 3, U)
        p_z = self.grad(data, 2, 3, U)

        f_u = u_t + (u * u_x + v * u_y + w * u_z) + p_x - 1. / self.Re * (u_xx + u_yy + u_zz)
        f_v = v_t + (u * v_x + v * v_y + w * v_z) + p_y - 1. / self.Re * (v_xx + v_yy + v_zz)
        f_w = w_t + (u * w_x + v * w_y + w * w_z) + p_z - 1. / self.Re * (w_xx + w_yy + w_zz)

        f_e = u_x + v_y + w_z

        return u, v, w, p, f_u, f_v, f_w, f_e

    def loss_fn(self, xb, yb, zb, tb, xi, yi, zi, ti, x_f, y_f, z_f, t_f, ub, vb, wb, ui, vi, wi):
        """
        loss_fn
        """
        Ub = self.net_u(xb, yb, zb, tb)
        ub_pred = Ub[:, 0:1]
        vb_pred = Ub[:, 1:2]
        wb_pred = Ub[:, 2:3]

        Ui = self.net_u(xi, yi, zi, ti)
        ui_pred = Ui[:, 0:1]
        vi_pred = Ui[:, 1:2]
        wi_pred = Ui[:, 2:3]

        _, _, _, _, f_u_pred, f_v_pred, f_w_pred, f_e_pred = self.net_f(x_f, y_f, z_f, t_f)

        loss_ub = ops.reduce_mean(ops.square(ub_pred - ub))
        loss_vb = ops.reduce_mean(ops.square(vb_pred - vb))
        loss_wb = ops.reduce_mean(ops.square(wb_pred - wb))

        loss_ui = ops.reduce_mean(ops.square(ui_pred - ui))
        loss_vi = ops.reduce_mean(ops.square(vi_pred - vi))
        loss_wi = ops.reduce_mean(ops.square(wi_pred - wi))

        alpha = ms.Tensor(100., dtype=self.msfloat_type)

        loss_f_u = ops.reduce_mean(ops.square(f_u_pred - ms.ops.zeros_like(f_u_pred)))
        loss_f_v = ops.reduce_mean(ops.square(f_v_pred - ms.ops.zeros_like(f_v_pred)))
        loss_f_w = ops.reduce_mean(ops.square(f_w_pred - ms.ops.zeros_like(f_w_pred)))
        loss_f_e = ops.reduce_mean(ops.square(f_e_pred - ms.ops.zeros_like(f_e_pred)))

        loss_b = loss_ub + loss_vb + loss_wb
        loss_i = loss_ui + loss_vi + loss_wi
        loss_f = loss_f_u + loss_f_v + loss_f_w + loss_f_e

        loss = alpha * loss_b + alpha * loss_i + loss_f
        if self.use_npu:
            loss = self.loss_scaler.scale(loss)
        return loss, loss_b, loss_i, loss_f
