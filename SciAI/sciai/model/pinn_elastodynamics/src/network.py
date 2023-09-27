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
import pickle

import mindspore as ms
from mindspore import nn, ops
from mindspore.common.initializer import XavierNormal
from sciai.architecture import MSE
from sciai.operators import grad
from sciai.utils import print_log


class DeepElasticWave(nn.Cell):
    """Loss for elastic wave"""
    def __init__(self, uv_layers, dtype, uv_path=''):
        super().__init__()

        # Mat. properties
        self.e = 2.5
        self.mu = 0.25
        self.rho = 1.0

        print_log("Loading uv NN ...")
        nn_uv = Net(uv_path, uv_layers, dtype)
        self.net_uv = NetUV(nn_uv)
        self.nn_e = NetGrad(self.net_uv)

        self.mse = MSE()

    def construct(self, *args):
        """Network forward pass"""
        x_c, y_c, t_c, x_src, y_src, t_src, u_src, v_src, x_ic, y_ic, t_ic, x_fix, y_fix, t_fix = args
        u_ic_pred, v_ic_pred, ut_ic_pred, vt_ic_pred, _, _, _ = self.net_uv(x_ic, y_ic, t_ic)
        u_fix_pred, v_fix_pred, _, _, _, _, _ = self.net_uv(x_fix, y_fix, t_fix)
        u_src_pred, v_src_pred, _, _, _, _, _ = self.net_uv(x_src, y_src, t_src)

        f_pred_u, f_pred_v, f_pred_ut, f_pred_vt, f_pred_s11, f_pred_s22, f_pred_s12 = self.net_f_sig(x_c, y_c, t_c)

        # Not included in training, just for evaluation
        loss_src = self.mse(u_src_pred - u_src) \
                   + self.mse(v_src_pred - v_src)
        loss_ic = self.mse(u_ic_pred) \
                  + self.mse(v_ic_pred) \
                  + self.mse(ut_ic_pred) \
                  + self.mse(vt_ic_pred)
        loss_fix = self.mse(u_fix_pred) \
                   + self.mse(v_fix_pred)

        loss_f_uv = self.mse(f_pred_u) \
                    + self.mse(f_pred_v) \
                    + self.mse(f_pred_ut) \
                    + self.mse(f_pred_vt)
        loss_f_s = self.mse(f_pred_s11) \
                   + self.mse(f_pred_s22) \
                   + self.mse(f_pred_s12)

        loss = 5 * loss_f_uv + 5 * loss_f_s + loss_src + loss_ic + loss_fix
        return loss, loss_f_uv, loss_f_s, loss_src, loss_ic, loss_fix

    def net_f_sig(self, x, y, t):
        """calculations for f"""
        e, mu, rho = self.e, self.mu, self.rho

        _, _, ut, vt, s11, s22, s12 = self.net_uv(x, y, t)
        # Strains
        (u_x, u_y, u_t), (v_x, v_y, v_t), (_, _, ut_t), (_, _, vt_t), (s11_x, _, _), (
            _, s22_y, _), (s12_x, s12_y, _) = self.nn_e(x, y, t)

        # Strains
        e11, e22, e12 = u_x, v_y, u_y + v_x

        # Plane stress problem
        sp11 = e / (1 - mu * mu) * e11 + e * mu / (1 - mu * mu) * e22
        sp22 = e * mu / (1 - mu * mu) * e11 + e / (1 - mu * mu) * e22
        sp12 = e / (2 * (1 + mu)) * e12

        # Plane strain problem
        coef = e / ((1 + mu) * (1 - 2 * mu))
        sp11 = coef * (1 - mu) * e11 + coef * mu * e22
        sp22 = coef * mu * e11 + coef * (1 - mu) * e22
        sp12 = e / (2 * (1 + mu)) * e12

        # Cauchy stress
        f_s11, f_s12, f_s22 = s11 - sp11, s12 - sp12, s22 - sp22

        f_ut, f_vt = u_t - ut, v_t - vt

        # f_u: Sxx_x+Sxy_y-rho*u_tt
        f_u = s11_x + s12_y - rho * ut_t
        f_v = s22_y + s12_x - rho * vt_t

        return f_u, f_v, f_ut, f_vt, f_s11, f_s22, f_s12

    def predict(self, x_star, y_star, t_star):
        grad_u, grad_v, _, _, _, _, _ = self.nn_e(x_star, y_star, t_star)
        e11_star, e22_star, e12_star = grad_u[0], grad_v[1], grad_u[1] + grad_v[0]
        u_star, v_star, _, _, s11_star, s22_star, s12_star = self.net_uv(x_star, y_star, t_star)
        return u_star.asnumpy(), v_star.asnumpy(), s11_star.asnumpy(), s22_star.asnumpy(), s12_star.asnumpy(), \
               e11_star.asnumpy(), e22_star.asnumpy(), e12_star.asnumpy()


class NetGrad(nn.Cell):
    """Grad network"""
    def __init__(self, net_uv):
        super().__init__()
        self.grad_net_u = grad(net_uv, output_index=0, input_index=-1)
        self.grad_net_v = grad(net_uv, output_index=1, input_index=-1)
        self.grad_net_ut = grad(net_uv, output_index=2, input_index=-1)
        self.grad_net_vt = grad(net_uv, output_index=3, input_index=-1)
        self.grad_net_s11 = grad(net_uv, output_index=4, input_index=-1)
        self.grad_net_s22 = grad(net_uv, output_index=5, input_index=-1)
        self.grad_net_s12 = grad(net_uv, output_index=6, input_index=-1)

    def construct(self, x, y, t):
        """Network forward pass"""
        u_x, u_y, u_t = self.grad_net_u(x, y, t)
        v_x, v_y, v_t = self.grad_net_v(x, y, t)
        ut_x, ut_y, ut_t = self.grad_net_ut(x, y, t)
        vt_x, vt_y, vt_t = self.grad_net_vt(x, y, t)
        s11_x, s11_y, s11_t = self.grad_net_s11(x, y, t)
        s22_x, s22_y, s22_t = self.grad_net_s22(x, y, t)
        s12_x, s12_y, s12_t = self.grad_net_s12(x, y, t)
        return (u_x, u_y, u_t), (v_x, v_y, v_t), (ut_x, ut_y, ut_t), (vt_x, vt_y, vt_t), (s11_x, s11_y, s11_t), (
            s22_x, s22_y, s22_t), (s12_x, s12_y, s12_t)


class Net(nn.Cell):
    """Network"""
    def __init__(self, pickle_path, layers, dtype):
        super().__init__()
        self.cell_list = nn.SequentialCell()
        if pickle_path:
            with open(pickle_path, 'rb') as f:
                uv_weights, uv_biases = pickle.load(f)
                # Stored model must have the same # of layers
                num_layers = len(layers)
                if num_layers != (len(uv_weights) + 1):
                    raise ValueError("number of layers are not compatible with pickle data")

                for num, (in_channel, out_channel) in enumerate(zip(layers[:-1], layers[1:])):
                    weight = ms.Tensor(uv_weights[num], dtype=dtype).transpose()
                    bias = ms.Tensor(uv_biases[num], dtype=dtype).squeeze()
                    activation = "tanh" if num < len(layers) - 2 else None
                    layer = nn.Dense(in_channels=in_channel, out_channels=out_channel,
                                     weight_init=weight, has_bias=True, bias_init=bias, activation=activation)
                    self.cell_list.append(layer)
        else:
            for num, (in_channel, out_channel) in enumerate(zip(layers[:-1], layers[1:])):
                activation = "tanh" if num < len(layers) - 2 else None
                layer = nn.Dense(in_channels=in_channel, out_channels=out_channel, weight_init=XavierNormal(),
                                 has_bias=True, activation=activation)
                self.cell_list.append(layer)

    def construct(self, x):
        """Network forward pass"""
        out = self.cell_list(x)
        return out


class NetUV(nn.Cell):
    """UV network"""
    def __init__(self, net):
        super().__init__()
        self.nn_uv = net

    def construct(self, x, y, t):
        """Network forward pass"""
        # This NN return sigma_phi
        inputs = ops.concat([x, y, t], axis=1)
        uv_sig = self.nn_uv(inputs)
        u, v, ut, vt, s11, s22, s12 = ops.split(uv_sig, split_size_or_sections=1, axis=1)
        return u, v, ut, vt, s11, s22, s12
