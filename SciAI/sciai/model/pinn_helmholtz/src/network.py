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
"""Network architectures for PINN helmholtz"""
from mindspore import nn, ops

from sciai.architecture import MLP, Normalize
from sciai.common.initializer import StandardUniform
from sciai.operators.derivatives import grad


class LossCellHelmholtz(nn.Cell):
    """Loss net"""
    def __init__(self, backbone, model_param):
        super().__init__()
        a_real, a_imag, b_real, b_imag, c_real, c_imag, ps_real, ps_imag, m, omega = model_param
        self._backbone = backbone
        self.a_real = a_real
        self.a_imag = a_imag
        self.b_real = b_real
        self.b_imag = b_imag
        self.c_real = c_real
        self.c_imag = c_imag
        self.ps_real = ps_real
        self.ps_imag = ps_imag
        self.omega_square_m = omega * omega * m

        self.du_real_func = grad(self._backbone, 0, (0, 1))  # output real(du/dx) real(du/dz)
        self.du_imag_func = grad(self._backbone, 1, (0, 1))  # output imag(du/dx) imag(du/dz)

        self.ddu_real_func1 = grad(self.du_real_func, 0, (0, 1))  # output real(d2u/dx2) real(d2u/dxdz)
        self.ddu_real_func2 = grad(self.du_real_func, 1, (0, 1))  # output real(d2u/dzdx) real(d2u/dz2)
        self.ddu_imag_func1 = grad(self.du_imag_func, 0, (0, 1))  # output imag(d2u/dx2) imag(d2u/dxdz)
        self.ddu_imag_func2 = grad(self.du_imag_func, 1, (0, 1))  # output imag(d2u/dzdx) imag(d2u/dz2)

    def construct(self, x, z):
        """Network forward pass"""
        u_pred_real, u_pred_imag = self._backbone(x, z)

        ddu_dxx_real, _ = self.ddu_real_func1(x, z)
        _, ddu_dzz_real = self.ddu_real_func2(x, z)
        ddu_dxx_imag, _ = self.ddu_imag_func1(x, z)
        _, ddu_dzz_imag = self.ddu_imag_func2(x, z)

        da_dudxx_real = self.a_real * ddu_dxx_real - self.a_imag * ddu_dxx_imag
        da_dudxx_imag = self.a_real * ddu_dxx_imag + self.a_imag * ddu_dxx_real
        db_dudzz_real = self.b_real * ddu_dzz_real - self.b_imag * ddu_dzz_imag
        db_dudzz_imag = self.b_real * ddu_dzz_imag + self.b_imag * ddu_dzz_real

        loss_real = self.omega_square_m * (self.c_real * u_pred_real - self.c_imag * u_pred_imag) \
                    + da_dudxx_real + db_dudzz_real - self.ps_real

        loss_imag = self.omega_square_m * (self.c_real * u_pred_imag + self.c_imag * u_pred_real) \
                    + da_dudxx_imag + db_dudzz_imag - self.ps_imag

        loss = ops.mean(ops.square(loss_real) + ops.square(loss_imag))
        return loss


class PhysicsInformedNN(nn.Cell):
    """Physics-informed neural net"""
    def __init__(self, layers, bounds):
        super(PhysicsInformedNN, self).__init__()
        self.normalize = Normalize(bounds[0], bounds[1])
        self.mlp = MLP(layers, weight_init=StandardUniform(), bias_init="zeros", activation=ops.Sin())

    def construct(self, x, z):
        """Network forward pass"""
        pos = ops.concat((x, z), axis=1)
        pos = self.normalize(pos)
        y = self.mlp(pos)
        return y[:, 0:1], y[:, 1:2]
