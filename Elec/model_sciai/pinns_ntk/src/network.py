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
"""pinns ntk network"""
from mindspore import nn, ops
from mindspore.common.initializer import Normal

from sciai.architecture import MSE, MLP
from sciai.common import LeCunNormal
from sciai.operators.jacobian_weights import JacobianWeights


class NTKNet:
    """NTKNet"""

    def __init__(self, net_u, net_r, shape_u, shape_r, dtype):
        super(NTKNet, self).__init__()
        self.net_u = net_u
        self.net_r = net_r
        self.jacobian_u = JacobianWeights(self.net_u, out_shape=shape_u, out_type=dtype)
        self.jacobian_r = JacobianWeights(self.net_r, out_shape=shape_r, out_type=dtype)

    def __call__(self, x_u_ntk_tf, x_r_ntk_tf):
        j_u = self.compute_jacobian(x_u_ntk_tf)
        j_r = self.compute_jacobian(x_r_ntk_tf)

        k_uu = self.compute_ntk(j_u, x_u_ntk_tf, j_u)
        k_ur = self.compute_ntk(j_u, x_u_ntk_tf, j_r)
        k_rr = self.compute_ntk(j_r, x_r_ntk_tf, j_r)
        return k_uu, k_ur, k_rr

    @staticmethod
    def compute_ntk(j1_list, x1, j2_list):
        """Compute the empirical NTK = J J^T"""
        d = x1.shape[0]
        n = len(j1_list)

        ker = ops.zeros((d, d))
        for k in range(n):
            j1 = ops.reshape(j1_list[k], (d, -1))
            j2 = ops.reshape(j2_list[k], (d, -1))

            k = ops.matmul(j1, j2.transpose())
            ker = ker + k
        return ker

    def compute_jacobian(self, ntk_pred):
        """Compute Jacobian for each weight and bias in each layer and return a list."""
        weights, biases = self.net_u.weights(), self.net_u.biases()
        weight_len = len(weights)
        bias_len = len(biases)
        j_list = []
        for i in range(weight_len):
            j_w = self.jacobian_u(ntk_pred, weights[i])
            j_list.append(j_w)

        for i in range(bias_len):
            j_b = self.jacobian_r(ntk_pred, biases[i])
            j_list.append(j_b)
        return j_list


class PINN(nn.Cell):
    """PINN"""

    def __init__(self, layers, x_u, y_u, x_r, y_r, dtype):
        super(PINN, self).__init__()
        self.mu_x, self.sigma_x = x_r.mean(0), x_r.std(0)
        self.mu_x, self.sigma_x = self.mu_x[0], self.sigma_x[0]
        # Normalize
        self.x_u = (x_u - self.mu_x) / self.sigma_x
        self.y_u = y_u
        self.x_r = (x_r - self.mu_x) / self.sigma_x
        self.y_r = y_r
        # Initialize network weights and biases
        self.net_u = MLP(layers, weight_init=LeCunNormal(), bias_init=Normal(sigma=1, mean=0), activation="tanh")
        self.grad = ops.GradOperation()
        self.net_grad_u_x = self.grad(self.net_u)
        self.net_grad_u_xx = self.grad(self.net_grad_u_x)
        self.ntks = NTKNet(self.net_u, self.net_r, self.x_u.shape, self.x_r.shape, dtype)
        self.loss_bcs_log = []
        self.loss_res_log = []
        # NTK logger
        self.k_uu_log = []
        self.k_rr_log = []
        self.k_ur_log = []
        # Weights logger
        self.weights_log = []
        self.biases_log = []
        self.mse = MSE()

    def construct(self, x_bc_tf, u_bc_tf, x_r_tf, r_tf):
        """Network forward pass"""
        # Evaluate predictions
        u_bc_pred = self.net_u(x_bc_tf)
        r_pred = self.net_r(x_r_tf)
        # Boundary loss
        loss_bcs = self.mse(u_bc_pred - u_bc_tf)
        # Residual loss
        loss_res = self.mse(r_tf - r_pred)
        # Total loss
        return loss_bcs, loss_res

    # Forward pass for the residual
    def net_r(self, x):
        u_xx = self.net_grad_u_xx(x) / self.sigma_x / self.sigma_x
        return u_xx

    def predict_u(self, x_star):
        x_star = (x_star - self.mu_x) / self.sigma_x
        u_star = self.net_u(x_star)
        return u_star.asnumpy()

    # Evaluates predictions at test points
    def predict_r(self, x_star):
        x_star = (x_star - self.mu_x) / self.sigma_x
        r_star = self.net_r(x_star)
        return r_star.asnumpy()
