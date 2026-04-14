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

"""HP-VPINNs network"""
import numpy as np
import mindspore as ms
from mindspore import ops, nn
from sciai.architecture import MLP, MSE
from sciai.common.initializer import XavierTruncNormal
from scipy.special import gamma, jacobi, roots_jacobi


class VPINN(nn.Cell):
    """VPINN"""
    def __init__(self, x_quad, w_quad, f_exact_total, grid, args, dtype):
        super().__init__()
        self.dtype = dtype
        self.xquad = ms.Tensor(x_quad, dtype=self.dtype)
        self.xquad_np = x_quad
        self.wquad = ms.Tensor(w_quad, dtype=self.dtype)
        self.f_ext_total = f_exact_total
        self.nelement = np.shape(self.f_ext_total)[0]
        self.grid = ms.Tensor(grid.astype(np.float), dtype=self.dtype)
        self.grid_np = grid.astype(np.float)

        self.net_u = MLP(args.layers, weight_init=XavierTruncNormal(), bias_init="zeros", activation=ops.Sin())
        self.net_u_grad = ops.grad(self.net_u)
        self.net_u_grad_second = ops.grad(self.net_u_grad)
        self.var_form = args.var_form
        self.lossb_weight = args.lossb_weight
        self.mse = MSE()

    def construct(self, x_tf, u_tf):
        """Network forward pass"""
        u_nn_pred = self.net_u(x_tf)
        varloss_total = 0
        for e in range(self.nelement):
            f_ext_element = self.f_ext_total[e]
            ntest_element = np.shape(f_ext_element)[0]
            x_quad_element = self.grid[e] + (self.grid[e + 1] - self.grid[e]) / 2 * (self.xquad + 1)
            x_b_element = ms.Tensor(np.array([[self.grid_np[e]], [self.grid_np[e + 1]]]), dtype=self.dtype)
            jacobian = (self.grid[e + 1] - self.grid[e]) / 2
            test_quad_element = self.test_fcn(ntest_element, self.xquad_np)
            d1test_quad_element, d2test_quad_element = self.d_test_fcn(ntest_element, self.xquad_np)
            u_nn_quad_element = self.net_u(x_quad_element)
            d1u_nn_quad_element = self.net_u_grad(x_quad_element)
            d2u_nn_quad_element = self.net_u_grad_second(x_quad_element)
            u_nn_bound_element = self.net_u(x_b_element)
            d1test_bound_element, _ = self.d_test_fcn(ntest_element, np.array([[-1], [1]], dtype=np.float))
            concat_list = []
            if self.var_form == 1:
                for i in range(ntest_element):
                    reduced = ops.reduce_sum(
                        self.wquad * d2u_nn_quad_element * ms.Tensor(test_quad_element[i], dtype=self.dtype))
                    res = -jacobian * reduced.expand_dims(axis=0)
                    concat_list.append(res)
            elif self.var_form == 2:
                for i in range(ntest_element):
                    reduced = ops.reduce_sum(self.wquad * d1u_nn_quad_element * d1test_quad_element[i])
                    concat_list.append(reduced)
            elif self.var_form == 3:
                for i in range(ntest_element):
                    reduced1 = -ops.reduce_sum(self.wquad * u_nn_quad_element * d2test_quad_element[i])
                    reduced2 = ops.reduce_sum(
                        u_nn_bound_element * np.array([-d1test_bound_element[i][0], d1test_bound_element[i][-1]]))
                    concat_list.append((reduced1 + reduced2) / jacobian)
            u_nn_element = ops.reshape(ops.concat(concat_list), (-1, 1))
            res_nn_element = u_nn_element - f_ext_element
            loss_element = ops.reduce_mean(ops.square(res_nn_element))
            varloss_total = varloss_total + loss_element

        lossb = self.mse(u_tf - u_nn_pred)
        lossv = varloss_total
        loss = self.lossb_weight * lossb + lossv
        return loss, lossb, lossv

    def test_fcn(self, n_test, x):
        test_total = []
        for n in range(1, n_test + 1):
            test = jacobi_poly(n + 1, 0, 0, x) - jacobi_poly(n - 1, 0, 0, x)
            test_total.append(test)
        return np.asarray(test_total)

    def d_test_fcn(self, n_test, x):
        """d test function"""
        d1test_total = []
        d2test_total = []
        for n in range(1, n_test + 1):
            if n == 1:
                d1test = ((n + 2) / 2) * jacobi_poly(n, 1, 1, x)
                d2test = ((n + 2) * (n + 3) / (2 * 2)) * jacobi_poly(n - 1, 2, 2, x)
            elif n == 2:
                d1test = ((n + 2) / 2) * jacobi_poly(n, 1, 1, x) - (n / 2) * jacobi_poly(n - 2, 1, 1, x)
                d2test = ((n + 2) * (n + 3) / (2 * 2)) * jacobi_poly(n - 1, 2, 2, x)
            else:
                d1test = ((n + 2) / 2) * jacobi_poly(n, 1, 1, x) - (n / 2) * jacobi_poly(n - 2, 1, 1, x)
                d2test = ((n + 2) * (n + 3) / (2 * 2)) * jacobi_poly(n - 1, 2, 2, x) \
                    - (n * (n + 1) / (2 * 2)) * jacobi_poly(n - 3, 2, 2, x)
            d1test_total.append(d1test)
            d2test_total.append(d2test)
        return np.asarray(d1test_total), np.asarray(d2test_total)


# Recursive generation of the Jacobi polynomial of order n
def jacobi_poly(n, a, b, x):
    return jacobi(n, a, b)(x)


# Weight coefficients
def gauss_lobatto_jacobi_weights(q: int, a, b):
    """gauss lobatto jacobi weights"""
    x = roots_jacobi(q - 2, a + 1, b + 1)[0]
    if a == 0 and b == 0:
        w = 2 / ((q - 1) * (q) * (jacobi_poly(q - 1, 0, 0, x) ** 2))
        wl = 2 / ((q - 1) * (q) * (jacobi_poly(q - 1, 0, 0, -1) ** 2))
        wr = 2 / ((q - 1) * (q) * (jacobi_poly(q - 1, 0, 0, 1) ** 2))
    else:
        w = 2 ** (a + b + 1) * gamma(a + q) * gamma(b + q) \
            / ((q - 1) * gamma(q) * gamma(a + b + q + 1) * (jacobi_poly(q - 1, a, b, x) ** 2))
        wl = (b + 1) * 2 ** (a + b + 1) * gamma(a + q) * gamma(b + q) \
            / ((q - 1) * gamma(q) * gamma(a + b + q + 1) * (jacobi_poly(q - 1, a, b, -1) ** 2))
        wr = (a + 1) * 2 ** (a + b + 1) * gamma(a + q) * gamma(b + q) \
            / ((q - 1) * gamma(q) * gamma(a + b + q + 1) * (jacobi_poly(q - 1, a, b, 1) ** 2))
    w = np.append(w, wr)
    w = np.append(wl, w)
    x = np.append(x, 1)
    x = np.append(-1, x)
    return [x, w]
