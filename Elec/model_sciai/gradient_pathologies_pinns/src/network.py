
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

"""Network architecture for gp_pinns"""
import numpy as np
import mindspore as ms
from mindspore import nn, ops
from sciai.architecture import MLP, MSE, MLPShortcut
from sciai.operators import grad
from sciai.utils import to_tensor, print_log


class HelmholtzEqn:
    """Helmholtz Equation"""

    def __init__(self, a1, a2, lam):
        self.a1 = a1
        self.a2 = a2
        self.lam = lam

    def u(self, x):
        return np.sin(self.a1 * np.pi * x[:, 0:1]) * np.sin(self.a2 * np.pi * x[:, 1:2])

    def u_xx(self, x):
        return - (self.a1 * np.pi) ** 2 * np.sin(self.a1 * np.pi * x[:, 0:1]) * np.sin(self.a2 * np.pi * x[:, 1:2])

    def u_yy(self, x):
        return - (self.a2 * np.pi) ** 2 * np.sin(self.a1 * np.pi * x[:, 0:1]) * np.sin(self.a2 * np.pi * x[:, 1:2])

    def f(self, x):
        """Forcing term"""
        return self.u_xx(x) + self.u_yy(x) + self.lam * self.u(x)


class NetU(nn.Cell):
    """Forward pass for u"""

    def __init__(self, layers, model):
        super(NetU, self).__init__()
        self.concat = ops.concat
        if model in ('M1', 'M2'):
            self.forward = MLP(layers, weight_init="XavierNormal", bias_init="zeros")
        if model in ('M3', 'M4'):
            self.forward = MLPShortcut(layers, weight_init="XavierNormal", bias_init="zeros", activation='tanh')

    def construct(self, x1, x2):
        """Network forward pass"""
        u = self.forward(self.concat([x1, x2], 1))
        return u


class NetR(nn.Cell):
    """Forward pass for residual"""

    def __init__(self, net_u, lam, sigma_x1, sigma_x2):
        super(NetR, self).__init__()
        self.net_u = net_u
        self.operator = Operator(self.net_u, lam, sigma_x1, sigma_x2)

    def construct(self, x1, x2):
        """Network forward pass"""
        u = self.net_u(x1, x2)
        residual = self.operator(u, x1, x2)
        return residual


class Operator(nn.Cell):
    """Operator Net"""
    def __init__(self, net_u, lam, sigma_x1, sigma_x2):
        super().__init__()
        self.net_u = net_u
        self.lam = lam
        self.sigma_x1 = sigma_x1
        self.sigma_x2 = sigma_x2
        self.u_x1 = grad(self.net_u, input_index=0)
        self.u_x2 = grad(self.net_u, input_index=1)
        self.u_x1_x1 = grad(self.u_x1, input_index=0)
        self.u_x2_x2 = grad(self.u_x2, input_index=1)

    def construct(self, u, x1, x2):
        """Network forward pass"""
        u_x1_x1 = self.u_x1_x1(x1, x2) / self.sigma_x1 / self.sigma_x1
        u_x2_x2 = self.u_x2_x2(x1, x2) / self.sigma_x2 / self.sigma_x2
        residual = u_x1_x1 + u_x2_x2 + self.lam * u
        return residual


class Helmholtz2D(nn.Cell):
    """Helmholtz2D Net"""

    def __init__(self, layers, res_sampler, lam, model, dtype):
        super().__init__()
        self.dtype = dtype

        # Normalization constants
        self.mu_x, self.sigma_x = res_sampler.normalization_constants(np.int32(1e5))

        # Helmoholtz constant
        self.lam = ms.Tensor(lam, dtype=self.dtype)

        # Model Type
        if model not in ('M1', 'M2', 'M3', 'M4'):
            raise ValueError('Wrong model type: should be one of M1, M2, M3, or M4')
        self.model = model

        self.layers = layers
        self.net_u = NetU(layers, model)
        self.net_r = NetR(self.net_u, lam, self.sigma_x[0], self.sigma_x[1])

        # Generate dicts for gradients storage
        self.dict_gradients_res_layers = self.generate_grad_dict(self.layers)
        self.dict_gradients_bcs_layers = self.generate_grad_dict(self.layers)

        self.mse = MSE()

        # Gradients Storage
        self.max_grad_res = ms.Tensor(0, dtype=self.dtype)
        self.mean_grad_bcs = ms.Tensor(0, dtype=self.dtype)

        self.grad_res_list = []
        self.grad_bcs_list = []
        self.max_grad_res_list = []
        self.mean_grad_bcs_list = []

    @staticmethod
    def generate_grad_dict(layers):
        """create dictionary to store gradients"""
        num = len(layers) - 1
        grad_dict = {}
        for i in range(num):
            grad_dict['layer_{}'.format(i + 1)] = []
        return grad_dict

    def construct(self, *inputs):
        """Network forward pass"""
        x1_bc1, x2_bc1, u_bc1, x1_bc2, x2_bc2, u_bc2, x1_bc3, x2_bc3, u_bc3, x1_bc4, x2_bc4, u_bc4, x1_r, x2_r, r, \
        adaptive_const = inputs

        # Evaluate predictions
        u_bc1_pred = self.net_u(x1_bc1, x2_bc1)
        u_bc2_pred = self.net_u(x1_bc2, x2_bc2)
        u_bc3_pred = self.net_u(x1_bc3, x2_bc3)
        u_bc4_pred = self.net_u(x1_bc4, x2_bc4)

        # Residual loss
        loss_res = self.mse(r - self.net_r(x1_r, x2_r))

        # Boundary loss
        loss_bc1 = self.mse(u_bc1 - u_bc1_pred)
        loss_bc2 = self.mse(u_bc2 - u_bc2_pred)
        loss_bc3 = self.mse(u_bc3 - u_bc3_pred)
        loss_bc4 = self.mse(u_bc4 - u_bc4_pred)
        loss_bcs = adaptive_const * (loss_bc1 + loss_bc2 + loss_bc3 + loss_bc4)

        return loss_res, loss_bcs

    def save_gradients(self):
        """save gradients of loss_res and loss_bcs"""
        num_layers = len(self.layers)
        for i in range(num_layers - 1):
            self.dict_gradients_res_layers.get('layer_' + str(i + 1)).append(self.grad_res_list[i].flatten())
            self.dict_gradients_bcs_layers.get('layer_' + str(i + 1)).append(self.grad_bcs_list[i].flatten())

    def clear_grad_list(self):
        """clear gradient list"""
        self.grad_res_list = []
        self.grad_bcs_list = []
        self.max_grad_res_list = []
        self.mean_grad_bcs_list = []

    def update_grad_list(self, grad_res, grad_bcs):
        """update gradient list"""
        self.clear_grad_list()

        for gradient in grad_res:
            self.grad_res_list.append(gradient)
            self.max_grad_res_list.append(ops.max(ops.abs(gradient))[0])
        for gradient in grad_bcs:
            self.grad_bcs_list.append(gradient)
            self.mean_grad_bcs_list.append(ops.mean(ops.abs(gradient)))

        if len(self.max_grad_res_list) >= 1:
            res = ops.max(ops.stack(self.max_grad_res_list))[0]
            self.max_grad_res = res
        if len(self.mean_grad_bcs_list) >= 1:
            bcs = ops.mean(ops.stack(self.mean_grad_bcs_list))
            self.mean_grad_bcs = bcs

    def predict_u(self, x_star):
        """evaluate u at test points"""
        x_star = (x_star - self.mu_x) / self.sigma_x
        x1_u, x2_u = to_tensor(x_star[:, 0:1]), to_tensor(x_star[:, 1:2])
        u_pred = self.net_u(x1_u, x2_u)
        return u_pred

    def predict_r(self, x_star):
        """evaluate r at test points"""
        x_star = (x_star - self.mu_x) / self.sigma_x
        x1_r, x2_r = to_tensor(x_star[:, 0:1]), to_tensor(x_star[:, 1:2])
        r_pred = self.net_r(x1_r, x2_r)
        return r_pred

    def evaluate(self, helm, x_star):
        """evaluate model at x_star"""
        # Exact solution
        u_star = helm.u(x_star)
        f_star = helm.f(x_star)
        # Predictions
        u_pred = self.predict_u(x_star).asnumpy()
        f_pred = self.predict_r(x_star).asnumpy()
        # Relative error
        error_u = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
        error_f = np.linalg.norm(f_star - f_pred, 2) / np.linalg.norm(f_star, 2)
        print_log('Relative L2 error_u: {:.2e}'.format(error_u))
        print_log('Relative L2 error_f: {:.2e}'.format(error_f))
        return u_pred, u_star


class TrainOneStep(nn.Cell):
    """TrainOneStep class"""
    def __init__(self, network, optimizer):
        super().__init__()
        self.network = network
        self.optimizer = optimizer
        self.params = self.optimizer.parameters
        self.grad = ops.GradOperation(get_by_list=True)(self.network, self.params)

        if self.network.model in ('M1', 'M2'):
            self.weights = []
            for name, param in network.parameters_dict().items():
                if name.endswith('weight'):
                    self.weights.append(param)

        elif self.network.model in ('M3', 'M4'):
            self.weights = network.net_u.forward.main_weights()

        self._grad = ops.GradOperation(get_by_list=True, sens_param=True)(self.network, self.weights)

    def construct(self, *inputs):
        """Network forward pass"""
        loss_res, loss_bcs = self.network(*inputs)
        zeros = ops.zeros_like(loss_res)
        ones = ops.ones_like(loss_res)

        grad_res = self._grad(*inputs, (ones, zeros))
        grad_bcs = self._grad(*inputs, (zeros, ones))

        grads = self.grad(*inputs)
        self.optimizer(grads)

        return loss_res, loss_bcs, grad_res, grad_bcs
