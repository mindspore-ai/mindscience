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

"""Networks structure"""
import numpy as np
from mindspore import nn, ops
from mindspore.common.initializer import XavierUniform
from sciai.architecture import MLP
from sciai.operators.derivatives import grad
from sciai.utils import to_tensor
from scipy.special import gamma


class PDENN(nn.Cell):
    """Network definition for PDE"""
    def __init__(self, fnn):
        super().__init__()
        self.fnn = fnn
        self.dy = grad(self.fnn, output_index=0, input_index=(0, 1))
        self.dy_dxx = grad(self.dy, output_index=0, input_index=0)

    def construct(self, x, t):
        """Network forward pass"""
        _, dy_dt = self.dy(x, t)
        dy_dxx = self.dy_dxx(x, t)
        res = ops.sub(dy_dt, dy_dxx)
        sin_res = ops.sin(np.pi * x)
        res1 = ops.mul(ops.exp(-t), ops.sub(sin_res, ops.mul(np.pi ** 2, sin_res)))
        res = ops.add(res, res1)
        return res


class FNN(nn.Cell):
    def __init__(self, layers):
        super(FNN, self).__init__()
        self.mlp = MLP(layers, weight_init=XavierUniform(), bias_init="zeros", activation="tanh")

    def construct(self, x, t):
        """Network forward pass"""
        out = ops.concat([t, x], 1)
        out = self.mlp(out)
        return out


class FNNWithTransform(nn.Cell):
    """FNN with transform"""
    def __init__(self, layers):
        super(FNNWithTransform, self).__init__()
        self.mlp = MLP(layers, weight_init=XavierUniform(), bias_init="zeros", activation="tanh")

    def construct(self, x, t):
        """Network forward pass"""
        out = ops.concat([t, x], 1)
        out = self.mlp(out)
        out = x * (1 - x) * t * out + ops.pow(x, 3) * ops.pow((1 - x), 3)
        return out


class Net(nn.Cell):
    """Net"""
    def __init__(self, layers):
        super().__init__()
        self.fnn = FNN(layers)
        self.pdenn = PDENN(self.fnn)

    def construct(self, t_train, t_bc, x_train, x_bc):
        """Network forward pass"""
        y_bc_pred = self.fnn(x_bc, t_bc)
        y_pred = self.pdenn(x_train, t_train)
        return y_pred, y_bc_pred


class MyWithLossCell(nn.Cell):
    """Loss Cell"""
    def __init__(self, backbone, loss_fn):
        super(MyWithLossCell, self).__init__()
        self._backbone = backbone
        self._loss_fn = loss_fn

    def construct(self, *inputs):
        """Network forward pass"""
        t_train, t_bc, x_train, x_bc, y_bc, y_train = inputs
        y_pred, y_bc_pred = self._backbone(t_train, t_bc, x_train, x_bc)
        return ops.add(self._loss_fn(y_pred, y_train),
                       self._loss_fn(y_bc, y_bc_pred))


class MyWithLossCellFPDE(nn.Cell):
    """Loss Cell for Fractional PDE"""
    def __init__(self, fnn, criterion, num_bcs, alpha, dtype):
        super().__init__()
        self.dtype = dtype
        self.fnn = fnn
        self.alpha = alpha
        self.idx = int(np.cumsum([0] + num_bcs)[-1])
        self.dy = grad(self.fnn, output_index=0, input_index=1)
        self._loss_fn = criterion

        self.coeff4_ms = self._calc_coeff(4)
        self.coeff5_ms = self._calc_coeff(5)
        self.coeff6_ms = self._calc_coeff(6)
        self.coeff7_ms = self._calc_coeff(7)

    def construct(self, x, t, int_mat):
        """Network forward pass"""
        y = self.fnn(x, t)
        lhs = - ops.matmul(int_mat, y)
        dy_dt = self.dy(x, t)
        rhs = - dy_dt - ops.exp(-t) * (
            ops.pow(x, 3) * ops.pow((1 - x), 3)
            + self.coeff4_ms * (ops.pow(x, (3 - self.alpha)) + ops.pow((1 - x), (3 - self.alpha)))
            - 3 * self.coeff5_ms * (ops.pow(x, (4 - self.alpha)) + ops.pow((1 - x), (4 - self.alpha)))
            + 3 * self.coeff6_ms * (ops.pow(x, (5 - self.alpha)) + ops.pow((1 - x), (5 - self.alpha)))
            - self.coeff7_ms * (ops.pow(x, (6 - self.alpha)) + ops.pow((1 - x), (6 - self.alpha)))
        )
        res = (lhs - rhs)[self.idx:]
        return self._loss_fn(res, ops.zeros_like(res))

    def _calc_coeff(self, n):
        coeff = gamma(n) / gamma(n - self.alpha)
        return to_tensor(coeff, dtype=self.dtype)
