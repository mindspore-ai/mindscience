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
"""inverse Navier-Stokes 2D Problem"""
from sympy import diff, Function, symbols

from mindspore import ops
from mindspore import numpy as mnp

from mindflow.loss import get_loss_metric
from mindflow.pde import PDEWithLoss
from mindflow.pde import sympy_to_mindspore


class InvNavierStokes(PDEWithLoss):
    r"""
    2D inverse NavierStokes equation problem based on PDEWithLoss.

    Args:
        model (mindspore.nn.Cell): network for training.
        params(mindspore.Tensor): parameter needs training
        loss_fn (Union[str, Cell]): Define the loss function. Default: mse.

    Supported Platforms:
        ``Ascend`` ``GPU``
    """

    def __init__(self, model, params, loss_fn="mse"):

        self.params_val = params[-1]
        self.theta1, self.theta2 = symbols('theta1 theta2')
        self.x, self.y, self.t = symbols('x y t')
        self.u = Function('u')(self.x, self.y, self.t)
        self.v = Function('v')(self.x, self.y, self.t)
        self.p = Function('p')(self.x, self.y, self.t)

        self.in_vars = [self.x, self.y, self.t]
        self.out_vars = [self.u, self.v, self.p]
        self.params = [self.theta1, self.theta2]

        super(InvNavierStokes, self).__init__(model, self.in_vars, self.out_vars, self.params, self.params_val)
        self.data_nodes = sympy_to_mindspore(self.data_loss(), self.in_vars, self.out_vars)
        if isinstance(loss_fn, str):
            self.loss_fn = get_loss_metric(loss_fn)
        else:
            self.loss_fn = loss_fn

    def pde(self):
        """
        Define governing equations based on sympy, abstract method.

        Returns:
            dict, user defined sympy symbolic equations.
        """
        momentum_x = self.u.diff(self.t) + \
                     self.theta1 * (self.u * self.u.diff(self.x) + self.v * self.u.diff(self.y)) + \
                     self.p.diff(self.x) - \
                     self.theta2 * (diff(self.u, (self.x, 2)) + diff(self.u, (self.y, 2)))
        momentum_y = self.v.diff(self.t) + \
                     self.theta1 * (self.u * self.v.diff(self.x) + self.v * self.v.diff(self.y)) + \
                     self.p.diff(self.y) - \
                     self.theta2 * (diff(self.v, (self.x, 2)) + diff(self.v, (self.y, 2)))
        continuty = self.u.diff(self.x) + self.v.diff(self.y)

        equations = {"momentum_x": momentum_x, "momentum_y": momentum_y, "continuty": continuty}
        return equations

    def data_loss(self):
        """
        Define governing equations based on sympy, abstract method.

        Returns:
            dict, user defined sympy symbolic equations.
        """
        velocity_u = self.u
        velocity_v = self.v
        p = self.p
        equations = {"velocity_u": velocity_u, "velocity_v": velocity_v, "p": p}
        return equations

    def get_loss(self, pde_data, data, label):
        """
        loss contains 2 parts,pde parts and data loss.
        """
        pde_res = self.parse_node(self.pde_nodes, inputs=pde_data)
        pde_residual = ops.Concat(1)(pde_res)
        pde_loss = self.loss_fn(pde_residual, mnp.zeros_like(pde_residual))

        data_res = self.parse_node(self.data_nodes, inputs=data)
        data_residual = ops.Concat(1)(data_res)
        train_data_loss = self.loss_fn(data_residual, label)

        return pde_loss + train_data_loss
