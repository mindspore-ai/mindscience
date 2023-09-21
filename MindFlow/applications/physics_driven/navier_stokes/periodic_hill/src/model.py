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
"""Navier-Stokes-RANS"""
import numpy as np
from sympy import diff, Function, symbols

from mindspore import ops, Tensor
from mindspore import dtype as mstype
from mindflow.pde import PDEWithLoss
from mindflow.pde.sympy2mindspore import sympy_to_mindspore
from mindflow.loss import get_loss_metric


class NavierStokesRANS(PDEWithLoss):
    r"""
    Reynold-Averaged NavierStokes equation problem based on PDEWithLoss.

    Args:
        model (mindspore.nn.Cell): Network for training.
        re (float): Reynolds number is the ratio of inertia force to viscous force of a fluid. it is a dimensionless
            quantity. Default: 100.0.
        rho (float): Density of fluid. Default: 1.0.
        loss_fn (str): Define the loss function. Default: mse.

    Supported Platforms:
        ``Ascend`` ``GPU``
    """
    def __init__(self, model, re=5600, rho=1., loss_fn="mse"):
        self.vis = np.float32(1.0 / re)
        self.rho = np.float32(rho)
        self.x, self.y = symbols('x y')
        # u, v, p, uu, uv, vv, rho, nu
        self.u = Function('u')(self.x, self.y)
        self.v = Function('v')(self.x, self.y)
        self.p = Function('p')(self.x, self.y)
        self.uu = Function('uu')(self.x, self.y)
        self.uv = Function('uv')(self.x, self.y)
        self.vv = Function('vv')(self.x, self.y)
        self.in_vars = [self.x, self.y]
        self.out_vars = [self.u, self.v, self.p, self.uu, self.uv, self.vv]
        super(NavierStokesRANS, self).__init__(model, self.in_vars, self.out_vars)
        if isinstance(loss_fn, str):
            self.loss_fn = get_loss_metric(loss_fn)
        else:
            self.loss_fn = loss_fn
        self.bc_nodes = sympy_to_mindspore(self.bc(), self.in_vars, self.out_vars)

    def pde(self):
        """
        Define governing equations based on sympy, abstract method

        returns:
            dict, user defined sympy symbolic equations.
        """
        momentum_x = self.u * self.u.diff(self.x) + self.v * self.u.diff(self.y) + \
                     (1/self.rho) * self.p.diff(self.x) - self.vis * (diff(self.u, (self.x, 2)) + \
                     diff(self.u, (self.y, 2))) + diff(self.uu, self.x) + diff(self.uv, self.y)
        momentum_y = self.u * self.v.diff(self.x) + self.v * self.v.diff(self.y) + \
                     (1/self.rho) * self.p.diff(self.y) - self.vis * (diff(self.v, (self.x, 2)) + \
                     diff(self.v, (self.y, 2))) + diff(self.vv, self.y) + diff(self.uv, self.x)
        continuty = self.u.diff(self.x) + self.v.diff(self.y)

        equations = {"momentum_x": momentum_x, "momentum_y": momentum_y, "continuty": continuty}
        return equations

    def bc(self):
        """
        Define boundary condition equations based on sympy, abstract method.
        """
        bc_u = self.u
        bc_v = self.v
        bc_p = self.p
        bc_uu = self.uu
        bc_uv = self.uv
        bc_vv = self.vv
        equations = {"bc_u": bc_u, "bc_v": bc_v, "bc_p": bc_p, "bc_uu": bc_uu, "bc_uv": bc_uv, "bc_vv": bc_vv}
        return equations

    def get_loss(self, pde_data, bc_data, bc_label):
        """
        Compute loss of 3 parts: governing equation, initial condition and boundary conditions.

        Args:
            pde_data (Tensor): the input data of governing equations.
            bc_data (Tensor): the input data of boundary condition.
            bc_label (Tensor): the true value at boundary.
            ic_data (Tensor): the input data of initial condition.
            ic_label (Tensor): the true value of initial state.
        """
        pde_res = self.parse_node(self.pde_nodes, inputs=pde_data)
        pde_residual = ops.Concat(1)(pde_res)
        pde_loss = self.loss_fn(pde_residual, Tensor(np.array([0.0]).astype(np.float32), mstype.float32))


        bc_res = self.parse_node(self.bc_nodes, inputs=bc_data)
        bc_residual = ops.Concat(1)(bc_res)
        bc_loss = self.loss_fn(bc_residual, bc_label)

        return pde_loss + bc_loss
