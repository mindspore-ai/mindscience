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
"""2d kovasznay problem with dirichlet boundary condition"""
import math

import sympy
from sympy import Function, diff, symbols
from mindspore import nn
from mindspore import numpy as ms_np
from mindspore import ops

from mindflow import PDEWithLoss, sympy_to_mindspore
from mindflow.loss import get_loss_metric


class Kovasznay(PDEWithLoss):
    """Define the loss of the Kovasznay equation."""

    def __init__(self, model, re=20, loss_fn=nn.MSELoss()):
        """Initialize."""
        self.re = re
        self.nu = 1 / self.re
        self.l = 1 / (2 * self.nu) - math.sqrt(
            1 / (4 * self.nu**2) + 4 * math.pi**2
        )
        self.x, self.y = symbols("x y")
        self.u = Function("u")(self.x, self.y)
        self.v = Function("v")(self.x, self.y)
        self.p = Function("p")(self.x, self.y)
        self.in_vars = [self.x, self.y]
        self.out_vars = [self.u, self.v, self.p]
        super(Kovasznay, self).__init__(model, self.in_vars, self.out_vars)
        self.bc_nodes = sympy_to_mindspore(self.bc(), self.in_vars, self.out_vars)
        if isinstance(loss_fn, str):
            self.loss_fn = get_loss_metric(loss_fn)
        else:
            self.loss_fn = loss_fn

    def pde(self):
        """Define the gonvering equation."""
        u, v, p = self.out_vars
        u_x = diff(u, self.x)
        u_y = diff(u, self.y)
        v_x = diff(v, self.x)
        v_y = diff(v, self.y)
        p_x = diff(p, self.x)
        p_y = diff(p, self.y)
        u_xx = diff(u_x, self.x)
        u_yy = diff(u_y, self.y)
        v_xx = diff(v_x, self.x)
        v_yy = diff(v_y, self.y)
        momentum_x = u * u_x + v * u_y + p_x - (1 / self.re) * (u_xx + u_yy)
        momentum_y = u * v_x + v * v_y + p_y - (1 / self.re) * (v_xx + v_yy)
        continuty = u_x + v_y
        equations = {
            "momentum_x": momentum_x,
            "momentum_y": momentum_y,
            "continuty": continuty,
        }
        return equations

    def u_func(self):
        """Define the analytical solution."""
        u = 1 - sympy.exp(self.l * self.x) * sympy.cos(2 * sympy.pi * self.y)
        return u

    def v_func(self):
        """Define the analytical solution."""
        v = (
            self.l
            / (2 * sympy.pi)
            * sympy.exp(self.l * self.x)
            * sympy.sin(2 * sympy.pi * self.y)
        )
        return v

    def p_func(self):
        """Define the analytical solution."""
        p = 1 / 2 * (1 - sympy.exp(2 * self.l * self.x))
        return p

    def bc(self):
        """Define the boundary condition."""
        bc_u = self.u - self.u_func()
        bc_v = self.v - self.v_func()
        bc_p = self.p - self.p_func()
        bcs = {"u": bc_u, "v": bc_v, "p": bc_p}
        return bcs

    def get_loss(self, pde_data, bc_data):
        """Define the loss function."""
        pde_res = self.parse_node(self.pde_nodes, inputs=pde_data)
        pde_residual = ops.Concat(axis=1)(pde_res)
        pde_loss = self.loss_fn(pde_residual, ms_np.zeros_like(pde_residual))
        bc_res = self.parse_node(self.bc_nodes, inputs=bc_data)
        bc_residual = ops.Concat(axis=1)(bc_res)
        bc_loss = self.loss_fn(bc_residual, ms_np.zeros_like(bc_residual))
        return pde_loss + bc_loss
