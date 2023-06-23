# ============================================================================
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
"""Define the Poisson equation."""
import sympy
from mindspore import numpy as ms_np
from mindflow import PDEWithLoss, MTLWeightedLoss, sympy_to_mindspore


class Poisson(PDEWithLoss):
    """Define the loss of the Poisson equation."""

    def __init__(self, model):
        self.x, self.y = sympy.symbols("x y")
        self.u = sympy.Function("u")(self.x, self.y)
        self.in_vars = [self.x, self.y]
        self.out_vars = [self.u,]
        self.alpha = 0.01  # kernel width
        super(Poisson, self).__init__(model, self.in_vars, self.out_vars)
        self.bc_nodes = sympy_to_mindspore(self.bc(), self.in_vars, self.out_vars)
        self.loss_fn = MTLWeightedLoss(num_losses=3)

    def pde(self):
        """Define the gonvering equation."""
        uu_xx = sympy.diff(self.u, (self.x, 2))
        uu_yy = sympy.diff(self.u, (self.y, 2))

        # Use Laplace probability density function to approximate the Dirac \delta function.
        x_src = sympy.pi / 2
        y_src = sympy.pi / 2
        force_term = 0.25 / self.alpha**2 * sympy.exp(-(
            sympy.Abs(self.x - x_src) + sympy.Abs(self.y - y_src)) / self.alpha)

        poisson = uu_xx + uu_yy + force_term
        equations = {"poisson": poisson}
        return equations

    def bc(self):
        """Define the boundary condition."""
        bc_eq = self.u

        equations = {"bc": bc_eq}
        return equations

    def get_loss(self, pde_data, bc_data, src_data):
        """Define the loss function."""
        res_pde = self.parse_node(self.pde_nodes, inputs=pde_data)
        res_bc = self.parse_node(self.bc_nodes, inputs=bc_data)
        res_src = self.parse_node(self.pde_nodes, inputs=src_data)

        loss_pde = ms_np.mean(ms_np.square(res_pde[0]))
        loss_bc = ms_np.mean(ms_np.square(res_bc[0]))
        loss_src = ms_np.mean(ms_np.square(res_src[0]))

        return self.loss_fn((loss_pde, loss_bc, loss_src))
