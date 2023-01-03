# Copyright 2022 Huawei Technologies Co., Ltd
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
"""Navier-Stokes 2D"""
import numpy as np

from mindspore import nn, ops, Tensor
from mindspore import dtype as mstype

from mindflow.pde import NavierStokes, sympy_to_mindspore


class NavierStokes2D(NavierStokes):
    r"""
    2D NavierStokes equation problem based on PDEWithLoss.

    Args:
        model (mindspore.nn.Cell): Network for training.
        re (float): Reynolds number is the ratio of inertia force to viscous force of a fluid. it is a dimensionless
            quantity. Default: 100.0.
        loss_fn (Union[None, mindspore.nn.Cell]): Define the loss function. If None, the `network` should have the loss
            inside. Default: mindspore.nn.MSELoss.

    Supported Platforms:
        ``Ascend`` ``GPU``
    """
    def __init__(self, model, re=100, loss_fn=nn.MSELoss()):
        super(NavierStokes2D, self).__init__(model, re=re, loss_fn=loss_fn)
        self.ic_nodes = sympy_to_mindspore(self.ic(), self.in_vars, self.out_vars)
        self.bc_nodes = sympy_to_mindspore(self.bc(), self.in_vars, self.out_vars)

    def ic(self):
        """
        Define initial condition equations based on sympy, abstract method.
        """
        ic_u = self.u
        ic_v = self.v
        equations = {"ic_u": ic_u, "ic_v": ic_v}
        return equations

    def bc(self):
        """
        Define boundary condition equations based on sympy, abstract method.
        """
        bc_u = self.u
        bc_v = self.v
        bc_p = self.p
        equations = {"bc_u": bc_u, "bc_v": bc_v, "bc_p": bc_p}
        return equations

    def get_loss(self, pde_data, ic_data, ic_label, bc_data, bc_label):
        """
        Compute loss of 3 parts: governing equation, initial condition and boundary conditions.

        Args:
            pde_data (Tensor): the input data of governing equations.
            ic_data (Tensor): the input data of initial condition.
            ic_label (Tensor): the true value of initial state.
            bc_data (Tensor): the input data of boundary condition.
            bc_label (Tensor): the true value at boundary.
        """
        pde_res = self.parse_node(self.pde_nodes, inputs=pde_data)
        pde_residual = ops.Concat(1)(pde_res)
        pde_loss = self.loss_fn(pde_residual, Tensor(np.array([0.0]).astype(np.float32), mstype.float32))

        ic_res = self.parse_node(self.ic_nodes, inputs=ic_data)
        ic_residual = ops.Concat(1)(ic_res)
        ic_loss = self.loss_fn(ic_residual, ic_label)

        bc_res = self.parse_node(self.bc_nodes, inputs=bc_data)
        bc_residual = ops.Concat(1)(bc_res)
        bc_loss = self.loss_fn(bc_residual, bc_label)

        return pde_loss + ic_loss + bc_loss
