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
"""Poisson 2D"""
import numpy as np
import sympy

from mindspore import nn, Tensor
from mindspore import dtype as mstype

from mindflow.pde import Poisson, sympy_to_mindspore


class Poisson2D(Poisson):
    r"""
    Poisson 2-D problem based on Poisson.

    Args:
        model (mindspore.nn.Cell): Network for training.
        loss_fn (Union[None, mindspore.nn.Cell]): Define the loss function. If None, the `model` should have the loss
            inside. Default: mindspore.nn.MSELoss.

    Supported Platforms:
        ``Ascend`` ``GPU``
    """

    def __init__(self, model, loss_fn=nn.MSELoss()):
        super(Poisson2D, self).__init__(model, loss_fn=loss_fn)
        self.bc_outer_nodes = sympy_to_mindspore(self.bc_outer(), self.in_vars, self.out_vars)
        self.bc_inner_nodes = sympy_to_mindspore(self.bc_inner(), self.in_vars, self.out_vars)

    def bc_outer(self):
        """
        Define Dirichlet boundary condition equations based on sympy, abstract method.
        """
        bc_outer_eq = self.u
        equations = {"bc_outer": bc_outer_eq}
        return equations

    def bc_inner(self):
        """
        Define Neumann boundary condition equations based on sympy, abstract method.
        """
        bc_inner_eq = sympy.Derivative(self.u, self.normal) - 0.5
        equations = {"bc_inner": bc_inner_eq}
        return equations

    def get_loss(self, pde_data, bc_outer_data, bc_inner_data, bc_inner_normal):
        """
        Compute loss of 3 parts: governing equation, inner boundary condition and outer boundary conditions.

        Args:
            pde_data (Tensor): the input data of governing equations.
            bc_outer_data (Tensor): the input data of Dirichlet boundary condition.
            bc_inner_data (Tensor): the input data of Neumann boundary condition.
            bc_inner_normal (Tensor): the normal of the surface at a Neumann boundary point P is a vector perpendicular
                to the tangent plane of the point.

        Supported Platforms:
            ``Ascend`` ``GPU``
        """
        pde_res = self.parse_node(self.pde_nodes, inputs=pde_data)
        pde_loss = self.loss_fn(pde_res[0], Tensor(np.array([0.0]), mstype.float32))

        bc_inner_res = self.parse_node(self.bc_inner_nodes, inputs=bc_inner_data, norm=bc_inner_normal)
        bc_inner_loss = self.loss_fn(bc_inner_res[0], Tensor(np.array([0.0]), mstype.float32))

        bc_outer_res = self.parse_node(self.bc_outer_nodes, inputs=bc_outer_data)
        bc_outer_loss = self.loss_fn(bc_outer_res[0], Tensor(np.array([0.0]), mstype.float32))

        return pde_loss + bc_inner_loss + bc_outer_loss
