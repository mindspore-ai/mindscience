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
"""Burgers1D"""
import numpy as np
import sympy

from mindspore import ops, Tensor, nn
from mindspore import dtype as mstype

from mindflow.pde import NavierStokes, Burgers, sympy_to_mindspore
from mindflow.pde import PDEWithLoss
from mindflow.loss import get_loss_metric
from mindflow.cell import MultiScaleFCSequential


class MoeMlp(nn.Cell):
    """
    The backbone network of MOE-PINNs

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        layers (int): The number of layers.
        neurons (int): The number of neurons in each layer.
        residual (bool): Whether to use residual connection.
        act (str): The activation function.
        num_scales (int): The number of scales.
        input_scale (float): The scale of input data.
        input_center (float): The center of input data.
        n_experts (int): The number of experts.
    """

    def __init__(self, in_channels, out_channels, layers, neurons, residual,
                 act, num_scales, input_scale, input_center, n_experts):
        super().__init__()
        self.n_experts = n_experts
        self.model = MultiScaleFCSequential(in_channels=in_channels,
                                            out_channels=neurons,
                                            layers=layers,
                                            neurons=neurons,
                                            residual=residual,
                                            act=act,
                                            num_scales=num_scales,
                                            input_scale=input_scale,
                                            input_center=input_center)
        self.experts = nn.CellList([nn.Dense(neurons, out_channels) for _ in range(n_experts)])
        self.gate = nn.Dense(neurons, n_experts)
        self.softmax = nn.Softmax(axis=-1)

    def construct(self, x):
        """
        Construct function of the network
        """
        x = self.model(x)
        gate = self.softmax(self.gate(x)).unsqueeze(-1)
        experts = ops.stack([expert(x) for expert in self.experts], -1)
        after_gate = ops.BatchMatMul()(experts, gate).squeeze(-1)
        return after_gate


class Burgers1D(Burgers):
    r"""
    Burgers 1-D problem based on PDEWithLoss

    Args:
        model (Cell): network for training.
        loss_fn (str): Define the loss function. Default: mse.

    Supported Platforms:
        ``Ascend`` ``GPU``

    """

    def __init__(self, model, loss_fn="mse"):
        super(Burgers1D, self).__init__(model, loss_fn=loss_fn)
        self.ic_nodes = sympy_to_mindspore(
            self.ic(), self.in_vars, self.out_vars)
        self.bc_nodes = sympy_to_mindspore(
            self.bc(), self.in_vars, self.out_vars)
        self.pde_loss_type = 0
        self.pde_ic_loss_type = 1
        self.ic_loss_type = 2
        self.bc_loss_type = 3
        self.ic_bc_loss_type = 4
        self.all_loss_type = 5

    def ic(self):
        """
        define initial condition equations based on sympy, abstract method.
        """
        ic_eq = self.u + sympy.sin(np.pi * self.x)
        equations = {"ic": ic_eq}
        return equations

    def bc(self):
        """
        define boundary condition equations based on sympy, abstract method.
        """
        bc_eq = self.u
        equations = {"bc": bc_eq}
        return equations

    def get_loss(self, pde_data, ic_data, bc_data, loss_type):
        """
        Compute loss of 3 parts: governing equation, initial condition and boundary conditions.

        Args:
            pde_data (Tensor): the input data of governing equations.
            ic_data (Tensor): the input data of initial condition.
            bc_data (Tensor): the input data of boundary condition.
            loss_type (int): the type of loss (pde-1, ic-2, bc-3, ic-bc-4, all-other)
        """
        if loss_type == self.pde_loss_type:
            pde_res = self.parse_node(self.pde_nodes, inputs=pde_data)
            pde_loss = self.loss_fn(pde_res[0], Tensor(
                np.array([0.0]), mstype.float32))
            result = pde_loss

        elif loss_type == self.pde_ic_loss_type:
            pde_res = self.parse_node(self.pde_nodes, inputs=pde_data)
            pde_loss = self.loss_fn(pde_res[0], Tensor(
                np.array([0.0]), mstype.float32))
            ic_res = self.parse_node(self.ic_nodes, inputs=ic_data)
            ic_loss = self.loss_fn(ic_res[0], Tensor(
                np.array([0.0]), mstype.float32))
            result = pde_loss + ic_loss

        elif loss_type == self.ic_loss_type:
            ic_res = self.parse_node(self.ic_nodes, inputs=ic_data)
            ic_loss = self.loss_fn(ic_res[0], Tensor(
                np.array([0.0]), mstype.float32))
            result = ic_loss

        elif loss_type == self.bc_loss_type:
            bc_res = self.parse_node(self.bc_nodes, inputs=bc_data)
            bc_loss = self.loss_fn(bc_res[0], Tensor(
                np.array([0.0]), mstype.float32))
            result = bc_loss

        elif loss_type == self.ic_bc_loss_type:
            ic_res = self.parse_node(self.ic_nodes, inputs=ic_data)
            ic_loss = self.loss_fn(ic_res[0], Tensor(
                np.array([0.0]), mstype.float32))

            bc_res = self.parse_node(self.bc_nodes, inputs=bc_data)
            bc_loss = self.loss_fn(bc_res[0], Tensor(
                np.array([0.0]), mstype.float32))

            ic_bc_loss = ic_loss + bc_loss
            result = ic_bc_loss

        else:
            pde_res = self.parse_node(self.pde_nodes, inputs=pde_data)
            pde_loss = self.loss_fn(pde_res[0], Tensor(
                np.array([0.0]), mstype.float32))

            ic_res = self.parse_node(self.ic_nodes, inputs=ic_data)
            ic_loss = self.loss_fn(ic_res[0], Tensor(
                np.array([0.0]), mstype.float32))

            bc_res = self.parse_node(self.bc_nodes, inputs=bc_data)
            bc_loss = self.loss_fn(bc_res[0], Tensor(
                np.array([0.0]), mstype.float32))

            all_loss = pde_loss + ic_loss + bc_loss
            result = all_loss

        return result


class NavierStokes2D(NavierStokes):
    r"""
    2D NavierStokes equation problem based on PDEWithLoss.

    Args:
        model (mindspore.nn.Cell): Network for training.
        re (float): Reynolds number is the ratio of inertia force to viscous force of a fluid. it is a dimensionless
            quantity. Default: 100.0.
        loss_fn (str): Define the loss function. Default: mse.

    Supported Platforms:
        ``Ascend`` ``GPU``
    """

    def __init__(self, model, re=100, loss_fn="mse"):
        super(NavierStokes2D, self).__init__(model, re=re, loss_fn=loss_fn)
        self.ic_nodes = sympy_to_mindspore(
            self.ic(), self.in_vars, self.out_vars)
        self.bc_nodes = sympy_to_mindspore(
            self.bc(), self.in_vars, self.out_vars)
        self.pde_loss_type = 0
        self.pde_ic_loss_type = 1
        self.ic_loss_type = 2
        self.bc_loss_type = 3
        self.ic_bc_loss_type = 4
        self.all_loss_type = 5

    def bc(self):
        """
        Define boundary condition equations based on sympy, abstract method.
        """
        bc_u = self.u
        bc_v = self.v
        equations = {"bc_u": bc_u, "bc_v": bc_v}
        return equations

    def ic(self):
        """
        Define initial condition equations based on sympy, abstract method.
        """
        ic_u = self.u
        ic_v = self.v
        ic_p = self.p
        equations = {"ic_u": ic_u, "ic_v": ic_v, "ic_p": ic_p}
        return equations

    def get_loss(self, pde_data, bc_data, bc_label, ic_data, ic_label, loss_type):
        """
        Compute loss of 3 parts: governing equation, initial condition and boundary conditions.

        Args:
            pde_data (Tensor): the input data of governing equations.
            bc_data (Tensor): the input data of boundary condition.
            bc_label (Tensor): the true value at boundary.
            ic_data (Tensor): the input data of initial condition.
            ic_label (Tensor): the true value of initial state.
            loss_type (int): the type of loss (pde-1, ic-2, bc-3, ic-bc-4, all-other value)
        """

        if loss_type == self.pde_loss_type:
            pde_res = self.parse_node(self.pde_nodes, inputs=pde_data)
            pde_residual = ops.Concat(1)(pde_res)
            pde_loss = self.loss_fn(pde_residual, Tensor(
                np.array([0.0]).astype(np.float32), mstype.float32))
            result = pde_loss

        elif loss_type == self.pde_ic_loss_type:
            pde_res = self.parse_node(self.pde_nodes, inputs=pde_data)
            pde_residual = ops.Concat(1)(pde_res)
            pde_loss = self.loss_fn(pde_residual, Tensor(
                np.array([0.0]), mstype.float32))

            ic_res = self.parse_node(self.ic_nodes, inputs=ic_data)
            ic_residual = ops.Concat(1)(ic_res)
            ic_loss = self.loss_fn(ic_residual, ic_label)
            result = pde_loss + ic_loss

        elif loss_type == self.ic_loss_type:
            ic_res = self.parse_node(self.ic_nodes, inputs=ic_data)
            ic_residual = ops.Concat(1)(ic_res)
            ic_loss = self.loss_fn(ic_residual, ic_label)
            result = ic_loss

        elif loss_type == self.bc_loss_type:
            bc_res = self.parse_node(self.bc_nodes, inputs=bc_data)
            bc_residual = ops.Concat(1)(bc_res)
            bc_loss = self.loss_fn(bc_residual, bc_label)
            result = bc_loss

        elif loss_type == self.ic_bc_loss_type:
            ic_res = self.parse_node(self.ic_nodes, inputs=ic_data)
            ic_residual = ops.Concat(1)(ic_res)
            ic_loss = self.loss_fn(ic_residual, ic_label)

            bc_res = self.parse_node(self.bc_nodes, inputs=bc_data)
            bc_residual = ops.Concat(1)(bc_res)
            bc_loss = self.loss_fn(bc_residual, bc_label)

            ic_bc_loss = ic_loss + bc_loss
            result = ic_bc_loss

        else:
            pde_res = self.parse_node(self.pde_nodes, inputs=pde_data)
            pde_residual = ops.Concat(1)(pde_res)
            pde_loss = self.loss_fn(pde_residual, Tensor(
                np.array([0.0]).astype(np.float32), mstype.float32))

            ic_res = self.parse_node(self.ic_nodes, inputs=ic_data)
            ic_residual = ops.Concat(1)(ic_res)
            ic_loss = self.loss_fn(ic_residual, ic_label)

            bc_res = self.parse_node(self.bc_nodes, inputs=bc_data)
            bc_residual = ops.Concat(1)(bc_res)
            bc_loss = self.loss_fn(bc_residual, bc_label)

            all_loss = pde_loss + ic_loss + bc_loss
            result = all_loss

        return result


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
        self.vis = np.float32(1.0 / (re + 1e-8))
        self.rho = np.float32(rho)
        self.x, self.y = sympy.symbols('x y')
        # u, v, p, uu, uv, vv, rho, nu
        self.u = sympy.Function('u')(self.x, self.y)
        self.v = sympy.Function('v')(self.x, self.y)
        self.p = sympy.Function('p')(self.x, self.y)
        self.uu = sympy.Function('uu')(self.x, self.y)
        self.uv = sympy.Function('uv')(self.x, self.y)
        self.vv = sympy.Function('vv')(self.x, self.y)
        self.in_vars = [self.x, self.y]
        self.out_vars = [self.u, self.v, self.p, self.uu, self.uv, self.vv]
        super(NavierStokesRANS, self).__init__(
            model, self.in_vars, self.out_vars)
        if isinstance(loss_fn, str):
            self.loss_fn = get_loss_metric(loss_fn)
        else:
            self.loss_fn = loss_fn
        self.bc_nodes = sympy_to_mindspore(
            self.bc(), self.in_vars, self.out_vars)
        self.pde_loss_type = 1
        self.bc_loss_type = 3
        self.all_loss_type = 5

    def pde(self):
        """
        Define governing equations based on sympy, abstract method

        returns:
            dict, user defined sympy symbolic equations.
        """
        momentum_x = self.u * self.u.diff(self.x) + self.v * self.u.diff(self.y) + \
            (1/(self.rho+1e-8)) * self.p.diff(self.x) - self.vis * (sympy.diff(self.u, (self.x, 2)) + \
                                                                    sympy.diff(self.u, (self.y, 2))) + \
                                                                    sympy.diff(self.uu, self.x) + \
                                                                    sympy.diff(self.uv, self.y)
        momentum_y = self.u * self.v.diff(self.x) + self.v * self.v.diff(self.y) + \
            (1/(self.rho+1e-8)) * self.p.diff(self.y) - self.vis * (sympy.diff(self.v, (self.x, 2)) + \
                                                                    sympy.diff(self.v, (self.y, 2))) + \
                                                                    sympy.diff(self.vv, self.y) + \
                                                                    sympy.diff(self.uv, self.x)
        continuty = self.u.diff(self.x) + self.v.diff(self.y)

        equations = {"momentum_x": momentum_x,
                     "momentum_y": momentum_y, "continuty": continuty}
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
        equations = {"bc_u": bc_u, "bc_v": bc_v, "bc_p": bc_p,
                     "bc_uu": bc_uu, "bc_uv": bc_uv, "bc_vv": bc_vv}
        return equations

    def get_loss(self, pde_data, bc_data, bc_label, loss_type):
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
        pde_loss = self.loss_fn(pde_residual, Tensor(
            np.array([0.0]).astype(np.float32), mstype.float32))

        bc_res = self.parse_node(self.bc_nodes, inputs=bc_data)
        bc_residual = ops.Concat(1)(bc_res)
        bc_loss = self.loss_fn(bc_residual, bc_label)

        if loss_type == self.pde_loss_type:
            return pde_loss + 1e-5 * bc_loss
        if loss_type == self.bc_loss_type:
            return 0.99995 * bc_loss
        return pde_loss + bc_loss
