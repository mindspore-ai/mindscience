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
"""init"""
import numpy as np
import sympy
from sympy import Function, symbols

from mindspore import ops, Tensor, Parameter
from mindspore import dtype as mstype

from mindflow.pde import NavierStokes, sympy_to_mindspore
from mindflow.pde import PDEWithLoss
from mindflow.loss import get_loss_metric

from .divide import divide_with_error


def constr_softmax(c_min, c):
    softplus = ops.Softplus()
    return softplus(c) + c_min


def constr_sigmoid(alpha_min, alpha_max, alpha):
    return ops.sigmoid(alpha) * (alpha_max - alpha_min) + alpha_min


class Burgers(PDEWithLoss):
    r"""
    Burgers equation problem based on PDEWithLoss.

    Args:
        model (mindspore.nn.Cell): Network for training.
        nu (float): Viscosity. Default: 0.1.
        loss_fn (str): Define the loss function. Default: mse.

    Supported Platforms:
        ``Ascend`` ``GPU``
    """

    def __init__(self, lamda, model, config, loss_fn="mse"):
        self.lamda = np.float32(lamda)
        self.x, self.t = symbols('x t')
        self.u = Function('u')(self.x, self.t)
        self.in_vars = [self.x, self.t]
        self.out_vars = [self.u]
        self.param_config = config["hyperparameter"]
        self.w_min = self.param_config["w_min"]
        self.w_max = self.param_config["w_max"]

        # define hyperparameters
        self.w1 = Parameter(Tensor([1], mstype.float32), name="w1")
        self.w2 = Parameter(Tensor([1], mstype.float32), name="w2")
        self.w3 = Parameter(Tensor([1], mstype.float32), name="w3")

        super(Burgers, self).__init__(model, self.in_vars, self.out_vars)
        if isinstance(loss_fn, str):
            self.loss_fn = get_loss_metric(loss_fn)
        else:
            self.loss_fn = loss_fn
        self.ic_nodes = sympy_to_mindspore(
            self.ic(), self.in_vars, self.out_vars)
        self.bc_nodes = sympy_to_mindspore(
            self.bc(), self.in_vars, self.out_vars)

    def pde(self):
        """
        Define the governing equation based on sympy, abstract method.

        Returns:
            dict, user defined sympy symbolic equations.
        """
        burgers_eq = self.u.diff(
            self.t) + self.u*self.u.diff(self.x) - self.lamda * self.u.diff(self.x, 2)

        equations = {"burgers_eq": burgers_eq}
        return equations

    def ic(self):
        """
        Define the initial condition based on sympy, abstract method.
        """
        ic_eq = self.u + sympy.sin(np.pi * self.x)
        equations = {"ic": ic_eq}
        return equations

    def bc(self):
        """
        Define the boundary condition equations based on sympy, abstract method.
        """
        bc_eq = self.u
        equations = {"bc": bc_eq}

        return equations

    def get_loss(self, *data, if_constr_sigmoid, if_weighted):
        """
        Compute the loss of the governing equation, boundary conditions and initial conditions.

        Args:
            pde_data (Tensor): The input data of the governing equation.
            bc_data (Tensor): The input data of the boundary condition.
            ic_data (Tensor): The input data of the initial condition.
        """
        pde_data, ic_data, bc_data = data

        pde_res = self.parse_node(self.pde_nodes, inputs=pde_data)
        pde_loss = self.loss_fn(pde_res[0], Tensor(
            np.array([0.0]), mstype.float32))

        ic_res = self.parse_node(self.ic_nodes, inputs=ic_data)
        ic_loss = self.loss_fn(ic_res[0], Tensor(
            np.array([0.0]), mstype.float32))

        bc_res = self.parse_node(self.bc_nodes, inputs=bc_data)
        bc_loss = self.loss_fn(bc_res[0], Tensor(
            np.array([0.0]), mstype.float32))

        if if_constr_sigmoid:
            self.w1 = constr_sigmoid(
                self.w_min, self.w_max, self.w1)
            self.w2 = constr_sigmoid(
                self.w_min, self.w_max, self.w2)
            self.w3 = constr_sigmoid(
                self.w_min, self.w_max, self.w3)

        if if_weighted:
            result = self.w1 * pde_loss + self.w2 * ic_loss + self.w3 * bc_loss
        else:
            result = pde_loss + ic_loss + bc_loss
        return result

    def set_hp_params(self, params):
        self.w1 = params[0]
        self.w2 = params[1]
        self.w3 = params[2]

    def get_params(self, if_value=False):
        if if_value:
            return [self.w1.asnumpy(), self.w2.asnumpy(), self.w3.asnumpy()]
        return [self.w1, self.w2, self.w3]


class LBurgers(PDEWithLoss):
    r"""
    Linear Burgers problem based on PDEWithLoss.

    Args:
        model (mindspore.nn.Cell): Network for training.
        lamda (float): Viscosity.
        loss_fn (str): Define the loss function. Default: mse.

    Supported Platforms:
        ``Ascend`` ``GPU``
    """

    def __init__(self, lamda, model, config, loss_fn="mse"):
        self.lamda = np.float32(lamda)
        self.x, self.t = symbols('x t')
        self.u = Function('u')(self.x, self.t)
        self.in_vars = [self.x, self.t]
        self.out_vars = [self.u]
        self.param_config = config["hyperparameter"]

        self.w_min = self.param_config["w_min"]
        self.w_max = self.param_config["w_max"]

        # define hyperparameters
        self.w1 = Parameter(Tensor([1], mstype.float32), name="w1")
        self.w2 = Parameter(Tensor([1], mstype.float32), name="w2")

        super(LBurgers, self).__init__(model, self.in_vars, self.out_vars)
        if isinstance(loss_fn, str):
            self.loss_fn = get_loss_metric(loss_fn)
        else:
            self.loss_fn = loss_fn
        self.ic_nodes = sympy_to_mindspore(
            self.ic(), self.in_vars, self.out_vars)

    def pde(self):
        """
        Define the governing equation based on sympy, abstract method.

        Returns:
            dict, user defined sympy symbolic equations.
        """
        custom_eq = self.u.diff(self.t) - self.lamda * \
            self.u.diff(self.x, 2) + self.u.diff(self.x, 1)

        equations = {"ac_eq": custom_eq}
        return equations

    def ic(self):
        """
        Define the initial condition based on sympy, abstract method.
        """
        ic_eq = self.u - 10 * sympy.exp(-(2*self.x)**2)
        equations = {"ic": ic_eq}
        return equations

    def get_loss(self, *data, if_constr_sigmoid, if_weighted):
        """
        Compute the loss of the governing equation and initial conditions.

        Args:
            pde_data (Tensor): The input data of the governing equation.
            ic_data (Tensor): The input data of the initial condition.
        """
        pde_data, ic_data = data
        pde_res = self.parse_node(self.pde_nodes, inputs=pde_data)
        pde_loss = self.loss_fn(pde_res[0], Tensor(
            np.array([0.0]), mstype.float32))

        ic_res = self.parse_node(self.ic_nodes, inputs=ic_data)
        ic_loss = self.loss_fn(ic_res[0], Tensor(
            np.array([0.0]), mstype.float32))

        if if_constr_sigmoid:
            self.w1 = constr_sigmoid(
                self.w_min, self.w_max, self.w1)
            self.w2 = constr_sigmoid(
                self.w_min, self.w_max, self.w2)

        if not if_weighted:
            self.w1 = 1.0
            self.w2 = 1.0

        return self.w1 * pde_loss + self.w2 * ic_loss

    def set_hp_params(self, params):
        self.w1 = params[0]
        self.w2 = params[1]

    def get_params(self, if_value=False):
        if if_value:
            return [self.w1.asnumpy(), self.w2.asnumpy()]
        return [self.w1, self.w2]


class ConvectionDiffusion(PDEWithLoss):
    r"""
    ConvectionDiffusion problem based on PDEWithLoss.

    Args:
        model (mindspore.nn.Cell): Network for training.
        lamda (float): Viscosity.
        loss_fn (str): Define the loss function. Default: mse.

    Supported Platforms:
        ``Ascend`` ``GPU``
    """

    def __init__(self, lamda, model, config, loss_fn="mse"):
        self.lamda = np.float32(lamda)
        self.x = symbols('x')
        self.u = Function('u')(self.x)
        self.in_vars = [self.x]
        self.out_vars = [self.u]
        self.param_config = config["hyperparameter"]
        self.w_min = self.param_config["w_min"]
        self.w_max = self.param_config["w_max"]
        # define hyperparameters
        self.w1 = Parameter(Tensor([1], mstype.float32), name="w1")
        self.w2 = Parameter(Tensor([1], mstype.float32), name="w2")

        super(ConvectionDiffusion, self).__init__(
            model, self.in_vars, self.out_vars)
        if isinstance(loss_fn, str):
            self.loss_fn = get_loss_metric(loss_fn)
        else:
            self.loss_fn = loss_fn
        self.bc_nodes = sympy_to_mindspore(
            self.bc(), self.in_vars, self.out_vars)

    # Function to generate analytical solution
    @staticmethod
    def eval_u(x):
        u = divide_with_error((1. - ops.exp(Tensor(
            np.array([6.0]), mstype.float32) * x)), (1. - ops.exp(Tensor(
                np.array([6.0]), mstype.float32))))
        return u

    def pde(self):
        """
        Define the governing equation based on sympy, abstract method.

        Returns:
            dict, user defined sympy symbolic equations.
        """
        custom_eq = 6 * self.u.diff(self.x) - self.lamda * \
            self.u.diff(self.x, 2)

        equations = {"ac_eq": custom_eq}
        return equations

    def bc(self):
        """
        Define the boundary condition based on sympy, abstract method.
        """
        bc_eq = self.u
        equations = {"bc": bc_eq}
        return equations

    def get_loss(self, *data, if_constr_sigmoid, if_weighted):
        """
        Compute the loss of the governing equation, boundary conditions and initial conditions.

        Args:
            pde_data (Tensor): The input data of the governing equation.
            bc_data (Tensor): The input data of the boundary condition.
        """
        pde_data, bc_data = data

        pde_res = self.parse_node(self.pde_nodes, inputs=pde_data)
        pde_res = ops.Concat(1)(pde_res)
        pde_loss = self.loss_fn(pde_res, Tensor(
            np.array([0.0]), mstype.float32))

        bc_res = self.parse_node(self.bc_nodes, inputs=bc_data)
        bc_res = ops.Concat(1)(bc_res)
        bc_label = self.eval_u(bc_data)
        bc_loss = self.loss_fn(bc_res, bc_label)

        if if_constr_sigmoid:
            self.w1 = constr_sigmoid(
                self.w_min, self.w_max, self.w1)
            self.w2 = constr_sigmoid(
                self.w_min, self.w_max, self.w2)

        if not if_weighted:
            self.w1 = 1.0
            self.w2 = 1.0

        return self.w1 * pde_loss + self.w2 * bc_loss

    def set_hp_params(self, params):
        self.w1 = params[0]
        self.w2 = params[1]

    def get_params(self, if_value=False):
        if if_value:
            return [self.w1.asnumpy(), self.w2.asnumpy()]
        return [self.w1, self.w2]


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

    def __init__(self, lamda, model, config, loss_fn="mse"):
        self.lamda = np.float32(lamda)
        self.param_config = config["hyperparameter"]
        super(NavierStokes2D, self).__init__(
            model, re=self.lamda, loss_fn=loss_fn)
        self.w_min = self.param_config["w_min"]
        self.w_max = self.param_config["w_max"]
        self.ic_nodes = sympy_to_mindspore(
            self.ic(), self.in_vars, self.out_vars)
        self.bc_nodes = sympy_to_mindspore(
            self.bc(), self.in_vars, self.out_vars)

        # define hyperparameters
        self.w1 = Parameter(Tensor([1], mstype.float32), name="w1")
        self.w2 = Parameter(Tensor([1], mstype.float32), name="w2")
        self.w3 = Parameter(Tensor([1], mstype.float32), name="w3")

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

    def get_loss(self, *data, if_constr_sigmoid, if_weighted):
        """
        Compute loss
        """
        pde_data, bc_data, bc_label, ic_data, ic_label = data

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

        if if_constr_sigmoid:
            self.w1 = constr_sigmoid(
                self.w_min, self.w_max, self.w1)
            self.w2 = constr_sigmoid(
                self.w_min, self.w_max, self.w2)
            self.w3 = constr_sigmoid(
                self.w_min, self.w_max, self.w3)

        if not if_weighted:
            self.w1 = 1.0
            self.w2 = 1.0
            self.w3 = 1.0

        return self.w1 * pde_loss + self.w2 * ic_loss + self.w3 * bc_loss

    def set_hp_params(self, params):
        self.w1 = params[0]
        self.w2 = params[1]
        self.w3 = params[2]

    def get_params(self, if_value=False):
        if if_value:
            return [self.w1.asnumpy(), self.w2.asnumpy(), self.w3.asnumpy()]
        return [self.w1, self.w2, self.w3]


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

    def __init__(self, lamda, model, config, loss_fn="mse"):
        re = 5600
        self.vis = np.float32(divide_with_error(1.0, re))
        self.rho = np.float32(lamda)
        self.x, self.y = sympy.symbols('x y')
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

        self.param_config = config["hyperparameter"]
        self.w_min = self.param_config["w_min"]
        self.w_max = self.param_config["w_max"]
        # define hyperparameters
        self.w1 = Parameter(Tensor([1], mstype.float32), name="w1")
        self.w2 = Parameter(Tensor([1], mstype.float32), name="w2")

    def pde(self):
        """
        Define governing equations based on sympy, abstract method

        returns:
            dict, user defined sympy symbolic equations.
        """
        momentum_x = self.u * self.u.diff(self.x) + self.v * self.u.diff(self.y) + \
            divide_with_error(1, self.rho) * self.p.diff(self.x) - self.vis * (sympy.diff(self.u, (self.x, 2)) +
                                                                               sympy.diff(self.u, (self.y, 2))) + \
            sympy.diff(self.uu, self.x) + sympy.diff(self.uv, self.y)
        momentum_y = self.u * self.v.diff(self.x) + self.v * self.v.diff(self.y) + \
            divide_with_error(1, self.rho) * self.p.diff(self.y) - self.vis * (sympy.diff(self.v, (self.x, 2)) +
                                                                               sympy.diff(self.v, (self.y, 2))) + \
            sympy.diff(self.vv, self.y) + sympy.diff(self.uv, self.x)
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

    def get_loss(self, *data, if_constr_sigmoid, if_weighted):
        """
        Args:
            pde_data (Tensor): the input data of governing equations.
            bc_data (Tensor): the input data of boundary condition.
            bc_label (Tensor): the true value at boundary.
            ic_data (Tensor): the input data of initial condition.
            ic_label (Tensor): the true value of initial state.
        """
        pde_data, bc_data, bc_label = data
        pde_res = self.parse_node(self.pde_nodes, inputs=pde_data)
        pde_residual = ops.Concat(1)(pde_res)
        pde_loss = self.loss_fn(pde_residual, Tensor(
            np.array([0.0]).astype(np.float32), mstype.float32))

        bc_res = self.parse_node(self.bc_nodes, inputs=bc_data)
        bc_residual = ops.Concat(1)(bc_res)
        bc_loss = self.loss_fn(bc_residual, bc_label)

        if if_constr_sigmoid:
            self.w1 = constr_sigmoid(
                self.w_min, self.w_max, self.w1)
            self.w2 = constr_sigmoid(
                self.w_min, self.w_max, self.w2)

        if not if_weighted:
            self.w1 = 1.0
            self.w2 = 1.0

        return self.w1 * pde_loss + self.w2 * bc_loss

    def set_hp_params(self, params):
        self.w1 = params[0]
        self.w2 = params[1]

    def get_params(self, if_value=False):
        if if_value:
            return [self.w1.asnumpy(), self.w2.asnumpy()]
        return [self.w1, self.w2]
