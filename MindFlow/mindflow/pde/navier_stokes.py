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
"""Navier-Stokes 2D Problem"""
import numpy as np
from sympy import diff, Function, symbols

from ..loss import get_loss_metric
from .sympy_pde import PDEWithLoss


class NavierStokes(PDEWithLoss):
    r"""
    2D NavierStokes equation problem based on PDEWithLoss.

    Args:
        model (mindspore.nn.Cell): network for training.
        re (float): reynolds number is the ratio of inertia force to viscous force of a fluid. It is a dimensionless
            quantity. Default: 100.0.
        loss_fn (str): Define the loss function. Default: mse.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> from mindflow.pde import NavierStokes
        >>> from mindspore import nn, ops
        >>> class Net(nn.Cell):
        ...     def __init__(self, cin=3, cout=3, hidden=10):
        ...         super().__init__()
        ...         self.fc1 = nn.Dense(cin, hidden)
        ...         self.fc2 = nn.Dense(hidden, hidden)
        ...         self.fcout = nn.Dense(hidden, cout)
        ...         self.act = ops.Tanh()
        ...
        ...     def construct(self, x):
        ...         x = self.act(self.fc1(x))
        ...         x = self.act(self.fc2(x))
        ...         x = self.fcout(x)
        ...         return x
        >>> model = Net()
        >>> problem = NavierStokes(model)
        >>> print(problem.pde())
        momentum_x: u(x, y, t)Derivative(u(x, y, t), x) + v(x, y, t)Derivative(u(x, y, t), y) +
        Derivative(p(x, y, t), x) + Derivative(u(x, y, t), t) - 0.00999999977648258Derivative(u(x, y, t), (x, 2)) -
        0.00999999977648258Derivative(u(x, y, t), (y, 2))
            Item numbers of current derivative formula nodes: 6
        momentum_y: u(x, y, t)Derivative(v(x, y, t), x) + v(x, y, t)Derivative(v(x, y, t), y) +
        Derivative(p(x, y, t), y) + Derivative(v(x, y, t), t) - 0.00999999977648258Derivative(v(x, y, t), (x, 2)) -
        0.00999999977648258Derivative(v(x, y, t), (y, 2))
            Item numbers of current derivative formula nodes: 6
        continuty: Derivative(u(x, y, t), x) + Derivative(v(x, y, t), y)
            Item numbers of current derivative formula nodes: 2
        {'momentum_x': u(x, y, t)Derivative(u(x, y, t), x) + v(x, y, t)Derivative(u(x, y, t), y) +
        Derivative(p(x, y, t), x) + Derivative(u(x, y, t), t) - 0.00999999977648258Derivative(u(x, y, t), (x, 2)) -
        0.00999999977648258Derivative(u(x, y, t), (y, 2)),
        'momentum_y': u(x, y, t)Derivative(v(x, y, t), x) + v(x, y, t)Derivative(v(x, y, t), y) +
        Derivative(p(x, y, t), y) + Derivative(v(x, y, t), t) - 0.00999999977648258Derivative(v(x, y, t), (x, 2)) -
        0.00999999977648258Derivative(v(x, y, t), (y, 2)),
        'continuty': Derivative(u(x, y, t), x) + Derivative(v(x, y, t), y)}
    """

    def __init__(self, model, re=100.0, loss_fn="mse"):
        self.number = np.float32(1.0 / re)
        self.x, self.y, self.t = symbols('x y t')
        self.u = Function('u')(self.x, self.y, self.t)
        self.v = Function('v')(self.x, self.y, self.t)
        self.p = Function('p')(self.x, self.y, self.t)
        self.in_vars = [self.x, self.y, self.t]
        self.out_vars = [self.u, self.v, self.p]
        super(NavierStokes, self).__init__(model, self.in_vars, self.out_vars)
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
        momentum_x = self.u.diff(self.t) + self.u * self.u.diff(self.x) + self.v * self.u.diff(self.y) + \
                     self.p.diff(self.x) - self.number * (diff(self.u, (self.x, 2)) + diff(self.u, (self.y, 2)))
        momentum_y = self.v.diff(self.t) + self.u * self.v.diff(self.x) + self.v * self.v.diff(self.y) + \
                     self.p.diff(self.y) - self.number * (diff(self.v, (self.x, 2)) + diff(self.v, (self.y, 2)))
        continuty = self.u.diff(self.x) + self.v.diff(self.y)

        equations = {"momentum_x": momentum_x, "momentum_y": momentum_y, "continuty": continuty}
        return equations
