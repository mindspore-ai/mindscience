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
"""Burgers 1D problem"""
import numpy as np
from sympy import diff, Function, symbols

from .sympy_pde import PDEWithLoss
from ..loss import get_loss_metric


class Burgers(PDEWithLoss):
    r"""
    Base class for Burgers 1-D problem based on PDEWithLoss.

    Args:
        model (mindspore.nn.Cell): Network for training.
        loss_fn (str): Define the loss function. Default: mse.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> from mindflow.pde import Burgers
        >>> from mindspore import nn, ops
        >>> class Net(nn.Cell):
        ...     def __init__(self, cin=2, cout=1, hidden=10):
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
        >>> problem = Burgers(model)
        >>> print(problem.pde())
        burgers: u(x, t)Derivative(u(x, t), x) + Derivative(u(x, t), t) - 0.00318309897556901Derivative(u(x, t), (x, 2))
            Item numbers of current derivative formula nodes: 3
        {'burgers': u(x, t)Derivative(u(x, t), x) + Derivative(u(x, t), t) - 0.00318309897556901Derivative(u(x, t),
        (x, 2))}
    """
    def __init__(self, model, loss_fn="mse"):
        self.mu = np.float32(0.01 / np.pi)
        self.x, self.t = symbols('x t')
        self.u = Function('u')(self.x, self.t)
        self.in_vars = [self.x, self.t]
        self.out_vars = [self.u]
        super(Burgers, self).__init__(model, self.in_vars, self.out_vars)
        if isinstance(loss_fn, str):
            self.loss_fn = get_loss_metric(loss_fn)
        else:
            self.loss_fn = loss_fn

    def pde(self):
        """
        Define Burgers 1-D governing equations based on sympy, abstract method.

        Returns:
            dict, user defined sympy symbolic equations.
        """
        burgers_eq = diff(self.u, (self.t, 1)) + self.u * diff(self.u, (self.x, 1)) - \
            self.mu * diff(self.u, (self.x, 2))

        equations = {"burgers": burgers_eq}
        return equations
