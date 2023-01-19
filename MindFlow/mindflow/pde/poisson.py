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
"""Poisson 2D problem"""
import sympy

from ..loss import get_loss_metric
from .sympy_pde import PDEWithLoss


class Poisson(PDEWithLoss):
    r"""
    Base class for Poisson 2-D problem based on PDEWithLoss.

    Args:
        model (mindspore.nn.Cell): network for training.
        loss_fn (str): Define the loss function. Default: mse.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> from mindflow.pde import Poisson
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
        >>> problem = Poisson(model)
        >>> print(problem.pde())
        poisson: Derivative(u(x, y), (x, 2)) + Derivative(u(x, y), (y, 2)) + 1.0
            Item numbers of current derivative formula nodes: 3
        {'poisson': Derivative(u(x, y), (x, 2)) + Derivative(u(x, y), (y, 2)) + 1.0}
    """
    def __init__(self, model, loss_fn="mse"):
        self.x = sympy.Symbol('x')
        self.y = sympy.Symbol('y')
        self.normal = sympy.Symbol('n')
        self.u = sympy.Function('u')(self.x, self.y)

        self.in_vars = [self.x, self.y]
        self.out_vars = [self.u]
        super(Poisson, self).__init__(model, self.in_vars, self.out_vars)
        if isinstance(loss_fn, str):
            self.loss_fn = get_loss_metric(loss_fn)
        else:
            self.loss_fn = loss_fn

    def pde(self):
        """
        Define Poisson 2-D governing equations based on sympy, abstract method.

        Returns:
            dict, user defined sympy symbolic equations.
        """
        poisson = sympy.diff(self.u, (self.x, 2)) + sympy.diff(self.u, (self.y, 2)) + 1.0

        equations = {"poisson": poisson}
        return equations
