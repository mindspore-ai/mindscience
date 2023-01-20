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
"""
Base class of user-defined pde problems.
"""
from __future__ import absolute_import

from mindspore import jit_class

from .sympy2mindspore import sympy_to_mindspore
from ..operators import batched_hessian, batched_jacobian


@jit_class
class PDEWithLoss:
    """
    Base class of user-defined pde problems.
    All user-defined problems to set constraint on each dataset should be inherited from this class.
    It is utilized to establish the mapping between each sub-dataset and used-defined loss functions.
    The loss will be calculated automatically by the constraint type of each sub-dataset. Corresponding member functions
    must be out_channels by user based on the constraint type in order to obtain the target label output. For example,
    for dataset1 the constraint type is "pde", so the member function "pde" must be overridden to tell that how to get
    the pde residual. The data(e.g. inputs) used to solve the residuals is passed to the parse_node, and the residuals
    of each equation can be automatically calculated.

    Args:
        model (mindspore.nn.Cell): Network for training.
        in_vars (List[sympy.core.Symbol]): Input variables of the `model`, represented by the sympy symbol.
        out_vars (List[sympy.core.Function]): Output variables of the `model`, represented by the sympy function.

    Note:
        - The member function, "pde", must be overridden to define the symbolic derivative equqtions based on sympy.
        - The member function, "get_loss", must be overridden to caluate the loss of symbolic derivative equqtions.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> from mindflow.pde import PDEWithLoss, sympy_to_mindspore
        >>> from mindspore import nn, ops, Tensor
        >>> from mindspore import dtype as mstype
        >>> from sympy import symbols, Function, diff
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
        >>> class MyProblem(PDEWithLoss):
        ...     def __init__(self, model, loss_fn=nn.MSELoss()):
        ...         self.x, self.y = symbols('x t')
        ...         self.u = Function('u')(self.x, self.y)
        ...         self.in_vars = [self.x, self.y]
        ...         self.out_vars = [self.u]
        ...         super(MyProblem, self).__init__(model, in_vars=self.in_vars, out_vars=self.out_vars)
        ...         self.loss_fn = loss_fn
        ...         self.bc_nodes = sympy_to_mindspore(self.bc(), self.in_vars, self.out_vars)
        ...
        ...     def pde(self):
        ...         my_eq = diff(self.u, (self.x, 2)) + diff(self.u, (self.y, 2)) - 4.0
        ...         equations = {"my_eq": my_eq}
        ...         return equations
        ...
        ...     def bc(self):
        ...         bc_eq = diff(self.u, (self.x, 1)) + diff(self.u, (self.y, 1)) - 2.0
        ...         equations = {"bc_eq": bc_eq}
        ...         return equations
        ...
        ...     def get_loss(self, pde_data, bc_data):
        ...         pde_res = self.parse_node(self.pde_nodes, inputs=pde_data)
        ...         pde_loss = self.loss_fn(pde_res[0], Tensor(np.array([0.0]), mstype.float32))
        ...         bc_res = self.parse_node(self.bc_nodes, inputs=bc_data)
        ...         bc_loss = self.loss_fn(bc_res[0], Tensor(np.array([0.0]), mstype.float32))
        ...         return pde_loss + bc_loss
        >>> problem = MyProblem(model)
        >>> print(problem.pde())
        >>> print(problem.bc())
        my_eq: Derivative(u(x, t), (t, 2)) + Derivative(u(x, t), (x, 2)) - 4.0
            Item numbers of current derivative formula nodes: 3
        bc_eq: Derivative(u(x, t), t) + Derivative(u(x, t), x) - 2.0
            Item numbers of current derivative formula nodes: 3
        {'my_eq': Derivative(u(x, t), (t, 2)) + Derivative(u(x, t), (x, 2)) - 4.0}
        {'bc_eq': Derivative(u(x, t), t) + Derivative(u(x, t), x) - 2.0}
    """

    def __init__(self, model, in_vars, out_vars):
        self.model = model
        self.jacobian = batched_jacobian(self.model)
        self.hessian = batched_hessian(self.model)
        pde_nodes = self.pde() or dict()
        if isinstance(pde_nodes, dict) and pde_nodes:
            self.pde_nodes = sympy_to_mindspore(pde_nodes, in_vars, out_vars)

    def pde(self):
        """
        Governing equation based on sympy, abstract method.
        This function must be overridden, if the corresponding constraint is governing equation.
        """
        return None

    def get_loss(self):
        """
        Compute all loss from user-defined derivative equations. This function must be overridden.
        """
        return None

    def parse_node(self, formula_nodes, inputs=None, norm=None):
        """
        Calculate the results for each formula node.

        Args:
            formula_nodes (list[FormulaNode]): List of expressions node can by identified by mindspore.
            inputs (Tensor): The input data of network. Default: None.
            norm (Tensor): The normal of the surface at a point P is a vector perpendicular to the tangent plane of the
                point. Default: None.

        Returns:
            List(Tensor), the results of the partial differential equations.
        """
        max_order = 0
        for formula_node in formula_nodes:
            max_order = max(formula_node.max_order, max_order)

        outputs = self.model(inputs)
        if max_order == 2:
            hessian = self.hessian(inputs)
            jacobian = self.jacobian(inputs)
        elif max_order == 1:
            hessian = None
            jacobian = self.jacobian(inputs)
        else:
            hessian = None
            jacobian = None
        data_map = {"inputs": inputs, "outputs": outputs, "jacobian": jacobian, "hessian": hessian, "norm": norm}
        res = []
        for formula_node in formula_nodes:
            cur_eq_ret = formula_node.compute(data_map)
            res.append(cur_eq_ret)
        return res
