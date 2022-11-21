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
"""2d darcy problem with dirichlet boundary condition"""
from scipy.constants import pi as PI
from mindspore import ms_function
from mindspore import ops
from mindspore import Tensor
import mindspore.common.dtype as mstype

from mindflow.operators import Grad
from mindflow.pde import Problem


class Darcy2D(Problem):
    r"""
    The steady-state 2D Darcy flow's equations with Dirichlet boundary condition

    Args:
      model (Cell): The solving network.
      domain_name (str): The corresponding column name of data which governed by maxwell's equation.
      bc_name (str): The corresponding column name of data which governed by boundary condition.
    """

    def __init__(self, model, domain_name=None, bc_name=None):
        super(Darcy2D, self).__init__()
        self.domain_name = domain_name
        self.bc_name = bc_name
        self.model = model
        self.grad = Grad(self.model)
        self.sin = ops.Sin()
        self.cos = ops.Cos()

        # constants
        self.pi = Tensor(PI, mstype.float32)

    def force_function(self, in_x, in_y):
        """"forcing function in Darcy Equation"""
        return 8 * self.pi**2 * self.sin(2 * self.pi * in_x) * self.cos(2 * self.pi * in_y)

    @ms_function
    def governing_equation(self, *output, **kwargs):
        """darcy equation"""
        u_x, u_y, _ = ops.split(output[0], axis=1, output_num=3)

        data = kwargs[self.domain_name]
        in_x = ops.Reshape()(data[:, 0], (-1, 1))
        in_y = ops.Reshape()(data[:, 1], (-1, 1))

        duxdx = ops.Cast()(self.grad(data, 0, 0, output[0]), mstype.float32)
        duydy = ops.Cast()(self.grad(data, 1, 1, output[0]), mstype.float32)
        dpdx = ops.Cast()(self.grad(data, 0, 2, output[0]), mstype.float32)
        dpdy = ops.Cast()(self.grad(data, 1, 2, output[0]), mstype.float32)

        loss_1 = -1 * (duxdx + duydy - self.force_function(in_x, in_y))
        loss_2 = 1 * (u_x + dpdx)
        loss_3 = 2 * self.pi * (u_y + dpdy)

        return ops.Concat(1)((loss_1, loss_2, loss_3))

    @ms_function
    def boundary_condition(self, *output, **kwargs):
        """Dirichlet boundary condition"""

        out_vars = output[0]
        u_x, u_y, pressure = ops.split(out_vars, axis=1, output_num=3)
        data = kwargs[self.bc_name]
        in_x = ops.Reshape()(data[:, 0], (-1, 1))
        in_y = ops.Reshape()(data[:, 1], (-1, 1))
        ux_boundary = -1 * (
            u_x - (-2 * self.pi * self.cos(2 * self.pi * in_x) * self.cos(2 * self.pi * in_y))
        )

        uy_boundary = 1 * (
            u_y - (2 * self.pi * self.sin(2 * self.pi * in_x) * self.sin(2 * self.pi * in_y))
        )

        p_boundary = (
            2 * self.pi * (pressure - self.sin(2 * self.pi * in_x) * self.cos(2 * self.pi * in_y))
        )
        return ops.Concat(1)((ux_boundary, uy_boundary, p_boundary))
