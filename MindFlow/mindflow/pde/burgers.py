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
from math import pi as PI

from mindspore import ops, Tensor
from mindspore import dtype as mstype

from .problem import Problem
from ..operators import Grad, SecondOrderGrad


class Burgers1D(Problem):
    """The 1D Burger's equations with constant boundary condition."""

    def __init__(self,
                 model,
                 domain_name=None,
                 bc_name=None,
                 ic_name=None):
        super(Burgers1D, self).__init__()

        self.model = model
        self.mu = Tensor(0.01 / PI, mstype.float32)
        self.pi = Tensor(PI, mstype.float32)

        self.domain_name = domain_name
        self.bc_name = bc_name
        self.ic_name = ic_name
        # define first order gradient and second order gradient
        self.first_grad = Grad(self.model)
        self.u_xx_cell = SecondOrderGrad(self.model, input_idx1=0, input_idx2=0, output_idx=0)

    def governing_equation(self, *output, **kwargs):
        """Burgers equation"""
        u = output[0]
        data = kwargs[self.domain_name]

        du_dxt = self.first_grad(data, None, 0, u)
        du_dx, du_dt = ops.split(du_dxt, axis=1, output_num=2)
        du_dxx = self.u_xx_cell(data)

        pde_residual = du_dt + u * du_dx - self.mu * du_dxx

        return pde_residual

    def boundary_condition(self, *output, **kwargs):
        """constant boundary condition"""
        bc_residual = output[0]
        return bc_residual

    def initial_condition(self, *output, **kwargs):
        """initial condition: u = - pi * sin(x)"""
        u = output[0]
        data = kwargs[self.ic_name]
        x = ops.reshape(data[:, 0], (-1, 1))
        ic_residual = u + ops.sin(self.pi * x)
        return ic_residual
    