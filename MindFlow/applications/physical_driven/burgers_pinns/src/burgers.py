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
"""Burgers 1D problem with constant bc"""
from math import pi as PI
from mindspore import ops
from mindspore import Tensor
import mindspore.common.dtype as mstype
from mindflow.solver import Problem

from mindflow.operator import SecondOrderGrad


class Burgers1D(Problem):
    """The 1D Burger's equations with constant boundary condition."""

    def __init__(self, model, config, domain_name=None, bc_name=None, bc_normal=None, ic_name=None):
        super(Burgers1D, self).__init__()
        self.domain_name = domain_name
        self.bc_name = bc_name
        self.ic_name = ic_name
        self.model = model
        self.first_grad = ops.grad(self.model)
        self.u_xx_cell = SecondOrderGrad(self.model, input_idx1=0, input_idx2=0, output_idx=0)
        self.reshape = ops.Reshape()
        self.split = ops.Split(1, 2)
        self.mu = Tensor(0.01 / PI, mstype.float32)
        self.pi = Tensor(PI, mstype.float32)
        self.config = config
        self.bc_normal = bc_normal

    def governing_equation(self, *output, **kwargs):
        """Burgers equation"""
        u = output[0]
        data = kwargs[self.domain_name]

        du_dxt = self.first_grad(data)
        du_dx, du_dt = self.split(du_dxt)
        du_dxx = self.u_xx_cell(data)

        pde_r = du_dt + u * du_dx - self.mu * du_dxx

        return pde_r

    def boundary_condition(self, *output, **kwargs):
        """constant boundary condition"""
        u = output[0]
        return u

    def initial_condition(self, *output, **kwargs):
        """initial condition: u = - sin(x)"""
        u = output[0]
        data = kwargs[self.ic_name]
        x = self.reshape(data[:, 0], (-1, 1))
        return u + ops.sin(self.pi * x)
