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
from mindspore import ops, Tensor
from mindspore import dtype as mstype

from .problem import Problem
from ..operators import Grad, SecondOrderGrad


class NavierStokes2D(Problem):
    """2D NavierStokes equation"""

    def __init__(self, model, re=100, domain_name=None, bc_label_name=None, ic_label_name=None):
        super(NavierStokes2D, self).__init__()
        self.model = model
        self.re = Tensor(re, mstype.float32)

        self.domain_name = domain_name
        self.bc_label_name = bc_label_name
        self.ic_label_name = ic_label_name

        # define first order gradient and second order gradient
        self.grad = Grad(self.model)
        self.gradux_xx = SecondOrderGrad(self.model, input_idx1=0, input_idx2=0, output_idx=0)
        self.gradux_yy = SecondOrderGrad(self.model, input_idx1=1, input_idx2=1, output_idx=0)
        self.graduy_xx = SecondOrderGrad(self.model, input_idx1=0, input_idx2=0, output_idx=1)
        self.graduy_yy = SecondOrderGrad(self.model, input_idx1=1, input_idx2=1, output_idx=1)

    def governing_equation(self, *output, **kwargs):
        """governing equation"""
        flow_vars = output[0]
        u, v, _ = ops.split(flow_vars, axis=1, output_num=3)
        domain_data = kwargs[self.domain_name]

        du_dx, du_dy, du_dt = ops.split(self.grad(domain_data, None, 0, flow_vars), axis=1, output_num=3)
        dv_dx, dv_dy, dv_dt = ops.split(self.grad(domain_data, None, 1, flow_vars), axis=1, output_num=3)
        dp_dx, dp_dy, _ = ops.split(self.grad(domain_data, None, 2, flow_vars), axis=1, output_num=3)
        du_dxx = self.gradux_xx(domain_data)
        du_dyy = self.gradux_yy(domain_data)
        dv_dxx = self.graduy_xx(domain_data)
        dv_dyy = self.graduy_yy(domain_data)

        eq1 = du_dt + (u * du_dx + v * du_dy) + dp_dx - 1.0 / self.re * (du_dxx + du_dyy)
        eq2 = dv_dt + (u * dv_dx + v * dv_dy) + dp_dy - 1.0 / self.re * (dv_dxx + dv_dyy)
        eq3 = du_dx + dv_dy
        pde_residual = ops.Concat(1)((eq1, eq2, eq3))
        return pde_residual

    def boundary_condition(self, *output, **kwargs):
        """boundary condition"""
        # select u, v and drop pressure
        flow_vars = output[0][:, :2]
        bc_label = kwargs[self.bc_label_name]
        bc_residual = flow_vars - bc_label
        return bc_residual

    def initial_condition(self, *output, **kwargs):
        """initial condition"""
        flow_vars = output[0]
        ic_label = kwargs[self.ic_label_name]
        ic_residual = flow_vars - ic_label
        return ic_residual
    