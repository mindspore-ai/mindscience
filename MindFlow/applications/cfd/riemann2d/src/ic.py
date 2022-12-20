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
# ==============================================================================
"""initial condition of riemann 2d flow"""
from mindspore import ops
from mindspore import numpy as mnp


def riemann2d_ic(mesh_x, mesh_y):
    """initial condition of riemann 2d flow."""
    rho = [1.5, 0.5323, 0.138, 0.5323]
    u = [0.0, 1.206, 1.206, 0.0]
    v = [0.0, 0.0, 1.206, 1.206]
    p = [1.5, 0.3, 0.029, 0.3]

    logical_and = ops.LogicalAnd()

    large_x = mnp.greater_equal(mesh_x, 0.5)
    small_x = mnp.less(mesh_x, 0.5)
    large_y = mnp.greater_equal(mesh_y, 0.5)
    small_y = mnp.less(mesh_y, 0.5)

    one = mnp.ones_like(mesh_x)
    zero = mnp.zeros_like(mesh_x)
    reg1 = logical_and(large_x, large_y)
    reg1 = mnp.where(reg1, one, zero)
    reg2 = logical_and(small_x, large_y)
    reg2 = mnp.where(reg2, one, zero)
    reg3 = logical_and(small_x, small_y)
    reg3 = mnp.where(reg3, one, zero)
    reg4 = logical_and(large_x, small_y)
    reg4 = mnp.where(reg4, one, zero)

    rho_0 = rho[0] * reg1 + rho[1] * reg2 + rho[2] * reg3 + rho[3] * reg4
    u_0 = u[0] * reg1 + u[1] * reg2 + u[2] * reg3 + u[3] * reg4
    v_0 = v[0] * reg1 + v[1] * reg2 + v[2] * reg3 + v[3] * reg4
    w_0 = mnp.zeros_like(u_0)
    p_0 = p[0] * reg1 + p[1] * reg2 + p[2] * reg3 + p[3] * reg4

    return mnp.stack([rho_0, u_0, v_0, w_0, p_0], axis=0)
