# ============================================================================
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
"""trainer"""
import numpy as np
from mindspore import nn, ops, Tensor, float32, jit_class
from .constant import dx_2d_op, dy_2d_op, lap_2d_op


@jit_class
class PhysicsLossGenerator:
    """Trainer"""

    def __init__(self, dx, dy, dt, nu, compute_dtype=float32):
        self.loss = nn.MSELoss()
        self.dx = dx
        self.dy = dy
        self.dt = dt
        self.nu = nu
        self.compute_dtype = compute_dtype
        self.dx_kernel = Tensor(np.array(dx_2d_op) /
                                self.dx, self.compute_dtype)
        self.dy_kernel = Tensor(np.array(dy_2d_op) /
                                self.dy, self.compute_dtype)
        self.lap_kernel = Tensor(
            np.array(lap_2d_op) / self.dx**2, self.compute_dtype)

    def get_phy_residual(self, output):
        """calculate the phy loss"""
        output = ops.concat(
            (output[:, :, :, -2:], output, output[:, :, :, 0:3]), axis=3)
        output = ops.concat(
            (output[:, :, -2:, :], output, output[:, :, 0:3, :]), axis=2)

        laplace_u = ops.conv2d(output[0:-2, 0:1, :, :], self.lap_kernel)
        laplace_v = ops.conv2d(output[0:-2, 1:2, :, :], self.lap_kernel)

        u_x = ops.conv2d(output[0:-2, 0:1, :, :], self.dx_kernel)
        u_y = ops.conv2d(output[0:-2, 0:1, :, :], self.dy_kernel)
        v_x = ops.conv2d(output[0:-2, 1:2, :, :], self.dx_kernel)
        v_y = ops.conv2d(output[0:-2, 1:2, :, :], self.dy_kernel)

        u_t = (output[1:-1, 0:1, 2:-2, 2:-2] -
               output[0:-2, 0:1, 2:-2, 2:-2]) / self.dt
        v_t = (output[1:-1, 1:2, 2:-2, 2:-2] -
               output[0:-2, 1:2, 2:-2, 2:-2]) / self.dt

        u = output[0:-2, 0:1, 2:-2, 2:-2]
        v = output[0:-2, 1:2, 2:-2, 2:-2]

        f_u = u_t - self.nu*laplace_u + u*u_x + v*u_y
        f_v = v_t - self.nu*laplace_v + u*v_x + v*v_y

        ones = ops.ones_like(u)

        library = {'f_u': f_u, 'f_v': f_v, 'ones': ones, 'u': u, 'v': v, 'u_t': u_t, 'v_t': v_t,
                   'u_x': u_x, 'u_y': u_y, 'v_x': v_x, 'v_y': v_y, 'lap_u': laplace_u, 'lap_v': laplace_v}

        return library

    def get_phy_loss(self, output):
        library = self.get_phy_residual(output)
        f_u, f_v = library['f_u'], library['f_v']
        return self.loss(f_u, ops.zeros_like(f_u)), self.loss(f_v, ops.zeros_like(f_v))
