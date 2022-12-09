# Copyright 2021 Huawei Technologies Co., Ltd
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
"""
maxwell function and its constraints
"""
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.numpy as ms_np
from mindspore import ms_function
from mindspore import Tensor

from mindelec.operators import SecondOrderGrad, Grad
from mindelec.common import PI


class Maxwell3DLoss(nn.Cell):
    """
    Define the PINNs loss network, which is also the loss of PDEs.
    """

    def __init__(self, net, config):
        super(Maxwell3DLoss, self).__init__(auto_prefix=False)
        self.net = net
        self.config = config

        self.grad = Grad(net)

        self.hessian_ex_xx = SecondOrderGrad(net, 0, 0, output_idx=0)
        self.hessian_ex_yy = SecondOrderGrad(net, 1, 1, output_idx=0)
        self.hessian_ex_zz = SecondOrderGrad(net, 2, 2, output_idx=0)

        self.hessian_ey_xx = SecondOrderGrad(net, 0, 0, output_idx=1)
        self.hessian_ey_yy = SecondOrderGrad(net, 1, 1, output_idx=1)
        self.hessian_ey_zz = SecondOrderGrad(net, 2, 2, output_idx=1)

        self.hessian_ez_xx = SecondOrderGrad(net, 0, 0, output_idx=2)
        self.hessian_ez_yy = SecondOrderGrad(net, 1, 1, output_idx=2)
        self.hessian_ez_zz = SecondOrderGrad(net, 2, 2, output_idx=2)

        self.reshape = ops.Reshape()
        self.concat = ops.Concat(1)
        self.cast = ops.Cast()
        self.mul = ops.Mul()

        self.zeros_like = ops.ZerosLike()
        self.reduce_mean = ops.ReduceMean()
        self.l2_loss = nn.MSELoss()

        self.eps0 = Tensor(config["eps0"], ms.dtype.float32)  # permitivity
        self.wave_number = Tensor(
            config["wave_number"], ms.dtype.float32)
        self.pi = Tensor(PI, ms.dtype.float32)
        self.eigenmode = Tensor(config["eigenmode"], ms.dtype.float32)

        # Coefficient used to adjust the different loss.
        self.gamma_domain = Tensor(
            1.0 / self.wave_number**2, ms.dtype.float32)
        self.gamma_bc = Tensor(100.0, ms.dtype.float32)

    def construct(self, in_domain, in_bc):
        """
        Calculate the loss according the input data.

        Args:
            in_domain: input sampling points of domain.
            in_bc: input sampling points of boundary.
        """
        # domain loss
        u_domain = self.net(in_domain)
        loss_domain = self.get_domain_loss(u_domain, in_domain)

        # boundary loss
        u_bc = self.net(in_bc)
        loss_bc = self.get_boundary_loss(u_bc, in_bc)

        return self.gamma_domain * loss_domain + self.gamma_bc * loss_bc

    @ms_function
    def get_domain_loss(self, u, data):
        """
        Calculate the inner loss according to 3D Maxwell equation.
        """
        # Get the second order derivative.
        ex_xx = self.hessian_ex_xx(data)
        ex_yy = self.hessian_ex_yy(data)
        ex_zz = self.hessian_ex_zz(data)

        ey_xx = self.hessian_ey_xx(data)
        ey_yy = self.hessian_ey_yy(data)
        ey_zz = self.hessian_ey_zz(data)

        ez_xx = self.hessian_ez_xx(data)
        ez_yy = self.hessian_ez_yy(data)
        ez_zz = self.hessian_ez_zz(data)

        # Get the residual of PDE.
        pde_rx = ex_xx + ex_yy + ex_zz + \
            self.wave_number**2 * self.eps0 * u[:, 0:1]
        pde_ry = ey_xx + ey_yy + ey_zz + \
            self.wave_number**2 * self.eps0 * u[:, 1:2]
        pde_rz = ez_xx + ez_yy + ez_zz + \
            self.wave_number**2 * self.eps0 * u[:, 2:3]
        pde_r = self.concat((pde_rx, pde_ry, pde_rz))
        loss_domain = self.reduce_mean(
            self.l2_loss(pde_r, self.zeros_like(pde_r)))

        # The divergence of no source region is zero.
        ex_x = self.grad(data, 0, 0, u)
        ey_y = self.grad(data, 1, 1, u)
        ez_z = self.grad(data, 2, 2, u)
        no_source_r = ex_x + ey_y + ez_z
        loss_no_source = self.reduce_mean(self.l2_loss(
            no_source_r, self.zeros_like(no_source_r)))

        return loss_domain + loss_no_source

    @ms_function
    def get_boundary_loss(self, u, data):
        """
        根据边界条件计算边界损失，包括左面的波导面约束
        """
        coord_min = self.config["coord_min"]
        coord_max = self.config["coord_max"]
        batch_size, _ = data.shape
        # mask is used to select data.
        mask = ms_np.zeros(shape=(batch_size, 17), dtype=ms.dtype.float32)

        mask[:, 0] = ms_np.where(ms_np.isclose(
            data[:, 0], coord_min[0]), 1.0, 0.0)  # left
        mask[:, 1] = ms_np.where(ms_np.isclose(
            data[:, 0], coord_min[0]), 1.0, 0.0)  # left
        mask[:, 2] = ms_np.where(ms_np.isclose(
            data[:, 0], coord_min[0]), 1.0, 0.0)  # left

        mask[:, 3] = ms_np.where(ms_np.isclose(
            data[:, 0], coord_max[0]), 1.0, 0.0)  # right
        mask[:, 4] = ms_np.where(ms_np.isclose(
            data[:, 0], coord_max[0]), 1.0, 0.0)  # right

        mask[:, 5] = ms_np.where(ms_np.isclose(
            data[:, 1], coord_min[1]), 1.0, 0.0)  # bottom
        mask[:, 6] = ms_np.where(ms_np.isclose(
            data[:, 1], coord_min[1]), 1.0, 0.0)  # bottom
        mask[:, 7] = ms_np.where(ms_np.isclose(
            data[:, 1], coord_min[1]), 1.0, 0.0)  # bottom

        mask[:, 8] = ms_np.where(ms_np.isclose(
            data[:, 1], coord_max[1]), 1.0, 0.0)  # top
        mask[:, 9] = ms_np.where(ms_np.isclose(
            data[:, 1], coord_max[1]), 1.0, 0.0)  # top
        mask[:, 10] = ms_np.where(ms_np.isclose(
            data[:, 1], coord_max[1]), 1.0, 0.0)  # top

        mask[:, 11] = ms_np.where(ms_np.isclose(
            data[:, 2], coord_min[2]), 1.0, 0.0)  # back
        mask[:, 12] = ms_np.where(ms_np.isclose(
            data[:, 2], coord_min[2]), 1.0, 0.0)  # back
        mask[:, 13] = ms_np.where(ms_np.isclose(
            data[:, 2], coord_min[2]), 1.0, 0.0)  # back

        mask[:, 14] = ms_np.where(ms_np.isclose(
            data[:, 2], coord_max[2]), 1.0, 0.0)  # front
        mask[:, 15] = ms_np.where(ms_np.isclose(
            data[:, 2], coord_max[2]), 1.0, 0.0)  # front
        mask[:, 16] = ms_np.where(ms_np.isclose(
            data[:, 2], coord_max[2]), 1.0, 0.0)  # front

        # Waveguide-port
        # Ground-truth: Ex = sin(m * pi * y / height) * sin(m * pi * y / length)
        height = coord_max[1]
        length = coord_max[2]
        # data[:,0]->x, data[:,1]->y, data[:,2]->z
        label_left = ops.sin(self.eigenmode * self.pi * data[:, 1:2] / height) * \
            ops.sin(self.eigenmode * self.pi * data[:, 2:3] / length)
        # Give a weight 10.0 to emphasize the loss.
        bc_r_waveguide = 10.0 * (u[:, 2:3] - label_left)
        bc_r_left = self.concat(
            [u[:, 0:1], u[:, 1:2], bc_r_waveguide])

        # The loss of right ABC plane
        # n x \nabla x E = 0 ==> dEz/dy - dEy/dz = 0
        ex_z = self.grad(data, 2, 0, u)
        ez_x = self.grad(data, 0, 2, u)
        ey_x = self.grad(data, 0, 1, u)
        ex_y = self.grad(data, 1, 0, u)
        bc_r_right = self.concat([ex_z - ez_x, ey_x - ex_y])

        # The loss of 4 PEC planes: n x E = 0
        # bottom: Ez=0, Ex = 0, dEy/dy=0
        # top: Ez, Ex = 0, dEy/dy=0
        # back: Ex, Ey = 0, dEz/dz=0
        # front: Ex, Ey = 0, dEz/dz=0
        ey_y = self.grad(data, 1, 1, u)
        ez_z = self.grad(data, 2, 2, u)
        bc_r_round = self.concat(
            (u[:, 2:3], u[:, 0:1], ey_y,
             u[:, 2:3], u[:, 0:1], ey_y,
             u[:, 0:1], u[:, 1:2], ez_z,
             u[:, 0:1], u[:, 1:2], ez_z),
        )

        bc_r_all = self.concat((bc_r_left, bc_r_right, bc_r_round))
        bc_r = self.mul(bc_r_all, mask)
        loss_bc = self.reduce_mean(self.l2_loss(bc_r, self.zeros_like(bc_r)))
        return loss_bc
