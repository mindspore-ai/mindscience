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
maxwell function and its loss
"""
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.numpy as ms_np
from mindspore import ms_function
from mindspore import Tensor

from mindelec.operators import SecondOrderGrad, Grad


class MaxwellCavity(nn.Cell):
    """
    Define the PINNs loss network, which is also the loss of PDEs.
    """

    def __init__(self, net, config):
        super(MaxwellCavity, self).__init__(auto_prefix=False)
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

        self.slab_len = Tensor(config["slab_len"], ms.dtype.float32)
        # permitivity in slab
        self.eps1 = Tensor(config["eps1"], ms.dtype.float32)
        # permitivity in vacuum
        self.eps0 = Tensor(config["eps0"], ms.dtype.float32)
        self.wave_number = Tensor(config["wave_number"], ms.dtype.float32)

        # the loss coefficients of different parts, self define
        self.gamma_domain = Tensor(
            1.0 / self.wave_number ** 2, ms.dtype.float32)
        self.gamma_bc = Tensor(100.0, ms.dtype.float32)
        self.gamma_port = Tensor(100.0, ms.dtype.float32)

    def construct(self, in_domain, in_bc, in_port):
        """
        Calculate the loss according to input data.
        """
        u_domain = self.net(in_domain)
        loss_domain = self.get_domain_loss(u_domain, in_domain)

        u_bc = self.net(in_bc)
        loss_bc = self.get_boundary_loss(u_bc, in_bc)

        u_port = self.net(in_port[:, 0:3])
        loss_port = self.get_port_loss(u_port, in_port)

        return self.gamma_domain * loss_domain + self.gamma_bc * loss_bc \
            + self.gamma_port * loss_port

    @ms_function
    def get_domain_loss(self, u, data):
        """
        Domain loss.
        """
        # Second order operator
        ex_xx = self.hessian_ex_xx(data)
        ex_yy = self.hessian_ex_yy(data)
        ex_zz = self.hessian_ex_zz(data)

        ey_xx = self.hessian_ey_xx(data)
        ey_yy = self.hessian_ey_yy(data)
        ey_zz = self.hessian_ey_zz(data)

        ez_xx = self.hessian_ez_xx(data)
        ez_yy = self.hessian_ez_yy(data)
        ez_zz = self.hessian_ez_zz(data)

        batch_size, _ = data.shape
        mask = ms_np.zeros(shape=(batch_size, 3), dtype=ms.dtype.float32)
        # The mask of slab area is 1.0
        mask = ms_np.where(data[:, 1] > -self.slab_len/2, 1.0, 0)
        mask += ms_np.where(data[:, 1] < self.slab_len/2, 1.0, 0)
        mask += ms_np.where(data[:, 2] > -self.slab_len/2, 1.0, 0)
        mask += ms_np.where(data[:, 2] < self.slab_len/2, 1.0, 0)
        # If a points is in slab, the upper 4 conditions is satisfied,
        # so the value is 4.0, reset it to 1.0 .
        mask = ms_np.where(ms_np.equal(mask, 4.0), 1.0, 0)
        # reshaape the mask to 1 column
        mask = self.reshape(mask, (batch_size, 1))

        # The permitivity in slab is eps1
        pde_rx_slab = ex_xx + ex_yy + ex_zz + \
            self.wave_number**2 * self.eps1 * u[:, 0:1]
        pde_ry_slab = ey_xx + ey_yy + ey_zz + \
            self.wave_number**2 * self.eps1 * u[:, 1:2]
        pde_rz_slab = ez_xx + ez_yy + ez_zz + \
            self.wave_number**2 * self.eps1 * u[:, 2:3]

        # The permitivity in vacuum is eps0
        pde_rx_vac = ex_xx + ex_yy + ex_zz + \
            self.wave_number**2 * self.eps0 * u[:, 0:1]
        pde_ry_vac = ey_xx + ey_yy + ey_zz + \
            self.wave_number**2 * self.eps0 * u[:, 1:2]
        pde_rz_vac = ez_xx + ez_yy + ez_zz + \
            self.wave_number**2 * self.eps0 * u[:, 2:3]

        pde_r_slab = self.concat((pde_rx_slab, pde_ry_slab, pde_rz_slab))
        pde_r_vac = self.concat((pde_rx_vac, pde_ry_vac, pde_rz_vac))

        # Multiply mask, mask.shape=(bs,1), pde_r_slab.shape=(bs,3), pde_r_vac.shape=(bs,3)
        # the broadcast mechanics is used.
        ones = ms_np.ones_like(mask)
        domain_r = mask * pde_r_slab + (ones - mask) * pde_r_vac

        loss_domain = self.reduce_mean(
            self.l2_loss(domain_r, self.zeros_like(domain_r)))

        # The divergence where no source is zero.
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
        The loss on boundary.
        """
        coord_min = self.config["coord_min"]
        coord_max = self.config["coord_max"]
        batch_size, _ = data.shape
        # Use mask to select the boundary points.
        mask = ms_np.zeros(shape=(batch_size, 14), dtype=ms.dtype.float32)

        mask[:, 0] = ms_np.where(ms_np.equal(
            data[:, 0], coord_max[0]), 1.0, 0.0)  # right
        mask[:, 1] = ms_np.where(ms_np.equal(
            data[:, 0], coord_max[0]), 1.0, 0.0)  # right

        mask[:, 2] = ms_np.where(ms_np.equal(
            data[:, 1], coord_min[1]), 1.0, 0.0)  # bottom
        mask[:, 3] = ms_np.where(ms_np.equal(
            data[:, 1], coord_min[1]), 1.0, 0.0)  # bottom
        mask[:, 4] = ms_np.where(ms_np.equal(
            data[:, 1], coord_min[1]), 1.0, 0.0)  # bottom

        mask[:, 5] = ms_np.where(ms_np.equal(
            data[:, 1], coord_max[1]), 1.0, 0.0)  # top
        mask[:, 6] = ms_np.where(ms_np.equal(
            data[:, 1], coord_max[1]), 1.0, 0.0)  # top
        mask[:, 7] = ms_np.where(ms_np.equal(
            data[:, 1], coord_max[1]), 1.0, 0.0)  # top

        mask[:, 8] = ms_np.where(ms_np.equal(
            data[:, 2], coord_min[2]), 1.0, 0.0)  # back
        mask[:, 9] = ms_np.where(ms_np.equal(
            data[:, 2], coord_min[2]), 1.0, 0.0)  # back
        mask[:, 10] = ms_np.where(ms_np.equal(
            data[:, 2], coord_min[2]), 1.0, 0.0)  # back

        mask[:, 11] = ms_np.where(ms_np.equal(
            data[:, 2], coord_max[2]), 1.0, 0.0)  # front
        mask[:, 12] = ms_np.where(ms_np.equal(
            data[:, 2], coord_max[2]), 1.0, 0.0)  # front
        mask[:, 13] = ms_np.where(ms_np.equal(
            data[:, 2], coord_max[2]), 1.0, 0.0)  # front

        # The ABC loss on right face.
        # n x \nabla x E = 0 ==> dEz/dy - dEy/dz = 0
        ex_z = self.grad(data, 2, 0, u)
        ez_x = self.grad(data, 0, 2, u)

        ey_x = self.grad(data, 0, 1, u)
        ex_y = self.grad(data, 1, 0, u)

        bc_r_right = self.concat([ex_z - ez_x, ey_x - ex_y])

        # The PEC loss, n x E = 0
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

        bc_r_all = self.concat((bc_r_right, bc_r_round))
        bc_r = self.mul(bc_r_all, mask)

        loss_bc = self.reduce_mean(self.l2_loss(bc_r, self.zeros_like(bc_r)))
        return loss_bc

    @ms_function
    def get_port_loss(self, u, data):
        """
        The loss on waveguide port.
        """
        uz = u[:, 2:3]          # Ez prediction.
        label = data[:, 3:4]    # Ez label.
        # Ground truth: Ex=0, Ey=0, Ez=label
        waveguide = self.concat([u[:, 0:1], u[:, 1:2], uz - label])

        loss_port = self.reduce_mean(self.l2_loss(
            waveguide, self.zeros_like(waveguide)))
        return loss_port
