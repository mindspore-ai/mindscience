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
"""
Model of Boltzmann equation
"""
import numpy as np

import mindspore as ms
from mindspore import nn, ops
from mindspore.ops.operations import math_ops

from src.utils import init_kernel_mode_vector, get_vdis, mesh_nd
from src.cells import Maxwellian, RhoUTheta, MtlLoss, JacFwd, PrimNorm


class BGKKernel(nn.Cell):
    """collision kernel for BGK model"""

    def __init__(self, vmin, vmax, nv):
        super().__init__()
        v, w = mesh_nd(vmin, vmax, nv)
        vdis = ms.Tensor(v.astype(np.float32))
        wdis = ms.Tensor(w.astype(np.float32))
        self.maxwellian_nd = Maxwellian(vdis)
        self.rho_u_theta = RhoUTheta(vdis, wdis)

    def construct(self, f, kn=1):
        """return the collision term"""
        rho, u, theta = self.rho_u_theta(f)
        f_m = self.maxwellian_nd(rho, u, theta)
        return 1 / kn * (f_m - f)


class FSMKernel(nn.Cell):
    """collision kernel for FSM model"""

    def __init__(self, vmin, vmax, nv, quad_num=64, omega=0.81, m=5):
        super().__init__()
        phi, psi, phipsi = init_kernel_mode_vector(
            vmax[0],
            vmin[0],
            nv[0],
            vmax[1],
            vmin[1],
            nv[1],
            vmax[2],
            vmin[2],
            nv[2],
            quad_num,
            omega,
            m,
            np.float32,
        )
        self.phi = ms.Tensor(phi.astype(np.float32))
        self.psi = ms.Tensor(psi.astype(np.float32))
        self.phipsi = ms.Tensor(phipsi.astype(np.float32))
        self.num = nv[0] * nv[1] * nv[2]
        self.nv = (nv[0], nv[1], nv[2])
        self.fft3d = math_ops.FFTWithSize(signal_ndim=3,
                                          inverse=False,
                                          real=False,
                                          norm="backward")
        self.ifft3d = math_ops.FFTWithSize(signal_ndim=3,
                                           inverse=True,
                                           real=False,
                                           norm="backward")
        self.m = m

    def construct(self, f_ms, kn_bzm=1):
        """return the collision term"""
        f_ms = f_ms.reshape((f_ms.shape[0],) + self.nv)
        f_spec = ops.Cast()(f_ms, ms.complex64)
        f_spec = self.ifft3d(f_spec)
        f_spec = f_spec / ops.Cast()(self.num, ms.complex64)

        f_spec = fftshift_ms(f_spec)
        f_temp = ops.Cast()(0.0, ms.complex64)
        m = self.m

        for i in range(1, M - 1 + 1):
            for j in range(1, M + 1):
                fc1 = f_spec * self.phi[:, :, :, i - 1, j - 1]
                fc2 = f_spec * self.psi[:, :, :, i - 1, j - 1]
                fc11 = self.fft3d(fc1)
                fc22 = self.fft3d(fc2)
                f_temp = f_temp + fc11 * fc22

        fc1 = f_spec * self.phipsi
        fc2 = f_spec
        fc11 = self.fft3d(fc1)
        fc22 = self.fft3d(fc2)
        f_temp = f_temp - fc11 * fc22
        q = 4.0 * 3.1415926**2.0 / kn_bzm / m**2 * ops.Real()(f_temp)
        return q.reshape((-1, self.num))


class BoltzmannBGK(nn.Cell):
    """The Boltzmann BGK model"""

    def __init__(self,
                 net,
                 kn,
                 vconfig,
                 iv_weight=100,
                 bv_weight=100,
                 pde_weight=10):
        super().__init__()
        self.net = net
        self.kn = kn

        vdis, wdis = get_vdis(vconfig)

        self.vdis = vdis
        loss_num = 3 * (vdis.shape[0] + 1 + 2 * vdis.shape[-1])
        self.mtl = MtlLoss(loss_num)
        self.jac = JacFwd(self.net)
        self.iv_weight = iv_weight
        self.bv_weight = bv_weight
        self.pde_weight = pde_weight
        self.maxwellian_nd = Maxwellian(vdis)
        self.rho_u_theta = RhoUTheta(vdis, wdis)
        self.criterion_norm = lambda x: ops.square(x).mean(axis=0)
        self.criterion = lambda x, y: ops.square(x - y).mean(axis=0)
        self.prim_norm = PrimNorm(vdis, wdis)
        self.collision = BGKKernel(vconfig["vmin"], vconfig["vmax"],
                                   vconfig["nv"])

    def governing_equation(self, inputs):
        f, fxft = self.jac(inputs)
        fx, ft = fxft[0], fxft[1]
        pde = ft + self.vdis[..., 0] * fx - self.collision(f, self.kn)
        return pde

    def boundary_condition(self, bv_points1, bv_points2):
        fl = self.net(bv_points1)
        fr = self.net(bv_points2)
        return fl - fr

    def initial_condition(self, inputs):
        iv_pred = self.net(inputs)
        iv_x = inputs[..., 0:1]
        rho_l = ops.sin(2 * np.pi * iv_x) * 0.5 + 1
        u_l = ops.zeros((iv_x.shape[0], 3), ms.float32)
        theta_l = ops.sin(2 * np.pi * iv_x + 0.2) * 0.5 + 1
        iv_truth = self.maxwellian_nd(rho_l, u_l, theta_l)
        return iv_pred - iv_truth

    def loss_fn(self, inputs):
        """the loss function"""
        return self.criterion_norm(inputs), self.prim_norm(inputs)

    def construct(self, domain_points, iv_points, bv_points1, bv_points2):
        """combined all loss function"""
        pde = self.governing_equation(domain_points)
        iv = self.initial_condition(iv_points)
        bv = self.boundary_condition(bv_points1, bv_points2)

        loss_pde = self.pde_weight * self.criterion_norm(pde)
        loss_pde2 = self.pde_weight * self.prim_norm(pde)

        loss_bv = self.bv_weight * self.criterion_norm(bv)
        loss_bv2 = self.bv_weight * self.prim_norm(bv)

        loss_iv = self.iv_weight * self.criterion_norm(iv)
        loss_iv2 = self.iv_weight * self.prim_norm(iv)

        loss_sum = self.mtl(
            ops.concat(
                [loss_iv, loss_iv2, loss_bv, loss_bv2, loss_pde, loss_pde2],
                axis=-1))
        return loss_sum, (loss_iv, loss_iv2, loss_bv, loss_bv2, loss_pde,
                          loss_pde2)


class BoltzmannFSM(BoltzmannBGK):
    """The BoltzmannFSM model"""

    def __init__(self,
                 net,
                 kn,
                 vconfig,
                 IV_weight=100,
                 BV_weight=100,
                 PDE_weight=10,
                 omega=0.81):
        super().__init__(net, kn, vconfig, IV_weight, BV_weight, PDE_weight)
        self.collision = FSMKernel(vconfig["vmin"],
                                   vconfig["vmax"],
                                   vconfig["nv"],
                                   omega=omega)
