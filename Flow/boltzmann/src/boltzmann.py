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
import scipy

import mindspore as ms
from mindspore import nn, ops

from src.utils import (
    init_kernel_mode_vector,
    get_vdis,
    mesh_nd,
    fftshift_ms,
    orthonormalize,
    get_new_kernel,
    get_vtuple,
)
from src.cells import (
    dis_lowrank_add,
    dis_lowrank_sub,
    Maxwellian,
    RhoUTheta,
    MtlLoss,
    JacFwd,
    PrimNorm,
    FFT3D,
    IFFT3D,
    ComplexMul,
    AdaptiveMSE,
    JacFwdLR,
    MaxwellianLR,
    RhoUThetaLr,
    PrimNormLR,
    rho_u_theta_lowrank,
)


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


class FBGKKernel(nn.Cell):
    """collision kernel for Full BGK model"""

    def __init__(self, vmin, vmax, nv, omega):
        super().__init__()
        v, w = mesh_nd(vmin, vmax, nv)
        vdis = ms.Tensor(v.astype(np.float32))
        wdis = ms.Tensor(w.astype(np.float32))
        self.maxwellian_nd = Maxwellian(vdis)
        self.rho_u_theta = RhoUTheta(vdis, wdis)
        self.omega = omega

    def construct(self, f, mu_ref=1):
        """return the collision term"""
        rho, u, theta = self.rho_u_theta(f)
        f_m = self.maxwellian_nd(rho, u, theta)
        tau = mu_ref * 2 / (theta * 2) ** (1 - self.omega) / rho
        return 1 / tau * (f_m - f)


class FSMKernel(nn.Cell):
    """Fast spectral method kernel"""
    def __init__(self, vmin, vmax, nv, quad_num=64, omega=0.81, m=5):
        super().__init__()
        phi, psi, phipsi = init_kernel_mode_vector(
            vmin, vmax, nv, quad_num, omega, m, np.float32
        )
        self.phi = ms.Tensor(phi.astype(np.float32))
        self.psi = ms.Tensor(psi.astype(np.float32))
        self.phipsi = ms.Tensor(phipsi.astype(np.float32))
        self.num = nv[0] * nv[1] * nv[2]
        self.nv = (nv[0], nv[1], nv[2])
        self.fft3d = FFT3D()
        self.ifft3d = IFFT3D()
        self.m = m
        self.cast = ops.Cast()
        self.complex = ops.Complex()
        self.view_as_real = ops.view_as_real
        self.complex_mul = ComplexMul()

    def construct(self, f_ms, kn_bzm=1):
        """construct"""
        f_ms = f_ms.reshape((f_ms.shape[0],) + self.nv)
        f_spec = self.cast(f_ms, ms.complex64)
        f_spec = self.view_as_real(f_spec)

        f_spec = self.ifft3d(f_spec)
        f_spec = f_spec / self.num

        f_spec = fftshift_ms(f_spec, axes=(-4, -3, -2))
        f_temp = 0.0

        for i in range(1, self.m - 1 + 1):
            for j in range(1, self.m + 1):
                fc1 = f_spec * self.phi[:, :, :, i - 1, j - 1, None]
                fc2 = f_spec * self.psi[:, :, :, i - 1, j - 1, None]
                fc11 = self.fft3d(fc1)
                fc22 = self.fft3d(fc2)
                f_temp = f_temp + self.complex_mul(fc11, fc22)

        fc1 = f_spec * self.phipsi[..., None]
        fc2 = f_spec
        fc11 = self.fft3d(fc1)
        fc22 = self.fft3d(fc2)
        f_temp = f_temp - self.complex_mul(fc11, fc22)
        q = 4.0 * 3.141592653589793**2.0 / \
            kn_bzm / self.m**2 * (f_temp[..., 0])
        return q.reshape((-1, self.num))


class FSMLAKernel(nn.Cell):
    """fsm lineat approximation kernel"""
    def __init__(self, f, k, g):
        super().__init__()
        self.f = f
        self.k = k
        self.g = g

    def construct(self, x, kn_bzm=1.0):
        ff = x @ self.f
        q = ops.einsum("...i,...j,ijk->...k", ff, ff, self.k)
        qr = q @ self.g.T
        return qr / kn_bzm


class BoltzmannEqu(nn.Cell):
    """The Boltzmann BGK model"""

    def __init__(
            self,
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
        self.collision = BGKKernel(
            vconfig["vmin"],
            vconfig["vmax"],
            vconfig["nv"])

    @ms.jit
    def pred(self, xt):
        f = self.net(xt)
        return self.rho_u_theta(f)

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
                [loss_iv, loss_iv2, loss_bv, loss_bv2, loss_pde, loss_pde2], axis=-1
            )
        )
        return loss_sum, (loss_iv, loss_iv2, loss_bv,
                          loss_bv2, loss_pde, loss_pde2)


class BoltzmannBGK(BoltzmannEqu):
    """The BoltzmannBGK model"""

    def __init__(
            self,
            net,
            kn,
            vconfig,
            iv_weight=100,
            bv_weight=100,
            pde_weight=10,
            omega=0.81):
        super().__init__(net, kn, vconfig, iv_weight, bv_weight, pde_weight)
        self.collision = BGKKernel(
            vconfig["vmin"],
            vconfig["vmax"],
            vconfig["nv"])
        self.omega = omega


class BoltzmannFBGK(BoltzmannEqu):
    """The BoltzmannFBGK model"""

    def __init__(
            self,
            net,
            kn,
            vconfig,
            iv_weight=100,
            bv_weight=100,
            pde_weight=10,
            omega=0.81):
        super().__init__(net, kn, vconfig, iv_weight, bv_weight, pde_weight)
        self.collision = FBGKKernel(
            vconfig["vmin"], vconfig["vmax"], vconfig["nv"], omega=omega
        )


class BoltzmannFSM(BoltzmannEqu):
    """The BoltzmannFSM model"""

    def __init__(
            self,
            net,
            kn,
            vconfig,
            iv_weight=100,
            bv_weight=100,
            pde_weight=10,
            omega=0.81):
        super().__init__(net, kn, vconfig, iv_weight, bv_weight, pde_weight)
        self.collision = FSMKernel(
            vconfig["vmin"], vconfig["vmax"], vconfig["nv"], omega=omega
        )


class BoltzmannLA(BoltzmannEqu):
    """linear approximation collision boltzmann equation"""
    def __init__(
            self,
            net,
            kn,
            vconfig,
            f,
            k,
            g,
            iv_weight=100,
            bv_weight=100,
            pde_weight=1,
            omega=0.81,
    ):
        super().__init__(net, kn, vconfig, iv_weight, bv_weight, pde_weight)
        self.collision = FSMLAKernel(f, k, g)
        self.omega = omega


class BoltzmannLR(nn.Cell):
    """low rank approximation collision boltzmann equation"""
    def __init__(
            self,
            net,
            kn,
            vconfig,
            iv_weight=100,
            bv_weight=100,
            pde_weight=10):
        super().__init__()
        self.net = net
        self.kn = kn
        vdis, _ = get_vdis(vconfig)
        vtuple, wtuple = get_vtuple(vconfig)

        self.vdis = vdis
        self.vtuple = vtuple
        self.wtuple = wtuple
        self.iv_weight = iv_weight
        self.bv_weight = bv_weight
        self.pde_weight = pde_weight

        self.nvx = self.vtuple[0].shape[0]
        self.nvy = self.vtuple[1].shape[0]
        self.nvz = self.vtuple[2].shape[0]
        self.adaptive_loss_pde = AdaptiveMSE(self.nvx, self.nvy, self.nvz)
        self.adaptive_loss_iv = AdaptiveMSE(self.nvx, self.nvy, self.nvz)
        self.adaptive_loss_bv = AdaptiveMSE(self.nvx, self.nvy, self.nvz)
        self.adaptive_loss_prim = MtlLoss(3 * 7)

        self.jac = JacFwdLR(self.net)
        self.maxwellian = MaxwellianLR(vtuple)
        self.rho_u_theta = RhoUThetaLr(self.vtuple, self.wtuple)
        self.criterion_norm = lambda x: ops.square(x).mean(axis=0)
        self.criterion = lambda x, y: ops.square(x - y).mean(axis=0)
        self.primnorm = PrimNormLR(vtuple, wtuple)
        self.collision = BGKKernel(
            vconfig["vmin"],
            vconfig["vmax"],
            vconfig["nv"])
        self.knr3 = kn ** (1 / 3)

    @ms.jit
    def pred(self, xt):
        ft = self.net(xt)
        return self.rho_u_theta(ft)

    def governing_equation(self, inputs):
        """governing equation"""
        f, fxft = self.jac(inputs)
        p, q, r = f
        vx = self.vtuple[0]
        knr3 = self.knr3
        p_x, q_x, r_x = fxft[0]
        p_t, q_t, r_t = fxft[1]
        f_t = ((p_t, q, r), (p, q_t, r), (p, q, r_t))
        vf_x = (
            (p_x * vx[..., None], q, r),
            (p * vx[..., None], q_x, r),
            (p * vx[..., None], q, r_x),
        )

        rho, u, theta = rho_u_theta_lowrank(f, self.vtuple, self.wtuple)
        f_mx, f_my, f_mz = self.maxwellian(rho, u, theta)

        f_m = ((-1 / knr3 * f_mx, -1 / knr3 * f_my, -1 / knr3 * f_mz),)
        f = ((1 / knr3 * p, 1 / knr3 * q, 1 / knr3 * r),)

        return dis_lowrank_add(f_t + vf_x + f_m + f)

    def boundary_condition(self, bv_points1, bv_points2):
        fl = self.net(bv_points1)
        fr = self.net(bv_points2)
        return dis_lowrank_sub(fl, fr)

    def initial_condition(self, inputs):
        iv_pred = self.net(inputs)
        iv_x = inputs[..., 0:1]
        rho_l = ops.sin(2 * np.pi * iv_x) * 0.5 + 1
        u_l = ops.zeros((iv_x.shape[0], 3), ms.float32)
        theta_l = ops.sin(2 * np.pi * iv_x + 0.2) * 0.5 + 1
        iv_truth = self.maxwellian(rho_l, u_l, theta_l)

        return dis_lowrank_sub(iv_pred, iv_truth)

    def construct(self, domain_points, iv_points, bv_points1, bv_points2):
        """construct"""
        pde = self.governing_equation(domain_points)
        iv = self.initial_condition(iv_points)
        bv = self.boundary_condition(bv_points1, bv_points2)

        loss_bv, loss_bv_w = self.adaptive_loss_bv(
            (self.bv_weight * bv[0], bv[1], bv[2])
        )
        loss_iv, loss_iv_w = self.adaptive_loss_iv(
            (self.iv_weight * iv[0], iv[1], iv[2])
        )
        loss_pde, loss_pde_w = self.adaptive_loss_pde(
            (self.pde_weight * pde[0], pde[1], pde[2])
        )

        loss_bv_p = self.primnorm((self.bv_weight * bv[0], bv[1], bv[2]))
        loss_iv_p = self.primnorm((self.iv_weight * iv[0], iv[1], iv[2]))
        loss_pde_p = self.primnorm((self.pde_weight * pde[0], pde[1], pde[2]))

        loss_sum = (
            self.adaptive_loss_prim(
                ops.cat(
                    [
                        loss_bv_p,
                        loss_iv_p,
                        loss_pde_p])) +
            loss_bv +
            loss_bv_w +
            loss_iv +
            loss_iv_w +
            loss_pde +
            loss_pde_w)

        return loss_sum, (loss_iv, loss_iv_p, loss_bv,
                          loss_bv_p, loss_pde, loss_pde_p)


def get_reduced_kernel(config, traindata):
    """get the approximation kernel by data."""
    vconfig = config["vmesh"]
    rank = config["rank"]
    collision = FSMKernel(
        vconfig["vmin"], vconfig["vmax"], vconfig["nv"], omega=config["omega"]
    )

    train_tensor = ms.Tensor(traindata, dtype=ms.float32)
    q_data = collision(train_tensor, 1.0)

    s, u, vh = ops.svd(train_tensor, full_matrices=False)
    s_2, u_2, vh_2 = ops.svd(q_data, full_matrices=False)

    s = s.asnumpy()
    s_2 = s_2.asnumpy()
    u = u.asnumpy()
    u_2 = u_2.asnumpy()
    vh = vh.asnumpy()
    vh_2 = vh_2.asnumpy()

    vdis, _ = get_vdis(vconfig)
    nvprod = vdis.shape[0]

    c_rho = ops.ones((1, nvprod))
    c_veloc = vdis.mT
    c_energy = (vdis.mT**2).sum(axis=0, keepdims=True)
    c_feature = ops.cat([c_rho, c_veloc, c_energy])

    cc_feat = c_feature.asnumpy().T

    vhs = vh * s
    cc = vhs.T
    cc = np.concatenate([cc_feat.T, cc])
    vhsc = orthonormalize(cc.T).T
    f_bases = vhsc[:rank].T

    vh2s = vh_2 * s_2
    cc2 = vh2s[:, :rank]
    cc2 = cc2 - cc_feat @ (cc_feat.T @ cc2)

    vh2sc = scipy.linalg.orth(cc2, rcond=1e-10)

    f_bases2 = vh2sc

    nv = vconfig["nv"]
    vmin = vconfig["vmin"]
    vmax = vconfig["vmax"]
    phi, psi, phipsi = init_kernel_mode_vector(vmin, vmax, nv, 5)
    nk = get_new_kernel(
        f_bases,
        f_bases2,
        nv[0],
        nv[1],
        nv[2],
        1.0,
        phi,
        psi,
        phipsi)
    return f_bases, f_bases2, nk
