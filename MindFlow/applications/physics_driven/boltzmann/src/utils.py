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
util functions
"""
import math
import numpy as np
from scipy import special
import matplotlib.pyplot as plt

import mindspore as ms
from mindspore.common.initializer import Initializer, _assignment
import mindspore.numpy as mnp


def fftshift_ms(x, axes=None):
    """fftshift function as the same in numpy"""
    if axes is None:
        axes = tuple(range(x.ndim))
        shift = [dim // 2 for dim in x.shape]
    elif isinstance(axes, int):
        shift = x.shape[axes] // 2
    else:
        shift = [x.shape[ax] // 2 for ax in axes]

    return mnp.roll(x, shift, axes)


def get_gamma(ck):
    """get the gamma for gas"""
    gamma = (ck + 5) / (ck + 3)
    return gamma


def get_potential(omega):
    """get the potential index for given omega"""
    if omega == 0.5:
        alpha = 1.0
    else:
        eta = 4.0 / (2.0 * omega - 1.0) + 1.0
        alpha = (eta - 5.0) / (eta - 1.0)
    return alpha


def get_mu(alpha, omega, kn):
    """get the viscosity of gas"""
    mu = (5 * (alpha + 1) * (alpha + 2) * math.sqrt(math.pi) /
          (4 * alpha * (5 - 2 * omega) * (7 - 2 * omega)) * kn)
    return mu


def get_kn_bzm(alpha, mu_ref):
    """return the normalized kn"""
    kn_bzm = (64 * math.sqrt(2.0)**alpha / 5.0 * math.gamma(
        (alpha + 3) / 2) * math.gamma(2.0) * math.sqrt(math.pi) * mu_ref)
    return kn_bzm


def _calculate_in(arr):
    """get the input channels"""
    n_in = arr.shape[-1]
    return n_in


class SimpleUniformInitializer(Initializer):

    def _initialize(self, arr):
        n_in = _calculate_in(arr)
        boundary = math.sqrt(1 / n_in)
        data = np.random.uniform(-boundary, boundary, arr.shape)
        _assignment(arr, data)


def lgwt(n: int, a: float, b: float):
    """Gauss-Legendre quadrature
    Args:
        N (int): points
        a (float): left
        b (float): right
    Returns:
        Tuple[np.Tensor,np.Tensor]: quadrature points, weight
    """
    x, w = np.polynomial.legendre.leggauss(n)
    x = 0.5 * (x + 1) * (b - a) + a
    w = w * 0.5 * (b - a)
    return x, w


def init_kernel_mode_vector(
        umax,
        umin,
        unum,
        vmax,
        vmin,
        vnum,
        wmax,
        wmin,
        wnum,
        quad_num=64,
        omega=0.81,
        m=5,
        dtype=np.float64,
):
    """Get the collision kernel."""
    pi = math.pi
    alpha = get_potential(omega)
    du, dv, dw = (
        (umax - umin) / (unum - 1),
        (vmax - vmin) / (vnum - 1),
        (wmax - wmin) / (wnum - 1),
    )
    supp = math.sqrt(2.0) * 2.0 * max(umax, vmax, wmax) / \
        (3.0 + math.sqrt(2.0))

    fre_vx = np.linspace(-pi / du, (unum / 2 - 1.0) * 2.0 * pi / unum / du,
                         unum)
    fre_vy = np.linspace(-pi / dv, (vnum / 2 - 1.0) * 2.0 * pi / vnum / dv,
                         vnum)
    fre_vz = np.linspace(-pi / dw, (wnum / 2 - 1.0) * 2.0 * pi / wnum / dw,
                         wnum)

    abscissa, gweight = lgwt(quad_num, 0.0, supp)
    theta = pi / m * np.arange(1, m - 1 + 1)
    theta2 = pi / m * np.arange(1, m + 1)

    s = ((fre_vx[:, None, None] * np.sin(theta)[None, :, None] *
          np.cos(theta2)[None, None, :])[:, None, None, :, :] +
         (fre_vy[:, None, None] * np.sin(theta)[None, :, None] *
          np.sin(theta2)[None, None, :])[None, :, None, :, :] +
         (fre_vz[:, None, None] * np.cos(theta)[None, :, None])[None,
                                                                None, :, :, :])

    int_temp = (
        2 * gweight[..., None, None, None, None, None] *
        np.cos(s[None, ...] * abscissa[..., None, None, None, None, None]) *
        (abscissa[..., None, None, None, None, None]**alpha)).sum(axis=0)
    phi2 = int_temp * np.sin(theta[None, None, None, :, None])

    s = ((fre_vx * fre_vx)[:, None, None, None, None] +
         (fre_vy * fre_vy)[None, :, None, None, None] +
         (fre_vz * fre_vz)[None, None, :, None, None] - s * s)

    psi2 = np.zeros((unum, vnum, wnum, m - 1, m), dtype=dtype)

    so = s.copy()
    s = np.abs(s)
    s = np.sqrt(s)
    bel = supp * s
    bessel = special.jv(1, bel)

    psi2 = pi * supp * supp * np.ones_like(s)
    np.divide(2.0 * pi * supp * bessel, s, out=psi2, where=so > 0)

    phipsi2 = (phi2 * psi2).sum(axis=(-1, -2))

    return phi2, psi2, phipsi2


def fvmlinspace(vmin, vmax, nv):
    dv = (vmax - vmin) / nv
    return np.linspace(vmin + dv / 2, vmax - dv / 2, nv)


def vmsh(vmin, vmax, nv):
    assert vmax > vmin
    assert nv > 0
    v = fvmlinspace(vmin, vmax, nv)
    w = (vmax - vmin) / nv
    return v, w


def mesh_nd(vmin, vmax, nv, return_list=False):
    """generate Nd mesh.

    Args:
        vmin (number or tuple of numbers): the lower bound of the velocity
        vmax (number or tuple of numbers): the higher bound of the velocity
        nv (int or tuple of ints): the number of discrete points in each direction
    """
    if (isinstance(vmin, (int, float)) and isinstance(vmax, (int, float))
            and isinstance(nv, int)):
        v, w = vmsh(vmin, vmax, nv)
        vlist, wlist = [
            v,
        ], [
            w,
        ]
    else:
        assert len(vmin) == len(vmax) == len(
            nv), "vmin,vmax,nv must be the same length"
        vlist, wlist = list(
            zip(*[
                vmsh(vmini, vmaxi, nvi)
                for vmini, vmaxi, nvi in zip(vmin, vmax, nv)
            ]))
        v = np.meshgrid(*vlist, indexing="ij")
        v = np.stack([vi.flatten() for vi in v], axis=-1)
        w = np.multiply.reduce(wlist)
    return (vlist, wlist) if return_list else (v, w)


def maxwellian_nd(v: np.array, rho: np.array, u: np.array, theta: np.array):
    """generate ND maxwellian VDF

    Args:
        v (np.array): [Nv,D] array
        rho (np.array): [N,1] array
        u (np.array): [N,D] array
        T (np.array): [N,1] array

    Returns:
        np.array: [N,Nv] array
    """
    return (rho / np.sqrt(2 * math.pi * theta)**v.shape[-1]) * np.exp(-(
        (u[..., None, :] - v)**2).sum(axis=-1) / (2 * theta))


def maxwellian(v: np.array, rho: np.array, u: np.array, theta: np.array):
    """generate 1D maxwellian VDF

    Args:
        v (np.array): [Nv] array
        rho (np.array): [N,1] array
        u (np.array): [N,1] array
        T (np.array): [N,1] array

    Returns:
        np.array: [N,Nv] array
    """
    f = rho / np.sqrt(2 * math.p * theta) * \
        np.exp(-((u - v) ** 2) / (2 * theta))
    return f


def get_mesh(vconfig, return_list=False):
    nv = vconfig["nv"]
    vmin = vconfig["vmin"]
    vmax = vconfig["vmax"]
    return mesh_nd(vmin, vmax, nv, return_list)


def get_vdis(vconfig):
    v, w = get_mesh(vconfig)
    vdis = ms.Tensor(v.astype(np.float32))
    wdis = ms.Tensor(w.astype(np.float32))
    return vdis, wdis


def get_vtuple(vconfig):
    vlist, wlist = get_mesh(vconfig, return_list=True)
    vtuple = tuple([ms.Tensor(v.astype(np.float32)) for v in vlist])
    wtuple = tuple([ms.Tensor(w) for w in wlist])
    return vtuple, wtuple


def visual(problem, resolution=100, filename="result.jpg"):
    """visualize the results"""
    x = np.linspace(-0.5, 0.5, resolution)
    t0 = 0.0 * np.ones_like(x)
    t1 = 0.1 * np.ones_like(x)
    xt0 = ms.Tensor(np.stack((x, t0), axis=-1).astype(np.float32))
    xt1 = ms.Tensor(np.stack((x, t1), axis=-1).astype(np.float32))
    rho0, u0, theta0 = problem.rho_u_theta(problem.net(xt0))
    rho1, u1, theta1 = problem.rho_u_theta(problem.net(xt1))
    fig, ax = plt.subplots(1, 2, figsize=(10, 3))
    ax[0].plot(x, rho0, label=r"$\rho$")
    ax[0].plot(x, u0[..., 0], label="$u_x$")
    ax[0].plot(x, theta0, label="T")
    ax[0].legend()
    ax[1].plot(x, rho1, label=r"$\rho$")
    ax[1].plot(x, u1[..., 0], label="$u_x$")
    ax[1].plot(x, theta1, label="T")
    ax[1].legend()
    fig.savefig(filename)
    return fig
