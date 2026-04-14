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
from numpy.fft import fftn, ifftn, fftshift
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
    mu = (
        5
        * (alpha + 1)
        * (alpha + 2)
        * math.sqrt(math.pi)
        / (4 * alpha * (5 - 2 * omega) * (7 - 2 * omega))
        * kn
    )
    return mu


def get_kn_bzm(alpha, mu_ref):
    """return the normalized kn"""
    kn_bzm = (
        64
        * math.sqrt(2.0) ** alpha
        / 5.0
        * math.gamma((alpha + 3) / 2)
        * math.gamma(2.0)
        * math.sqrt(math.pi)
        * mu_ref
    )
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
        vmin,
        vmax,
        nv,
        quad_num: int = 64,
        omega: float = 0.81,
        m=5,
        dtype=np.float64,
):
    """Get the collision kernel."""
    pi = math.pi
    alpha = get_potential(omega)
    umax, vmax, wmax = vmax
    umin, vmin, wmin = vmin
    unum, vnum, wnum = nv
    du, dv, dw = (
        (umax - umin) / (unum - 1),
        (vmax - vmin) / (vnum - 1),
        (wmax - wmin) / (wnum - 1),
    )
    supp = math.sqrt(2.0) * 2.0 * max(umax, vmax, wmax) / (3.0 + math.sqrt(2.0))

    fre_vx = np.linspace(-pi / du, (unum / 2 - 1.0) * 2.0 * pi / unum / du, unum)
    fre_vy = np.linspace(-pi / dv, (vnum / 2 - 1.0) * 2.0 * pi / vnum / dv, vnum)
    fre_vz = np.linspace(-pi / dw, (wnum / 2 - 1.0) * 2.0 * pi / wnum / dw, wnum)

    abscissa, gweight = lgwt(quad_num, 0.0, supp)
    theta = pi / m * np.arange(1, m - 1 + 1)
    theta2 = pi / m * np.arange(1, m + 1)

    s = (
        (
            fre_vx[:, None, None]
            * np.sin(theta)[None, :, None]
            * np.cos(theta2)[None, None, :]
        )[:, None, None, :, :]
        + (
            fre_vy[:, None, None]
            * np.sin(theta)[None, :, None]
            * np.sin(theta2)[None, None, :]
        )[None, :, None, :, :]
        + (fre_vz[:, None, None] * np.cos(theta)[None, :, None])[None, None, :, :, :]
    )

    int_temp = (
        2
        * gweight[..., None, None, None, None, None]
        * np.cos(s[None, ...] * abscissa[..., None, None, None, None, None])
        * (abscissa[..., None, None, None, None, None] ** alpha)
    ).sum(axis=0)
    phi2 = int_temp * np.sin(theta[None, None, None, :, None])

    s = (
        (fre_vx * fre_vx)[:, None, None, None, None]
        + (fre_vy * fre_vy)[None, :, None, None, None]
        + (fre_vz * fre_vz)[None, None, :, None, None]
        - s * s
    )

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


def collision_fft(f_spec, kn_bzm, phi, psi, phipsi) -> np.array:
    """calculate collition by fft method."""
    unum, vnum, wnum = phi.shape[:3]

    def ifft3d(x):
        return ifftn(x, axes=(-3, -2, -1), norm="forward")

    def fft3d(x):
        return fftn(x, axes=(-3, -2, -1), norm="backward")

    f_spec = ifft3d(f_spec)
    f_spec = f_spec / (unum * vnum * wnum)

    f_spec = fftshift(f_spec, axes=(-3, -2, -1))
    f_temp = 0
    m = phi.shape[-1]
    for i in range(1, m - 1 + 1):
        for j in range(1, m + 1):
            fc1 = f_spec * phi[:, :, :, i - 1, j - 1]
            fc2 = f_spec * psi[:, :, :, i - 1, j - 1]
            fc11 = fft3d(fc1)
            fc22 = fft3d(fc2)
            f_temp = f_temp + fc11 * fc22
    fc1 = f_spec * phipsi
    fc2 = f_spec
    fc11 = fft3d(fc1)
    fc22 = fft3d(fc2)
    f_temp = f_temp - fc11 * fc22
    q = 4.0 * np.pi**2 / kn_bzm / m**2 * f_temp.real
    return q


def collision_fft_fg(f_spec, g_spec, kn_bzm, phi, psi, phipsi) -> np.array:
    """calculate collition by fft method. (as a biliner function)"""
    unum, vnum, wnum = phi.shape[:3]

    def ifft3d(x):
        return ifftn(x, axes=(-3, -2, -1), norm="forward")

    def fft3d(x):
        return fftn(x, axes=(-3, -2, -1), norm="backward")

    f_spec = ifft3d(f_spec)
    g_spec = ifft3d(g_spec)
    f_spec = f_spec / (unum * vnum * wnum)
    g_spec = g_spec / (unum * vnum * wnum)

    f_spec = fftshift(f_spec, axes=(-3, -2, -1))
    g_spec = fftshift(g_spec, axes=(-3, -2, -1))
    f_temp = 0
    m = phi.shape[-1]
    for i in range(1, m - 1 + 1):
        for j in range(1, m + 1):
            fc1 = f_spec * phi[:, :, :, i - 1, j - 1]
            fc2 = g_spec * psi[:, :, :, i - 1, j - 1]
            fc11 = fft3d(fc1)
            fc22 = fft3d(fc2)
            f_temp = f_temp + fc11 * fc22
    fc1 = f_spec * phipsi
    fc2 = g_spec
    fc11 = fft3d(fc1)
    fc22 = fft3d(fc2)
    f_temp = f_temp - fc11 * fc22
    q = 4.0 * np.pi**2 / kn_bzm / m**2 * f_temp.real
    return q


def orthonormalize(vectors: np.array) -> np.array:
    """
    Orthonormalizes the vectors using gram schmidt procedure.

    Parameters:
        vectors: torch tensor, size (dimension, n_vectors)
                they must be linearly independent
    Returns:
        orthonormalized_vectors: torch tensor, size (dimension, n_vectors)
    """
    assert (
        vectors.shape[1] <= vectors.shape[0]
    ), "number of vectors must be smaller or equal to the dimension"
    orthonormalized_vectors = np.zeros_like(vectors)
    orthonormalized_vectors[:, 0] = vectors[:, 0] / np.linalg.norm(
        vectors[:, 0], axis=0, ord=2
    )

    for i in range(1, orthonormalized_vectors.shape[1]):
        vector = vectors[:, i]
        v = orthonormalized_vectors[:, :i]
        pv_vector = v @ (v.T @ vector)
        orthonormalized_vectors[:, i] = (vector - pv_vector) / np.linalg.norm(
            vector - pv_vector, axis=0, ord=2
        )

    return orthonormalized_vectors


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
    if (
            isinstance(vmin, (int, float))
            and isinstance(vmax, (int, float))
            and isinstance(nv, int)
    ):
        v, w = vmsh(vmin, vmax, nv)
        vlist, wlist = [
            v,
        ], [
            w,
        ]
    else:
        assert len(vmin) == len(vmax) == len(nv), "vmin,vmax,nv must be the same length"
        vlist, wlist = list(
            zip(*[vmsh(vmini, vmaxi, nvi) for vmini, vmaxi, nvi in zip(vmin, vmax, nv)])
        )
        v = np.meshgrid(*vlist, indexing="ij")
        v = np.stack([vi.flatten() for vi in v], axis=-1)
        w = np.multiply.reduce(wlist)
        wlist = [w * np.ones_like(v) for v, w in zip(vlist, wlist)]
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
    return (rho / np.sqrt(2 * math.pi * theta) ** v.shape[-1]) * np.exp(
        -((u[..., None, :] - v) ** 2).sum(axis=-1) / (2 * theta)
    )


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
    f = rho / np.sqrt(2 * math.p * theta) * np.exp(-((u - v) ** 2) / (2 * theta))
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
    wtuple = tuple([ms.Tensor(w.astype(np.float32)) for w in wlist])
    return vtuple, wtuple


def visual(problem, resolution=100, filename="result.jpg"):
    """visualize the results"""
    x = np.linspace(-0.5, 0.5, resolution)
    t0 = 0.0 * np.ones_like(x)
    t1 = 0.1 * np.ones_like(x)
    xt0 = ms.Tensor(np.stack((x, t0), axis=-1).astype(np.float32))
    xt1 = ms.Tensor(np.stack((x, t1), axis=-1).astype(np.float32))
    rho0, u0, theta0 = problem.pred(xt0)
    rho1, u1, theta1 = problem.pred(xt1)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3))
    ax[0].plot(x, rho0.asnumpy(), label=r"$\rho$")
    ax[0].plot(x, u0[..., 0].asnumpy(), label="$u_x$")
    ax[0].plot(x, theta0.asnumpy(), label="T")
    ax[0].legend()
    ax[1].plot(x, rho1.asnumpy(), label=r"$\rho$")
    ax[1].plot(x, u1[..., 0].asnumpy(), label="$u_x$")
    ax[1].plot(x, theta1.asnumpy(), label="T")
    ax[1].legend()
    fig.savefig(filename)
    return fig


def valid_model(config, problem):
    """valid the model."""
    ref_solution0 = np.load(config["ref_solution"])
    rho0_ref, u0_ref, theta0_ref = (
        ref_solution0["rho0"],
        ref_solution0["u0"],
        ref_solution0["T0"],
    )
    rho1_ref, u1_ref, theta1_ref = (
        ref_solution0["rho1"],
        ref_solution0["u1"],
        ref_solution0["T1"],
    )
    resolution = rho0_ref.shape[0]
    x = np.linspace(-0.5, 0.5, resolution)
    t0 = 0.0 * np.ones_like(x)
    t1 = 0.1 * np.ones_like(x)
    xt0 = ms.Tensor(np.stack((x, t0), axis=-1).astype(np.float32))
    xt1 = ms.Tensor(np.stack((x, t1), axis=-1).astype(np.float32))
    rho0, u0, theta0 = problem.pred(xt0)
    rho1, u1, theta1 = problem.pred(xt1)

    err1 = (
        ((rho0.asnumpy()[..., 0] - rho0_ref) ** 2).mean() / ((rho0_ref) ** 2).mean()
    ) ** 0.5
    err2 = (
        ((u0.asnumpy()[..., 0] - u0_ref) ** 2).mean() / (1 + (u0_ref) ** 2).mean()
    ) ** 0.5
    err3 = (
        ((theta0.asnumpy()[..., 0] - theta0_ref) ** 2).mean() / (theta0_ref**2).mean()
    ) ** 0.5
    print(f"err at t=0.0: {err1.item():.3e}\t{err2.item():.3e}\t{err3.item():.3e}\t")

    err1 = (
        ((rho1.asnumpy()[..., 0] - rho1_ref) ** 2).mean() / ((rho1_ref) ** 2).mean()
    ) ** 0.5
    err2 = (
        ((u1.asnumpy()[..., 0] - u1_ref) ** 2).mean() / (1 + (u1_ref) ** 2).mean()
    ) ** 0.5
    err3 = (
        ((theta1.asnumpy()[..., 0] - theta1_ref) ** 2).mean() / (theta1_ref**2).mean()
    ) ** 0.5
    print(f"err at t=0.1: {err1.item():.3e}\t{err2.item():.3e}\t{err3.item():.3e}\t")


def save_points(problem, points=1000, filename="points.npz"):
    x = np.random.rand(points) - 0.5
    t = 0.1 * np.random.rand(points)
    xt0 = ms.Tensor(np.stack((x, t), axis=-1).astype(np.float32))
    f = problem.net(xt0)
    np.savez(filename, f=f.asnumpy())


def get_new_kernel(f_bases, f_bases2, nx, ny, nz, kn_bzm, phi, psi, phipsi):
    """get the kernel tensor"""
    k = f_bases.shape[1]
    t = np.zeros((k, k, k))
    for i in range(k):
        for j in range(k):
            f1 = f_bases[:, i].reshape((nx, ny, nz))
            f2 = f_bases[:, j].reshape((nx, ny, nz))
            f3 = collision_fft_fg(f1, f2, kn_bzm, phi, psi, phipsi).reshape(
                (nx * ny * nz, 1)
            )
            coef = f_bases2.T @ f3
            t[i, j, :] = coef[..., 0]
    return t
