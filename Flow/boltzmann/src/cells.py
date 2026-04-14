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
Basic blocks for boltzmann equation
"""
import math
import numpy as np

import mindspore as ms
from mindspore import nn, ops
import mindspore.numpy as mnp
from mindspore.nn import Jvp
from mindspore.ops.operations import math_ops
from mindspore.common.parameter import Parameter

from mindflow.cell import FCSequential

from src.utils import SimpleUniformInitializer


class FFT3D(nn.Cell):
    """class to do the fft3d"""
    def __init__(self, auto_prefix=True, flags=None):
        super().__init__(auto_prefix, flags)
        self.fft3d = math_ops.FFTWithSize(
            signal_ndim=3, inverse=False, real=False, norm="backward"
        )
        self.complex = ops.Complex()
        self.view_as_real = ops.view_as_real

    def construct(self, x):
        inputs = self.complex(x[..., 0], x[..., 1])
        y = self.fft3d(inputs)
        return self.view_as_real(y)


class IFFT3D(nn.Cell):
    """class to do the inverse fft3d"""
    def __init__(self, auto_prefix=True, flags=None):
        super().__init__(auto_prefix, flags)
        self.ifft3d = math_ops.FFTWithSize(
            signal_ndim=3, inverse=True, real=False, norm="forward"
        )
        self.complex = ops.Complex()
        self.view_as_real = ops.view_as_real

    def construct(self, x):
        inputs = self.complex(x[..., 0], x[..., 1])
        y = self.ifft3d(inputs)
        return self.view_as_real(y)


class ComplexMul(nn.Cell):
    """complex number multiply"""

    def construct(self, x, y):
        r = x[..., 0] * y[..., 0] - x[..., 1] * y[..., 1]
        i = x[..., 0] * y[..., 1] + x[..., 1] * y[..., 0]
        return ops.stack((r, i), axis=-1)


class Maxwellian(nn.Cell):
    """the maxwellian function for given v"""

    def __init__(self, v):
        super().__init__()
        self.v = v
        self.dim = v.shape[-1]
        self.pi = math.pi

    def construct(self, rho, u, theta):
        return (rho / ops.sqrt(2 * self.pi * theta) ** self.dim) * ops.exp(
            -((u[..., None, :] - self.v) ** 2).sum(axis=-1) / (2 * theta)
        )


@ms.jit
def fsum(f, w):
    """return the weighted sum of w"""
    return (f * w).sum(-1, keepdims=True)


class _M0(nn.Cell):
    """the 0-order moment"""

    def __init__(self, v, w):
        super().__init__()
        self.v = v
        self.w = w

    def construct(self, f):
        return fsum(f, self.w)


@ms.jit
def _m1_d1(f, v, w):
    """the 1-order moment for 1d distribution"""
    return fsum(f * v[..., 0], w)


@ms.jit
def _m1_d2(f, v, w):
    """the 1-order moment for 2d distribution"""
    return ops.concat([fsum(f * v[..., 0], w), fsum(f * v[..., 1], w)], axis=-1)


@ms.jit
def _m1_d3(f, v, w):
    """the 1-order moment for 3d distribution"""
    return ops.concat(
        [fsum(f * v[..., 0], w), fsum(f * v[..., 1], w), fsum(f * v[..., 2], w)],
        axis=-1,
    )


class _M1(nn.Cell):
    """the 1-order moment"""

    def __init__(self, v, w):
        super().__init__()
        self.v = v
        self.w = w
        self.dim = v.shape[-1]
        if self.dim == 1:
            self._m1 = _m1_d1
        elif self.dim == 2:
            self._m1 = _m1_d2
        elif self.dim == 3:
            self._m1 = _m1_d3
        else:
            raise ValueError("v.shape[-1] should be 1,2 or 3.")

    def construct(self, f):
        return self._m1(f, self.v, self.w)


@ms.jit
def _m2_d1(f, v, w):
    """the 2-order moment for 1d distribution"""
    return fsum(f * v[..., 0] ** 2, w)


@ms.jit
def _m2_d2(f, v, w):
    """the 2-order moment for 2d distribution"""
    return ops.concat(
        [fsum(f * v[..., 0] ** 2, w), fsum(f * v[..., 1] ** 2, w)], axis=-1
    )


@ms.jit
def _m2_d3(f, v, w):
    """the 2-order moment for 3d distribution"""
    return ops.concat(
        [
            fsum(f * v[..., 0] ** 2, w),
            fsum(f * v[..., 1] ** 2, w),
            fsum(f * v[..., 2] ** 2, w),
        ],
        axis=-1,
    )


class _M2(nn.Cell):
    """the 2-order moment"""

    def __init__(self, v, w):
        super().__init__()
        self.v = v
        self.w = w
        self.dim = v.shape[-1]
        if self.dim == 1:
            self._m2 = _m2_d1
        elif self.dim == 2:
            self._m2 = _m2_d2
        elif self.dim == 3:
            self._m2 = _m2_d3
        else:
            raise ValueError("v.shape[-1] should be 1,2 or 3.")

    def construct(self, f):
        return self._m2(f, self.v, self.w)


class _M012(nn.Cell):
    """return the 0,1,2 order moment"""

    def __init__(self, v, w):
        super().__init__()
        self._m0 = _M0(v, w)
        self._m1 = _M1(v, w)
        self._m2 = _M2(v, w)

    def construct(self, f):
        return self._m0(f), self._m1(f), self._m2(f)


class RhoUTheta(nn.Cell):
    """return the density,velocity and temperature of given f"""

    def __init__(self, v, w, eps=1e-3):
        super().__init__()
        self._m0 = _M0(v, w)
        self._m1 = _M1(v, w)
        self._m2 = _M2(v, w)
        self.eps = eps
        self.dim = v.shape[-1]

    def construct(self, f):
        m0, m1, m2 = self._m0(f), self._m1(f), self._m2(f)
        density = ops.maximum(m0, self.eps)
        veloc = m1 / m0
        v2 = (veloc**2).sum(axis=-1, keepdims=True)
        temperature = (m2.sum(axis=-1, keepdims=True) / m0 - v2) / self.dim
        temperature = ops.maximum(temperature, self.eps)
        return density, veloc, temperature


class PrimNorm(nn.Cell):
    """return the primary loss function"""

    def __init__(self, v, w, criterion_norm=None):
        super().__init__()
        self._m012 = _M012(v, w)
        self.dim = v.shape[-1]
        if criterion_norm:
            self.criterion_norm = criterion_norm
        else:
            self.criterion_norm = lambda x: ops.square(x).mean(axis=0)

    def construct(self, f):
        m1, m2, m3 = self._m012(f)
        return ops.concat(
            [self.criterion_norm(m1), self.criterion_norm(m2), self.criterion_norm(m3)],
            axis=-1,
        )


class MultiResInput(nn.Cell):
    """multiply the input with different numbers for multi-resolution"""

    def __init__(self, freq):
        super().__init__()
        self.freq = freq
        self.reshape = ops.shape

    def construct(self, x):
        xf = x[..., None] * self.freq
        y = xf.reshape(xf.shape[:-2] + (xf.shape[-2] * xf.shape[-1],))
        return y


class MultiRes(nn.Cell):
    """the multi-resolution network"""

    def __init__(self, in_channel, out_channel, layers, neurons, freq=(1, 4, 16)):
        super().__init__()
        self.minput = MultiResInput(freq)
        self.net = FCSequential(
            in_channel * len(freq),
            out_channel,
            layers,
            neurons,
            residual=False,
            act="sin",
            weight_init=SimpleUniformInitializer(),
            has_bias=True,
            bias_init=SimpleUniformInitializer(),
            weight_norm=False,
        )

    def construct(self, x):
        x = self.minput(x)
        y = self.net(x)
        return y


class SplitNet(nn.Cell):
    """the network combined the maxwellian and non-maxwellian"""

    def __init__(self, in_channel, layers, neurons, vdis, alpha=0.01):
        super().__init__()
        self.net_eq = MultiRes(in_channel, 5, layers, neurons)
        self.net_neq = MultiRes(in_channel, vdis.shape[0], layers, neurons)
        self.maxwellian = Maxwellian(vdis)
        self.alpha = alpha

    def construct(self, xt):
        www = self.net_eq(xt)
        rho, u, theta = www[..., 0:1], www[..., 1:4], www[..., 4:5]
        rho = ops.exp(-rho)
        theta = ops.exp(-theta)
        x1 = self.maxwellian(rho, u, theta)
        x2 = self.net_neq(xt)
        y = x1 * (x1 + self.alpha * x2)
        return y


class JacNet(nn.Cell):
    """jacbian by JVP"""

    def __init__(self, net):
        super().__init__()
        self.net = net
        self.dx = Jvp(self.net)
        self.dt = Jvp(self.net)

    def construct(self, xt):
        vx1 = ops.zeros_like(xt)
        vx1[..., 0] = 1
        out, f_x = self.dx(xt, (vx1))
        vx2 = ops.zeros_like(xt)
        vx2[..., 1] = 1
        _, f_t = self.dt(xt, (vx2))
        return out, ops.stack((f_x, f_t))


class JacFwd(nn.Cell):
    """jacbian by jvp"""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def construct(self, x):
        v = mnp.broadcast_to(
            mnp.eye(x.shape[-1])[:, None, :], (x.shape[1], x.shape[0], x.shape[1])
        )
        output, g = ms.jvp(self.net, x, v[0])
        ga = (g,)
        for i in range(1, v.shape[0]):
            ga += (ms.jvp(self.net, x, v[i])[1],)
        return output, ops.stack(ga)


class MtlLoss(nn.Cell):
    """multi target loss"""

    def __init__(self, num_losses, eta=1e-3):
        super().__init__()
        self.num_losses = num_losses
        self.params = ms.Parameter(
            ms.Tensor(np.ones(num_losses), ms.float32), requires_grad=True
        )
        self.eta = ms.Tensor(eta)

    def construct(self, losses):
        ww = self.params**2 + self.eta**2
        loss = 0.5 / ww * losses + ops.log(1 + ww)
        return loss.sum() / self.num_losses


##### Cells for LR
@ms.jit
def maxwellian_lr_1d(v, rho, u, theta):
    """
    v [nv]
    rho [nx,1]
    u [nx,1]
    T [nx,1]
    """
    f = rho / ops.sqrt(2 * np.pi * theta) * ops.exp(-((u - v) ** 2) / (2 * theta))
    return f


@ms.jit
def maxwellian_lr_3d(vtuple, rho, u, theta):
    vx, vy, vz = vtuple
    f1 = maxwellian_lr_1d(vx, rho ** (1 / 3), u[..., 0:1], theta)
    f2 = maxwellian_lr_1d(vy, rho ** (1 / 3), u[..., 1:2], theta)
    f3 = maxwellian_lr_1d(vz, rho ** (1 / 3), u[..., 2:3], theta)
    return f1[..., None], f2[..., None], f3[..., None]


class MaxwellianLR(nn.Cell):
    def __init__(self, vtuple):
        super().__init__()
        self.v = vtuple
        self.maxwellian_lr = maxwellian_lr_3d

    def construct(self, rho, u, theta):
        return self.maxwellian_lr(self.v, rho, u, theta)


@ms.jit
def f_sum_lowrank(ft, wt):
    fx, fy, fz = ft
    wx, wy, wz = wt
    sx = ops.einsum("...ir,i->...r", fx, wx)
    sy = ops.einsum("...jr,j->...r", fy, wy)
    sz = ops.einsum("...kr,k->...r", fz, wz)
    s = ops.einsum("...r,...r,...r->...", sx, sy, sz)
    return s


@ms.jit
def f_m0_lowrank(ft, wt):
    return f_sum_lowrank(ft, wt)[..., None]


@ms.jit
def f_m1_lowrank(ft, vt, wt):
    fx, fy, fz = ft
    vx, vy, vz = vt
    mux = f_sum_lowrank((fx * vx[..., None], fy, fz), wt)
    muy = f_sum_lowrank((fx, fy * vy[..., None], fz), wt)
    muz = f_sum_lowrank((fx, fy, fz * vz[..., None]), wt)
    return ops.stack([mux, muy, muz], axis=-1)


@ms.jit
def f_m2_lowrank(ft, vt, wt):
    fx, fy, fz = ft
    vx, vy, vz = vt
    mux = f_sum_lowrank((fx * vx[..., None] ** 2, fy, fz), wt)
    muy = f_sum_lowrank((fx, fy * vy[..., None] ** 2, fz), wt)
    muz = f_sum_lowrank((fx, fy, fz * vz[..., None] ** 2), wt)
    return ops.stack([mux, muy, muz], axis=-1)


# @ms.jit
def f_m012_lowrank(f, v, w):
    m0, m1, m2 = f_m0_lowrank(f, w), f_m1_lowrank(f, v, w), f_m2_lowrank(f, v, w)
    return m0, m1, m2


@ms.jit
def fsum_lr(f, w):
    return f_sum_lowrank(f, w)


class M0Lr(nn.Cell):
    def __init__(self, v, w):
        super().__init__()
        self.v = v
        self.w = w
        self.pi = math.pi

    def construct(self, f):
        return fsum_lr(f, self.w)[..., None]


@ms.jit
def _m1_d3_lr(f, v, w):
    fx, fy, fz = f
    vx, vy, vz = v
    return ops.stack(
        [
            fsum_lr((fx * vx[..., None], fy, fz), w),
            fsum_lr((fx, fy * vy[..., None], fz), w),
            fsum_lr((fx, fy, fz * vz[..., None]), w),
        ],
        axis=-1,
    )


class M1Lr(nn.Cell):
    """the 1 order moment of low rank"""
    def __init__(self, v, w):
        super().__init__()
        self.v = v
        self.w = w
        self.pi = math.pi
        self.dim = 3
        self._m1 = _m1_d3_lr

    def construct(self, f):
        return self._m1(f, self.v, self.w)


@ms.jit
def _m2_d3_lr(f, v, w):
    fx, fy, fz = f
    vx, vy, vz = v
    return ops.stack(
        [
            fsum_lr((fx * (vx[..., None] ** 2), fy, fz), w),
            fsum_lr((fx, (fy * vy[..., None] ** 2), fz), w),
            fsum_lr((fx, fy, (fz * vz[..., None] ** 2)), w),
        ],
        axis=-1,
    )


class M2Lr(nn.Cell):
    """the 2 order moment of low rank"""
    def __init__(self, v, w):
        super().__init__()
        self.v = v
        self.w = w
        self.pi = math.pi
        self.dim = 3
        self._m2 = _m2_d3_lr

    def construct(self, f):
        return self._m2(f, self.v, self.w)


class M012Lr(nn.Cell):
    def __init__(self, v, w):
        super().__init__()
        self._m0 = M0Lr(v, w)
        self._m1 = M1Lr(v, w)
        self._m2 = M2Lr(v, w)

    def construct(self, f):
        return self._m0(f), self._m1(f), self._m2(f)


class RhoUThetaLr(nn.Cell):
    """the rho,u,theta of low rank"""
    def __init__(self, v, w, eps=1e-3):
        super().__init__()
        self._m0 = M0Lr(v, w)
        self._m1 = M1Lr(v, w)
        self._m2 = M2Lr(v, w)
        self.eps = eps
        self.dim = 3

    def construct(self, f):
        m0, m1, m2 = self._m0(f), self._m1(f), self._m2(f)
        density = ops.maximum(m0, self.eps)
        veloc = m1 / m0
        v2 = (veloc**2).sum(axis=-1, keepdims=True)
        temperature = (m2.sum(axis=-1, keepdims=True) / m0 - v2) / self.dim
        temperature = ops.maximum(temperature, self.eps)
        return density, veloc, temperature


@ms.jit
def rho_u_theta_lowrank(ft, vt, wt):
    """the rho,u,theta of low rank"""
    eps_r = 1e-4
    m0 = f_m0_lowrank(ft, wt)
    m1 = f_m1_lowrank(ft, vt, wt)
    m2 = f_m2_lowrank(ft, vt, wt)
    density = ops.maximum(m0, eps_r)
    veloc = m1 * (1.0 / m0)
    v2 = (veloc**2).sum(axis=-1, keepdims=True)
    temperature = (m2.sum(axis=-1, keepdims=True) / m0 - v2) * (1.0 / len(ft))
    return density, veloc, temperature


class PrimNormLR(nn.Cell):
    """the primary norm of low rank"""
    def __init__(self, v, w, criterion_norm=None):
        super().__init__()
        self.v = v
        self.w = w
        self.dim = len(v)
        if criterion_norm:
            self.criterion_norm = criterion_norm
        else:
            self.criterion_norm = lambda x: ops.square(x).mean(axis=0)

    def construct(self, f):
        m1, m2, m3 = f_m012_lowrank(f, self.v, self.w)
        return ops.concat(
            [self.criterion_norm(m1), self.criterion_norm(m2), self.criterion_norm(m3)],
            axis=-1,
        )


@ms.jit
def dis_lowrank_add(dislist):
    return (
        ops.concat((d[0] for d in dislist), axis=-1),
        ops.concat((d[1] for d in dislist), axis=-1),
        ops.concat((d[2] for d in dislist), axis=-1),
    )


@ms.jit
def dis_lowrank_sub(dis1, dis2):
    d1x, d1y, d1z = dis1
    d2x, d2y, d2z = dis2
    return (
        ops.concat([d1x, -d2x], axis=-1),
        ops.concat([d1y, d2y], axis=-1),
        ops.concat([d1z, d2z], axis=-1),
    )


class CustomConcat(nn.Cell):
    """custom concat (for double backward)"""
    def construct(self, x, y):
        return ops.concat((x, y), axis=-1)

    def bprop(self, x, y, out, dout):
        _ = (y, out)
        dx = dout[..., : x.shape[-1]]
        dy = dout[..., x.shape[-1] :]
        return dx, dy


class JacFwdLR(nn.Cell):
    """jacbian (forward auto grad) for lor rank net."""
    def __init__(self, net):
        super().__init__()
        self.net = net

    def construct(self, x):
        v = mnp.broadcast_to(
            mnp.eye(x.shape[-1])[:, None, :], (x.shape[1], x.shape[0], x.shape[1])
        )
        output, g = ms.jvp(self.net, x, v[0])
        ga = (g,)
        for i in range(1, v.shape[0]):
            ga += (ms.jvp(self.net, x, v[i])[1],)
        return output, ga


class SplitNetLR(nn.Cell):
    """the split network for low rank"""
    def __init__(self, in_channel, layers, neurons, vtuple, rank=40, alpha=0.01):
        super().__init__()
        self.net_eq = MultiRes(in_channel, 5, layers, neurons)
        self.vt = vtuple
        self.rank = rank
        self.out_channel_1 = self.vt[0].shape[0]
        self.out_channel_2 = self.vt[1].shape[0]
        self.out_channel_3 = self.vt[2].shape[0]
        self.net_neq = MultiRes(
            in_channel,
            (self.out_channel_1 + self.out_channel_2 + self.out_channel_3) * rank,
            layers,
            neurons,
        )

        self.net_neq1 = MultiRes(in_channel, self.out_channel_1 * rank, layers, neurons)
        self.net_neq2 = MultiRes(in_channel, self.out_channel_2 * rank, layers, neurons)
        self.net_neq3 = MultiRes(in_channel, self.out_channel_3 * rank, layers, neurons)

        self.maxwellian = MaxwellianLR(self.vt)
        self.alpha = alpha
        self.custom_concat = CustomConcat()

    def construct(self, xt):
        """construct"""
        www = self.net_eq(xt)
        rho, u, theta = www[..., 0:1], www[..., 1:4], www[..., 4:5]
        rho = ops.exp(-rho)
        theta = ops.exp(-theta)

        fmx, fmy, fmz = self.maxwellian(rho, u, theta)

        f2x = self.net_neq1(xt)
        f2y = self.net_neq2(xt)
        f2z = self.net_neq3(xt)

        f2x = f2x.reshape(f2x.shape[:-1] + (self.out_channel_1, self.rank))
        f2x = (0.01**0.33) * fmx * f2x

        f2y = f2y.reshape(f2y.shape[:-1] + (self.out_channel_2, self.rank))
        f2y = (0.01**0.33) * fmy * f2y

        f2z = f2z.reshape(f2z.shape[:-1] + (self.out_channel_3, self.rank))
        f2z = (0.01**0.33) * fmz * f2z

        return (
            self.custom_concat(fmx, f2x),
            self.custom_concat(fmy, f2y),
            self.custom_concat(fmz, f2z),
        )


@ms.jit
def lrmse_adap(p, q, r, w1, w2, w3):
    w1 = w1[None, ..., None].sqrt()
    w2 = w2[None, ..., None].sqrt()
    w3 = w3[None, ..., None].sqrt()
    p = p * w1
    q = q * w2
    r = r * w3
    return 0.5 * ((p.mT @ p) * (q.mT @ q) * (r.mT @ r)).sum() * (1.0 / p.shape[0])


class AdaptiveMSE(nn.Cell):
    """mse with adaptive weight"""
    def __init__(self, nx, ny, nz, eta=1e-6):
        super().__init__()
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.w1 = Parameter(ops.ones(nx, dtype=ms.float32))
        self.w2 = Parameter(ops.ones(ny, dtype=ms.float32))
        self.w3 = Parameter(ops.ones(nz, dtype=ms.float32))
        self.eta = eta

    def construct(self, f):
        """construct"""
        p, q, r = f
        w1 = self.eta**2 + self.w1**2
        w2 = self.eta**2 + self.w2**2
        w3 = self.eta**2 + self.w3**2
        w = w1[:, None, None] * w2[None, :, None] * w3[None, None, :]
        loss_1, loss_w = (
            lrmse_adap(p, q, r, 0.5 / w1, 0.5 / w2, 0.5 / w3),
            ops.log(1 + w).sum(),
        )
        return loss_1, loss_w
