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

from mindflow.cell import FCSequential

from src.utils import SimpleUniformInitializer


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
    return f.sum(-1, keepdims=True) * w


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
        return loss.mean()
