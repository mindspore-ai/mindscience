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
traditional pde solver to obtain data
"""
import numpy as np


def _initgen_periodic(mesh_size, freq=3):
    """"generate initial field on periodic boundary"""
    dim = len(mesh_size)
    x = np.random.randn(*mesh_size)
    coe = np.fft.ifftn(x)
    freqs = np.random.randint(freq, 2 * freq, size=[dim])
    for i in range(dim):
        perm = [i for i in range(dim)]
        perm[i] = 0
        perm[0] = i
        coe = coe.transpose(*perm)
        coe[freqs[i] + 1: - freqs[i]] = 0
        coe = coe.transpose(*perm)
    x = np.fft.fftn(coe)
    x = x.real
    return x


def initgen(mesh_size, freq=3, boundary='Periodic'):
    """
    initial value generator
    """
    if np.iterable(freq):
        return freq
    x = _initgen_periodic(mesh_size, freq=freq)
    x = x * 100
    if boundary.upper() == 'DIRICHLET':
        dim = x.ndim
        for i in range(dim):
            mesh_size_i = mesh_size[i]
            y = [i / mesh_size_i * (1 - i / mesh_size_i) for i in range(mesh_size_i)]
            s = [1 for _ in range(dim)]
            s[i] = mesh_size[i]
            y = np.reshape(y, s)
            x = x * y
        x = x[[slice(1, None)] * dim]
        x = x * 16
    return x


class _PDESolver:
    """base class for pde solver"""

    def step(self, init, dt):
        raise NotImplementedError

    def predict(self, init, time_step):
        """predict field based on initial condition"""
        res = 0
        if not hasattr(self, 'max_dt'):
            res = self.step(init, time_step)
        else:
            n = int(np.ceil(time_step / self.max_dt))
            dt = time_step / n
            u = init
            for _ in range(n):
                u = self.step(u, dt)
            res = u
        return res


def _coe_modify(coe_a, coe_b, m):
    coe_a[:m, :m] = coe_b[:m, :m]
    coe_a[:m, - m + 1:] = coe_b[:m, - m + 1:]
    coe_a[- m + 1:, :m] = coe_b[- m + 1:, :m]
    coe_a[- m + 1:, - m + 1:] = coe_b[- m + 1:, - m + 1:]


class VariantCoeLinear2d(_PDESolver):
    """
    pde solvers in traditional algorithm
    """

    def __init__(self, spectral_size, max_dt=5e-3, variant_coe_magnitude=1):
        self.max_dt = max_dt
        self.spectral_size = spectral_size
        self._coe_mag = variant_coe_magnitude
        freq_shift_coe = np.zeros((spectral_size,))
        freq_shift_coe[:spectral_size // 2] = range(spectral_size // 2)
        freq_shift_coe[:-spectral_size // 2 - 1:-1] = range(-spectral_size // 2, 0)[::-1]
        self.k0 = np.reshape(freq_shift_coe, (spectral_size, 1))
        self.k1 = np.reshape(freq_shift_coe, (1, spectral_size))

        def b10(x):
            y = np.reshape(x, [-1, 2])
            tmp = np.cos(y[:, 0]) + y[:, 1] * (2 * np.pi - y[:, 1]) * np.sin(y[:, 1])
            return self._coe_mag * 0.5 * np.reshape(tmp, x.shape[:-1]) + 0.6

        def b01(x):
            y = np.reshape(x, [-1, 2])
            return self._coe_mag * 2 * np.reshape(np.cos(y[:, 0]) + np.sin(y[:, 1]), x.shape[:-1]) + 0.8

        self.a = np.ndarray([5, 5], dtype=object)
        self.a[0, 0] = lambda x: np.zeros(x.shape[:-1])
        self.a[0, 1] = b01
        self.a[1, 0] = b10
        self.a[0, 2] = lambda x: np.zeros(x.shape[:-1]) + 0.3
        self.a[1, 1] = lambda x: np.zeros(x.shape[:-1])
        self.a[2, 0] = lambda x: np.zeros(x.shape[:-1]) + 0.2
        b00 = lambda x: np.zeros(x.shape[:-1])
        self.a[list(range(4)), list(range(3, -1, -1))] = b00
        self.a[list(range(5)), list(range(4, -1, -1))] = b00
        self.a_fourier_coe = np.ndarray([5, 5], dtype=object)
        self.a_smooth = np.ndarray([5, 5], dtype=object)

        xx = [k * 2 * np.pi / spectral_size for k in range(spectral_size)]
        yy = xx.copy()
        yy, xx = np.meshgrid(xx, yy)
        xx = np.expand_dims(xx, axis=-1)
        yy = np.expand_dims(yy, axis=-1)
        xy = np.concatenate([xx, yy], axis=2)
        m = spectral_size // 2
        for k in range(3):
            for j in range(k + 1):
                tmp_fourier = np.fft.ifft2(self.a[j, k - j](xy))
                self.a_fourier_coe[j, k - j] = tmp_fourier
                tmp = np.zeros([m * 3, m * 3], dtype=np.complex128)
                _coe_modify(tmp, tmp_fourier, m)
                self.a_smooth[j, k - j] = np.fft.fft2(tmp).real

    def vc_conv(self, order, coe):
        m = self.spectral_size // 2
        vc_smooth = self.a_smooth[order[0], order[1]]
        tmp = np.zeros(vc_smooth.shape, dtype=np.complex128)
        _coe_modify(tmp, coe, m)
        c_aug = np.fft.ifft2(vc_smooth * np.fft.fft2(tmp))
        c = np.zeros(coe.shape, dtype=np.complex128)
        _coe_modify(c, c_aug, m)
        return c

    def rhs_fourier(self, l):
        rhsl = np.zeros(l.shape, dtype=np.complex128)
        rhsl += self.vc_conv([1, 0], -1j * self.k0 * l)
        rhsl += self.vc_conv([0, 1], -1j * self.k1 * l)
        rhsl += self.vc_conv([2, 0], -self.k0 ** 2 * l)
        rhsl += self.vc_conv([1, 1], -self.k0 * self.k1 * l)
        rhsl += self.vc_conv([0, 2], -self.k1 ** 2 * l)
        return rhsl

    def step(self, init, dt):
        y = np.zeros([self.spectral_size, self.spectral_size], dtype=np.complex128)
        m = self.spectral_size // 2
        l = np.fft.ifft2(init)
        _coe_modify(y, l, m)
        rhsl1 = self.rhs_fourier(y)
        rhsl2 = self.rhs_fourier(y + 0.5 * dt * rhsl1)
        rhsl3 = self.rhs_fourier(y + 0.5 * dt * rhsl2)
        rhsl4 = self.rhs_fourier(y + dt * rhsl3)

        y = y + (rhsl1 + 2 * rhsl2 + 2 * rhsl3 + rhsl4) * dt / 6
        _coe_modify(l, y, m)
        x_tmp = np.fft.fft2(l)
        x = x_tmp.real
        return x
