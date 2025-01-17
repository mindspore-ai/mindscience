# Copyright 2025 Huawei Technologies Co., Ltd
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
"""The CBS (convergen Born series) API"""
from math import factorial
import numpy as np
import mindspore as ms
from mindspore import Tensor, nn, ops, numpy as mnp, lazy_inline

from .dft import MyDFTn, MyiDFTn


class CBSBlock(nn.Cell):
    ''' The computation procedures for each iteration in CBS '''
    @lazy_inline
    def __init__(self, shape):
        '''
        No trainable parameters, but the dft cells needs initialization
        Args:
            shape: tuple of int, only the spatial shape, not including the batch and channel dimensions
        '''
        super().__init__()
        self.dft_cell = MyDFTn(shape)
        self.idft_cell = MyiDFTn(shape)

    # Scattering potential calculation for real and imaginary parts
    def op_v(self, ur, ui, vr, vi):
        wr = ur * vr - ui * vi
        wi = ur * vi + ui * vr
        return wr, wi

    # Vectorized Helmholtz Green function for real and imaginary parts
    def op_g(self, ur, ui, gr, gi):
        fur, fui = self.dft_cell(ur, ui)
        gur = gr * fur - gi * fui
        gui = gi * fur + gr * fui
        wr, wi = self.idft_cell(gur, gui)
        return wr, wi

    # Vectorized Born iteration for real and imaginary parts
    def construct(self, ur, ui, vr, vi, gr, gi, rhs, eps):
        ''' run one iteration and return the incremental '''
        vur, vui = self.op_v(ur, ui, vr, vi)
        gvr, gvi = self.op_g(vur + rhs, vui, gr, gi)
        vgr, vgi = self.op_v(gvr - ur, gvi - ui, vr, vi)

        # eps > 0: Convergent Born series; eps == 0: Original Born Series
        cond = ops.broadcast_to(eps, ur.shape) > 0
        dur = ops.select(cond, -vgi / (eps + 1e-8), gvr - ur) # '* (-1.)' comes from imag part multiplying i/eps
        dui = ops.select(cond, vgr / (eps + 1e-8), gvi - ui)

        return ops.stack([dur, dui]) # return a single Tensor for compatibility with nn.SequentialCell

class CBS(nn.Cell):
    ''' The CBS cell for solving 2D acoustic equation '''
    def __init__(self,
                 shape,
                 n_iter=20,
                 pml_size=60,
                 alpha=1.0,
                 rampup=12,
                 remove_pml=True,
                 epsilon=None,
                 ):
        """Configurations of the CBS solver

        Args:
            shape (tuple[int]): only the spatial shape, not including the batch and channel dimensions
            n_iter (int, optional): number of iterations in a single call. Defaults to 20.
            pml_size (int, optional): number of grid layers to pad on each boundary for the wave to attenuate.
                Defaults to 60.
            alpha (float, optional): the strength of wave attenuation in PML layers. Defaults to 1.0.
            rampup (int, optional): the smoothness of transition from interior domain to PML layers. Defaults to 12.
            remove_pml (bool, optional): whether to remove the PML layers for the output. Defaults to True.
            epsilon (float, optional): the small value to stabilize the iteration.
                Defaults to None, calculating epsilon automatically.
        """
        super().__init__()

        self.n_iter = n_iter
        self.pml_size = pml_size
        self.alpha = alpha
        self.rampup = rampup
        self.remove_pml = remove_pml
        self.epsilon = epsilon

        shape_padded = tuple(n + 2 * pml_size for n in shape)

        dxs = (1.0, 1.0)
        p_sq = sum(np.meshgrid(
            *[np.fft.fftfreq(n, d)**2 for n, d in zip(shape_padded, dxs)],
            indexing="ij")) * (2 * np.pi)**2
        self.p_sq = Tensor(p_sq, dtype=ms.float32, const_arg=True)

        pml_mask = 1 - np.pad(np.ones(shape), pml_size)
        self.pml_mask = Tensor(pml_mask, dtype=ms.float32, const_arg=True)

        self.cbs_block = CBSBlock(shape_padded)

    def cbs_params(self, c_star, f_star):
        ''' compute constant variables for CBS iteration '''
        pml_size = self.pml_size
        nz, nx = c_star.shape[-2:]
        dxs = (1.0, 1.0)
        omg = 1.0

        # source field
        rhs = ops.pad(f_star / c_star**2, [pml_size] * 4) # (batch, 1, nz_padded, nx_padded)

        # homogeneous k field
        k_max = omg / ops.amin(c_star, axis=(-2, -1), keepdims=True)
        k_min = omg / ops.amax(c_star, axis=(-2, -1), keepdims=True)
        k0 = ops.sqrt(0.5 * (k_max**2 + k_min**2)) # (batch, 1, 1, 1)

        # heterogeneous k field
        ksq_r, ksq_i = self.cbs_pml(
            (nz, nx), dxs, k_max, pml_size, self.alpha, self.rampup) # (batch, 1, nz_padded, nx_padded)

        ksq_r = ksq_r * self.pml_mask + ops.pad((omg / c_star)**2, [pml_size] * 4) * (1 - self.pml_mask)
        ksq_i = ksq_i * self.pml_mask

        eps = ops.amax((ksq_r - k0**2)**2 + ksq_i**2, axis=(-2, -1), keepdims=True)**.5 # (batch, 1, 1, 1)

        # if epsilon given by user, use original BS instead of CBS
        if isinstance(self.epsilon, (float, int)):
            eps = self.epsilon * ops.ones_like(eps)

        # field variables needed by operator V & G
        vr = ksq_r - k0**2 # (batch, 1, nz_padded, nx_padded)
        vi = ksq_i - eps   # (batch, 1, nz_padded, nx_padded)
        gr = 1. / ((self.p_sq - k0**2)**2 + eps**2) * (self.p_sq - k0**2) # (batch, 1, nz_padded, nx_padded)
        gi = 1. / ((self.p_sq - k0**2)**2 + eps**2) * eps                 # (batch, 1, nz_padded, nx_padded)

        return vr, vi, gr, gi, rhs, eps * (self.epsilon is None)

    @staticmethod
    def cbs_pml(shape, dxs, k0, pml_size, alpha, rampup):
        ''' construct the heterogeneous k field with PML BC embedded '''
        shape_padded = tuple(n + 2 * pml_size for n in shape)

        def num(x):
            num_real = (alpha ** 2) * (rampup - alpha * x) * ((alpha * x) ** (rampup - 1))
            num_imag = (alpha ** 2) * (2 * k0 * x) * ((alpha * x) ** (rampup - 1))
            return num_real, num_imag

        def den(x):
            return sum([(alpha * x) ** i / float(factorial(i)) for i in range(rampup + 1)]) * factorial(rampup)

        def transform_fun(x):
            num_real, num_imag = num(x)
            den_x = den(x)
            transform_real, transform_imag = num_real / den_x, num_imag / den_x
            return transform_real, transform_imag

        diff = ops.stack(mnp.meshgrid(
            *[((ops.abs(mnp.linspace(1 - n, n - 1, n)) - n) / 2 + pml_size) * d for n, d in zip(shape_padded, dxs)],
            indexing="ij"), axis=0)

        diff *= (diff > 0).astype(ms.float32) / 4.

        dist = ops.norm(diff, dim=0)
        k_k0_real, k_k0_imag = transform_fun(dist)
        ksq_r = k_k0_real + k0 ** 2
        ksq_i = k_k0_imag

        return ksq_r, ksq_i

    def construct(self, c_star, f_star, ur_init=None, ui_init=None):
        '''
        Run the solver to solve non-dimensionalized 2D acoustic equation for given c* and f*
        Args:
            c_star: float (batch_size, 1, nz, nx), the non-dimensionalized velocity field
            f_star: float (batch_size, 1, nz, nx), the mask marking out the source locations
            ur_init, ui_init: float (batch_size, 1, NZ, NX), initial wave field for iteration, real & imag parts.
              If remove_pml is True, NZ = nz, NX = nx, otherwise NZ = nz + 2 * pml_size, NX = nx + 2 * pml_size.
              Default is None, which means initialize from 0.
        '''
        vr, vi, gr, gi, rhs, eps = self.cbs_params(c_star, f_star)

        n0 = self.remove_pml * self.pml_size
        n1 = (ur_init is None or self.remove_pml) * self.pml_size
        n2 = (ui_init is None or self.remove_pml) * self.pml_size

        # construct initial field
        if ur_init is None:
            ur_init = ops.zeros_like(c_star, dtype=ms.float32) # (batch, 1, nz, nx)
        if ui_init is None:
            ui_init = ops.zeros_like(c_star, dtype=ms.float32) # (batch, 1, nz, nx)

        # pad initial field
        ur = ops.pad(ur_init, padding=[n1] * 4, value=0) # note: better padding (with gradual damping) can be applied
        ui = ops.pad(ui_init, padding=[n2] * 4, value=0) # (batch, 1, nz_padded, nx_padded)

        # start iteration
        errs_list = []

        for _ in range(self.n_iter):
            dur, dui = self.cbs_block(ur, ui, vr, vi, gr, gi, rhs, eps)
            ur += dur
            ui += dui

            # calculate iteration residual
            errs = (ops.sum(dur**2 + dui**2, dim=(-2, -1)) / ops.sum(ur**2 + ui**2, dim=(-2, -1)))**.5
            errs_list.append(errs)

        # remove pml layer
        nz, nx = ur.shape[-2:]
        ur = ur[..., n0:nz - n0, n0:nx - n0]
        ui = ui[..., n0:nz - n0, n0:nx - n0]
        ui *= -1.
        # note: the conjugate here is because we define Fourier modes differently to JAX in that the frequencies
        # are opposite, leading to opposite attenuation in PML, and finally the conjugation in results

        return ur, ui, errs_list

    def solve(self,
              c_star,
              f_star,
              ur_init=None,
              ui_init=None,
              tol=1e-3,
              max_iter=10000,
              remove_pml=True,
              print_info=True,
              ):
        """A convenient method for solving the equation to a given tolerance

        Args:
            tol (float, optional): the tolerance for the relative error. Defaults to 1e-3.
        """
        msg = 'PML layers cannot be removed during iteration, but can be removed for the final result'
        assert not self.remove_pml, msg

        ur, ui, errs_list = self(c_star, f_star, ur_init, ui_init)

        for ep in range(max_iter // self.n_iter):
            err_max = float(errs_list[-1].max())
            err_min = float(errs_list[-1].min())
            err_ave = float(errs_list[-1].mean())

            if print_info:
                print(f'step {(ep + 1) * self.n_iter}, max error {err_max:.6f}', end=', ')
                print(f'min error {err_min:.6f}, mean error {err_ave:.6f}')

            if err_max < tol:
                break

            ur, ui, errs = self(c_star, f_star, ur, -ui)
            errs_list += errs

        if remove_pml and self.pml_size:
            ur = ur[..., self.pml_size:-self.pml_size, self.pml_size:-self.pml_size]
            ui = ui[..., self.pml_size:-self.pml_size, self.pml_size:-self.pml_size]

        return ur, ui, errs_list
