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
"""CBS solver for solving acoustic equation in frequency domain"""
import warnings
from math import factorial
from time import time as toc
import numpy as np
import mindspore as ms
from mindspore import Tensor, nn, ops, mint, numpy as mnp, lazy_inline
from mindscience import DFTn, IDFTn, DST, IDST


class MixedDSTDFTn(nn.Cell):
    ''' The mixed computation of DST and DFT in CBS iterations '''
    def __init__(self, shape):
        """Configurations of mixed DST and DFT

        Args:
            shape (tuple[int]): Only the spatial shape, not including the batch and channel dimensions.
        """
        super().__init__()
        self.dst_cell = DST(shape[:1])
        self.dft_cell = DFTn(shape[1:])

    def construct(self, ar, ai):
        '''
        Get mixed DST and DFT of the input
        Args:
            ar: float (batch_size, 1, nz, nx) for 2D,
                float (batch_size, 1, nz, ny, nx) for 3D,
                the real part of the input tensor.
            ai: float (batch_size, 1, nz, nx) for 2D,
                float (batch_size, 1, nz, ny, nx) for 3D,
                the imaginary part of the input tensor.
        '''
        dft_ar, dft_ai = self.dft_cell(ar, ai)
        dft_ar = ops.swapdims(dft_ar, 2, -1) # Move z to last axis for DST
        dft_ar = self.dst_cell(dft_ar)
        dst_dft_ar = ops.swapdims(dft_ar, 2, -1)
        dft_ai = ops.swapdims(dft_ai, 2, -1)
        dft_ai = self.dst_cell(dft_ai)
        dst_dft_ai = ops.swapdims(dft_ai, 2, -1)

        return dst_dft_ar, dst_dft_ai


class MixedIDSTDFTn(nn.Cell):
    ''' The mixed computation of the inverse of DST and DFT in CBS iterations '''
    def __init__(self, shape):
        """Configurations of the inverse of mixed DST and DFT

        Args:
            shape (tuple[int]): Only the spatial shape, not including the batch and channel dimensions.
        """
        super().__init__()
        self.idst_cell = IDST(shape[:1])
        self.idft_cell = IDFTn(shape[1:])

    def construct(self, ar, ai):
        '''
        Get the inverse of mixed DST and DFT of the input
        Args:
            ar: float (batch_size, 1, nz, nx) for 2D,
                float (batch_size, 1, nz, ny, nx) for 3D,
                the real part of the input tensor.
            ai: float (batch_size, 1, nz, nx) for 2D,
                float (batch_size, 1, nz, ny, nx) for 3D,
                the imaginary part of the input tensor.
        '''
        ar = ops.swapdims(ar, 2, -1)
        ar = self.idst_cell(ar)
        idst_ar = ops.swapdims(ar, 2, -1)
        ai = ops.swapdims(ai, 2, -1)
        ai = self.idst_cell(ai)
        idst_ai = ops.swapdims(ai, 2, -1)
        idft_idst_ar, idft_idst_ai = self.idft_cell(idst_ar, idst_ai)

        return idft_idst_ar, idft_idst_ai


class CBSBlock(nn.Cell):
    ''' The computation procedures for each iteration in CBS '''
    @lazy_inline
    def __init__(self, shape, btype):
        '''
        No trainable parameters, but the dft cells needs initialization
        Args:
            shape: Tuple of int, only the spatial shape, not including the batch and channel dimensions
        '''
        super().__init__()
        self.btype = btype
        if self.btype == "freesurface":
            self.dft_cell = MixedDSTDFTn(shape)
            self.idft_cell = MixedIDSTDFTn(shape)
        else:
            self.dft_cell = DFTn(shape)
            self.idft_cell = IDFTn(shape)

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
        ''' Run one iteration and return the incremental '''
        vur, vui = self.op_v(ur, ui, vr, vi)
        gvr, gvi = self.op_g(vur + rhs, vui, gr, gi)
        vgr, vgi = self.op_v(gvr - ur, gvi - ui, vr, vi)

        # eps > 0: Convergent Born series; eps == 0: Original Born Series
        cond = ops.broadcast_to(eps, ur.shape) > 0
        dur = ops.select(cond, -vgi / (eps + 1e-8), gvr - ur) # '* (-1.)' comes from imag part multiplying i/eps
        dui = ops.select(cond, vgr / (eps + 1e-8), gvi - ui)

        return ops.stack([dur, dui])

class CBS(nn.Cell):
    ''' The CBS cell for solving 2D/3D Helmholtz equation using the Convergent Born Series (CBS)
        method described in `Osnabrugge et al, 2016 <https://doi.org/10.1016/j.jcp.2016.06.034>`_ .
        It is an iterative solver with spectral precision due to the Fourier approximation to
        the differential terms. This cell can be used for acoustic wave simulation, or be adapted to
        physics loss for self-supervised training of neural operators. '''
    def __init__(self,
                 shape,
                 dxs=None,
                 n_iter=20,
                 pml_size=None,
                 alpha=1.0,
                 rampup=12,
                 btype="pml",
                 remove_pml=True,
                 epsilon=None,
                 ):
        """Configurations of the CBS solver

        Args:
            shape (tuple[int]): Only the spatial shape, not including the batch and channel dimensions.
            dxs (tuple[float], optional): The grid interval along z & x (2D) or z & y & x (3D) directions
                under nondimensionalization. Defaults to None, it represents assigning a default value
                `(1,)*len(shape)`.
            n_iter (int, optional): Number of iterations in a single call. Defaults to 20.
            pml_size (int, tuple[int], optional): Number of grid layers to pad on each boundary for the wave
            to attenuate. The order is (z_up, z_down, x_left, x_right) for 2D and (z_up, z_down, y_left, y_right,
            x_front, x_back) for 3D. Defaults to None.
                1) If None, assign a default value (If `btype` is `"pml"`, the size of absorption boundary
                   in each direction is `shape[-1]//4`, if `btype` is `"freesurface"`, except for the upper
                   surface absorption boundary size of 0, the rest of the absorption boundary sizes are
                   `shape[-1]//4`).
                2) If an int, apply this `pml_size` to all boundaries.
                3) If a tuple[int] (length 2*len(shape)), specify `pml_size` for each boundary
                   individually.
            alpha (float, optional): The strength of wave attenuation in PML layers. Defaults to 1.0.
            rampup (int, optional): The smoothness of transition from interior domain to PML layers. Defaults to 12.
            btype (string, optional): Boundary type (choose one of) pml freesurface. Defaults to 'pml'.
            remove_pml (bool, optional): Whether to remove the PML layers for the output. Defaults to True.
            epsilon (float, optional): The small value to stabilize the iteration.
                Defaults to None, calculating epsilon automatically.
        """
        super().__init__()

        if (dxs is not None) and (len(dxs) != len(shape)):
            raise ValueError("Length of the shape and dxs should be equal.")
        if btype not in ["pml", "freesurface"]:
            raise ValueError("The boundary type should be one of `pml` and `freesurface`.")

        self.shape = shape
        self.n_iter = n_iter
        self.alpha = alpha
        self.rampup = rampup
        self.btype = btype
        self.remove_pml = remove_pml
        self.epsilon = epsilon
        self.dim = len(shape)

        if pml_size is None:
            s = shape[-1] // 4
            self.pml_size = (s * int(btype == "pml"),) + (s,) * (2 * self.dim - 1)
        elif isinstance(pml_size, int):
            self.pml_size = (pml_size * int(btype == "pml"),) + (pml_size,) * (2 * self.dim - 1)
        else:
            self.pml_size = (int(btype == "pml") * pml_size[0],) + pml_size[1:]

        self.pml_size = list(self.pml_size)
        for i, (a, b) in enumerate(zip(self.pml_size, [d for pair in zip(shape, shape) for d in pair])):
            if a > b:
                self.pml_size[i] = b
                warnings.warn("The size of absorbing boundary should not be larger than the original "
                              f"grid ({a} > {b}), CBS.pml_size[{i}] will be assigned as {b}.",
                              category=UserWarning)
        self.pml_size = tuple(self.pml_size)

        if dxs is None:
            self.dxs = (1.0,) * self.dim
        else:
            self.dxs = dxs

        # Coordination reverse (x, y, z) -> (z, y, x)
        self.pml_size_xyz = [
            num
            for i in range(len(self.pml_size)-2, -2, -2)
            for num in self.pml_size[i:i+2]
        ]

        self.shape_padded = tuple(n + s1 + s2 for n, s1, s2 in zip(shape, self.pml_size[0::2], self.pml_size[1::2]))

        if self.btype == "freesurface":
            # DST wave number for 1st-order derivative
            fftfreq = [(np.pi / self.dxs[0] / self.shape_padded[0] * np.arange(1, self.shape_padded[0]+1))**2]
            fftfreq += [4 * np.pi**2 * np.fft.fftfreq(n, d)**2 for n, d in zip(self.shape_padded[1:], self.dxs[1:])]
            p_sq = sum(np.meshgrid(*fftfreq, indexing="ij"))
        else:
            p_sq = sum(np.meshgrid(
                *[np.fft.fftfreq(n, d)**2 for n, d in zip(self.shape_padded, self.dxs)],
                indexing="ij")) * (2 * np.pi)**2

        self.p_sq = Tensor(p_sq, dtype=ms.float32, const_arg=True)

        self.pml_mask = 1 - ops.pad(ops.ones(shape), self.pml_size_xyz)

        self.cbs_block = CBSBlock(self.shape_padded, self.btype)

    def cbs_params(self, c_star, f_star):
        ''' Compute constant variables for CBS iteration '''
        omg = 1.0

        # Source field
        # Shape (batch, 1, nz_padded, nx_padded) for 2D or (batch, 1, nz_padded, ny_padded, nx_padded) for 3D
        rhs = ops.pad(f_star / c_star**2, self.pml_size_xyz)

        # Homogeneous k field
        axis = tuple(range(-self.dim, 0, 1))
        k_max = omg / ops.amin(c_star, axis=axis, keepdims=True)
        k_min = omg / ops.amax(c_star, axis=axis, keepdims=True)
        k0 = ops.sqrt(0.5 * (k_max**2 + k_min**2)) # (batch, 1, 1, 1)
        # Mirror padding gradually approaching `values`
        k_padded = self.extrapolate(omg / c_star, self.pml_size_xyz, values=k_max)

        # Heterogeneous k field
        # Shape (batch, 1, nz_padded, nx_padded) for 2D or (batch, 1, nz_padded, ny_padded, nx_padded) for 3D
        ksq_r, ksq_i = self.cbs_pml(self.shape, self.dxs, k_padded, self.pml_size,
                                    self.alpha, self.rampup)

        ksq_r = ksq_r * self.pml_mask + ops.pad((omg / c_star)**2, self.pml_size_xyz) * (1 - self.pml_mask)
        ksq_i = ksq_i * self.pml_mask

        # Shape (batch, 1, 1, 1) for 2D, (batch, 1, 1, 1, 1) for 3D
        eps = ops.amax((ksq_r - k0**2)**2 + ksq_i**2, axis=axis, keepdims=True)**.5

        # If epsilon given by user, use original BS instead of CBS
        if isinstance(self.epsilon, (float, int)):
            eps = self.epsilon * ops.ones_like(eps)

        # Field variables needed by operator V & G
        # Shape (batch, 1, nz_padded, nx_padded) for 2D or (batch, 1, nz_padded, ny_padded, nx_padded) for 3D
        vr = ksq_r - k0**2
        vi = ksq_i - eps
        gr = 1. / ((self.p_sq - k0**2)**2 + eps**2) * (self.p_sq - k0**2)
        gi = 1. / ((self.p_sq - k0**2)**2 + eps**2) * eps

        return vr, vi, gr, gi, rhs, eps * (self.epsilon is None)

    @staticmethod
    def cbs_pml(shape, dxs, k0, pml_size, alpha, rampup):
        ''' Construct the heterogeneous k field with PML BC embedded '''
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

        def pml_padding(n, d, s1, s2):
            original = (ops.abs(mnp.linspace(1 - n, n - 1, n)) - n) * (d / 2)
            left = ops.arange((s1 - 0.5)*d, -d/2, -d, dtype=ms.float32)
            right = ops.arange(d/2, (s2 + 0.5)*d, d, dtype=ms.float32)
            return ops.concat((left, original, right))

        diff = ops.stack(mnp.meshgrid(
            *[pml_padding(n, d, s1, s2)
              for n, d, s1, s2 in zip(shape, dxs, pml_size[0::2], pml_size[1::2])], indexing="ij"), axis=0)

        # Reference: https://github.com/ucl-bug/jwave/blob/9d114ac6acade3c866e78a9984be47833119a9f9/
        #            jwave/acoustics/time_harmonic.py#L153
        diff *= (diff > 0).astype(ms.float32) / 4.

        dist = ops.norm(diff, dim=0)
        k_k0_real, k_k0_imag = transform_fun(dist)
        ksq_r = k_k0_real + k0 ** 2
        ksq_i = k_k0_imag

        return ksq_r, ksq_i

    @staticmethod
    def extrapolate(data, padding, values):
        ''' Mirror padding gradually approaching `values` '''
        ndim = len(padding) // 2
        for d in range(ndim):
            n0 = data.shape[-1 - d] # current dimension length
            n1 = padding[2 * d] # front padding
            n2 = padding[2 * d + 1] # back padding
            n = n1 + n0 + n2 # padded length

            # build matrix accessing mirror elements near boundary and applying attenuation
            weights = np.ones(n)
            weights[:n1] = np.arange(n1) / (n1 - 1)
            weights[n1+n0:] = np.arange(n2)[::-1] / (n2 - 1)
            indices = n0 - 1 - np.abs(n0 - 1 - np.abs(np.arange(n) - n1))
            indices[:n1] -= 1
            indices[n1+n0:] += 1
            flipmat = np.zeros([n, n0])
            flipmat[range(n), indices] = weights

            weights = ms.Tensor(weights, dtype=data.dtype)
            flipmat = ms.Tensor(flipmat, dtype=data.dtype)

            # implement padding as a matmul
            data = ops.swapaxes(data, -1 - d, -2)
            data = mint.matmul(flipmat, data) + values * (1 - weights)[:, None]
            data = ops.swapaxes(data, -1 - d, -2)

        return data

    def compute_error(self, dur, dui, ur, ui):
        ''' Compute the relative error of du '''
        dim = tuple(range(-self.dim, 0, 1))
        return (ops.sum(dur**2 + dui**2, dim=dim) / ops.sum(ur**2 + ui**2, dim=dim))**.5

    def construct(self, c_star, f_star, ur_init=None, ui_init=None):
        '''
        Run the solver to solve non-dimensionalized 2D/3D acoustic equation for given c* and f*
        Args:
            c_star: float (batch_size, 1, nz, nx) for 2D,
                    float (batch_size, 1, nz, ny, nx) for 3D,
                    the non-dimensionalized velocity field.
            f_star: float (batch_size, 1, nz, nx) for 2D,
                    float (batch_size, 1, nz, ny, nx) for 3D,
                    the mask marking out the source locations.
            ur_init, ui_init: float (batch_size, 1, NZ, NX) for 2D,
                              float (batch_size, 1, NZ, NY, NX) for 3D,
                              initial wave field for iteration, real & imag parts.
              In 2D, if remove_pml is True, NZ = nz, NX = nx, otherwise
              NZ = nz + pml_size[0] + pml_size[1], NX = nx + pml_size[2] + pml_size[3].
              In 3D, if remove_pml is True, NZ = nz, NY = ny, NX = nx, otherwise
              NZ = nz + pml_size[0] + pml_size[1], NY = ny + pml_size[2] + pml_size[3],
              NX = nx + pml_size[4] + pml_size[5].
              Default is None, which means initialize from 0.
        '''
        vr, vi, gr, gi, rhs, eps = self.cbs_params(c_star, f_star)

        # Padding under x-y-z coordination
        n0 = [self.remove_pml * s for s in self.pml_size]
        paddingr = [(ur_init is None or self.remove_pml) * s for s in self.pml_size_xyz]
        paddingi = [(ui_init is None or self.remove_pml) * s for s in self.pml_size_xyz]

        # Construct initial field
        if ur_init is None:
            ur_init = ops.zeros_like(c_star, dtype=ms.float32) # (batch, 1, nz, nx)
        if ui_init is None:
            ui_init = ops.zeros_like(c_star, dtype=ms.float32) # (batch, 1, nz, nx)

        # Pad initial field
        # Note: here u_init is conjugated, because the output is also conjugated
        ur = ops.pad(ur_init, padding=paddingr, value=0) # Note: better padding (with gradual damping) can be applied
        # Shape (batch, 1, nz_padded, nx_padded) for 2D, (batch, 1, nz_padded, ny_padded, nx_padded) for 3D
        ui = ops.pad(0 - ui_init, padding=paddingi, value=0)

        # Start iteration
        errs_list = []

        for _ in range(self.n_iter):
            dur, dui = self.cbs_block(ur, ui, vr, vi, gr, gi, rhs, eps)
            ur = ur + dur
            ui = ui + dui

            # Calculate iteration residual
            errs = self.compute_error(dur, dui, ur, ui)
            errs_list.append(errs)

        # Remove pml layer
        if self.dim == 2:
            nz, nx = ur.shape[-self.dim:]
            ur = ur[..., n0[0]:nz - n0[1], n0[2]:nx - n0[3]]
            ui = ui[..., n0[0]:nz - n0[1], n0[2]:nx - n0[3]]
        else:
            nz, ny, nx = ur.shape[-self.dim:]
            ur = ur[..., n0[0]:nz - n0[1], n0[2]:ny - n0[3], n0[4]:nx - n0[5]]
            ui = ui[..., n0[0]:nz - n0[1], n0[2]:ny - n0[3], n0[4]:nx - n0[5]]
        ui *= -1.
        # Note: the conjugate here is because we define Fourier modes differently to JAX in that the frequencies
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
            c_star: float (batch_size, 1, nz, nx) for 2D,
                    float (batch_size, 1, nz, ny, nx) for 3D,
                    the non-dimensionalized velocity field.
            f_star: float (batch_size, 1, nz, nx) for 2D,
                    float (batch_size, 1, nz, ny, nx) for 3D,
                    the mask marking out the source locations.
            ur_init, ui_init: float (batch_size, 1, NZ, NX) for 2D,
                              float (batch_size, 1, NZ, NY, NX) for 3D,
                              initial wave field for iteration, real & imag parts.
              In 2D, if remove_pml is True, NZ = nz, NX = nx, otherwise
              NZ = nz + pml_size[0] + pml_size[1], NX = nx + pml_size[2] + pml_size[3].
              In 3D, if remove_pml is True, NZ = nz, NY = ny, NX = nx, otherwise
              NZ = nz + pml_size[0] + pml_size[1], NY = ny + pml_size[2] + pml_size[3],
              NX = nx + pml_size[4] + pml_size[5].
              Default is None, which means initialize from 0.
            tol (float, optional): The tolerance for the relative error. Defaults to 1e-3.
            max_iter (int, optional): The maximum iterations. Defaults to 10000.
            remove_pml (bool, optional): Whether to remove the PML layers for the output. Defaults to True.
            print_info (bool, optional): Whether to print the information of solution. Defaults to True.
        """
        msg = 'PML layers cannot be removed during iteration, but can be removed for the final result'
        assert not self.remove_pml, msg

        tic = toc()

        ur, ui, errs_list = self(c_star, f_star, ur_init, ui_init)

        for ep in range(max_iter // self.n_iter):
            err_max = float(errs_list[-1].max())
            err_min = float(errs_list[-1].min())
            err_ave = float(errs_list[-1].mean())

            if print_info:
                print(f'step {(ep + 1) * self.n_iter}, max error {err_max:.6f}', end=', ')
                print(f'min error {err_min:.6f}, mean error {err_ave:.6f}', end=', ')
                print(f'mean step time {(toc() - tic) / self.n_iter:.4f}s')
                tic = toc()

            if err_max < tol:
                print(f"CBS solve converged within TOL {err_max}")
                break

            ur, ui, errs = self(c_star, f_star, ur, ui)
            errs_list += errs

        if remove_pml:
            if self.dim == 2:
                ur = ur[..., self.pml_size[0]:self.pml_size[0]+self.shape[0],
                        self.pml_size[2]:self.pml_size[2]+self.shape[1]]
                ui = ui[..., self.pml_size[0]:self.pml_size[0]+self.shape[0],
                        self.pml_size[2]:self.pml_size[2]+self.shape[1]]
            else:
                ur = ur[..., self.pml_size[0]:self.pml_size[0]+self.shape[0],
                        self.pml_size[2]:self.pml_size[2]+self.shape[1],
                        self.pml_size[4]:self.pml_size[4]+self.shape[2]]
                ui = ui[..., self.pml_size[0]:self.pml_size[0]+self.shape[0],
                        self.pml_size[2]:self.pml_size[2]+self.shape[1],
                        self.pml_size[4]:self.pml_size[4]+self.shape[2]]

        return ur, ui, errs_list
