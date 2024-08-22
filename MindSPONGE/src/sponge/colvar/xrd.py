# Copyright 2021-2024 @ Shenzhen Bay Laboratory &
#                       Peking University &
#                       Huawei Technologies Co., Ltd
#
# This code is a part of MindSPONGE:
# MindSpore Simulation Package tOwards Next Generation molecular modelling.
#
# MindSPONGE is open-source software based on the AI-framework:
# MindSpore (https://www.mindspore.cn/)
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
XRD3D
"""
import numpy as np
import mindspore as ms
from mindspore import Tensor, ops
from . import Colvar
from ..function import get_ms_array

class XRD3D(Colvar):
    """
    XRD3D
    """
    def __init__(self,
                 theta: float = 0,
                 lamb: float = 1,
                 pbc_box=None,
                 index=None,
                 qi=None,
                 s: float = 1,) -> None:
        super().__init__(name='xrd3d', shape=(1,))
        self.theta = np.array(theta).reshape(-1, 1)
        self.lamb = np.array(lamb).reshape(-1, 1)
        self.att_coeff = np.array(s)
        self.k0 = 4*np.pi/self.lamb*np.sin(self.theta*np.pi/180)
        self.index = get_ms_array(index, ms.int32).reshape(-1)
        self.xatom_numbers = self.index.shape[0]
        self.qi = get_ms_array(qi, ms.float32).reshape(1, -1, 1)
        self.pbc_box = pbc_box
        nfft = pbc_box[0] * 10 // 4 * 4
        self.set_nfft(nfft)
        self.fft_fijk = self.build_fft_fijk_list(self.nfft.asnumpy()[0, 0], pbc_box.asnumpy()*10)

        ma = [1.0 / 6.0, -0.5, 0.5, -1.0 / 6.0]
        ma = get_ms_array([[ma[i], ma[j], ma[k]]for i in range(4) for j in range(4) for k in range(4)], ms.float32)
        self.ma = ma.reshape(1, 1, 64, 3)
        mb = [0, 0.5, -1, 0.5]
        mb = get_ms_array([[mb[i], mb[j], mb[k]]for i in range(4) for j in range(4) for k in range(4)], ms.float32)
        self.mb = mb.reshape(1, 1, 64, 3)
        mc = [0, 0.5, 0, -0.5]
        mc = get_ms_array([[mc[i], mc[j], mc[k]]for i in range(4) for j in range(4) for k in range(4)], ms.float32)
        self.mc = mc.reshape(1, 1, 64, 3)
        md = [0, 1.0 / 6.0, 4.0 / 6.0, 1.0 / 6.0]
        md = get_ms_array([[md[i], md[j], md[k]]for i in range(4) for j in range(4) for k in range(4)], ms.float32)
        self.md = md.reshape(1, 1, 64, 3)
        self.base_grid = get_ms_array([[i, j, k] for i in range(4) for j in range(4) for k in range(4)],
                                      ms.int32).reshape(1, 1, 64, 3)

    def set_nfft(self, nfft: Tensor):
        """set nfft"""
        self.nfft = get_ms_array(nfft, ms.int32).reshape((-1, 1, 3))
        self.fftx = int(self.nfft[0][0][0])
        self.ffty = int(self.nfft[0][0][1])
        self.fftz = int(self.nfft[0][0][2])
        if self.fftx % 4 != 0 or self.ffty % 4 != 0 or self.fftz % 4 != 0:
            raise ValueError("The FFT grid number for PME must be a multiple of 4")

    def build_fft_fijk_list(self, grid_dimension, pbc_box):
        """build_fft_fijk_list"""
        volume = np.prod(pbc_box)

        # 构建B样条线插值的卷积格点
        b_spline_grid = np.zeros(grid_dimension, dtype=np.float32)
        temp_b_spline = np.array([1. / 6., 2. / 3., 1. / 6.])
        grid_serial = np.array([[i, j, k] for i in range(-1, 2) for j in range(-1, 2)
                                for k in range(-1, 2)]) % grid_dimension
        weights = temp_b_spline.reshape((-1, 1, 1))*temp_b_spline.reshape((1, -1, 1))*temp_b_spline.reshape((1, 1, -1))
        b_spline_grid[grid_serial[:, 0], grid_serial[:, 1], grid_serial[:, 2]] = weights.flatten()
        fft_b_spline = np.fft.rfftn(b_spline_grid, norm='backward').reshape(-1)
        scalor = np.pi*np.sqrt(np.pi) / volume / self.k0 / np.sqrt(self.att_coeff)/self.xatom_numbers

        grid_dimension_lower_half_plus_one = (grid_dimension // 2) + 1
        grid_dimension_upper_half = (grid_dimension + 1) // 2
        nzny_nx21_numbers = grid_dimension[0] * grid_dimension[1] * grid_dimension_lower_half_plus_one[2]

        grid_i = np.arange(nzny_nx21_numbers).reshape(-1, 1).repeat(3, 1)
        fft_fijk = np.zeros(nzny_nx21_numbers)
        grid_layer_numbers = grid_dimension[1]*grid_dimension_lower_half_plus_one[2]
        grid_i[:, 2] = grid_i[:, 2] % grid_dimension_lower_half_plus_one[2]
        grid_i[:, 1] = np.int32((grid_i[:, 1] % grid_layer_numbers) / grid_dimension_lower_half_plus_one[2])
        grid_i[:, 0] = np.int32(grid_i[:, 0] / grid_layer_numbers)
        temp = grid_i - grid_dimension_upper_half
        temp = np.right_shift(temp, 31)
        grid_fft_serial = (temp & grid_i) + ((~temp) & (grid_dimension - grid_i))
        pi2ijk = 2. * np.pi * np.linalg.norm(grid_fft_serial / pbc_box, axis=-1)
        pi2ijk_2 = pi2ijk**2 + self.k0**2
        rk0 = 2.*pi2ijk*self.k0
        fft_fijk = scalor * 1. / pi2ijk*(np.exp(-0.25/self.att_coeff*(pi2ijk_2 - rk0))
                                         - np.exp(-0.25/self.att_coeff*(pi2ijk_2 + rk0))) / (np.real(fft_b_spline))**2
        fft_fijk[:, 0] = 0
        return get_ms_array(fft_fijk.reshape(-1, grid_dimension[0], grid_dimension[1],
                                             grid_dimension_lower_half_plus_one[2]), ms.float32)

    def construct(self, coordinate: Tensor, pbc_box: Tensor = None):

        c = coordinate[:, self.index]
        pbc_box = self.pbc_box.reshape((-1, 1, 3))
        frac = c / ops.stop_gradient(pbc_box) % 1.0 * self.nfft
        grid = ops.Cast()(frac, ms.int32)
        frac = frac - ops.floor(frac)

        # (B,A,64,3) <- (B,A,1,3) + (1,1,64,3)
        neibor_grids = ops.expand_dims(grid, 2) - self.base_grid
        neibor_grids %= ops.expand_dims(self.nfft, 2)

        # (B,A,64,3) <- (B,A,1,3) * (1,1,64,3)
        frac = ops.expand_dims(frac, 2)
        neibor_q = frac * frac * frac * self.ma + frac * \
            frac * self.mb + frac * self.mc + self.md

        # (B,A,64) <- (B,A,1) * reduce (B,A,64,3)
        neibor_q = self.qi * ops.ReduceProd()(neibor_q, -1)

        # (B,A,64,4) <- concat (B,A,64,1) (B,A,64,3)
        batch_constant = ops.ones((c.shape[0], c.shape[1], 64, 1), dtype=ms.int32)
        batch_constant = batch_constant * ops.arange(0, c.shape[0], dtype=ms.int32).reshape(-1, 1, 1, 1)
        neibor_grids = ops.concat((batch_constant, neibor_grids), -1)

        # (B, fftx, ffty, fftz)
        q_matrix = ops.zeros([c.shape[0], self.fftx, self.ffty, self.fftz], ms.float32)
        q_matrix = ops.tensor_scatter_add(q_matrix, neibor_grids.reshape(-1, 4), neibor_q.reshape(-1))

        # XRD
        fq = ops.FFTWithSize(3, False, True, norm='backward')(q_matrix)
        fq_real = self.fft_fijk * ops.stop_gradient(fq.real())
        fq_imag = self.fft_fijk * ops.stop_gradient(fq.imag())
        fq_fijk = ops.Complex()(fq_real, fq_imag)
        # fq_fijk = fq*self.fft_fijk.unsqueeze(0)
        phi = ops.FFTWithSize(3, True, True, norm='forward')(fq_fijk)
        intensity = phi * q_matrix
        return intensity.sum(axis=(1, 2, 3)).unsqueeze(-1)
