# Copyright 2021 Huawei Technologies Co., Ltd
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
'''common'''
import math

import numpy as onp

from mindspore import ops
from mindspore.ops import constexpr
import mindspore.numpy as np

PI = 3.1415926

@constexpr
def get_excluded_index(atom_numbers, excluded_list_size):
    return np.tile(np.arange(excluded_list_size), (atom_numbers, 1))

@constexpr
def get_neighbour_index(atom_numbers, neighbour_list_size):
    return np.tile(np.arange(neighbour_list_size), (atom_numbers, 1))

@constexpr
def get_atom_list_tensor(atom_numbers):
    return np.arange(atom_numbers).reshape(-1, 1)

@constexpr
def get_zero_tensor(shape, dtype=np.float32):
    return np.zeros(shape, dtype)

@constexpr
def get_full_tensor(shape, fill_value, dtype=np.float32):
    return np.full(shape, fill_value, dtype)

@constexpr
def get_range_tensor(stop):
    return np.arange(stop)

def get_periodic_displacement(x, y, scaler):
    int_dr = x - y
    int_dr = int_dr.astype('int32')
    int_dr = ops.depend(int_dr, int_dr)
    return int_dr * scaler

def reform_excluded_list(excluded_list, excluded_list_start, exlcuded_list_number):
    """Re-format excluded list: (E,) -> (N, MAX_NEIGHBOUR)"""
    atom_numbers = excluded_list_start.shape[0]
    max_neighbour = exlcuded_list_number.max()
    excluded_list_end = excluded_list_start + exlcuded_list_number
    excluded_matrix = onp.full((atom_numbers, max_neighbour), -1, onp.int32)
    for i in range(atom_numbers):
        excluded_matrix[i, :exlcuded_list_number[i]] = excluded_list[excluded_list_start[i]:excluded_list_end[i]]
    return excluded_matrix

def cuCdiv(real1, imag1, real2, imag2):
    """
    Compute the quotient of the given complex numbers.
    Assuming that `x` is the complex number composed of
    the real part `real1` and imaginary part `imag1`,
    while `y` consists of `real2` and `imag2`,
    the result is computed by the equation `x / y`.
    """
    divisor = real2 * real2 + imag2 * imag2
    res_real = (real1 * real2 + imag1 * imag2) / divisor
    res_imag = (imag1 * real2 - real1 * imag2) / divisor
    return res_real, res_imag

def expc(real, imag):
    t = math.exp(real)
    res_real = math.cos(imag) * t
    res_imag = math.sin(imag) * t
    return res_real, res_imag

def M_(u, n):
    if n == 2:
        if u > 2 or u < 0:
            return 0
        return 1 - abs(u - 1)
    return u / (n - 1) * M_(u, n - 1) + (n - u) / (n - 1) * M_(u - 1, n - 1)

def getb(k, nfft, b_order):
    '''getb'''
    tempc2_real = 0.
    tempc2_imag = 0.
    tempc_real = 0.
    tempc_imag = 2 * (b_order - 1) * PI * k / nfft
    res_real, res_imag = expc(tempc_real, tempc_imag)
    for kk in range(b_order - 1):
        tempc_real = 0
        tempc_imag = 2 * PI * k / nfft * kk
        tempc_real, tempc_imag = expc(tempc_real, tempc_imag)
        tempf = M_(kk + 1, b_order)
        tempc2_real += tempf * tempc_real
        tempc2_imag += tempf * tempc_imag
    res_real, res_imag = cuCdiv(res_real, res_imag, tempc2_real, tempc2_imag)
    return res_real * res_real + res_imag * res_imag

def get_pme_bc(fftx, ffty, fftz, box, beta):
    '''get pme_bc'''
    z_range = fftz // 2 + 1

    B1 = list(map(lambda i: getb(i, fftx, 4), range(fftx)))
    B2 = list(map(lambda i: getb(i, ffty, 4), range(ffty)))
    B3 = list(map(lambda i: getb(i, fftz, 4), range(z_range)))

    mprefactor = PI * PI / -beta / beta
    volume = box[0] * box[1] * box[2]

    kxend = int(fftx / 2)
    kxrp_l = onp.arange(kxend + 1)
    kxrp_r = onp.arange(fftx - kxend -1, 0, -1)
    # (fftx, 1, 1)
    kxrp = onp.hstack((kxrp_l, kxrp_r)).reshape(-1, 1, 1) / box[0]

    kyend = int(ffty / 2)
    kyrp_l = onp.arange(kyend + 1)
    kyrp_r = onp.arange(ffty - kyend -1, 0, -1)
    # (1, ffty, 1)
    kyrp = onp.hstack((kyrp_l, kyrp_r)).reshape(1, -1, 1) / box[1]

    # (1, 1, fftz / 2 + 1)
    kzrp = onp.arange(z_range).reshape(1, 1, -1) / box[2]

    # (fftx, ffty, fftz / 2 + 1)
    msq = kxrp * kxrp + kyrp * kyrp + kzrp * kzrp
    B1 = onp.broadcast_to(
        onp.array(B1).reshape(-1, 1, 1), msq.shape).ravel() # pylint: disable=too-many-function-args
    B2 = onp.broadcast_to(
        onp.array(B2).reshape(1, -1, 1), msq.shape).ravel() # pylint: disable=too-many-function-args
    B3 = onp.broadcast_to(
        onp.array(B3).reshape(1, 1, -1), msq.shape).ravel() # pylint: disable=too-many-function-args
    msq = msq.ravel()

    PME_BC = onp.zeros_like(msq)
    msq = msq[1:]
    PME_BC[1:] = 1.0 / PI / msq * onp.exp(mprefactor * msq) / volume
    PME_BC = PME_BC * B1 * B2 * B3

    return PME_BC
