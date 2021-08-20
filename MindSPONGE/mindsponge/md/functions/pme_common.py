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
'''pme common'''
import mindspore.numpy as mnp
from mindspore import ops
from mindspore.ops import functional as F
from mindspore.ops import constexpr

from .common import get_neighbour_index, get_periodic_displacement

PERIODIC_FACTOR_INVERSE = 2.32830643e-10
PME_Ma = mnp.array([1.0 / 6.0, -0.5, 0.5, -1.0 / 6.0])
PME_Mb = mnp.array([0, 0.5, -1, 0.5])
PME_Mc = mnp.array([0, 0.5, 0, -0.5])
PME_Md = mnp.array([0, 1.0 / 6.0, 4.0 / 6.0, 1.0 / 6.0])
fft3d = ops.FFT3D()
ifft3d = ops.IFFT3D()


@constexpr
def to_tensor(args, dtype=mnp.float32):
    return mnp.array(args)


def Scale_List(element_numbers, tensor, scaler):
    """Scale values in `tensor`."""
    if tensor.ndim > 0 and len(tensor) > element_numbers:
        tensor = tensor[:element_numbers]
    return tensor * scaler


def PME_Atom_Near(uint_crd, PME_atom_near, PME_Nin, periodic_factor_inverse_x,
                  periodic_factor_inverse_y, periodic_factor_inverse_z, atom_numbers,
                  fftx, ffty, fftz, PME_kxyz, PME_uxyz):
    '''pme atom near'''
    periodic_factor_inverse_xyz = to_tensor(
        (periodic_factor_inverse_x, periodic_factor_inverse_y, periodic_factor_inverse_z))
    # (N, 3)
    tempf = uint_crd.astype('float32') * periodic_factor_inverse_xyz
    tempu = tempf.astype('int32')
    tempu = ops.depend(tempu, tempu)
    PME_frxyz = tempf - tempu

    cond = mnp.not_equal(PME_uxyz.astype(mnp.int32), tempu).any(1, True)
    PME_uxyz = mnp.where(cond, tempu, PME_uxyz)

    tempu = tempu.reshape(atom_numbers, 1, 3)
    kxyz = tempu - PME_kxyz.astype(mnp.int32)
    kxyz_plus = kxyz + mnp.array([fftx, ffty, fftz])
    kxyz = ops.select(kxyz < 0, kxyz_plus, kxyz)

    kxyz = kxyz * to_tensor((PME_Nin, fftz, 1), mnp.int32).reshape(1, 1, 3)
    temp_near = mnp.sum(kxyz.astype(mnp.float32), -1).astype(mnp.int32)
    PME_atom_near = mnp.where(cond, temp_near, PME_atom_near)

    return PME_frxyz, PME_uxyz, PME_atom_near


def PME_Q_Spread(PME_atom_near, charge, PME_frxyz, PME_Q, PME_kxyz, atom_numbers):
    '''pme q spread'''
    PME_kxyz = PME_kxyz.astype(mnp.int32)
    pme_ma = PME_Ma[PME_kxyz]
    pme_mb = PME_Mb[PME_kxyz]
    pme_mc = PME_Mc[PME_kxyz]
    pme_md = PME_Md[PME_kxyz]

    tempf = PME_frxyz.reshape(atom_numbers, 1, 3) # (N, 1, 3)
    tempf2 = tempf * tempf # (N, 1, 3)
    temp_charge = charge.reshape(atom_numbers, 1) # (N, 1)

    tempf = pme_ma * tempf * tempf2 + pme_mb * tempf2 + pme_mc * tempf + pme_md # (N, 64, 3)

    tempQ = temp_charge * tempf[..., 0] * tempf[..., 1] * tempf[..., 2] # (N, 64)
    index = PME_atom_near.ravel() # (N * 64,)
    tempQ = tempQ.ravel() # (N * 64,)
    PME_Q = ops.tensor_scatter_add(PME_Q, mnp.expand_dims(index, -1), tempQ)

    return PME_Q


def PME_Direct_Energy(atom_numbers, nl_numbers, nl_serial, uint_crd, boxlength, charge, beta, cutoff_square):
    '''pme direct energy'''
    r2 = uint_crd[nl_serial]

    dr_xyz = get_periodic_displacement(r2, mnp.expand_dims(uint_crd, 1), boxlength)
    dr2 = mnp.sum(dr_xyz * dr_xyz, -1)

    dr_abs = mnp.sqrt(dr2)
    charge_i = charge.reshape(-1, 1)
    charge_j = charge[nl_serial]

    ene_temp = charge_i * charge_j * ops.erfc(beta * dr_abs)
    where_zeros = dr_abs == 0.
    dr_abs[where_zeros] = 1.
    ene_temp = ene_temp / dr_abs

    idx = get_neighbour_index(atom_numbers, nl_serial.shape[1])
    mask = mnp.logical_and(dr2 < cutoff_square, idx < mnp.expand_dims(nl_numbers, -1))

    ene_lin = mnp.sum(ene_temp * mask)
    return ene_lin


@constexpr
def get_pme_kxyz():
    k = mnp.arange(4)
    x = mnp.repeat(k, 16).reshape(64, 1)
    y = F.tile(mnp.repeat(k, 4), (4, 1)).reshape(64, 1)
    z = F.tile(k, (16, 1)).reshape(64, 1)
    pme_kxyz = mnp.column_stack((x, y, z)).astype('uint32')
    return pme_kxyz


def PME_Energy_Reciprocal(real, imag, BC):
    return mnp.sum((real * real + imag * imag) * BC)


def PME_Energy_Product(tensor1, tensor2):
    return mnp.sum(tensor1 * tensor2)


def PME_Excluded_Energy_Correction(atom_numbers, uint_crd, scaler, charge, pme_beta, sqrt_pi, excluded_matrix):
    '''pme excluded energy correction'''
    mask = (excluded_matrix > -1)
    # (N, 3)[N, M]-> (N, M, 3)
    excluded_crd = uint_crd[excluded_matrix]

    # (N, M, 3)
    dr_xyz = get_periodic_displacement(excluded_crd, mnp.expand_dims(uint_crd, 1), scaler)
    # (N, M)
    dr2 = mnp.sum(dr_xyz * dr_xyz, -1)
    dr_abs = mnp.sqrt(dr2)
    beta_dr = pme_beta * dr_abs

    # (N,)[N, M] -> (N, M)
    excluded_charge = charge[excluded_matrix]
    # (N, 1) * (N, M) -> (N, M)
    charge_mul = mnp.expand_dims(charge, 1) * excluded_charge

    # (N, M)
    ene_lin = charge_mul * ops.erf(beta_dr)
    where_zeros = dr_abs == 0.
    dr_abs[where_zeros] = 1.
    ene_lin = ene_lin / dr_abs * mask

    return 0. - mnp.sum(ene_lin * mask)
