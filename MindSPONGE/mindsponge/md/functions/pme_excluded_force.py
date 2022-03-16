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
'''pme excluded force'''
import math
import mindspore.numpy as np
import mindspore.ops as ops
from mindspore.ops.operations import _csr_ops
from mindspore.common import CSRTensor

from .common import get_periodic_displacement

sqrt_pi = 2/math.sqrt(3.141592654)
zero_tensor = np.array(0, dtype=np.float32)
csr_reducesum = _csr_ops.CSRReduceSum()

def pme_excluded_force(atom_numbers, beta, uint_crd, scaler, charge, excluded_csr, excluded_row):
    """
    Calculate the excluded part of long-range Coulumb force using
    PME(Particle Meshed Ewald) method. Assume the number of atoms is
    N, and the length of excluded list is E.

    Args:
        atom_numbers (int): the number of atoms, N.
        excluded_matrix (Tensor, int32): [N, M], the excluded list for each atom.
        beta (float): the PME beta parameter, determined by the
          non-bond cutoff value and simulation precision tolerance.
        uint_crd (Tensor, uint32): [N, 3], the unsigned int coordinates value of each atom.
        scaler (Tensor, float32): [3,], the scale factor between real space
          coordinates and its unsigned int value.
        charge (Tensor, float32): [N,], the charge carried by each atom.
        excluded_matrix (Tensor, int32): [N, k] containing the excluded atoms for each atom, where k
          is the maximum number of excluded atoms for all atoms.

    Outputs:
        force (Tensor, float32): [N, 3], the force felt by each atom.

    Supported Platforms:
        ``GPU``
    """
    # (N, 3)[M]-> (M, 3)
    excluded_crd = uint_crd[excluded_csr.values]
    # (N, 3)[M]-> (M, 3)
    excluded_crd_row = uint_crd[excluded_row]
    # (M, 3) - (M, 3) -> (M, 3)
    crd_d = get_periodic_displacement(excluded_crd, excluded_crd_row, scaler)
    crd_2 = crd_d ** 2
    # (M, 3) -> (M,)
    crd_sum = np.sum(crd_2, -1)
    crd_abs = np.sqrt(crd_sum)
    crd_beta = crd_abs * beta
    frc_abs = crd_beta * sqrt_pi * np.exp(-crd_beta ** 2) + ops.erfc(crd_beta)
    frc_abs = (frc_abs - 1.) / crd_sum / crd_abs
    # (N,)[M] -> (M)
    excluded_charge = charge[excluded_csr.values]
    excluded_charge_row = charge[excluded_row]
    # (M)
    charge_mul = excluded_charge * excluded_charge_row
    frc_abs = -charge_mul * frc_abs
    # (M, 1) * (M, 3) -> (M, 3)
    frc_lin = np.expand_dims(frc_abs, -1) * crd_d
    # construct CSRTensors
    indptr = excluded_csr.indptr
    indices = excluded_csr.indices
    shape = (atom_numbers, excluded_row.shape[0])
    x, y, z = np.split(-frc_lin, 3, -1)
    # (N, M) -> (N, 1)
    frc_lin_x = csr_reducesum(CSRTensor(indptr, indices, x.ravel(), shape), 1)
    frc_lin_y = csr_reducesum(CSRTensor(indptr, indices, y.ravel(), shape), 1)
    frc_lin_z = csr_reducesum(CSRTensor(indptr, indices, z.ravel(), shape), 1)
    frc_outer = np.concatenate((frc_lin_x, frc_lin_y, frc_lin_z), -1)
    res = ops.tensor_scatter_add(frc_outer, excluded_csr.values.reshape(-1, 1), frc_lin)
    return res
