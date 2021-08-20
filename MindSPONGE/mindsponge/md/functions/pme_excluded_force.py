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

from .common import get_periodic_displacement

excluded_numbers = 2719
sqrt_pi = 2/math.sqrt(3.141592654)
zero_tensor = np.array(0, dtype=np.float32)
zero_indices = np.zeros((excluded_numbers, 1), dtype=np.int32)
one_indices = zero_indices + 1
two_indices = one_indices + 1
def pme_excluded_force(atom_numbers, beta, uint_crd, scaler, charge, excluded_matrix):
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
    # (N, M)
    mask = (excluded_matrix > -1)
    # (N, 3)[N, M]-> (N, M, 3)
    excluded_crd = uint_crd[excluded_matrix]
    # (N, M, 3) - (N, 1, 3) -> (N, M, 3)
    crd_d = get_periodic_displacement(excluded_crd, np.expand_dims(uint_crd, 1), scaler)
    crd_2 = crd_d ** 2
    # (N, M, 3) -> (N, M)
    crd_sum = np.sum(crd_2, -1)
    crd_abs = np.sqrt(crd_sum)
    crd_beta = crd_abs * beta
    frc_abs = crd_beta * sqrt_pi * np.exp(-crd_beta ** 2) + ops.erfc(crd_beta)
    frc_abs = (frc_abs - 1.) / crd_sum / crd_abs
    frc_abs = np.where(mask, frc_abs, zero_tensor)
    # (N,)[N, M] -> (N, M)
    excluded_charge = charge[excluded_matrix]
    # (N, 1) * (N, M) -> (N, M)
    charge_mul = np.expand_dims(charge, 1) * excluded_charge
    frc_abs = -charge_mul * frc_abs
    # (N, M, 1) * (N, M, 3) -> (N, M, 3)
    frc_lin = np.expand_dims(frc_abs, 2) * crd_d
    # (N, M, 3) -> (N, 3)
    frc_outer = np.sum(frc_lin, axis=1)
    # (N, M, 3) -> (N*M, 3)
    frc_inner = -frc_lin.reshape(-1, 3)
    excluded_list = excluded_matrix.reshape(-1, 1)
    res = ops.tensor_scatter_add(frc_outer, excluded_list, frc_inner)
    return res
