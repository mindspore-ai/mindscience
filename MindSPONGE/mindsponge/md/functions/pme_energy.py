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
'''pme energy'''
import mindspore.numpy as mnp
from .common import PI, get_zero_tensor, get_full_tensor
from .pme_common import Scale_List, PME_Atom_Near, PME_Q_Spread, PME_Direct_Energy, PME_Energy_Reciprocal, \
    PME_Excluded_Energy_Correction, PME_Energy_Product, get_pme_kxyz, PERIODIC_FACTOR_INVERSE, \
    fft3d

cutoff = 10.0

def pme_energy(atom_numbers, beta, fftx, ffty, fftz, pme_bc, uint_crd, charge,
               nl_numbers, nl_serial, scaler, excluded_matrix):
    """
    Calculate the Coulumb energy of the system using PME method.

    .. math::

        E = sum_{ij} q_iq_j/r_{ij}

    Args:
        atom_numbers (int): the number of atoms, N.
        beta (float): the PME beta parameter, determined by the
                       non-bond cutoff value and simulation precision tolerance.
        fftx (int): the number of points for Fourier transform in dimension X.
        ffty (int): the number of points for Fourier transform in dimension Y.
        fftz (int): the number of points for Fourier transform in dimension Z.
        uint_crd (Tensor, uint32): [N, 3], the unsigned int coordinates value of each atom.
        charge (Tensor, float32): [N,], the charge carried by each atom.
        nl_numbers - (Tensor, int32): [N,], the each atom.
        nl_serial - (Tensor, int32): [N, 800], the neighbor list of each atom, the max number is 800.
        scaler (Tensor, float32): [3,], the scale factor between real space
          coordinates and its unsigned int value.
        excluded_matrix (Tensor, int32): [N, k] containing the excluded atoms for each atom, where k
          is the maximum number of excluded atoms for all atoms.

    Outputs:
        reciprocal_ene  (float) - the reciprocal term of PME energy.
        self_ene (float) - the self term of PME energy.
        direct_ene (float) - the direct term of PME energy.
        correction_ene (float) - the correction term of PME energy.

    Supported Platforms:
        ``GPU``
    """
    PME_Nin = ffty * fftz
    PME_Nall = fftx * ffty * fftz

    PME_kxyz = get_pme_kxyz() # (64, 3)

    PME_uxyz = get_full_tensor((atom_numbers, 3), 2 ** 30, mnp.uint32)
    PME_atom_near = get_zero_tensor((atom_numbers, 64), mnp.int32)
    PME_frxyz, PME_uxyz, PME_atom_near = PME_Atom_Near(uint_crd, PME_atom_near, PME_Nin,
                                                       PERIODIC_FACTOR_INVERSE * fftx,
                                                       PERIODIC_FACTOR_INVERSE * ffty,
                                                       PERIODIC_FACTOR_INVERSE * fftz, atom_numbers,
                                                       fftx, ffty, fftz, PME_kxyz, PME_uxyz)

    PME_Q = get_full_tensor(PME_Nall, 0, mnp.float32)
    PME_Q = PME_Q_Spread(PME_atom_near, charge, PME_frxyz, PME_Q, PME_kxyz, atom_numbers)

    PME_Q = PME_Q.reshape(fftx, ffty, fftz).astype('float32')
    real, imag = fft3d(PME_Q)

    reciprocal_ene = PME_Energy_Reciprocal(real.ravel(), imag.ravel(), pme_bc)

    self_ene = PME_Energy_Product(charge, charge)
    self_ene = Scale_List(1, self_ene, -beta / mnp.sqrt(PI))

    direct_ene = PME_Direct_Energy(atom_numbers, nl_numbers, nl_serial, uint_crd, scaler, charge, beta,
                                   cutoff * cutoff)

    correction_ene = PME_Excluded_Energy_Correction(atom_numbers, uint_crd, scaler, charge, beta, mnp.sqrt(PI),
                                                    excluded_matrix)

    return mnp.atleast_1d(reciprocal_ene), mnp.atleast_1d(self_ene), \
           mnp.atleast_1d(direct_ene), mnp.atleast_1d(correction_ene)
