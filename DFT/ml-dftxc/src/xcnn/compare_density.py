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
"""compare density"""
from __future__ import print_function
import numpy as np
from pyscf import gto, scf, dft


def eval_rho(mol, dm, coords, deriv=0, xctype="LDA"):
    ao_value = dft.numint.eval_ao(mol, coords, deriv=deriv)
    rhos = dft.numint.eval_rho(mol, ao_value, dm, xctype=xctype)
    return rhos


def float2str(v):
    return "%+16.8e" % (v)


def _calculate_I_space_resolve(mol, dm0, dm1, coords, blk_size=1000, coord_system="cart"):
    # Bochevarov, A. D., et al. J. Chem. Phys., 128, 034102 (2008)
    # The densities produced by the density functional theory:
    # Comparison to full configuration interaction
    # Eqn 2.2
    # return density difference on grid points rather than integrated result
    #
    # coord_system == "cyl": cylinder coordinate system
    #   dxdydz = rdr d\theta dz
    
    if coord_system != "cyl":
        raise NotImplementedError

    total_length = len(coords)
    n_blk = total_length / blk_size
    res = total_length - n_blk * blk_size
    
    I = np.zeros([total_length], dtype=np.float)
    accumulated_I = np.zeros([total_length], dtype=np.float)
    for iBlk in range(n_blk):
        index = slice(iBlk*blk_size, (iBlk+1)*blk_size)

        rho0 = eval_rho(mol, dm0, coords[index])
        rho1 = eval_rho(mol, dm1, coords[index])
        drho = rho0 - rho1
        
        I[index] = drho * drho
        if coord_system == "cyl":
            accumulated_I[index] = drho * drho * coords[index, 0]


    if res > 0:
        rho0 = eval_rho(mol, dm0, coords[-res:])
        rho1 = eval_rho(mol, dm1, coords[-res:])
        drho = rho0 - rho1

        I[-res:] = drho * drho
        if coord_system == "cyl":
            accumulated_I[-res:] = drho * drho * coords[-res:, 0]
       
    return I, accumulated_I


def _calculate_I(mol, dm0, dm1, grids, blk_size=1000, full_output=False):
    # Bochevarov, A. D., et al. J. Chem. Phys., 128, 034102 (2008)
    # The densities produced by the density functional theory:
    # Comparison to full configuration interaction
    # Eqn 2.2
    
    coords = grids.coords
    weights = grids.weights
    
    total_length = len(coords)
    n_blk = total_length / blk_size
    res = total_length - n_blk * blk_size
    
    numerator = 0.
    denominator = 0.

    for iBlk in range(n_blk):
        index = slice(iBlk*blk_size, (iBlk+1)*blk_size)

        rho0 = eval_rho(mol, dm0, coords[index])
        rho1 = eval_rho(mol, dm1, coords[index])
        drho = rho0 - rho1

        denominator += np.sum(rho0 * rho0 * weights[index])
        denominator += np.sum(rho1 * rho1 * weights[index])

        numerator += np.sum(drho * drho * weights[index])

    if res > 0:
        rho0 = eval_rho(mol, dm0, coords[-res:])
        rho1 = eval_rho(mol, dm1, coords[-res:])
        drho = rho0 - rho1

        denominator += np.sum(rho0 * rho0 * weights[-res:])
        denominator += np.sum(rho1 * rho1 * weights[-res:])

        numerator += np.sum(drho * drho * weights[-res:])
       
    if full_output:
        return numerator, denominator
    else:
        return numerator / denominator
    return


def _calculate_rms2(mol, dm0, dm1, grids, blk_size=1000, full_output=False):
    # Bochevarov, A. D., et al. J. Chem. Phys., 128, 034102 (2008)
    # The densities produced by the density functional theory:
    # Comparison to full configuration interaction
    # Eqn 2.3
    
    coords = grids.coords
    weights = grids.weights
    
    total_length = len(coords)
    n_blk = total_length / blk_size
    res = total_length - n_blk * blk_size
    
    numerator = 0.
    denominator = float(total_length)

    for iBlk in range(n_blk):
        index = slice(iBlk*blk_size, (iBlk+1)*blk_size)

        rho0 = eval_rho(mol, dm0, coords[index])
        rho1 = eval_rho(mol, dm1, coords[index])
        drho = rho0 - rho1

        numerator += np.sum(drho * drho)

    if res > 0:
        rho0 = eval_rho(mol, dm0, coords[-res:])
        rho1 = eval_rho(mol, dm1, coords[-res:])
        drho = rho0 - rho1

        numerator += np.sum(drho * drho)
       
    if full_output:
        return numerator, denominator
    else:
        return numerator / denominator
    return


def _calculate_wI1(mol, dm0, dm1, grids, blk_size=1000, full_output=False):
    # Bochevarov, A. D., et al. J. Chem. Phys., 128, 034102 (2008)
    # The densities produced by the density functional theory:
    # Comparison to full configuration interaction
    # modified version of Eqn 2.2
    
    coords = grids.coords
    weights = grids.weights
    
    total_length = len(coords)
    n_blk = total_length / blk_size
    res = total_length - n_blk * blk_size
    
    numerator = 0.
    denominator = 0.

    for iBlk in range(n_blk):
        index = slice(iBlk*blk_size, (iBlk+1)*blk_size)

        rho0 = eval_rho(mol, dm0, coords[index])
        rho1 = eval_rho(mol, dm1, coords[index])
        drho = rho0 - rho1

        denominator += np.sum(rho0 * rho0 * weights[index])
        denominator += np.sum(rho1 * rho1 * weights[index])

        numerator += np.sum(drho * drho * rho1 * weights[index])

    if res > 0:
        rho0 = eval_rho(mol, dm0, coords[-res:])
        rho1 = eval_rho(mol, dm1, coords[-res:])
        drho = rho0 - rho1

        denominator += np.sum(rho0 * rho0 * weights[-res:])
        denominator += np.sum(rho1 * rho1 * weights[-res:])

        numerator += np.sum(drho * drho * rho1 * weights[-res:])
       
    if full_output:
        return numerator, denominator
    else:
        return numerator / denominator
    return


def _calculate_wI2(mol, dm0, dm1, grids, blk_size=1000, full_output=False):
    # Bochevarov, A. D., et al. J. Chem. Phys., 128, 034102 (2008)
    # The densities produced by the density functional theory:
    # Comparison to full configuration interaction
    # modified version of Eqn 2.2
    
    coords = grids.coords
    weights = grids.weights
    
    total_length = len(coords)
    n_blk = total_length / blk_size
    res = total_length - n_blk * blk_size
    
    numerator = 0.
    denominator = 0.

    for iBlk in range(n_blk):
        index = slice(iBlk*blk_size, (iBlk+1)*blk_size)

        rho0 = eval_rho(mol, dm0, coords[index])
        rho1 = eval_rho(mol, dm1, coords[index])
        drho = rho0 - rho1

        denominator += np.sum(rho0 * rho0 * weights[index])
        denominator += np.sum(rho1 * rho1 * weights[index])

        numerator += np.sum(drho * drho * rho1 * rho1 * weights[index])

    if res > 0:
        rho0 = eval_rho(mol, dm0, coords[-res:])
        rho1 = eval_rho(mol, dm1, coords[-res:])
        drho = rho0 - rho1

        denominator += np.sum(rho0 * rho0 * weights[-res:])
        denominator += np.sum(rho1 * rho1 * weights[-res:])

        numerator += np.sum(drho * drho * rho1 * rho1 * weights[-res:])
       
    if full_output:
        return numerator, denominator
    else:
        return numerator / denominator
    return


def _calculate_mad(mol, dm0, dm1, grids, blk_size=1000, full_output=False):
    
    coords = grids.coords
    weights = grids.weights
    
    total_length = len(coords)
    n_blk = total_length / blk_size
    res = total_length - n_blk * blk_size
    
    numerator = 0.
    denominator = 0.

    for iBlk in range(n_blk):
        index = slice(iBlk*blk_size, (iBlk+1)*blk_size)

        rho0 = eval_rho(mol, dm0, coords[index])
        rho1 = eval_rho(mol, dm1, coords[index])
        drho = np.abs(rho0 - rho1)

        denominator += np.sum(rho0 * weights[index])
        denominator += np.sum(rho1 * weights[index])

        numerator += np.sum(drho * weights[index])

    if res > 0:
        rho0 = eval_rho(mol, dm0, coords[-res:])
        rho1 = eval_rho(mol, dm1, coords[-res:])
        drho = np.abs(rho0 - rho1)

        denominator += np.sum(rho0 * weights[-res:])
        denominator += np.sum(rho1 * weights[-res:])

        numerator += np.sum(drho * weights[-res:])
       
    if full_output:
        return numerator, denominator
    else:
        return numerator / denominator
    return


def _calculate_wmad(mol, dm0, dm1, grids, blk_size=1000, full_output=False):
    
    coords = grids.coords
    weights = grids.weights
    
    total_length = len(coords)
    n_blk = total_length / blk_size
    res = total_length - n_blk * blk_size
    
    numerator = 0.
    denominator = 0.

    for iBlk in range(n_blk):
        index = slice(iBlk*blk_size, (iBlk+1)*blk_size)

        rho0 = eval_rho(mol, dm0, coords[index])
        rho1 = eval_rho(mol, dm1, coords[index])
        drho = np.abs(rho0 - rho1)

        denominator += np.sum(rho0 * weights[index])
        denominator += np.sum(rho1 * weights[index])

        numerator += np.sum(drho * rho1 * weights[index])

    if res > 0:
        rho0 = eval_rho(mol, dm0, coords[-res:])
        rho1 = eval_rho(mol, dm1, coords[-res:])
        drho = np.abs(rho0 - rho1)

        denominator += np.sum(rho0 * weights[-res:])
        denominator += np.sum(rho1 * weights[-res:])

        numerator += np.sum(drho * rho1 * weights[-res:])
       
    if full_output:
        return numerator, denominator
    else:
        return numerator / denominator
    return


def _calculate_wmad2(mol, dm0, dm1, grids, blk_size=1000, full_output=False):
    
    coords = grids.coords
    weights = grids.weights
    
    total_length = len(coords)
    n_blk = total_length / blk_size
    res = total_length - n_blk * blk_size
    
    numerator = 0.
    denominator = 0.

    for iBlk in range(n_blk):
        index = slice(iBlk*blk_size, (iBlk+1)*blk_size)

        rho0 = eval_rho(mol, dm0, coords[index])
        rho1 = eval_rho(mol, dm1, coords[index])
        drho = np.abs(rho0 - rho1)

        denominator += np.sum(rho0 * weights[index])
        denominator += np.sum(rho1 * weights[index])

        numerator += np.sum(drho * rho1 * rho1 * weights[index])

    if res > 0:
        rho0 = eval_rho(mol, dm0, coords[-res:])
        rho1 = eval_rho(mol, dm1, coords[-res:])
        drho = np.abs(rho0 - rho1)

        denominator += np.sum(rho0 * weights[-res:])
        denominator += np.sum(rho1 * weights[-res:])

        numerator += np.sum(drho * rho1 * rho1 * weights[-res:])
       
    if full_output:
        return numerator, denominator
    else:
        return numerator / denominator
    return


def _calculate_N(mol, dm, grids, blk_size=1000):
    # calculate total number of electrons.
    # N = \int \rho(r) dr

    coords = grids.coords
    weights = grids.weights
    
    total_length = len(coords)
    n_blk = total_length / blk_size
    res = total_length - n_blk * blk_size
    
    N = 0.
    for iBlk in range(n_blk):
        index = slice(iBlk*blk_size, (iBlk+1)*blk_size)

        rho = eval_rho(mol, dm, coords[index])
        N += np.sum(rho * weights[index])

    if res > 0:
        rho = eval_rho(mol, dm, coords[-res:])
        N += np.sum(rho * weights[-res:])
    
    return N


def compare_density(mol, dm0, dm1, grids, cmp_value, blk_size=1000, full_output=False):
    if cmp_value == "I":
        return _calculate_I(mol, dm0, dm1, grids, blk_size=blk_size, full_output=full_output)
    elif cmp_value == "Is":
        return _calculate_I_space_resolve(mol, dm0, dm1, grids, blk_size=blk_size, coord_system="cyl")
    elif cmp_value == "wI1":
        return _calculate_wI1(mol, dm0, dm1, grids, blk_size=blk_size, full_output=full_output)
    elif cmp_value == "wI2":
        return _calculate_wI2(mol, dm0, dm1, grids, blk_size=blk_size, full_output=full_output)
    elif cmp_value == "N":
        return _calculate_N(mol, dm0, grids, blk_size=blk_size)
    elif cmp_value == "rms2":
        return _calculate_rms2(mol, dm0, dm1, grids, blk_size=blk_size, full_output=full_output)
    elif cmp_value == "mad":
        return _calculate_mad(mol, dm0, dm1, grids, blk_size=blk_size, full_output=full_output)
    elif cmp_value == "wmad":
        return _calculate_wmad(mol, dm0, dm1, grids, blk_size=blk_size, full_output=full_output)
    elif cmp_value == "wmad2":
        return _calculate_wmad2(mol, dm0, dm1, grids, blk_size=blk_size, full_output=full_output)
    else:
        raise NotImplementedError

def main():
    mol = gto.M(atom="H 0 0 +0.25; H 0 0 -0.25", basis="augccpvqz")
    mf = scf.RKS(mol)
    mf.grids.level = 9
    mf.grids.build()

    dm_ccsd = np.load("dm_ccsd.npy")
    dm_rks = np.load("dm_rks_LDA.npy")
    dm_nn = np.load("dm_iter_08.npy")

    print("N CCSD", compare_density(mol, dm_ccsd, None, mf.grids, "N"))
    print("N RKS", compare_density(mol, dm_rks, None, mf.grids, "N"))
    print("N NN", compare_density(mol, dm_nn, None, mf.grids, "N"))

    numerator, denominator = compare_density(
            mol, dm_rks, dm_ccsd, mf.grids, "I",
            full_output=True)
    print("RKS vs CCSD", numerator, denominator)

    numerator, denominator = compare_density(
            mol, dm_nn, dm_ccsd, mf.grids, "I", 
            full_output=True)
    print("NN vs CCSD", numerator, denominator)

if __name__ == "__main__":
    main()

