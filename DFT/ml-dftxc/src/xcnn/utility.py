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
"""utility"""
import numpy
import pyscf

from src.xcnn.density import _density_on_grids

einsum = numpy.einsum


def dft(mol, mr=None, xc=None, level=3):
    if mr is None:
        mr = pyscf.scf.RKS(mol)
    if xc is not None:
        mr.xc = xc
    mr.grids.level = level
    mr.run()
    return mr.make_rdm1()


def rhf(mol, mf=None):
    if mf is None:
        mf = pyscf.scf.RHF(mol)
    mf.kernel()
    return mf.make_rdm1()


def ccsd(mol, mcc=None, mf=None):
    if mf is None:
        mf = pyscf.scf.RHF(mol)
        mf.kernel()
    c = mf.mo_coeff
    if mcc is None:
        mcc = pyscf.cc.CCSD(mf)
    ecc, t1, t2 = mcc.kernel()
    rdm1 = mcc.make_rdm1()
    rdm1_ao = einsum('pi,ij,qj->pq', c, rdm1, c.conj())
    return rdm1_ao


def print_mat(mat, name=None):
    if name is None:
        print(mat.shape)
        print(mat)
    else:
        print(name, mat.shape)
        print(mat)


def I(mol, dm1, dm2, coords, weights):
    rho1 = _density_on_grids(mol, coords, dm1)
    rho2 = _density_on_grids(mol, coords, dm2)
    drho = rho1 - rho2

    a = numpy.sum(drho * drho * weights) 
    b = numpy.sum(rho1 * rho1 * weights) + numpy.sum(rho2 * rho2 * weights)
    return a / b

