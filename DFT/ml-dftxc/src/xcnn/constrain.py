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
"""constrain"""
from __future__ import print_function
import numpy

from src.xcnn.density import _density_on_grids2
from src.xcnn.density import _density_grad_on_grids_a
from src.xcnn.density import eval_rho_on_grids


def improve_wxc_zf(mol, dm, coords, weights, wxc, f=numpy.zeros([3])):
    print('Construct vxc based on ZF constrain.')

    # rho = eval_rho_on_grids(mol, dm, coords, deriv=1, xctype='GGA')
    rho = _density_on_grids2(mol, coords, dm, deriv=1)
    
    N = len(coords)
    AU = numpy.zeros([N, 3])
    AL = numpy.zeros([3, N])

    for k in range(3):
        AU[:, k] = rho[k+1, :]
        AL[k, :] = rho[k+1, :] * weights

    B = wxc

    ALAU = numpy.einsum('ij,jk->ik', AL, AU)
    assert(ALAU.shape == (3, 3))
    a = -numpy.linalg.inv(ALAU)

    LB = numpy.einsum('ij,j->i', AL, B)
    aLB = numpy.einsum('ij,j->i', a, LB)
    UaLB = numpy.einsum('ij,j->i', AU, aLB)

    # TODO 
    # check sign of f
    af = numpy.einsum('ij,j->i', a, f)
    Uaf = numpy.einsum('ij,j->i', AU, af)

    vxc = B + UaLB + Uaf

    return vxc[:N]


def improve_wxc_zfzt(mol, dm, coords, weights, wxc, f=numpy.zeros([3]), tau=numpy.zeros([3])):
    print('Construct vxc based on ZF and ZT constrain.')

    rho = _density_on_grids2(mol, coords, dm, deriv=0)
    rho1 = _density_grad_on_grids_a(mol, coords, dm)
    rho_1 = _density_on_grids2(mol, coords, dm, deriv=1)

    # drho_cross_r = numpy.cross(rho[1:].T, coords)
    # assert(drho_cross_r.shape == coords.shape)
    r_cross_rho1_a = numpy.empty([mol.natm, len(coords), 3])
    for ia in range(mol.natm):
        r_cross_rho1_a[ia] = numpy.cross(coords, rho1[ia].T)
    # r_cross_rho1 = numpy.sum(r_cross_rho1_a, axis=0)
    # assert(r_cross_rho1.shape == (len(coords), 3))

    r_cross_rho1 = numpy.cross(coords, rho_1[1:].T)

    # numpy.save('R_cross_rho1', R_cross_rho1)
    # numpy.save('rho1', rho1)
    # numpy.save('xyz', mol.atom_coords())
    tmat = numpy.einsum('i,axi,i->ax', wxc, rho1, weights)
    print(tmat)
    # tmat = numpy.einsum('i,aix,i->ax', wxc, r_cross_rho1_a, weights)
    # print(tmat)
    tmat = numpy.einsum('i,ix,i->x', wxc, r_cross_rho1, weights)
    print(tmat)
    # exit(1)
    

    N = len(coords)
    AU = numpy.zeros([N, 6])
    AL = numpy.zeros([6, N])
   
    for k in range(3):
        # AU[:, k] = numpy.sum(rho1[:, k, :], axis=0)
        # AL[k, :] = numpy.sum(rho1[:, k, :], axis=0) * weights
        AU[:, k] = rho_1[k+1, :]
        AL[k, :] = rho_1[k+1, :] * weights
        
        AU[:, 3+k] = r_cross_rho1[:, k]
        AL[3+k, :] = r_cross_rho1[:, k] * weights
        
    B = wxc
    Bc = numpy.concatenate((f, tau))
    # TODO 
    # check sign of f

    ALAU = numpy.einsum('ij,jk->ik', AL, AU)
    assert(ALAU.shape == (6, 6))
    a = -numpy.linalg.inv(ALAU)

    LB = numpy.einsum('ij,j->i', AL, B)
    aLB = numpy.einsum('ij,j->i', a, LB)
    UaLB = numpy.einsum('ij,j->i', AU, aLB)
    
    aBc = numpy.einsum('ij,j->i', a, Bc)
    UaBc = numpy.einsum('ij,j->i', AU, aBc)

    vxc = B + UaLB + UaBc

    tmat = numpy.einsum('i,axi,i->ax', vxc[:N], rho1, weights)
    print(tmat)
    # tmat = numpy.einsum('i,aix,i->ax', vxc[:N], r_cross_rho1_a, weights)
    # print(tmat)
    tmat = numpy.einsum('i,ix,i->x', vxc[:N], r_cross_rho1, weights)
    print(tmat)
    # exit(1)

    return vxc[:N]
    

