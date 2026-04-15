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
"""scf"""
from __future__ import print_function
import numpy
import scipy

from src.xcnn.xc import eval_xc_mat
from src.xcnn.utility import dft, ccsd, rhf


def initialize(nnks):
    if nnks.opts['InitDensityMatrix'].lower() == 'rks':
        print('[35mUse RKS with xc ``%s" density as init[0m' % (nnks.opts['xcFunctional']))
        if nnks.dm_rks is None:
            nnks.dm_rks = dft(nnks.mol, xc=nnks.opts['xcFunctional'])
        nnks.dm0 = nnks.dm_rks.copy()
    elif nnks.opts['InitDensityMatrix'].lower() == 'ccsd':
        print('[35mUse CC density as init[0m')
        if nnks.dm_cc is None:
            nnks.dm_cc = ccsd(nnks.mol)
        nnks.dm0 = nnks.dm_cc
    elif nnks.opts['InitDensityMatrix'].lower() == 'rhf':
        print('[35mUse RHF density as init[0m')
        if nnks.dm_rhf is None:
            nnks.dm_rhf = rhf(nnks.mol)
        nnks.dm0 = nnks.dm_rhf
    elif nnks.opts['InitDensityMatrix'].find('load') != -1:
        print('[35mLoad %s as init[0m' % (nnks.opts['InitDensityMatrix'].split()[1]))
        nnks.dm0 = numpy.load(nnks.opts['InitDensityMatrix'].split()[1])
    print('Initialize done.')


def scf(nnks):
    """
    Perform SCF calculation using vbg predicted from 3D-CNN.

    Args:
        nnks (NNKS): an NNKS instance.
    """

    print()
    print('================')
    print('SCF calculation')
    print('================')
    
    nnks.dm = nnks.dm0
    conveged = False
    for i_iter in range(nnks.scf_max_iter):
        vxc = eval_xc_mat(nnks, niter=i_iter)
        
        nnks.e, nnks.c, dm_new = solve_KS(nnks.H+vxc, nnks.S, nnks.mo_occ)
        numpy.save('%s/dm_iter_%02d' % (nnks.chk_path, i_iter+1), dm_new)

        ddm = dm_new - nnks.dm
        print('Iter %3d\tmae(max abs diff in dm): %.8e\t sum abs diff in dm: %.8e\t# elec: %.8e' % 
                (i_iter, 
                    numpy.max(abs(ddm)), 
                    numpy.sum(abs(ddm)), 
                    numpy.einsum('ij,ji->', dm_new, nnks.S),
                    ))
        if numpy.linalg.norm(ddm) < nnks.scf_crit:
            print('SCF converged.')
            conveged = True
        
        nnks.dm = dm_new
        if conveged: break

    if not conveged:
        print('SCF convergence fails.')
    # END of scf()


def solve_KS(f, s, mo_occ):
    e, c = scipy.linalg.eigh(f, s)

    mocc = c[:, mo_occ>0]
    dm = numpy.dot(mocc * mo_occ[mo_occ>0], mocc.T.conj())

    return e, c, dm



