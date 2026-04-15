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
"""nnks"""
from __future__ import print_function
import os.path
import numpy

import pyscf.gto, pyscf.scf

from src.utils import ExtendModel
from src.xcnn.grid import Grids
from src.xcnn.scf import *
from src.xcnn.IO import parse_str
from src.xcnn.force import calc_force


class NNKS(object):
    def __init__(self, opts):
        self.opts = opts
        self.chk_path = opts['CheckPointPath']

        self._generate_mol(opts['Structure'], opts['OrbitalBasis'])

        self._get_fixed_matrix(opts['ReferencePotential'], opts['CheckPointPath'])

        self._load_nn_model(opts['Model'], opts['ModelPath'])

        self._get_model_info()

        # == DEBUG ==
        # mr = pyscf.scf.RKS(self.mol)
        # mr.xc = 'b3lypg'
        # # mr.kernel()
        # mr.grids.build()
        # print(mr.grids.coords.shape)
        # self.grids = Grids(
        #         mr.grids.coords, 
        #         mr.grids.weights, 
        #         self.opts)
        # == DEBUG ==
        self.grids = Grids(
                self.mr.grids.coords, 
                self.mr.grids.weights, 
                self.opts)

        self.scf_crit = opts['ConvergenceCriterion']
        self.scf_max_iter = opts['MaxIteration']

        self.constrains = list()
        if opts['ZeroForceConstrain']: self.constrains.append('ZeroForce')
        if opts['ZeroTorqueConstrain']: self.constrains.append('ZeroTorque')

        self.dm_cc = None
        self.dm_rks = None
        self.dm_rhf = None
        self.dm_hfx = None
        self.dm_nn = None

        print('NNKS::__init__() done.')
        # END of __init__()


    def _generate_mol(self, fn, basis, grid_level=3):
        atom_list, atom_str, charge, spin = parse_str(fn)    
        
        self.mol = pyscf.gto.M(
                atom=atom_str, 
                basis=basis, 
                charge=charge, 
                spin=spin)
        
        n_elec = self.mol.nelectron
        n_ao = self.mol.nao_nr()
        self.mo_occ = numpy.zeros(n_ao)
        self.mo_occ[:n_elec//2] = 2
        self.n_occ = n_elec//2

        self.mf = pyscf.scf.RHF(self.mol)
        self.mr = pyscf.scf.RKS(self.mol)
        self.mr.grids.level = grid_level
        # self.mr.grids.build()

        print('Structure info:')
        for a in self.mol._atom:
            print('%2s\t%+12.8e\t%+12.8e\t%+12.8e  Bohr' % 
                    (a[0], a[1][0], a[1][1], a[1][2]))
        print('charge = %d\nspin = %d' % (self.mol.charge, self.mol.spin))
        print('NNKS::_generate_mol() done.')
        # END of _generate_mol()

    def _get_fixed_matrix(self, ref_pot, chk_path):
        mol = self.mol
        self.S = mol.intor('int1e_ovlp_sph')
        self.T = mol.intor('cint1e_kin_sph')
        self.vn = mol.intor('cint1e_nuc_sph')
        
        if ref_pot[0].lower() == 'hfx':
            self.reference_potential = 'hfx'
            print("[31mUse HFX reference potential[0m")
            self.mr.xc = 'hf,'
            if False and os.path.isfile('%s/dm_hfx.npy' % (chk_path)):
                print("[31mLoad dm from %s/dm_hfx.npy[0m" % (chk_path))
                dm_hfx = numpy.load('%s/dm_hfx.npy' % (chk_path))
                self.mr.kernel(dm0=dm_hfx)
            else:
                self.mr.kernel()
                dm_hfx = self.mr.make_rdm1()
                print("[31mSave dm to %s/dm_hfx.npy[0m" % (chk_path))
                numpy.save('%s/dm_hfx' % (chk_path), dm_hfx)
                numpy.save('%s/dm_rks_%s' % (chk_path, self.mr.xc), dm_hfx)
            from pyscf import dft
            self.v0 = dft.rks.get_veff(self.mr, self.mol, dm_hfx)
            from pyscf.grad import rks as rks_grad
            g = rks_grad.Gradients(self.mr)
            vj1, vk1 = g.get_jk(dm=self.mr.make_rdm1()) 
            self.v01 = vj1 - vk1 * 0.5
            self.H1 = g.get_hcore()
            self.S1 = g.get_ovlp()

            # self._int2e_irxp1 = mol.intor('int2e_irxp1')
            # self.vj_irxp1 = -numpy.einsum('xijkl,kl->xij', self._int2e_irxp1, self.mr.make_rdm1())
            # self.vk_irxp1 = -numpy.einsum('xijkl,jk->xil', self._int2e_irxp1, self.mr.make_rdm1())
            # self.v0_irxp1 = self.vj_irxp1 - self.vk_irxp1 * 0.5
        else:
            raise NotImplementedError
        
        self.H = self.T + self.vn + self.v0
        print('NNKS::_get_fixed_matrix() done.')
        # END of _get_fixed_matrix()

    def _load_nn_model(self, model, model_path):
        self.nn_model = ExtendModel(model, 1)
        self.nn_model.load_model(model_path)
        print('NNKS::_load_nn_model() done.')
        # END of _load_nn_model()

    def _get_model_info(self):
        self.a = self.opts['CubeLength']
        self.na = self.opts['CubePoint']
        print('NNKS::_get_model_info() done.')
        # END of _get_model_info()

    def scf(self):
        initialize(self)
        scf(self)

    def force(self, verbose=False):
        return calc_force(self, debug=verbose)


