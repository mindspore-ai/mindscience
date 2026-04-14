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
"""force"""
from __future__ import print_function
import numpy
import scipy

from src.xcnn.utility import print_mat


einsum = numpy.einsum


def _HF_elec(mol, dm):
    """
    Hellmann-Feynman force, electronic part.

    Args:
        mol (Mole): PySCF Mole class.
        dm (ndarray): density matrix in AO.

    Returns:
        HF force at three directions.
    """
    
    offsetdic = mol.offset_nr_by_atom()
    HF_elec = numpy.zeros([mol.natm, 3])

    for k, ia in enumerate(range(mol.natm)):
        shl0, shl1, p0, p1 = offsetdic[ia]
        mol.set_rinv_origin(mol.atom_coord(ia))
        vrinv = -mol.atom_charge(ia) * mol.intor('int1e_iprinv', comp=3)
        HF_elec[k] += einsum("xij,ij->x", vrinv, dm) * 2

    return HF_elec


def _HF_nuc(mol):
    """
    Hellmann-Feynman force, nuclear part.

    Args:
        mol (Mole): PySCF Mole class.

    Returns:
        HF force at three directions.
    """

    HF_nuc = numpy.zeros((mol.natm, 3))
    atom_coords = mol.atom_coords()
    Z = mol.atom_charges()
    for i in range(mol.natm):
        for j in range(mol.natm):
            if j == i: continue
            Z_pair = Z[i] * Z[j]

            dis_r = numpy.sum((atom_coords[i] - atom_coords[j]) * (atom_coords[i] - atom_coords[j]))
            dis_r = numpy.sqrt(dis_r)
            dis_r3 = dis_r * dis_r * dis_r
            for k in range(3):
                HF_nuc[i][k] += Z_pair * (atom_coords[j][k] - atom_coords[i][k]) / dis_r3

    return HF_nuc


def HF_force(mol, dm, debug=False):
    HF_elec = _HF_elec(mol, dm)
    HF_nuc = _HF_nuc(mol)

    if debug:
        print_mat(HF_elec, "HF_elec")
        print_mat(HF_nuc, "HF_nuc")
        print_mat(HF_elec+HF_nuc, "HF")

    HF_force = HF_elec + HF_nuc
    return HF_force


# Pulay part
def _make_rdm1e(e, c, mo_occ):
    mo0 = c[:, mo_occ>0]
    mo0e = mo0 * (e[mo_occ>0] * mo_occ[mo_occ>0])
    dme_rks = numpy.dot(mo0e, mo0.T.conj())

    return dme_rks


def Pulay_term(mol, v1, dm):
    t = numpy.zeros([mol.natm, 3])
    offsetdic = mol.offset_nr_by_atom()
    for ia in range(mol.natm):
        shl0, shl1, p0, p1 = offsetdic[ia]
        t[ia] += einsum('xij,ij->x', v1[:, p0:p1], dm[p0:p1]) * 2 
    return t


def Pulay_force(mol, dm, dme, H1, v01, vbg1, S1, debug=False):
    veff1 = v01 + vbg1
    F1 = H1 + veff1
    
    offsetdic = mol.offset_nr_by_atom()

    Pulay_tf = numpy.zeros([mol.natm, 3])
    Pulay_ts = numpy.zeros([mol.natm, 3])
    if debug:
        th = numpy.zeros([mol.natm, 3])
        t0 = numpy.zeros([mol.natm, 3])
        tbg = numpy.zeros([mol.natm, 3])
    
    for ia in range(mol.natm):
        shl0, shl1, p0, p1 = offsetdic[ia]
        Pulay_tf[ia] += einsum('xij,ij->x', F1[:, p0:p1], dm[p0:p1]) * 2
        Pulay_ts[ia] -= einsum('xij,ij->x', S1[:, p0:p1], dme[p0:p1]) * 2
        
        if debug:
            th[ia] += einsum('xij,ij->x', H1[:, p0:p1], dm[p0:p1]) * 2 
            t0[ia] += einsum('xij,ij->x', v01[:, p0:p1], dm[p0:p1]) * 2 
            tbg[ia] += einsum('xij,ij->x', vbg1[:, p0:p1], dm[p0:p1]) * 2 
    
    if debug:
        print_mat(th, 'Pulay_th')
        print_mat(t0, 'Pulay_t0')
        print_mat(tbg, 'Pulay_tbg')
        print_mat(Pulay_tf, 'Pulay_tf')
        print_mat(Pulay_ts, 'Pulay_ts')
        print_mat(Pulay_tf+Pulay_ts, 'Pulay')
    
    return Pulay_tf + Pulay_ts


def calc_force(nnks, debug=False):
    from src.xcnn.constrain import improve_wxc_zf
    from src.xcnn.xc import eval_xc_on_grids, eval_xc_mat, eval_xc_grad_mat

    wxc = eval_xc_on_grids(nnks)
    if 'ZeroForce' in nnks.constrains:
        force_v0_atom = Pulay_term(nnks.mol, nnks.v01, nnks.dm)
        force_v0 = -numpy.sum(force_v0_atom, axis=0)
        # need to use original coordinates
        vxc = improve_wxc_zf(nnks.mol, nnks.dm, nnks.grids.original_coords, nnks.grids.weights, wxc, force_v0)
    else:
        vxc = wxc
    
    vxc_mat = eval_xc_mat(nnks, wxc=wxc, vxc=vxc, niter=99)
    vxc1_mat = eval_xc_grad_mat(nnks, wxc=wxc, vxc=vxc, niter=99)
    nnks.e, nnks.c = scipy.linalg.eigh(nnks.H+vxc_mat, nnks.S)
    mocc = nnks.c[:, nnks.mo_occ>0]
    dm = numpy.dot(mocc * nnks.mo_occ[nnks.mo_occ>0], mocc.T.conj())

    ddm = dm - nnks.dm
    print('Iter %3d\tmae(max abs diff in dm): %.8e\t sum abs diff in dm: %.8e\t# elec: %.8e' % 
            (99, 
                numpy.max(abs(ddm)), 
                numpy.sum(abs(ddm)), 
                numpy.einsum('ij,ji->', dm, nnks.S),
                ))
    dme = _make_rdm1e(nnks.e, nnks.c, nnks.mo_occ)

    HF = HF_force(nnks.mol, dm, debug)
    Pulay = Pulay_force(nnks.mol,
            dm, dme, 
            nnks.H1, nnks.v01, vxc1_mat, nnks.S1,
            debug=debug)


    return HF + Pulay


