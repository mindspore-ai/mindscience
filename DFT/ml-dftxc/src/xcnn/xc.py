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
"""xc"""
from __future__ import division
import numpy
import pyscf.dft
from tqdm import tqdm
import mindspore as ms
import mindspore.ops as ops

from src.xcnn.density import _density_on_grids2, eval_rho_on_grids
from src.xcnn.force import Pulay_term
from src.xcnn.constrain import improve_wxc_zf, improve_wxc_zfzt


einsum = numpy.einsum
BLK_SIZE = 200


def _eval_xc_from_nn(model, rho):
    """
    Feed local density into 3D-CNN model to get vbg.
    
    Args:
        model (torch.nn.Model): 3D-CNN model.
        rho (ndarray): density on grids.
    Returns:
        predicted vbg corresponding to rho.
    """

    inputs = ms.Tensor(rho)
    outputs = model(inputs)
    
    vbg = outputs.asnumpy().reshape(-1)
    
    return vbg
    # END of _eval_xc_from_nn()


def _eval_rho_on_grids(mol, coords, dm, na):
    """
    Evaluate density on grids and re-arrange to NN input.

    Args:
        mol (pyscf.Mole): PySCF Mole instance.
        coords (ndarray): coordinates.
        dm (ndarray): density matrix.
        na (int): number of cube points along each direction.
    Returns:
        electron densities of size (... x 4 x na x na x na)
    """

    n_samples = len(coords) // (na * na * na)
    # rho = _density_on_grids2(mol, coords, dm, deriv=1)
    rho = eval_rho_on_grids(mol, dm, coords, deriv=1, xctype='GGA')
    rho = rho.reshape([4, n_samples, na, na, na]).astype(numpy.float32)
    rho = rho.transpose([1, 0, 2, 3, 4])
    return rho
    # END of _eval_rho_on_grids()


def eval_xc_on_grids(nnks):
    """
    Evaluate vbg using 3D-CNN and current electron densities.
    Args:
        nnks (NNKS): a NNKS instance containing necessary informations.

    Returns:
        vbg on grids.
    """
    mol = nnks.mol
    dm = nnks.dm
    coords = nnks.grids.coords
    extended_coords = nnks.grids.extended_coords
    na = nnks.na
    na3 = na * na * na

    total_size = len(coords)
    assert(total_size * na3 == len(extended_coords))

    n_blk = total_size // BLK_SIZE
    res = total_size - BLK_SIZE * n_blk

    print('Evaluate xc potential on grids. Block size: %d Total: %d Number of blocks: %d Residual: %d' % 
            (BLK_SIZE, total_size, n_blk, res))

    wxc = numpy.empty(total_size)
    with tqdm(total=total_size) as pbar:
        for i in range(n_blk):
            idx = slice(BLK_SIZE * i, BLK_SIZE * (i + 1))
            ext_idx = slice(BLK_SIZE * i * na3, BLK_SIZE * (i + 1) * na3)
            rho = _eval_rho_on_grids(mol, extended_coords[ext_idx], dm, na)
            wxc[idx] = _eval_xc_from_nn(nnks.nn_model, rho)
            pbar.update(BLK_SIZE)

        if res > 0:
            rho = _eval_rho_on_grids(mol, extended_coords[-res*na3:], dm, na)
            wxc[-res:] = _eval_xc_from_nn(nnks.nn_model, rho)
            pbar.update(res)

    return wxc
    # END of eval_xc_on_grids()


def eval_xc_mat(nnks, wxc=None, vxc=None, niter=0):
    """
    Evaluate vbg using 3D-CNN and current electron densities,
    apply required constrain, 
    and generate matrix representation.
    Args:
        nnks (NNKS): a NNKS instance containing necessary informations.
        niter (int): number of iteration

    Returns:
        matrix vbg
    """

    weights = nnks.grids.weights
    
    if wxc is None:
        wxc = eval_xc_on_grids(nnks)

    if vxc is None:
        if 'ZeroForce' in nnks.constrains and 'ZeroTorque' in nnks.constrains:
            force_v0_atom = Pulay_term(nnks.mol, nnks.v01, nnks.dm)
            print(force_v0_atom)
            force_v0 = -numpy.sum(force_v0_atom, axis=0)
            torque_v0_atom = numpy.cross(nnks.mol.atom_coords(), force_v0_atom)
            # print(torque_v0_atom)
            # torque_v0 = numpy.sum(torque_v0_atom, axis=0)
            torque_v0 = Pulay_term(nnks.mol, nnks.v0_irxp1, nnks.dm)
            torque_v0 = numpy.sum(torque_v0, axis=0)
            print(torque_v0)
            vH_irxp1 = -einsum('xijkl,lk->xij', nnks._int2e_irxp1, nnks.dm)
            torque_vH = Pulay_term(nnks.mol, vH_irxp1, nnks.dm)
            torque_vH = numpy.sum(torque_vH, axis=0)
            print(torque_vH)

            torque_tmp = einsum(
                    'xij,ij->x', 
                    nnks.v0_irxp1-vH_irxp1,
                    nnks.dm)
            print(torque_tmp)

            # if niter == 1: exit(1) 

            # vxc = improve_wxc_zfzt(nnks.mol, nnks.dm, nnks.grids.original_coords, weights, wxc, force_v0, torque_v0-torque_vH)
            vxc = improve_wxc_zfzt(nnks.mol, nnks.dm, nnks.grids.original_coords, weights, wxc, force_v0, torque_tmp)

        elif 'ZeroForce' in nnks.constrains:
            force_v0_atom = Pulay_term(nnks.mol, nnks.v01, nnks.dm)
            force_v0 = -numpy.sum(force_v0_atom, axis=0)
            # need to use original coordinates
            vxc = improve_wxc_zf(nnks.mol, nnks.dm, nnks.grids.original_coords, weights, wxc, force_v0)
        else:
            vxc = wxc


    total_size = len(nnks.grids.original_coords)

    n_blk = total_size // BLK_SIZE
    res = total_size - BLK_SIZE * n_blk

    xc_mat = numpy.zeros(nnks.dm.shape, dtype=nnks.dm.dtype) 
    ao_loc = nnks.mol.ao_loc_nr()
    shls_slice = (0, nnks.mol.nbas)

    for i in range(n_blk):
        idx = slice(BLK_SIZE * i, BLK_SIZE * (i + 1))

        ao = pyscf.dft.numint.eval_ao(nnks.mol, nnks.grids.original_coords[idx], deriv=0)
        n_grids, n_ao = ao.shape
        wv = weights[idx] * vxc[idx] * 0.5
        aow = einsum('pi,p->pi', ao, wv)
        xc_mat += pyscf.dft.numint._dot_ao_ao(nnks.mol, ao, aow, numpy.ones((n_grids, nnks.mol.nbas), dtype=numpy.int8), shls_slice, ao_loc)
    
    if res > 0:
        ao = pyscf.dft.numint.eval_ao(nnks.mol, nnks.grids.original_coords[-res:], deriv=0)
        n_grids, n_ao = ao.shape
        wv = weights[-res:] * vxc[-res:] * 0.5
        aow = einsum('pi,p->pi', ao, wv)
        xc_mat += pyscf.dft.numint._dot_ao_ao(nnks.mol, ao, aow, numpy.ones((n_grids, nnks.mol.nbas), dtype=numpy.int8), shls_slice, ao_loc)

    numpy.save("%s/xc_w_iter_%02d" % (nnks.chk_path, niter + 1), wxc)
    numpy.save("%s/xc_v_iter_%02d" % (nnks.chk_path, niter + 1), vxc)
    numpy.save("%s/coords_nn" % (nnks.chk_path), nnks.grids.original_coords)

    return xc_mat + xc_mat.T


def eval_xc_grad_mat(nnks, wxc=None, vxc=None, niter=0):
    """
    Evaluate vbg using 3D-CNN and current electron densities,
    apply required constrain, 
    and generate matrix representation of 
    its first derivative with respect to nuclear coordinates.
    Args:
        nnks (NNKS): a NNKS instance containing necessary informations.
        niter (int): number of iteration

    Returns:
        matrix vbg_1
    """

    weights = nnks.grids.weights
    
    if wxc is None:
        wxc = eval_xc_on_grids(nnks)

    if vxc is None:
        if 'ZeroForce' in nnks.constrains:
            force_v0_atom = Pulay_term(nnks.mol, nnks.v01, nnks.dm)
            force_v0 = -numpy.sum(force_v0_atom, axis=0)
            # need to use original coordinates
            vxc = improve_wxc_zf(nnks.mol, nnks.dm, nnks.grids.original_coords, weights, wxc, force_v0)
        else:
            vxc = wxc

    total_size = len(nnks.grids.original_coords)

    n_blk = total_size // BLK_SIZE
    res = total_size - BLK_SIZE * n_blk
    
    xc_mat = numpy.zeros((3,)+nnks.dm.shape, dtype=nnks.dm.dtype)
    ao_loc = nnks.mol.ao_loc_nr()
    shls_slice = (0, nnks.mol.nbas)
    
    for i in range(n_blk):
        idx = slice(BLK_SIZE * i, BLK_SIZE * (i + 1))
        ao = pyscf.dft.numint.eval_ao(nnks.mol, nnks.grids.original_coords[idx], deriv=1)
        n_grids, n_ao = ao[0].shape

        wv = weights[idx] * vxc[idx] # NOTE no *.5
        aow = einsum('pi,p->pi', ao[0], wv)

        xc_mat[0] += pyscf.dft.numint._dot_ao_ao(nnks.mol, ao[1], aow, numpy.ones((n_grids, nnks.mol.nbas), dtype=numpy.int8), shls_slice, ao_loc)
        xc_mat[1] += pyscf.dft.numint._dot_ao_ao(nnks.mol, ao[2], aow, numpy.ones((n_grids, nnks.mol.nbas), dtype=numpy.int8), shls_slice, ao_loc)
        xc_mat[2] += pyscf.dft.numint._dot_ao_ao(nnks.mol, ao[3], aow, numpy.ones((n_grids, nnks.mol.nbas), dtype=numpy.int8), shls_slice, ao_loc)

    if res > 0:
        ao = pyscf.dft.numint.eval_ao(nnks.mol, nnks.grids.original_coords[-res:], deriv=1)
        n_grids, n_ao = ao[0].shape

        wv = weights[-res:] * vxc[-res:] # NOTE no *.5
        aow = einsum('pi,p->pi', ao[0], wv)

        xc_mat[0] += pyscf.dft.numint._dot_ao_ao(nnks.mol, ao[1], aow, numpy.ones((n_grids, nnks.mol.nbas), dtype=numpy.int8), shls_slice, ao_loc)
        xc_mat[1] += pyscf.dft.numint._dot_ao_ao(nnks.mol, ao[2], aow, numpy.ones((n_grids, nnks.mol.nbas), dtype=numpy.int8), shls_slice, ao_loc)
        xc_mat[2] += pyscf.dft.numint._dot_ao_ao(nnks.mol, ao[3], aow, numpy.ones((n_grids, nnks.mol.nbas), dtype=numpy.int8), shls_slice, ao_loc)
        
    return -xc_mat


