# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ============================================================================
#
# Copyright 2022
# Georgia Tech Research Corporation
# Atlanta, Georgia 30332-4024
# All Rights Reserved
#
# This file is part of the ML-DFT originally developed by Georgia Tech Research Corporation (GTRC).
# Modifications have been made by Cen Jianhuan at Huawei Technologies Co., Ltd.
#
# This Program is licensed under the GENERAL PUBLIC USE LICENSE AGREEMENT provided by GTRC.
# You may not use this file except in compliance with the License.
#
# You should have received a copy of the GENERAL PUBLIC USE LICENSE AGREEMENT along with this program.
# If not, contact Georgia Tech Research Corporation for further information.
#
# Modifications made:
# - Replaced the neural network implementation originally based on Keras with an implementation using Huawei's MindSpore framework.
# - Completed training and inference on the Ascend NPU 910B.
#
# DISCLAIMER OF WARRANTIES AND LIMITATION ON LIABILITY:
# The Program is provided "AS IS", without warranty of any kind, express or implied, including but not limited to
# the warranties of merchantability, fitness for a particular purpose, and noninfringement. In no event shall the
# authors or copyright holders be liable for any claim, damages, or other liability, whether in an action of contract,
# tort, or otherwise, arising from, out of, or in connection with the software or the use or other dealings in the software.
#
# ============================================================================
"""molecular figerprints"""

import warnings
warnings.filterwarnings('ignore')
import numpy as np
import math
import os
import sys
sys.path.append(os.getcwd())
from operator import itemgetter
from joblib import dump, load

import mindspore as ms
import mindspore.ops as ops
import itertools

CONFIG_PATH1 = './checkpoints/Scale_model_C_39.joblib'
CONFIG_PATH2 = './checkpoints/Scale_model_H_39.joblib'
CONFIG_PATH3 = './checkpoints/Scale_model_N_39.joblib'
CONFIG_PATH4 = './checkpoints/Scale_model_O_39.joblib'
class Input_parameters:
    cut_off_rad        = 5.0
    batch_size_fp      = 10000
    widest_gaussian    = 6
    narrowest_gaussian = 0.5
    num_gamma          = 18
    list_sigma = np.logspace(math.log10(narrowest_gaussian),math.log10(widest_gaussian), num_gamma)
    list_gamma = 0.5 / list_sigma ** 2

inp_args = Input_parameters()

def fp_atom(poscar_data,supercell,elems_list):
    cart_grid=[]
    frac_grid=[]
    at_elem=[]
    sites_elem=[]
    num_elems = len(elems_list)
    element_fingerprint = []
    unique_cart_list=[]
    for elem in elems_list:
        elem_supercell = supercell.copy()
        list_remove = []
        for elem_remove in elems_list:
            if elem_remove != elem:
                list_remove.append(elem_remove)
        elem_supercell.remove_species(list_remove)
        at_elem.append(elem_supercell.num_sites)
        sites_elem.append(elem_supercell.sites.copy())
        list_neigh = elem_supercell.get_all_neighbors(float(inp_args.cut_off_rad),include_index=True)
        flat_list_neigh = list(itertools.chain.from_iterable(list_neigh))
        list_frac = [x[0].frac_coords for x in flat_list_neigh]
        extra_list=[x for x in elem_supercell.frac_coords]
        if list_frac:
            list_frac=np.concatenate((list_frac,extra_list), axis=0)
        else:
            list_frac=extra_list
        frac_array = np.array(list_frac)
        frac_array = np.ascontiguousarray(frac_array)
        cart_grid.append(np.array(elem_supercell.cart_coords))
        frac_grid.append(np.array(elem_supercell.frac_coords))

        if frac_array.shape[0]==0:
            frac_array=np.array(elem_supercell.frac_coords)
        unique_a = np.unique(frac_array.view([('', frac_array.dtype)]*frac_array.shape[1]))
        frac_array_unique = unique_a.view(frac_array.dtype).reshape((unique_a.shape[0], frac_array.shape[1]))
        unique_cart = np.dot(frac_array_unique, elem_supercell.lattice._matrix)
        unique_cart_list.append(unique_cart)
    cart_grid_K=np.concatenate(cart_grid, axis=0)
    frac_grid_K=np.concatenate(frac_grid, axis=0)
    padding_size=max(at_elem)

#     def quad(cart_grid_ms):
#         quad_list=[]
#         for num_e in range(len(unique_cart_list)):
#             cart_atoms_ms = ms.Tensor(unique_cart_list[num_e],ms.float32)
#             rad_diff = cart_grid_ms - cart_atoms_ms[:, None]
#             rad = ops.sqrt(ops.reduce_sum(rad_diff ** 2, axis=-1))
#             rad_inv = ops.reciprocal(rad)
#             cut_off_rad = float(inp_args.cut_off_rad)

#             del cart_atoms_ms

#             cut_off_func_ms = (ops.cos((ops.minimum(rad, cut_off_rad) / cut_off_rad) * np.pi) + 1) / 2.0
#             exp_term = []
#             exp_term_vec=[]
#             exp_term_ten=[]
#             rad_list = []

#             for nth_fp, gamma in enumerate(inp_args.list_gamma):
#                 norm = np.power(gamma / np.pi, 1.5)
#                 norm = ms.Tensor(norm,ms.float32)
#                 gamma = ms.Tensor(gamma,ms.float32)
#                 exp_term.append(norm * ops.exp(-gamma * rad ** 2))
#                 tens1=norm * ops.exp(-gamma * rad ** 2)
#                 tens2=rad**2
#                 p = ops.where(rad == 0.0,   0.0, ops.div(tens1, rad))
#                 h = ops.where(tens2 == 0.0, 0.0, ops.div(tens1, tens2))

#                 exp_term_vec.append(p)
#                 exp_term_ten.append(h)
#                 exp_term[nth_fp] = exp_term[nth_fp] * cut_off_func_ms
#                 exp_term_vec[nth_fp]=exp_term_vec[nth_fp]*cut_off_func_ms
#                 exp_term_ten[nth_fp]=exp_term_ten[nth_fp]*cut_off_func_ms
#                 rad_list.append(ops.sum(exp_term[nth_fp], dim=0))
# #
#             del cut_off_func_ms
#             rad_fp = ops.stack(rad_list)

#             rad_dir_list=[]
#             for i in [0,1,2]:
#                 rad_list = []
#                 for nth_fp, gamma in enumerate(inp_args.list_gamma):
#                     rad_list.append(ops.sum((rad_diff[:,:,i] * exp_term_vec[nth_fp]), dim=0))

#                 rad_dir_list.append(ops.stack(rad_list))
# #
#             dipole = ops.sqrt(rad_dir_list[0]**2 + rad_dir_list[1]**2 + rad_dir_list[2]**2)
# #
#             rad_dir_list = []
# #
#             rad_1 = [0, 1, 2, 0, 1, 2]
#             rad_2 = [0, 1, 2, 1, 2, 0]
# #
#             for i in [0, 1, 2, 3, 4, 5]:
#                 rad_list = []
#                 for nth_fp, gamma in enumerate(inp_args.list_gamma):
    
#                     rad_list.append(ops.sum(rad_diff[:,:,rad_1[i]] * rad_diff[:,:,rad_2[i]] * exp_term_ten[nth_fp], dim=0))
# #
#                 rad_dir_list.append(ops.stack(rad_list))
# #
#             del rad_list
#             quad_1 = rad_dir_list[0] + rad_dir_list[1] + rad_dir_list[2]
# #
#             quad_2 = ops.sqrt(ops.abs(rad_dir_list[0] * rad_dir_list[1] + rad_dir_list[2] * rad_dir_list[0] + rad_dir_list[2] * rad_dir_list[1] - rad_dir_list[3] ** 2 - rad_dir_list[5] ** 2 - rad_dir_list[4] ** 2))
#             quad_3 = ops.pow(ops.abs(rad_dir_list[0] * (rad_dir_list[1] * rad_dir_list[2] - rad_dir_list[4] ** 2) - rad_dir_list[3] * (
#                         rad_dir_list[3]* rad_dir_list[2]- rad_dir_list[4] * rad_dir_list[5]) + rad_dir_list[5] * (
#                         rad_dir_list[3] * rad_dir_list[4]- rad_dir_list[1] * rad_dir_list[5])), 1. / 3.)

#             del rad_dir_list
#             quad = ops.cat([rad_fp, dipole, quad_1, quad_2, quad_3], axis=0)
#             quad_list.append(ops.transpose(quad,(1,0)))
# #
#         all_elem_quad=ops.cat(quad_list,axis=-1)

#         return all_elem_quad
# #
#     cart_grid_K = ms.Tensor(cart_grid_K,dtype=ms.float32)
#     Y = quad(cart_grid_K)
#     Y = Y.asnumpy()
#     num_atoms = cart_grid_K.shape[0]

    def quad(cart_grid_np):
        quad_list = []
        for num_e in range(len(unique_cart_list)):
            cart_atoms_np = np.array(unique_cart_list[num_e], dtype=np.float32)
            rad_diff = cart_grid_np - cart_atoms_np[:, None]
            rad = np.sqrt(np.sum(rad_diff ** 2, axis=-1))
            rad_inv = 1.0 / rad
            cut_off_rad = float(inp_args.cut_off_rad)

            cut_off_func_np = (np.cos(np.minimum(rad, cut_off_rad) / cut_off_rad * np.pi) + 1) / 2.0
            exp_term = []
            exp_term_vec = []
            exp_term_ten = []
            rad_list = []

            for nth_fp, gamma in enumerate(inp_args.list_gamma):
                norm = (gamma / np.pi) ** 1.5
                gamma = np.array(gamma, dtype=np.float32)
                exp = norm * np.exp(-gamma * rad ** 2)
                exp_term.append(exp)

                tens1 = exp
                tens2 = rad ** 2
                p = np.where(rad == 0.0, 0.0, tens1 / rad)
                h = np.where(tens2 == 0.0, 0.0, tens1 / tens2)

                exp_term_vec.append(p)
                exp_term_ten.append(h)

                exp_term[nth_fp] *= cut_off_func_np
                exp_term_vec[nth_fp] *= cut_off_func_np
                exp_term_ten[nth_fp] *= cut_off_func_np
                rad_list.append(np.sum(exp_term[nth_fp], axis=0))

            rad_fp = np.stack(rad_list)

            rad_dir_list = []
            for i in [0, 1, 2]:
                rad_list = []
                for nth_fp, gamma in enumerate(inp_args.list_gamma):
                    rad_list.append(np.sum(rad_diff[:, :, i] * exp_term_vec[nth_fp], axis=0))

                rad_dir_list.append(np.stack(rad_list))

            dipole = np.sqrt(rad_dir_list[0]**2 + rad_dir_list[1]**2 + rad_dir_list[2]**2)

            rad_dir_list = []

            rad_1 = [0, 1, 2, 0, 1, 2]
            rad_2 = [0, 1, 2, 1, 2, 0]

            for i in [0, 1, 2, 3, 4, 5]:
                rad_list = []
                for nth_fp, gamma in enumerate(inp_args.list_gamma):
                    rad_list.append(np.sum(rad_diff[:, :, rad_1[i]] * rad_diff[:, :, rad_2[i]] * exp_term_ten[nth_fp], axis=0))

                rad_dir_list.append(np.stack(rad_list))

            quad_1 = rad_dir_list[0] + rad_dir_list[1] + rad_dir_list[2]

            quad_2 = np.sqrt(np.abs(
                rad_dir_list[0] * rad_dir_list[1] + rad_dir_list[2] * rad_dir_list[0] + rad_dir_list[2] * rad_dir_list[1]
                - rad_dir_list[3] ** 2 - rad_dir_list[5] ** 2 - rad_dir_list[4] ** 2))

            quad_3 = np.abs(
                rad_dir_list[0] * (rad_dir_list[1] * rad_dir_list[2] - rad_dir_list[4] ** 2)
                - rad_dir_list[3] * (rad_dir_list[3] * rad_dir_list[2] - rad_dir_list[4] * rad_dir_list[5])
                + rad_dir_list[5] * (rad_dir_list[3] * rad_dir_list[4] - rad_dir_list[1] * rad_dir_list[5])) ** (1. / 3.)

            quad = np.concatenate([rad_fp, dipole, quad_1, quad_2, quad_3], axis=0)
            quad_list.append(np.transpose(quad, (1, 0)))

        all_elem_quad = np.concatenate(quad_list, axis=-1)

        return all_elem_quad

    cart_grid_np = np.array(cart_grid_K, dtype=np.float32)
    Y = quad(cart_grid_np)
    num_atoms = cart_grid_np.shape[0]


    if num_elems==2:
        fp_el=Y.shape[1]
        zr=np.zeros((int(num_atoms),int(fp_el)))
        Y=np.concatenate((Y,zr),axis=-1)
        num_elems=4
        at_elem.append(0)
        at_elem.append(0)
    if num_elems==3:
        if elems_list[2]=='N':
            fp_el=Y.shape[1]/3
            zr=np.zeros((int(num_atoms),int(fp_el)))
            Y=np.concatenate((Y,zr),axis=-1)
            num_elems=4
            at_elem.append(0)
        else:
            fp_el=Y.shape[1]/3
            zr=np.zeros((int(num_atoms),int(fp_el)))
            Y_ch=Y[:,:int(2*fp_el)]
            Y_chn=np.concatenate((Y_ch,zr),axis=-1)
            Y_chno=np.concatenate((Y_chn,Y[:,int(2*fp_el):]),axis=-1)
            Y=Y_chno
            num_elems=4
            sites_elem.append(sites_elem[2])
            sites_elem[2]=0
            new_at_elem=[at_elem[0],at_elem[1],0,at_elem[2]]
            at_elem=new_at_elem
    print('at_elem', at_elem)
    num_atoms=cart_grid_K.shape[0]
    matrix_tot=[]
    matrixT_tot=[]
    cutoff_distance=5.0
    for pp in range(0,2):
        for x in sites_elem[pp]:
            pos=x.coords
            pos_frac=x.frac_coords
            neighs_list=list(itertools.chain.from_iterable(poscar_data.structure.get_neighbors(x,cutoff_distance)))
            group_lst=[neighs_list[i:i+4] for i in range(0, len(neighs_list), 4)]
            sorted_list=sorted(group_lst, key=itemgetter(1))
            v1=sorted_list[0][0].coords-pos
            v2=sorted_list[1][0].coords-pos
            u3=np.cross(v1,v2)
            u2=np.cross(v1,u3)
            u1=v1/np.linalg.norm(v1)
            u2=u2/np.linalg.norm(u2)
            u3=u3/np.linalg.norm(u3)
            matrx=np.transpose(np.array([u1,u2,u3]))
            matrix_tot.append(matrx)
    if at_elem[2] !=0:
        for x in sites_elem[2]:
            pos=x.coords
            pos_frac=x.frac_coords
            neighs_list=list(itertools.chain.from_iterable(poscar_data.structure.get_neighbors(x,cutoff_distance)))
            group_lst=[neighs_list[i:i+4] for i in range(0, len(neighs_list), 4)]
            sorted_list=sorted(group_lst, key=itemgetter(1))
            v1=sorted_list[0][0].coords-pos
            v2=sorted_list[1][0].coords-pos
            u3=np.cross(v1,v2)
            u2=np.cross(v1,u3)
            u1=v1/np.linalg.norm(v1)
            u2=u2/np.linalg.norm(u2)
            u3=u3/np.linalg.norm(u3)
            matrx=np.transpose(np.array([u1,u2,u3]))
            matrix_tot.append(matrx)
    if at_elem[3] !=0:
        for x in sites_elem[3]:
            pos=x.coords
            pos_frac=x.frac_coords
            neighs_list=list(itertools.chain.from_iterable(poscar_data.structure.get_neighbors(x,cutoff_distance)))
            group_lst=[neighs_list[i:i+4] for i in range(0, len(neighs_list), 4)]
            sorted_list=sorted(group_lst, key=itemgetter(1))
            v1=sorted_list[0][0].coords-pos
            v2=sorted_list[1][0].coords-pos
            u3=np.cross(v1,v2)
            u2=np.cross(v1,u3)
            u1=v1/np.linalg.norm(v1)
            u2=u2/np.linalg.norm(u2)
            u3=u3/np.linalg.norm(u3)
            matrx=np.transpose(np.array([u1,u2,u3]))
            matrix_tot.append(matrx)
    final_mat=np.vstack(np.array(matrix_tot))
    final_mat=np.reshape(final_mat,(num_atoms, 3, 3))

    return Y,final_mat,sites_elem,num_atoms,at_elem

def fp_chg_norm(Coef_at1,Coef_at2,Coef_at3,Coef_at4,X_3D1,X_3D2,X_3D3,X_3D4,padding_size):
    print(Coef_at1.shape, Coef_at2.shape,Coef_at3.shape,Coef_at4.shape)
    padCHG1 = np.pad(Coef_at1.T, pad_width=((0, 0), (0, padding_size - Coef_at1.T.shape[1]), (0, 0)), mode='constant', constant_values=(0, 0))
    padCHG2 = np.pad(Coef_at2.T, pad_width=((0, 0), (0, padding_size - Coef_at2.T.shape[1]), (0, 0)), mode='constant', constant_values=(0, 0))
    padCHG3 = np.pad(Coef_at3.T, pad_width=((0, 0), (0, padding_size - Coef_at3.T.shape[1]), (0, 0)), mode='constant', constant_values=(0, 0))
    padCHG4 = np.pad(Coef_at4.T, pad_width=((0, 0), (0, padding_size - Coef_at4.T.shape[1]), (0, 0)), mode='constant', constant_values=(0, 0))
    X_C=np.concatenate((X_3D1,padCHG1.T),axis=-1)
    X_H=np.concatenate((X_3D2,padCHG2.T),axis=-1)
    X_N=np.concatenate((X_3D3,padCHG3.T),axis=-1)
    X_O=np.concatenate((X_3D4,padCHG4.T),axis=-1)
    scalerC = load(CONFIG_PATH1)
    scalerH = load(CONFIG_PATH2)
    scalerN = load(CONFIG_PATH3)
    scalerO = load(CONFIG_PATH4)
    X_C=X_C.reshape(padding_size,X_C.shape[-1])
    X_H=X_H.reshape(padding_size,X_H.shape[-1])
    X_N=X_N.reshape(padding_size,X_N.shape[-1])
    X_O=X_O.reshape(padding_size,X_O.shape[-1])
    X_C = scalerC.transform(X_C)
    X_H = scalerH.transform(X_H)
    X_N = scalerN.transform(X_N)
    X_O = scalerO.transform(X_O)
    X_C=X_C.reshape(1,padding_size,X_C.shape[-1])
    X_H=X_H.reshape(1,padding_size,X_H.shape[-1])
    X_N=X_N.reshape(1,padding_size,X_N.shape[-1])
    X_O=X_O.reshape(1,padding_size,X_O.shape[-1])
    return(X_C,X_H,X_N,X_O)

def fp_norm(X_C,X_H,X_N,X_O,padding_size):
    a=X_C.shape[0]
    scalerC = load(CONFIG_PATH1)
    scalerH = load(CONFIG_PATH2)
    scalerN = load(CONFIG_PATH3)
    scalerO = load(CONFIG_PATH4)
    X_C=X_C.reshape(a*padding_size,X_C.shape[-1])
    X_H=X_H.reshape(a*padding_size,X_H.shape[-1])
    X_N=X_N.reshape(a*padding_size,X_N.shape[-1])
    X_O=X_O.reshape(a*padding_size,X_O.shape[-1])
    X_C = scalerC.transform(X_C)
    X_H = scalerH.transform(X_H)
    X_N = scalerN.transform(X_N)
    X_O = scalerO.transform(X_O)
    X_C=X_C.reshape(a,padding_size,X_C.shape[-1])
    X_H=X_H.reshape(a,padding_size,X_H.shape[-1])
    X_N=X_N.reshape(a,padding_size,X_N.shape[-1])
    X_O=X_O.reshape(a,padding_size,X_O.shape[-1])
    return(X_C,X_H,X_N,X_O)
