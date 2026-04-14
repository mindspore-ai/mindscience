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
"""CHG implement"""

import warnings
warnings.filterwarnings('ignore')
import os
import numpy as np
import sys
import math
import matplotlib
import mindspore as ms
import mindspore.ops as ops
import mindspore.nn as nn
from mindspore import Tensor
import pymatgen
from pymatgen.io.vasp.outputs import Chgcar
import random
from operator import itemgetter
import itertools


CONFIG_PATH = './checkpoints/weights_CHG.ckpt'

# define model
class model_atom_chg(nn.Cell):
    def __init__(self,out_channels,split1,split2):
        super(model_atom_chg, self).__init__()
        self.denseSeq = nn.SequentialCell(
            nn.Dense(360, 200,activation='tanh'),
            nn.Dense(200, 200,activation='tanh'),
            nn.Dense(200, 200,activation='tanh'),
            nn.Dense(200, 200,activation='tanh'),
            nn.Dense(200, out_channels)
        )
        self.split1 = split1
        self.split2 = split2
    def construct(self,input):
        model_out_all = self.denseSeq(input)
        model_out_exp, model_out_coefs = ops.split(model_out_all, [self.split1,self.split2], axis=1)
        model_out_exp = ops.abs(model_out_exp)
        model_out_all = ops.cat((model_out_exp, model_out_coefs), axis=-1)
        return model_out_all

class modelCHG(nn.Cell):
    def __init__(self,padding_size):
        super(modelCHG, self).__init__()
        self.time_distributed_1 = nn.TimeDistributed(model_atom_chg(340,93,247),time_axis=1, reshape_with_axis=0)
        self.time_distributed_2 = nn.TimeDistributed(model_atom_chg(208,58,150),time_axis=1, reshape_with_axis=0)
        self.time_distributed_3 = nn.TimeDistributed(model_atom_chg(340,93,247),time_axis=1, reshape_with_axis=0)
        self.time_distributed_4 = nn.TimeDistributed(model_atom_chg(340,93,247),time_axis=1, reshape_with_axis=0)
        self.padding_size = padding_size

    def construct(self,input1,input2,input3,input4):
        model_out_c1 = self.time_distributed_1(input1)
        model_out_c2 = self.time_distributed_2(input2)
        model_out_c3 = self.time_distributed_3(input3)
        model_out_c4 = self.time_distributed_4(input4)
        return model_out_c1,model_out_c2,model_out_c3,model_out_c4

class init_chgmod():
    def __init__(self,padding_size):
        self._network = modelCHG(padding_size)
        self._network.set_train(False)
    def predict(self, *inputs):
        return self._network(*inputs)

def model_weights(modelCHG):
    param_dict = ms.load_checkpoint(CONFIG_PATH)
    ms.load_param_into_net(modelCHG._network, param_dict)
    return modelCHG


## CHG prediction
def s_chg(n, exps, coefs, k_r):
    expss = np.reshape(exps, (exps.shape[0], 1))
    k_rr = np.reshape(k_r, (1, k_r.shape[0]))
    k_rr_n = np.power(k_rr, n)
    krr = np.power(k_rr, 2)
    coef = np.reshape(coefs, (1, coefs.shape[0]))
    exps = np.matmul(expss, krr)
    expss = -1.0 * exps
    exps = np.exp(expss)
    expss = np.multiply(exps, k_rr_n)
    cont = np.squeeze(np.asarray(np.matmul(coef, expss)))
    return cont

def p_chg(n, exps, coefsx, coefsy, coefsz, new_coords, k_r):
    expss = np.reshape(exps, (exps.shape[0], 1))
    k_rr = np.reshape(k_r, (1, k_r.shape[0]))
    k_rr_n = np.power(k_rr, n)
    krr = np.power(k_rr, 2)
    coefx = np.reshape(coefsx, (1, coefsx.shape[0]))
    coefy = np.reshape(coefsy, (1, coefsy.shape[0]))
    coefz = np.reshape(coefsz, (1, coefsz.shape[0]))
    exps = np.matmul(expss, krr)
    expss = -1.0 * exps
    exps = np.exp(expss)
    expsx = np.matmul(coefx, exps)
    expsy = np.matmul(coefy, exps)
    expsz = np.matmul(coefz, exps)
    expss = np.hstack((expsx.T, expsy.T, expsz.T))
    exps = np.multiply(expss, new_coords)
    expss = np.sum(exps, axis=1)
    cont = np.squeeze(np.asarray(np.multiply(expss, k_rr_n.T)))
    return cont

def d_chg(n, exps, coefs1, coefs2, coefs3, coefs4, coefs5, coords, k_r):
    expss = np.reshape(exps, (exps.shape[0], 1))
    k_rr = np.reshape(k_r, (1, k_r.shape[0]))
    k_rr_n = np.power(k_rr, n)
    krr = np.power(k_rr, 2)
    coef1 = np.reshape(coefs1, (1, coefs1.shape[0]))
    coef2 = np.reshape(coefs2, (1, coefs2.shape[0]))
    coef3 = np.reshape(coefs3, (1, coefs3.shape[0]))
    coef4 = np.reshape(coefs4, (1, coefs4.shape[0]))
    coef5 = np.reshape(coefs5, (1, coefs5.shape[0]))
    exps = np.matmul(expss, krr)
    expss = -1.0 * exps
    exps = np.exp(expss)
    exps1 = np.matmul(coef1, exps)
    exps2 = np.matmul(coef2, exps)
    exps3 = np.matmul(coef3, exps)
    exps4 = np.matmul(coef4, exps)
    exps5 = np.matmul(coef5, exps)
    expss = np.hstack((exps1.T, exps2.T, exps3.T, exps4.T, exps5.T))
    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]
    xy = np.multiply(x, y)
    xz = np.multiply(x, z)
    yz = np.multiply(y, z)
    xx = np.multiply(x, x)
    yy = np.multiply(y, y)
    zz = np.multiply(z, z)
    zz = 3.0 * zz
    xxyy = np.subtract(xx, yy)
    xxyy = 0.5 * xxyy
    rr = np.multiply(k_r, k_r)
    zzrr = np.subtract(zz, rr)
    zzrr = 0.288675135 * zzrr
    new_coords = np.hstack((xy, xz, yz, xxyy, zzrr))
    exps = np.multiply(expss, new_coords)
    expss = np.sum(exps, axis=1)
    cont = np.squeeze(np.asarray(np.multiply(expss, k_rr_n.T)))
    return cont

def f_chg(n, exps, coefs1, coefs2, coefs3, coefs4, coefs5, coefs6, coefs7, coords, k_r):
    expss = np.reshape(exps, (exps.shape[0], 1))
    k_rr = np.reshape(k_r, (1, k_r.shape[0]))
    k_rr_n = np.power(k_rr, n)
    krr = np.power(k_rr, 2)
    coef1 = np.reshape(coefs1, (1, coefs1.shape[0]))
    coef2 = np.reshape(coefs2, (1, coefs2.shape[0]))
    coef3 = np.reshape(coefs3, (1, coefs3.shape[0]))
    coef4 = np.reshape(coefs4, (1, coefs4.shape[0]))
    coef5 = np.reshape(coefs5, (1, coefs5.shape[0]))
    coef6 = np.reshape(coefs6, (1, coefs6.shape[0]))
    coef7 = np.reshape(coefs7, (1, coefs7.shape[0]))
    exps = np.matmul(expss, krr)
    expss = -1.0 * exps
    exps = np.exp(expss)
    exps1 = np.matmul(coef1, exps)
    exps2 = np.matmul(coef2, exps)
    exps3 = np.matmul(coef3, exps)
    exps4 = np.matmul(coef4, exps)
    exps5 = np.matmul(coef5, exps)
    exps6 = np.matmul(coef6, exps)
    exps7 = np.matmul(coef7, exps)
    expss = np.hstack((exps1.T, exps2.T, exps3.T, exps4.T, exps5.T, exps6.T, exps7.T))
    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]
    xy = np.multiply(x, y)
    yy = np.multiply(y, y)
    xx = np.multiply(x, x)
    zz = np.multiply(z, z)
    coef1 = np.multiply(xy, z)
    coef2 = 3.0 * yy
    coef2 = np.subtract(xx, coef2)
    coef2 = np.multiply(x, coef2)
    coef2 = 0.204124145 * coef2
    coef3 = 3.0 * xx
    coef3 = np.subtract(coef3, yy)
    coef3 = np.multiply(y, coef3)
    coef3 = 0.204124145 * coef3
    coef4 = np.subtract(xx, yy)
    coef4 = np.multiply(z, coef4)
    coef4 = 0.50 * coef4
    coef5 = 4.0 * zz
    coef5 = np.subtract(coef5, xx)
    term = np.subtract(coef5, yy)
    coef5 = np.multiply(x, term)
    coef5 = 0.158113883 * coef5
    coef6 = np.multiply(y, term)
    coef6 = 0.316227766 * coef6
    mzz = 2.0 * 3.0 * zz
    mxx = 3.0 * xx
    myy = 3.0 * yy
    coef7 = np.subtract(mzz, mxx)
    coef7 = np.subtract(coef7, myy)
    coef7 = np.multiply(z, coef7)
    coef7 = 0.129099445 * coef7
    new_coords = np.hstack((coef1, coef2, coef3, coef4, coef5, coef6, coef7))
    del coef1, coef2, coef3, coef4, coef5, coef6, coef7
    exps = np.multiply(expss, new_coords)
    expss = np.sum(exps, axis=1)
    cont = np.squeeze(np.asarray(np.multiply(expss, k_rr_n.T)))
    return cont

def g_chg(n, exps, coefs1, coefs2, coefs3, coefs4, coefs5, coefs6, coefs7, coefs8, coefs9, coords, k_r):
    expss = np.reshape(exps, (exps.shape[0], 1))
    k_rr = np.reshape(k_r, (1, k_r.shape[0]))
    k_rr_n = np.power(k_rr, n)
    krr = np.power(k_rr, 2)
    coef1 = np.reshape(coefs1, (1, coefs1.shape[0]))
    coef2 = np.reshape(coefs2, (1, coefs2.shape[0]))
    coef3 = np.reshape(coefs3, (1, coefs3.shape[0]))
    coef4 = np.reshape(coefs4, (1, coefs4.shape[0]))
    coef5 = np.reshape(coefs5, (1, coefs5.shape[0]))
    coef6 = np.reshape(coefs6, (1, coefs6.shape[0]))
    coef7 = np.reshape(coefs7, (1, coefs7.shape[0]))
    coef8 = np.reshape(coefs8, (1, coefs8.shape[0]))
    coef9 = np.reshape(coefs9, (1, coefs9.shape[0]))
    exps = np.matmul(expss, krr)
    expss = -1.0 * exps
    exps = np.exp(expss)
    exps1 = np.matmul(coef1, exps)
    exps2 = np.matmul(coef2, exps)
    exps3 = np.matmul(coef3, exps)
    exps4 = np.matmul(coef4, exps)
    exps5 = np.matmul(coef5, exps)
    exps6 = np.matmul(coef6, exps)
    exps7 = np.matmul(coef7, exps)
    exps8 = np.matmul(coef8, exps)
    exps9 = np.matmul(coef9, exps)
    expss = np.hstack((exps1.T, exps2.T, exps3.T, exps4.T, exps5.T, exps6.T, exps7.T, exps8.T, exps9.T))
    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]
    xy = np.multiply(x, y)
    yy = np.multiply(y, y)
    xx = np.multiply(x, x)
    zz = np.multiply(z, z)
    mg1 = np.multiply(xx, xy)
    mg2 = np.multiply(xy, yy)
    coef_f2 = 3.0 * yy
    coef_f2 = np.subtract(xx, coef_f2)
    coef_f2 = np.multiply(x, coef_f2)
    coef_f3 = 3.0 * xx
    coef_f3 = np.subtract(coef_f3, yy)
    coef_f3 = np.multiply(y, coef_f3)
    coef1 = np.subtract(mg1, mg2)
    del mg1
    del mg2
    mg3 = 7.0 * zz
    mg3 = np.subtract(mg3, krr.T)
    mg4 = 2.0 * krr.T
    mg4 = np.subtract(mg3, mg4)
    coef1 = 2.50334294 * coef1
    coef2 = np.multiply(z, coef_f3)
    coef2 = 1.77013077 * coef2
    coef3 = np.multiply(mg3, xy)
    del xy
    coef3 = 0.946174696 * coef3
    yz = np.multiply(y, z)
    coef4 = np.multiply(mg4, yz)
    coef4 = 0.669046544 * coef4
    del yz
    mg5 = np.power(k_rr.T, 4)
    mg5 = 3.0 * mg5
    mg6 = np.multiply(zz, zz)
    mg6 = 35.0 * mg6
    coef5 = np.add(mg6, mg5)
    del mg5
    del mg6
    mg7 = np.multiply(zz, krr.T)
    mg7 = 30.0 * mg7
    xxyy = np.subtract(xx, yy)
    del zz
    coef5 = np.subtract(coef5, mg7)
    coef5 = 0.105785547 * coef5
    xz = np.multiply(x, z)
    coef6 = np.multiply(mg4, xz)
    del xz
    coef6 = 0.669046544 * coef6
    del mg4
    coef7 = np.multiply(mg3, xxyy)
    coef7 = 0.946174696 * coef7
    del mg3
    coef8 = np.multiply(z, coef_f2)
    coef8 = 1.77013077 * coef8
    coef9 = np.multiply(xx, yy)
    coef9 = 4.0 * coef9
    mg8 = np.multiply(xxyy, xxyy)
    del xxyy
    coef9 = np.subtract(mg8, coef9)
    coef9 = 0.625835735 * coef9
    new_coords = np.hstack((coef1, coef2, coef3, coef4, coef5, coef6, coef7, coef8, coef9))
    del coef1, coef2, coef3, coef4, coef5, coef6, coef7, coef8, coef9
    exps = np.multiply(expss, new_coords)
    expss = np.sum(exps, axis=1)
    cont = np.squeeze(np.asarray(np.multiply(expss, k_rr_n.T)))
    return cont

def meshgrid2(*arrs):
    arrs = tuple(reversed(arrs))
    lens = list(map(len, arrs))
    dim = len(arrs)
    sz = 1
    for s in lens:
        sz *= s
        ans = []
    for i, arr in enumerate(arrs):
        slc = [1] * dim
        slc[i] = lens[i]
        arr2 = np.asarray(arr).reshape(slc)
        for j, sz in enumerate(lens):
            if j != i:
                arr2 = arr2.repeat(sz, axis=j)
        ans.append(arr2)
    return tuple(ans)

def chg_pts(poscar_data, supercell, grid_spacing):
    supercell_size = [1, 1, 1]
    centering = []
    centering.append(0.5 - 1 / (2 * supercell_size[0]))
    centering.append(0.5 - 1 / (2 * supercell_size[1]))
    centering.append(0.5 - 1 / (2 * supercell_size[2]))
    lengths = poscar_data.structure.lattice.abc
    num_pts = [int(np.round(lengths[i] / grid_spacing)) for i in range(0, 3)]
    xgrid = np.array([i / num_pts[0] * lengths[0] for i in range(num_pts[0])]) / supercell.lattice.a + centering[0]
    ygrid = np.array([i / num_pts[1] * lengths[1] for i in range(num_pts[1])]) / supercell.lattice.b + centering[1]
    zgrid = np.array([i / num_pts[2] * lengths[2] for i in range(num_pts[2])]) / supercell.lattice.c + centering[2]
    g = meshgrid2(xgrid, ygrid, zgrid)

    positions = np.vstack([x.flatten() for x in g])
    positions[[0, 2]] = positions[[2, 0]]
    list_grid_pts = positions.T

    return list_grid_pts, num_pts

def chg_ref(file_loc, vol, supercell):
    chgcar_file = os.path.join(file_loc, "CHGCAR")
    chgcar_data = Chgcar.from_file(chgcar_file)
    density = chgcar_data.data['total'].flatten('F') / vol
    supercell_size = [1, 1, 1]
    centering = []
    centering.append(0.5 - 1 / (2 * supercell_size[0]))
    centering.append(0.5 - 1 / (2 * supercell_size[1]))
    centering.append(0.5 - 1 / (2 * supercell_size[2]))
    lengths = chgcar_data.poscar.structure.lattice.abc
    xgrid = np.array(chgcar_data.get_axis_grid(0)) / supercell.lattice.a + centering[0]
    ygrid = np.array(chgcar_data.get_axis_grid(1)) / supercell.lattice.b + centering[1]
    zgrid = np.array(chgcar_data.get_axis_grid(2)) / supercell.lattice.c + centering[2]
    num_pts = [int(np.array(chgcar_data.get_axis_grid(i)).shape[0]) for i in range(0, 3)]
    g = meshgrid2(xgrid, ygrid, zgrid)

    positions = np.vstack([x.flatten() for x in g])
    positions[[0, 2]] = positions[[2, 0]]
    list_grid_pts = positions.T

    hh = [(x, i) for (i, x) in enumerate(density)]

    hhh = hh[:]
    chg_coor = []
    chg_den = []  # Coment out later
    for (x, i) in hhh:
        chg_den.append(density[i])
        chg_coor.append(list_grid_pts[i])
    chg_coor = np.array(chg_coor)
    chg_den = np.array(chg_den)
    return chg_coor, chg_den, num_pts

def chg_train(file_loc, vol, supercell, sites_elem, num_atoms, at_elem):
    chgcar_file = os.path.join(file_loc, "CHGCAR")
    chgcar_data = Chgcar.from_file(chgcar_file)
    density = chgcar_data.data['total'].flatten('F') / vol
    dim = supercell.lattice.matrix
    supercell_size = [1, 1, 1]
    centering = []
    centering.append(0.5 - 1 / (2 * supercell_size[0]))
    centering.append(0.5 - 1 / (2 * supercell_size[1]))
    centering.append(0.5 - 1 / (2 * supercell_size[2]))
    lengths = chgcar_data.poscar.structure.lattice.abc

    xgrid = np.array(chgcar_data.get_axis_grid(0)) / supercell.lattice.a + centering[0]
    ygrid = np.array(chgcar_data.get_axis_grid(1)) / supercell.lattice.b + centering[1]
    zgrid = np.array(chgcar_data.get_axis_grid(2)) / supercell.lattice.c + centering[2]

    g = meshgrid2(xgrid, ygrid, zgrid)

    positions = np.vstack([x.flatten() for x in g])
    positions[[0, 2]] = positions[[2, 0]]
    list_grid_pts = positions.T

    ff = density.shape[0]
    mm1 = int(ff * 0.005)
    mm2 = int(ff * 0.01)
    mm = int(ff * 0.05)
    zz = int(ff * 0.10)
    yy = int(ff * 0.20)
    xx = int(ff * 0.30)
    oo = int(ff * 0.40)
    uu = int(ff * 0.50)
    ee = int(ff * 0.70)
    jj = int(ff * 0.90)
    hh = sorted([(x, i) for (i, x) in enumerate(density)], reverse=True)
    my_randoms = random.sample(hh[:mm1], 50) + random.sample(hh[mm1:mm2], 50) + random.sample(hh[mm2:mm],
                                                                                              150) + random.sample(
        hh[mm:zz], 200) + random.sample(hh[zz:yy], 150) + random.sample(hh[yy:xx], 100) + random.sample(hh[xx:oo],
                                                                                                        100) + random.sample(
        hh[oo:uu], 100) + random.sample(hh[uu:ee], 50) + random.sample(hh[ee:jj], 50)

    chg_coor = []
    chg_den = []
    for (x, i) in my_randoms:
        chg_coor.append(list_grid_pts[i])
        chg_den.append(density[i])
    chg_coor = np.array(chg_coor)
    chg_den = np.array(chg_den)
    final_coords = []
    cutoff_distance = 5.0
    for pp in range(0, 2):
        for x in sites_elem[pp]:
            pos = x.coords
            pos_frac = x.frac_coords
            neighs_list = list(itertools.chain.from_iterable(chgcar_data.structure.get_neighbors(x, cutoff_distance)))
            group_lst = [neighs_list[i:i + 4] for i in range(0, len(neighs_list), 4)]
            sorted_list = sorted(group_lst, key=itemgetter(1))
            v1 = sorted_list[0][0].coords - pos
            v2 = sorted_list[1][0].coords - pos
            u3 = np.cross(v1, v2)
            u2 = np.cross(v1, u3)
            u1 = v1 / np.linalg.norm(v1)
            u2 = u2 / np.linalg.norm(u2)
            u3 = u3 / np.linalg.norm(u3)
            matrx = np.transpose(np.array([u1, u2, u3]))
            chg_atoms = chg_coor - np.tile(pos_frac, (chg_coor.shape[0], 1))
            chg_atoms[chg_atoms > 0.5] -= 1.0
            chg_atoms[chg_atoms < -0.5] += 1.0
            chg_atoms = np.matrix(chg_atoms)
            dim = np.matrix(dim)
            matrx = np.matrix(matrx)
            new_coords = chg_atoms * dim * matrx
            final_coords.append(new_coords)
    if at_elem[2] != 0:
        for x in sites_elem[2]:
            pos = x.coords
            pos_frac = x.frac_coords
            neighs_list = list(itertools.chain.from_iterable(chgcar_data.structure.get_neighbors(x, cutoff_distance)))
            group_lst = [neighs_list[i:i + 4] for i in range(0, len(neighs_list), 4)]
            sorted_list = sorted(group_lst, key=itemgetter(1))
            v1 = sorted_list[0][0].coords - pos
            v2 = sorted_list[1][0].coords - pos
            u3 = np.cross(v1, v2)
            u2 = np.cross(v1, u3)
            u1 = v1 / np.linalg.norm(v1)
            u2 = u2 / np.linalg.norm(u2)
            u3 = u3 / np.linalg.norm(u3)
            matrx = np.transpose(np.array([u1, u2, u3]))
            chg_atoms = chg_coor - np.tile(pos_frac, (chg_coor.shape[0], 1))
            chg_atoms[chg_atoms > 0.5] -= 1.0
            chg_atoms[chg_atoms < -0.5] += 1.0
            chg_atoms = np.matrix(chg_atoms)
            dim = np.matrix(dim)
            matrx = np.matrix(matrx)
            new_coords = chg_atoms * dim * matrx
            final_coords.append(new_coords)
    if at_elem[3] != 0:
        for x in sites_elem[3]:
            pos = x.coords
            pos_frac = x.frac_coords
            neighs_list = list(itertools.chain.from_iterable(chgcar_data.structure.get_neighbors(x, cutoff_distance)))
            group_lst = [neighs_list[i:i + 4] for i in range(0, len(neighs_list), 4)]
            sorted_list = sorted(group_lst, key=itemgetter(1))
            v1 = sorted_list[0][0].coords - pos
            v2 = sorted_list[1][0].coords - pos
            u3 = np.cross(v1, v2)
            u2 = np.cross(v1, u3)
            u1 = v1 / np.linalg.norm(v1)
            u2 = u2 / np.linalg.norm(u2)
            u3 = u3 / np.linalg.norm(u3)
            matrx = np.transpose(np.array([u1, u2, u3]))
            chg_atoms = chg_coor - np.tile(pos_frac, (chg_coor.shape[0], 1))
            chg_atoms[chg_atoms > 0.5] -= 1.0
            chg_atoms[chg_atoms < -0.5] += 1.0
            chg_atoms = np.matrix(chg_atoms)
            dim = np.matrix(dim)
            matrx = np.matrix(matrx)
            new_coords = chg_atoms * dim * matrx
            final_coords.append(new_coords)
    local = np.vstack(np.array(final_coords))
    local_coords = np.reshape(local, (num_atoms, 1000, 3))
    return chg_den, local_coords

def chg_dat_prep(at_elem, dataset1, dataset2, i1, i2, i3, i4, num_chg_bins):
    val_conf1 = []
    rad_conf1 = []
    val_conf2 = []
    rad_conf2 = []
    val_conf3 = []
    rad_conf3 = []
    val_conf4 = []
    rad_conf4 = []
    num_atoms = dataset2.shape[0]
    for h in range(at_elem[0]):
        i = h
        val_atom = []
        rad_atom = []
        for j in range(dataset2.shape[1]):
            coords = dataset2[i][j]
            r = np.sqrt(coords[0] ** 2 + coords[1] ** 2 + coords[2] ** 2)
            values = np.array(dataset2[i][j])
            val_atom.append(values)
            rad_atom.append(r)
        val_atom = np.vstack(val_atom)
        rad_atom = np.stack(rad_atom)
        val_conf1.append(val_atom)
        rad_conf1.append(rad_atom)
    for h in range(at_elem[1]):
        val_atom = []
        rad_atom = []
        i = at_elem[0] + h
        for j in range(dataset2.shape[1]):
            coords = dataset2[i][j]
            r = np.sqrt(coords[0] ** 2 + coords[1] ** 2 + coords[2] ** 2)
            values = np.array(dataset2[i][j])
            val_atom.append(values)
            rad_atom.append(r)
        val_atom = np.vstack(val_atom)
        rad_atom = np.stack(rad_atom)
        val_conf2.append(val_atom)
        rad_conf2.append(rad_atom)
    if at_elem[2] != 0:
        for h in range(at_elem[2]):
            val_atom = []
            rad_atom = []
            i = at_elem[0] + at_elem[1] + h
            for j in range(dataset2.shape[1]):
                coords = dataset2[i][j]
                r = np.sqrt(coords[0] ** 2 + coords[1] ** 2 + coords[2] ** 2)
                values = np.array(dataset2[i][j])
                val_atom.append(values)
                rad_atom.append(r)
            val_atom = np.vstack(val_atom)
            rad_atom = np.stack(rad_atom)
            val_conf3.append(val_atom)
            rad_conf3.append(rad_atom)
        val_conf3 = np.vstack(val_conf3)
        val_conf3 = np.array(val_conf3)
        rad_conf3 = np.vstack(rad_conf3)
        rad_conf3 = np.array(rad_conf3)
        dataset2_at3 = np.reshape(val_conf3, (i3, num_chg_bins, 3))
    if at_elem[3] != 0:
        for h in range(at_elem[3]):
            val_atom = []
            rad_atom = []
            i = at_elem[0] + at_elem[1] + h
            for j in range(dataset2.shape[1]):
                coords = dataset2[i][j]
                r = np.sqrt(coords[0] ** 2 + coords[1] ** 2 + coords[2] ** 2)
                values = np.array(dataset2[i][j])
                val_atom.append(values)
                rad_atom.append(r)
            val_atom = np.vstack(val_atom)
            rad_atom = np.stack(rad_atom)
            val_conf4.append(val_atom)
            rad_conf4.append(rad_atom)
        val_conf4 = np.vstack(val_conf4)
        val_conf4 = np.array(val_conf4)
        rad_conf4 = np.vstack(rad_conf4)
        rad_conf4 = np.array(rad_conf4)
        dataset2_at4 = np.reshape(val_conf4, (i4, num_chg_bins, 3))
    val_conf1 = np.vstack(val_conf1)
    val_conf1 = np.array(val_conf1)
    rad_conf1 = np.vstack(rad_conf1)
    rad_conf1 = np.array(rad_conf1)
    val_conf2 = np.vstack(val_conf2)
    val_conf2 = np.array(val_conf2)
    rad_conf2 = np.vstack(rad_conf2)
    rad_conf2 = np.array(rad_conf2)
    dataset2_at1 = np.reshape(val_conf1, (i1, num_chg_bins, 3))
    dataset2_at2 = np.reshape(val_conf2, (i2, num_chg_bins, 3))
    dataset_at1 = []
    dataset_at2 = []
    dataset_at3 = []
    dataset_at4 = []
    for i in range(i1):
        con = np.append(dataset1[i][0:360], dataset2_at1[i])
        con2 = np.append(con, rad_conf1[i])
        dataset_at1.append(con2)
    for i in range(i2):
        con = np.append(dataset1[i1 + i][0:360], dataset2_at2[i])
        con2 = np.append(con, rad_conf2[i])
        #        print('shape con2:', con2.shape)
        dataset_at2.append(con2)
    if at_elem[2] != 0:
        for i in range(i3):
            con = np.append(dataset1[i1 + i2 + i][0:360], dataset2_at3[i])
            con2 = np.append(con, rad_conf3[i])
            dataset_at3.append(con2)
    else:
        dataset_at3.append([0] * 3560)
    if at_elem[3] != 0:
        for i in range(i4):
            con = np.append(dataset1[i1 + i2 + i3 + i][0:360], dataset2_at4[i])
            con2 = np.append(con, rad_conf4[i])
            dataset_at4.append(con2)
    else:
        dataset_at4.append([0] * 3560)

    dataset_at1 = np.vstack(dataset_at1)
    dataset_at2 = np.vstack(dataset_at2)
    dataset_at3 = np.vstack(dataset_at3)
    dataset_at4 = np.vstack(dataset_at4)
    global padding_size
    X_tot_at1 = np.pad(dataset_at1.T, pad_width=((0, 0), (0, padding_size - dataset_at1.T.shape[1])), mode='constant',constant_values=(0, 0))
    X_tot_at2 = np.pad(dataset_at2.T, pad_width=((0, 0), (0, padding_size - dataset_at2.T.shape[1])), mode='constant',constant_values=(0, 0))
    X_tot_at3 = np.pad(dataset_at3.T, pad_width=((0, 0), (0, padding_size - dataset_at3.T.shape[1])), mode='constant',constant_values=(0, 0))
    X_tot_at4 = np.pad(dataset_at4.T, pad_width=((0, 0), (0, padding_size - dataset_at4.T.shape[1])), mode='constant',constant_values=(0, 0))
    return X_tot_at1, X_tot_at2, X_tot_at3, X_tot_at4

def C_coef_split(Coef_at1):
    expss, coefss = np.split(Coef_at1, [93])
    exps_1s, exps_2s, exps_3s, exps_4s, exps_5s, exps_6s, exps_7s, exps_2p, exps_3p, exps_4p, exps_5p, exps_6p, exps_3d, exps_4d, exps_5d, exps_6d, exps_4f, exps_5f, exps_5g = np.split(
        expss, [12, 24, 34, 40, 44, 46, 47, 57, 65, 69, 71, 72, 78, 82, 84, 85, 89, 91])
    coefs_1s, coefs_2s, coefs_3s, coefs_4s, coefs_5s, coefs_6s, coefs_7s, coefs_2px, coefs_2py, coefs_2pz, coefs_3px, coefs_3py, coefs_3pz, coefs_4px, coefs_4py, coefs_4pz, coefs_5px, coefs_5py, coefs_5pz, coefs_6px, coefs_6py, coefs_6pz, coefs_3d1, coefs_3d2, coefs_3d3, coefs_3d4, coefs_3d5, coefs_4d1, coefs_4d2, coefs_4d3, coefs_4d4, coefs_4d5, coefs_5d1, coefs_5d2, coefs_5d3, coefs_5d4, coefs_5d5, coefs_6d1, coefs_6d2, coefs_6d3, coefs_6d4, coefs_6d5, coefs_4f1, coefs_4f2, coefs_4f3, coefs_4f4, coefs_4f5, coefs_4f6, coefs_4f7, coefs_5f1, coefs_5f2, coefs_5f3, coefs_5f4, coefs_5f5, coefs_5f6, coefs_5f7, coefs_5g1, coefs_5g2, coefs_5g3, coefs_5g4, coefs_5g5, coefs_5g6, coefs_5g7, coefs_5g8, coefs_5g9 = np.split(
        coefss,
        [12, 24, 34, 40, 44, 46, 47, 57, 67, 77, 85, 93, 101, 105, 109, 113, 115, 117, 119, 120, 121, 122, 128, 134,
         140, 146, 152, 156, 160, 164, 168, 172, 174, 176, 178, 180, 182, 183, 184, 185, 186, 187, 191, 195, 199, 203,
         207, 211, 215, 217, 219, 221, 223, 225, 227, 229, 231, 233, 235, 237, 239, 241, 243, 245])
    return exps_1s, exps_2s, exps_3s, exps_4s, exps_5s, exps_6s, exps_7s, exps_2p, exps_3p, exps_4p, exps_5p, exps_6p, exps_3d, exps_4d, exps_5d, exps_6d, exps_4f, exps_5f, exps_5g, coefs_1s, coefs_2s, coefs_3s, coefs_4s, coefs_5s, coefs_6s, coefs_7s, coefs_2px, coefs_2py, coefs_2pz, coefs_3px, coefs_3py, coefs_3pz, coefs_4px, coefs_4py, coefs_4pz, coefs_5px, coefs_5py, coefs_5pz, coefs_6px, coefs_6py, coefs_6pz, coefs_3d1, coefs_3d2, coefs_3d3, coefs_3d4, coefs_3d5, coefs_4d1, coefs_4d2, coefs_4d3, coefs_4d4, coefs_4d5, coefs_5d1, coefs_5d2, coefs_5d3, coefs_5d4, coefs_5d5, coefs_6d1, coefs_6d2, coefs_6d3, coefs_6d4, coefs_6d5, coefs_4f1, coefs_4f2, coefs_4f3, coefs_4f4, coefs_4f5, coefs_4f6, coefs_4f7, coefs_5f1, coefs_5f2, coefs_5f3, coefs_5f4, coefs_5f5, coefs_5f6, coefs_5f7, coefs_5g1, coefs_5g2, coefs_5g3, coefs_5g4, coefs_5g5, coefs_5g6, coefs_5g7, coefs_5g8, coefs_5g9

def H_coef_split(Coef_at2):
    expss, coefss = np.split(Coef_at2, [58])
    exps_1s, exps_2s, exps_3s, exps_4s, exps_2p, exps_3p, exps_4p, exps_3d, exps_4d, exps_5d, exps_4f, exps_5f, exps_5g = np.split(
        expss, [12, 24, 28, 30, 38, 44, 46, 50, 52, 53, 55, 57])
    coefs_1s, coefs_2s, coefs_3s, coefs_4s, coefs_2px, coefs_2py, coefs_2pz, coefs_3px, coefs_3py, coefs_3pz, coefs_4px, coefs_4py, coefs_4pz, coefs_3d1, coefs_3d2, coefs_3d3, coefs_3d4, coefs_3d5, coefs_4d1, coefs_4d2, coefs_4d3, coefs_4d4, coefs_4d5, coefs_5d1, coefs_5d2, coefs_5d3, coefs_5d4, coefs_5d5, coefs_4f1, coefs_4f2, coefs_4f3, coefs_4f4, coefs_4f5, coefs_4f6, coefs_4f7, coefs_5f1, coefs_5f2, coefs_5f3, coefs_5f4, coefs_5f5, coefs_5f6, coefs_5f7, coefs_5g1, coefs_5g2, coefs_5g3, coefs_5g4, coefs_5g5, coefs_5g6, coefs_5g7, coefs_5g8, coefs_5g9 = np.split(
        coefss,
        [12, 24, 28, 30, 38, 46, 54, 60, 66, 72, 74, 76, 78, 82, 86, 90, 94, 98, 100, 102, 104, 106, 108, 109, 110, 111,
         112, 113, 115, 117, 119, 121, 123, 125, 127, 129, 131, 133, 135, 137, 139, 141, 142, 143, 144, 145, 146, 147,
         148, 149])
    return exps_1s, exps_2s, exps_3s, exps_4s, exps_2p, exps_3p, exps_4p, exps_3d, exps_4d, exps_5d, exps_4f, exps_5f, exps_5g, coefs_1s, coefs_2s, coefs_3s, coefs_4s, coefs_2px, coefs_2py, coefs_2pz, coefs_3px, coefs_3py, coefs_3pz, coefs_4px, coefs_4py, coefs_4pz, coefs_3d1, coefs_3d2, coefs_3d3, coefs_3d4, coefs_3d5, coefs_4d1, coefs_4d2, coefs_4d3, coefs_4d4, coefs_4d5, coefs_5d1, coefs_5d2, coefs_5d3, coefs_5d4, coefs_5d5, coefs_4f1, coefs_4f2, coefs_4f3, coefs_4f4, coefs_4f5, coefs_4f6, coefs_4f7, coefs_5f1, coefs_5f2, coefs_5f3, coefs_5f4, coefs_5f5, coefs_5f6, coefs_5f7, coefs_5g1, coefs_5g2, coefs_5g3, coefs_5g4, coefs_5g5, coefs_5g6, coefs_5g7, coefs_5g8, coefs_5g9

def C_coef_s(Coef_at1):
    exps_1s, exps_2s, exps_3s, exps_4s, exps_5s, exps_6s, exps_7s, exps_2p, exps_3p, exps_4p, exps_5p, exps_6p, exps_3d, exps_4d, exps_5d, exps_6d, exps_4f, exps_5f, exps_5g, coefs_1s, coefs_2s, coefs_3s, coefs_4s, coefs_5s, coefs_6s, coefs_7s, coefs_2px, coefs_2py, coefs_2pz, coefs_3px, coefs_3py, coefs_3pz, coefs_4px, coefs_4py, coefs_4pz, coefs_5px, coefs_5py, coefs_5pz, coefs_6px, coefs_6py, coefs_6pz, coefs_3d1, coefs_3d2, coefs_3d3, coefs_3d4, coefs_3d5, coefs_4d1, coefs_4d2, coefs_4d3, coefs_4d4, coefs_4d5, coefs_5d1, coefs_5d2, coefs_5d3, coefs_5d4, coefs_5d5, coefs_6d1, coefs_6d2, coefs_6d3, coefs_6d4, coefs_6d5, coefs_4f1, coefs_4f2, coefs_4f3, coefs_4f4, coefs_4f5, coefs_4f6, coefs_4f7, coefs_5f1, coefs_5f2, coefs_5f3, coefs_5f4, coefs_5f5, coefs_5f6, coefs_5f7, coefs_5g1, coefs_5g2, coefs_5g3, coefs_5g4, coefs_5g5, coefs_5g6, coefs_5g7, coefs_5g8, coefs_5g9 = C_coef_split(
        Coef_at1)
    return exps_1s, exps_2s, exps_3s, exps_4s, exps_5s, exps_6s, exps_7s, coefs_1s, coefs_2s, coefs_3s, coefs_4s, coefs_5s, coefs_6s, coefs_7s

def H_coef_s(Coef_at2):
    exps_1s, exps_2s, exps_3s, exps_4s, exps_2p, exps_3p, exps_4p, exps_3d, exps_4d, exps_5d, exps_4f, exps_5f, exps_5g, coefs_1s, coefs_2s, coefs_3s, coefs_4s, coefs_2px, coefs_2py, coefs_2pz, coefs_3px, coefs_3py, coefs_3pz, coefs_4px, coefs_4py, coefs_4pz, coefs_3d1, coefs_3d2, coefs_3d3, coefs_3d4, coefs_3d5, coefs_4d1, coefs_4d2, coefs_4d3, coefs_4d4, coefs_4d5, coefs_5d1, coefs_5d2, coefs_5d3, coefs_5d4, coefs_5d5, coefs_4f1, coefs_4f2, coefs_4f3, coefs_4f4, coefs_4f5, coefs_4f6, coefs_4f7, coefs_5f1, coefs_5f2, coefs_5f3, coefs_5f4, coefs_5f5, coefs_5f6, coefs_5f7, coefs_5g1, coefs_5g2, coefs_5g3, coefs_5g4, coefs_5g5, coefs_5g6, coefs_5g7, coefs_5g8, coefs_5g9 = H_coef_split(
        Coef_at2)
    return exps_1s, exps_2s, exps_3s, exps_4s, coefs_1s, coefs_2s, coefs_3s, coefs_4s

def coef_predict(X_3D1,X_3D2,X_3D3,X_3D4,i1,i2,i3,i4,modelCHG):
    modelCHGt=model_weights(modelCHG)
    modelInput = [X_3D1, X_3D2, X_3D3, X_3D4]
    modelInput = [Tensor(inp, dtype=ms.float32) for inp in modelInput]
    Coef_at1, Coef_at2, Coef_at3, Coef_at4 = modelCHGt.predict(*modelInput)
    Coef_at1, Coef_at2, Coef_at3, Coef_at4 = Coef_at1.asnumpy(), Coef_at2.asnumpy(), Coef_at3.asnumpy(), Coef_at4.asnumpy()
    Coef_at1=Coef_at1[:,0:i1,:]
    Coef_at2=Coef_at2[:,0:i2,:]
    if i3!=0:
        Coef_at3=Coef_at3[:,0:i3,:]
    else:
        Coef_at3=[0]*340
        Coef_at3=np.reshape(Coef_at3,(1,1,340))
    if i4!=0:
        Coef_at4=Coef_at4[:,0:i4,:]
    else:
        Coef_at4=[0]*340
        Coef_at4=np.reshape(Coef_at4,(1,1,340))
    return Coef_at1,Coef_at2,Coef_at3,Coef_at4

def chg_predict(X_3D1, X_3D2, X_3D3, X_3D4, i1, i2, i3, i4, sites_elem, modelCHG, at_elem):
    modelCHGt = model_weights(modelCHG)
    modelInput = [X_3D1, X_3D2, X_3D3, X_3D4]
    modelInput = [Tensor(inp, dtype=ms.float32) for inp in modelInput]
    Coef_at1, Coef_at2, Coef_at3, Coef_at4 = modelCHGt.predict(*modelInput)
    Coef_at1, Coef_at2, Coef_at3, Coef_at4 = Coef_at1.asnumpy(), Coef_at2.asnumpy(), Coef_at3.asnumpy(), Coef_at4.asnumpy()
    Coef_at1 = Coef_at1[:, 0:i1, :]
    Coef_at2 = Coef_at2[:, 0:i2, :]
    if i3 != 0:
        Coef_at3 = Coef_at3[:, 0:i3, :]
    else:
        Coef_at3 = [0] * 340
        Coef_at3 = np.reshape(Coef_at3, (1, 1, 340))
    if i4 != 0:
        Coef_at4 = Coef_at4[:, 0:i4, :]
    else:
        Coef_at4 = [0] * 340
        Coef_at4 = np.reshape(Coef_at4, (1, 1, 340))
    C_at_charge = []
    H_at_charge = []
    N_at_charge = []
    O_at_charge = []
    jj = 0
    for x in sites_elem[0]:
        exps_1s, exps_2s, exps_3s, exps_4s, exps_5s, exps_6s, exps_7s, coefs_1s, coefs_2s, coefs_3s, coefs_4s, coefs_5s, coefs_6s, coefs_7s = C_coef_s(
            Coef_at1[0, jj, :])
        charge = np.sum((np.pi) ** (3 / 2) * coefs_1s / (exps_1s) ** (3 / 2))
        charge = charge + np.sum(np.pi * 2.0 * coefs_2s / (exps_2s) ** 2)
        charge = charge + np.sum((np.pi) ** (3 / 2) * 3.0 * coefs_3s / (2.0 * (exps_3s) ** (5 / 2)))
        charge = charge + np.sum(np.pi * 4.0 * coefs_4s / (exps_4s) ** 3)
        charge = charge + np.sum((np.pi) ** (3 / 2) * 15.0 * coefs_5s / (4.0 * (exps_5s) ** (7 / 2)))
        charge = charge + np.sum(np.pi * 12.0 * coefs_6s / (exps_6s) ** 4)
        charge = charge + np.sum((np.pi) ** (3 / 2) * 105.0 * coefs_7s / (8.0 * (exps_7s) ** (9 / 2)))
        C_at_charge.append(charge)
        jj = jj + 1
    jj = 0
    for x in sites_elem[1]:
        exps_1s, exps_2s, exps_3s, exps_4s, coefs_1s, coefs_2s, coefs_3s, coefs_4s = H_coef_s(Coef_at2[0, jj, :])
        charge = np.sum((np.pi) ** (3 / 2) * coefs_1s / (exps_1s) ** (3 / 2))
        charge = charge + np.sum(np.pi * 2.0 * coefs_2s / (exps_2s) ** 2)
        charge = charge + np.sum((np.pi) ** (3 / 2) * 3.0 * coefs_3s / (2.0 * (exps_3s) ** (5 / 2)))
        charge = charge + np.sum(np.pi * 4.0 * coefs_4s / (exps_4s) ** 3)
        H_at_charge.append(charge)
        jj = jj + 1
    if at_elem[2] != 0:
        jj = 0
        for x in sites_elem[2]:
            exps_1s, exps_2s, exps_3s, exps_4s, exps_5s, exps_6s, exps_7s, coefs_1s, coefs_2s, coefs_3s, coefs_4s, coefs_5s, coefs_6s, coefs_7s = C_coef_s(
                Coef_at3[0, jj, :])
            charge = np.sum((np.pi) ** (3 / 2) * coefs_1s / (exps_1s) ** (3 / 2))
            charge = charge + np.sum(np.pi * 2.0 * coefs_2s / (exps_2s) ** 2)
            charge = charge + np.sum((np.pi) ** (3 / 2) * 3.0 * coefs_3s / (2.0 * (exps_3s) ** (5 / 2)))
            charge = charge + np.sum(np.pi * 4.0 * coefs_4s / (exps_4s) ** 3)
            charge = charge + np.sum((np.pi) ** (3 / 2) * 15.0 * coefs_5s / (4.0 * (exps_5s) ** (7 / 2)))
            charge = charge + np.sum(np.pi * 12.0 * coefs_6s / (exps_6s) ** 4)
            charge = charge + np.sum((np.pi) ** (3 / 2) * 105.0 * coefs_7s / (8.0 * (exps_7s) ** (9 / 2)))
            N_at_charge.append(charge)
            jj = jj + 1
    if at_elem[3] != 0:
        jj = 0
        for x in sites_elem[3]:
            exps_1s, exps_2s, exps_3s, exps_4s, exps_5s, exps_6s, exps_7s, coefs_1s, coefs_2s, coefs_3s, coefs_4s, coefs_5s, coefs_6s, coefs_7s = C_coef_s(
                Coef_at4[0, jj, :])
            charge = np.sum((np.pi) ** (3 / 2) * coefs_1s / (exps_1s) ** (3 / 2))
            charge = charge + np.sum(np.pi * 2.0 * coefs_2s / (exps_2s) ** 2)
            charge = charge + np.sum((np.pi) ** (3 / 2) * 3.0 * coefs_3s / (2.0 * (exps_3s) ** (5 / 2)))
            charge = charge + np.sum(np.pi * 4.0 * coefs_4s / (exps_4s) ** 3)
            charge = charge + np.sum((np.pi) ** (3 / 2) * 15.0 * coefs_5s / (4.0 * (exps_5s) ** (7 / 2)))
            charge = charge + np.sum(np.pi * 12.0 * coefs_6s / (exps_6s) ** 4)
            charge = charge + np.sum((np.pi) ** (3 / 2) * 105.0 * coefs_7s / (8.0 * (exps_7s) ** (9 / 2)))
            O_at_charge.append(charge)
            jj = jj + 1
    return Coef_at1, Coef_at2, Coef_at3, Coef_at4, C_at_charge, H_at_charge, N_at_charge, O_at_charge

def C_chg_print(x, cutoff_distance, coefs, sites_elem, poscar_data, chg_coor, dim, vol, tot_C, count, jj, iden,
                tot_chg):
    charge = 0
    pos = x.coords
    pos_frac = x.frac_coords
    neighs_list = list(itertools.chain.from_iterable(poscar_data.structure.get_neighbors(x, cutoff_distance)))
    group_lst = [neighs_list[i:i + 4] for i in range(0, len(neighs_list), 4)]
    sorted_list = sorted(group_lst, key=itemgetter(1))
    v1 = sorted_list[0][0].coords - pos
    v2 = sorted_list[1][0].coords - pos
    u3 = np.cross(v1, v2)
    u2 = np.cross(v1, u3)
    u1 = v1 / np.linalg.norm(v1)
    u2 = u2 / np.linalg.norm(u2)
    u3 = u3 / np.linalg.norm(u3)
    matrx = np.transpose(np.array([u1, u2, u3]))
    chg_atoms = chg_coor - np.tile(pos_frac, (chg_coor.shape[0], 1))
    chg_atoms[chg_atoms > 0.5] -= 1.0
    chg_atoms[chg_atoms < -0.5] += 1.0

    chg_atoms = np.matrix(chg_atoms)
    dim = np.matrix(dim)

    matrx = np.matrix(matrx)
    new_coords = chg_atoms * dim * matrx
    squ_coords = np.square(new_coords)
    kr = np.sum(squ_coords, axis=1)
    k_r = np.sqrt(kr)
    exps_1s, exps_2s, exps_3s, exps_4s, exps_5s, exps_6s, exps_7s, exps_2p, exps_3p, exps_4p, exps_5p, exps_6p, exps_3d, exps_4d, exps_5d, exps_6d, exps_4f, exps_5f, exps_5g, coefs_1s, coefs_2s, coefs_3s, coefs_4s, coefs_5s, coefs_6s, coefs_7s, coefs_2px, coefs_2py, coefs_2pz, coefs_3px, coefs_3py, coefs_3pz, coefs_4px, coefs_4py, coefs_4pz, coefs_5px, coefs_5py, coefs_5pz, coefs_6px, coefs_6py, coefs_6pz, coefs_3d1, coefs_3d2, coefs_3d3, coefs_3d4, coefs_3d5, coefs_4d1, coefs_4d2, coefs_4d3, coefs_4d4, coefs_4d5, coefs_5d1, coefs_5d2, coefs_5d3, coefs_5d4, coefs_5d5, coefs_6d1, coefs_6d2, coefs_6d3, coefs_6d4, coefs_6d5, coefs_4f1, coefs_4f2, coefs_4f3, coefs_4f4, coefs_4f5, coefs_4f6, coefs_4f7, coefs_5f1, coefs_5f2, coefs_5f3, coefs_5f4, coefs_5f5, coefs_5f6, coefs_5f7, coefs_5g1, coefs_5g2, coefs_5g3, coefs_5g4, coefs_5g5, coefs_5g6, coefs_5g7, coefs_5g8, coefs_5g9 = C_coef_split(
        coefs[0, jj, :])

    coords_limit = 1000000
    coords_shape = kr.shape[0]
    partsCHG = math.ceil(coords_shape / coords_limit)
    if coords_shape > coords_limit:
        if tot_chg:
            if iden == 1:
                c_coef_1s = np.array(
                    [116.578026855869, 112.510252969745, 41.165379246020, 56.773590524042, 86.402676630032,
                     63.577812378632, 0.637955734166, 44.707906728918, 71.946789359638, 105.619710421794,
                     38.996281296036, 88.894683580824, 9.202881074009])
                c_exp_1s = np.array(
                    [324.387996761133, 177.399068268376, 50.940492722610, 4073.371026571095, 95.863341632517,
                     110803.674121156117, 14.516932727395, 8458.147167515093, 2075.892101482822, 593.471652826499,
                     20211.367729746886, 1097.306310034251, 27.454141193838])
            elif iden == 2:
                c_coef_1s = np.array(
                    [162.741399115522, 1.000590152363, 120.717723038064, 65.800446492523, 176.972282692795,
                     70.279948577080, 97.856043902832, 127.188202750892, 15.158261318857, 78.581279593310,
                     147.872961853726, 121.986697165618, 169.430095505507])
                c_exp_1s = np.array(
                    [224.998268113168, 20.137330366020, 113070.822303298963, 70.515523980315, 392.151336096174,
                     20996.404792315752, 4456.523128073991, 128.364544225048, 38.128008571196, 9020.825868609380,
                     1255.100280831314, 2325.453330503726, 694.286492554629])
            elif iden == 3:
                c_coef_1s = np.array(
                    [3.894908509869, 142.249270911306, 50.657031837569, 126.375241189918, 218.051260068246,
                     192.172719533969, 0.408943286657, 115.196635211020, 210.000163764055, 242.782693209721,
                     250.696858077352, 121.091043858451, 8.165223094803, 44.126392088493, 216.928704406869,
                     114.664105986775])
                c_exp_1s = np.array(
                    [2664.774694202190, 5147.982644234844, 2747.036781808482, 115.306011217834, 998.024205222061,
                     1664.421679207663, 22.251540530064, 2920.109248055150, 115613.602533650555, 600.292587465404,
                     355.059222702842, 9850.273163919423, 39.286718666097, 67.119899157727, 204.056327537959,
                     21979.010168625922])
            core_chg = np.sum([s_chg(0, c_exp_1s, c_coef_1s, k_r[:coords_limit])], axis=0)
        chg_1s = s_chg(0, exps_1s, coefs_1s, k_r[:coords_limit])
        chg_2s = s_chg(1, exps_2s, coefs_2s, k_r[:coords_limit])
        chg_3s = s_chg(2, exps_3s, coefs_3s, k_r[:coords_limit])
        chg_4s = s_chg(3, exps_4s, coefs_4s, k_r[:coords_limit])
        chg_5s = s_chg(4, exps_5s, coefs_5s, k_r[:coords_limit])
        chg_6s = s_chg(5, exps_6s, coefs_6s, k_r[:coords_limit])
        chg_7s = s_chg(6, exps_7s, coefs_7s, k_r[:coords_limit])
        chg_2p = p_chg(0, exps_2p, coefs_2px, coefs_2py, coefs_2pz, new_coords[:coords_limit], k_r[:coords_limit])
        chg_3p = p_chg(1, exps_3p, coefs_3px, coefs_3py, coefs_3pz, new_coords[:coords_limit], k_r[:coords_limit])
        chg_4p = p_chg(2, exps_4p, coefs_4px, coefs_4py, coefs_4pz, new_coords[:coords_limit], k_r[:coords_limit])
        chg_5p = p_chg(3, exps_5p, coefs_5px, coefs_5py, coefs_5pz, new_coords[:coords_limit], k_r[:coords_limit])
        chg_6p = p_chg(4, exps_6p, coefs_6px, coefs_6py, coefs_6pz, new_coords[:coords_limit], k_r[:coords_limit])
        chg_3d = d_chg(0, exps_3d, coefs_3d1, coefs_3d2, coefs_3d3, coefs_3d4, coefs_3d5, new_coords[:coords_limit],
                       k_r[:coords_limit])
        chg_4d = d_chg(1, exps_4d, coefs_4d1, coefs_4d2, coefs_4d3, coefs_4d4, coefs_4d5, new_coords[:coords_limit],
                       k_r[:coords_limit])
        chg_5d = d_chg(2, exps_5d, coefs_5d1, coefs_5d2, coefs_5d3, coefs_5d4, coefs_5d5, new_coords[:coords_limit],
                       k_r[:coords_limit])
        chg_6d = d_chg(3, exps_6d, coefs_6d1, coefs_6d2, coefs_6d3, coefs_6d4, coefs_6d5, new_coords[:coords_limit],
                       k_r[:coords_limit])
        chg_4f = f_chg(0, exps_4f, coefs_4f1, coefs_4f2, coefs_4f3, coefs_4f4, coefs_4f5, coefs_4f6, coefs_4f7,
                       new_coords[:coords_limit], k_r[:coords_limit])
        chg_5f = f_chg(1, exps_5f, coefs_5f1, coefs_5f2, coefs_5f3, coefs_5f4, coefs_5f5, coefs_5f6, coefs_5f7,
                       new_coords[:coords_limit], k_r[:coords_limit])
        chg_5g = g_chg(0, exps_5g, coefs_5g1, coefs_5g2, coefs_5g3, coefs_5g4, coefs_5g5, coefs_5g6, coefs_5g7,
                       coefs_5g8, coefs_5g9, new_coords[:coords_limit], k_r[:coords_limit])
        lpart = 1
        for lpart in range(2, partsCHG):
            if tot_chg:
                if iden == 1:
                    c_coef_1s = np.array(
                        [116.578026855869, 112.510252969745, 41.165379246020, 56.773590524042, 86.402676630032,
                         63.577812378632, 0.637955734166, 44.707906728918, 71.946789359638, 105.619710421794,
                         38.996281296036, 88.894683580824, 9.202881074009])
                    c_exp_1s = np.array(
                        [324.387996761133, 177.399068268376, 50.940492722610, 4073.371026571095, 95.863341632517,
                         110803.674121156117, 14.516932727395, 8458.147167515093, 2075.892101482822, 593.471652826499,
                         20211.367729746886, 1097.306310034251, 27.454141193838])
                elif iden == 2:
                    c_coef_1s = np.array(
                        [162.741399115522, 1.000590152363, 120.717723038064, 65.800446492523, 176.972282692795,
                         70.279948577080, 97.856043902832, 127.188202750892, 15.158261318857, 78.581279593310,
                         147.872961853726, 121.986697165618, 169.430095505507])
                    c_exp_1s = np.array(
                        [224.998268113168, 20.137330366020, 113070.822303298963, 70.515523980315, 392.151336096174,
                         20996.404792315752, 4456.523128073991, 128.364544225048, 38.128008571196, 9020.825868609380,
                         1255.100280831314, 2325.453330503726, 694.286492554629])
                elif iden == 3:
                    c_coef_1s = np.array(
                        [3.894908509869, 142.249270911306, 50.657031837569, 126.375241189918, 218.051260068246,
                         192.172719533969, 0.408943286657, 115.196635211020, 210.000163764055, 242.782693209721,
                         250.696858077352, 121.091043858451, 8.165223094803, 44.126392088493, 216.928704406869,
                         114.664105986775])
                    c_exp_1s = np.array(
                        [2664.774694202190, 5147.982644234844, 2747.036781808482, 115.306011217834, 998.024205222061,
                         1664.421679207663, 22.251540530064, 2920.109248055150, 115613.602533650555, 600.292587465404,
                         355.059222702842, 9850.273163919423, 39.286718666097, 67.119899157727, 204.056327537959,
                         21979.010168625922])
                core_chg = np.append(core_chg, np.sum(
                    [s_chg(0, c_exp_1s, c_coef_1s, k_r[(coords_limit) * (lpart - 1):coords_limit * lpart])], axis=0))
            chg_1s = np.append(chg_1s,
                               s_chg(0, exps_1s, coefs_1s, k_r[(coords_limit) * (lpart - 1):coords_limit * lpart]))
            chg_2s = np.append(chg_2s,
                               s_chg(1, exps_2s, coefs_2s, k_r[(coords_limit) * (lpart - 1):coords_limit * lpart]))
            chg_3s = np.append(chg_3s,
                               s_chg(2, exps_3s, coefs_3s, k_r[(coords_limit) * (lpart - 1):coords_limit * lpart]))
            chg_4s = np.append(chg_4s,
                               s_chg(3, exps_4s, coefs_4s, k_r[(coords_limit) * (lpart - 1):coords_limit * lpart]))
            chg_5s = np.append(chg_5s,
                               s_chg(4, exps_5s, coefs_5s, k_r[(coords_limit) * (lpart - 1):coords_limit * lpart]))
            chg_6s = np.append(chg_6s,
                               s_chg(5, exps_6s, coefs_6s, k_r[(coords_limit) * (lpart - 1):coords_limit * lpart]))
            chg_7s = np.append(chg_7s,
                               s_chg(6, exps_7s, coefs_7s, k_r[(coords_limit) * (lpart - 1):coords_limit * lpart]))
            chg_2p = np.append(chg_2p, p_chg(0, exps_2p, coefs_2px, coefs_2py, coefs_2pz,
                                             new_coords[(coords_limit) * (lpart - 1):coords_limit * lpart],
                                             k_r[(coords_limit) * (lpart - 1):coords_limit * lpart]))
            chg_3p = np.append(chg_3p, p_chg(1, exps_3p, coefs_3px, coefs_3py, coefs_3pz,
                                             new_coords[(coords_limit) * (lpart - 1):coords_limit * lpart],
                                             k_r[(coords_limit) * (lpart - 1):coords_limit * lpart]))
            chg_4p = np.append(chg_4p, p_chg(2, exps_4p, coefs_4px, coefs_4py, coefs_4pz,
                                             new_coords[(coords_limit) * (lpart - 1):coords_limit * lpart],
                                             k_r[(coords_limit) * (lpart - 1):coords_limit * lpart]))
            chg_5p = np.append(chg_5p, p_chg(3, exps_5p, coefs_5px, coefs_5py, coefs_5pz,
                                             new_coords[(coords_limit) * (lpart - 1):coords_limit * lpart],
                                             k_r[(coords_limit) * (lpart - 1):coords_limit * lpart]))
            chg_6p = np.append(chg_6p, p_chg(4, exps_6p, coefs_6px, coefs_6py, coefs_6pz,
                                             new_coords[(coords_limit) * (lpart - 1):coords_limit * lpart],
                                             k_r[(coords_limit) * (lpart - 1):coords_limit * lpart]))
            chg_3d = np.append(chg_3d, d_chg(0, exps_3d, coefs_3d1, coefs_3d2, coefs_3d3, coefs_3d4, coefs_3d5,
                                             new_coords[(coords_limit) * (lpart - 1):coords_limit * lpart],
                                             k_r[(coords_limit) * (lpart - 1):coords_limit * lpart]))
            chg_4d = np.append(chg_4d, d_chg(1, exps_4d, coefs_4d1, coefs_4d2, coefs_4d3, coefs_4d4, coefs_4d5,
                                             new_coords[(coords_limit) * (lpart - 1):coords_limit * lpart],
                                             k_r[(coords_limit) * (lpart - 1):coords_limit * lpart]))
            chg_5d = np.append(chg_5d, d_chg(2, exps_5d, coefs_5d1, coefs_5d2, coefs_5d3, coefs_5d4, coefs_5d5,
                                             new_coords[(coords_limit) * (lpart - 1):coords_limit * lpart],
                                             k_r[(coords_limit) * (lpart - 1):coords_limit * lpart]))
            chg_6d = np.append(chg_6d, d_chg(3, exps_6d, coefs_6d1, coefs_6d2, coefs_6d3, coefs_6d4, coefs_6d5,
                                             new_coords[(coords_limit) * (lpart - 1):coords_limit * lpart],
                                             k_r[(coords_limit) * (lpart - 1):coords_limit * lpart]))
            chg_4f = np.append(chg_4f,
                               f_chg(0, exps_4f, coefs_4f1, coefs_4f2, coefs_4f3, coefs_4f4, coefs_4f5, coefs_4f6,
                                     coefs_4f7, new_coords[(coords_limit) * (lpart - 1):coords_limit * lpart],
                                     k_r[(coords_limit) * (lpart - 1):coords_limit * lpart]))
            chg_5f = np.append(chg_5f,
                               f_chg(1, exps_5f, coefs_5f1, coefs_5f2, coefs_5f3, coefs_5f4, coefs_5f5, coefs_5f6,
                                     coefs_5f7, new_coords[(coords_limit) * (lpart - 1):coords_limit * lpart],
                                     k_r[(coords_limit) * (lpart - 1):coords_limit * lpart]))
            chg_5g = np.append(chg_5g,
                               g_chg(0, exps_5g, coefs_5g1, coefs_5g2, coefs_5g3, coefs_5g4, coefs_5g5, coefs_5g6,
                                     coefs_5g7, coefs_5g8, coefs_5g9,
                                     new_coords[(coords_limit) * (lpart - 1):coords_limit * lpart],
                                     k_r[(coords_limit) * (lpart - 1):coords_limit * lpart]))

        if tot_chg:
            if iden == 1:
                c_coef_1s = np.array(
                    [116.578026855869, 112.510252969745, 41.165379246020, 56.773590524042, 86.402676630032,
                     63.577812378632, 0.637955734166, 44.707906728918, 71.946789359638, 105.619710421794,
                     38.996281296036, 88.894683580824, 9.202881074009])
                c_exp_1s = np.array(
                    [324.387996761133, 177.399068268376, 50.940492722610, 4073.371026571095, 95.863341632517,
                     110803.674121156117, 14.516932727395, 8458.147167515093, 2075.892101482822, 593.471652826499,
                     20211.367729746886, 1097.306310034251, 27.454141193838])
            elif iden == 2:
                c_coef_1s = np.array(
                    [162.741399115522, 1.000590152363, 120.717723038064, 65.800446492523, 176.972282692795,
                     70.279948577080, 97.856043902832, 127.188202750892, 15.158261318857, 78.581279593310,
                     147.872961853726, 121.986697165618, 169.430095505507])
                c_exp_1s = np.array(
                    [224.998268113168, 20.137330366020, 113070.822303298963, 70.515523980315, 392.151336096174,
                     20996.404792315752, 4456.523128073991, 128.364544225048, 38.128008571196, 9020.825868609380,
                     1255.100280831314, 2325.453330503726, 694.286492554629])
            elif iden == 3:
                c_coef_1s = np.array(
                    [3.894908509869, 142.249270911306, 50.657031837569, 126.375241189918, 218.051260068246,
                     192.172719533969, 0.408943286657, 115.196635211020, 210.000163764055, 242.782693209721,
                     250.696858077352, 121.091043858451, 8.165223094803, 44.126392088493, 216.928704406869,
                     114.664105986775])
                c_exp_1s = np.array(
                    [2664.774694202190, 5147.982644234844, 2747.036781808482, 115.306011217834, 998.024205222061,
                     1664.421679207663, 22.251540530064, 2920.109248055150, 115613.602533650555, 600.292587465404,
                     355.059222702842, 9850.273163919423, 39.286718666097, 67.119899157727, 204.056327537959,
                     21979.010168625922])
            core_chg = np.append(core_chg, np.sum([s_chg(0, c_exp_1s, c_coef_1s, k_r[coords_limit * lpart:])], axis=0))
        chg_1s = np.append(chg_1s, s_chg(0, exps_1s, coefs_1s, k_r[coords_limit * lpart:]))
        chg_2s = np.append(chg_2s, s_chg(1, exps_2s, coefs_2s, k_r[coords_limit * lpart:]))
        chg_3s = np.append(chg_3s, s_chg(2, exps_3s, coefs_3s, k_r[coords_limit * lpart:]))
        chg_4s = np.append(chg_4s, s_chg(3, exps_4s, coefs_4s, k_r[coords_limit * lpart:]))
        chg_5s = np.append(chg_5s, s_chg(4, exps_5s, coefs_5s, k_r[coords_limit * lpart:]))
        chg_6s = np.append(chg_6s, s_chg(5, exps_6s, coefs_6s, k_r[coords_limit * lpart:]))
        chg_7s = np.append(chg_7s, s_chg(6, exps_7s, coefs_7s, k_r[coords_limit * lpart:]))
        chg_2p = np.append(chg_2p, p_chg(0, exps_2p, coefs_2px, coefs_2py, coefs_2pz, new_coords[coords_limit * lpart:],
                                         k_r[coords_limit * lpart:]))
        chg_3p = np.append(chg_3p, p_chg(1, exps_3p, coefs_3px, coefs_3py, coefs_3pz, new_coords[coords_limit * lpart:],
                                         k_r[coords_limit * lpart:]))
        chg_4p = np.append(chg_4p, p_chg(2, exps_4p, coefs_4px, coefs_4py, coefs_4pz, new_coords[coords_limit * lpart:],
                                         k_r[coords_limit * lpart:]))
        chg_5p = np.append(chg_5p, p_chg(3, exps_5p, coefs_5px, coefs_5py, coefs_5pz, new_coords[coords_limit * lpart:],
                                         k_r[coords_limit * lpart:]))
        chg_6p = np.append(chg_6p, p_chg(4, exps_6p, coefs_6px, coefs_6py, coefs_6pz, new_coords[coords_limit * lpart:],
                                         k_r[coords_limit * lpart:]))
        chg_3d = np.append(chg_3d, d_chg(0, exps_3d, coefs_3d1, coefs_3d2, coefs_3d3, coefs_3d4, coefs_3d5,
                                         new_coords[coords_limit * lpart:], k_r[coords_limit * lpart:]))
        chg_4d = np.append(chg_4d, d_chg(1, exps_4d, coefs_4d1, coefs_4d2, coefs_4d3, coefs_4d4, coefs_4d5,
                                         new_coords[coords_limit * lpart:], k_r[coords_limit * lpart:]))
        chg_5d = np.append(chg_5d, d_chg(2, exps_5d, coefs_5d1, coefs_5d2, coefs_5d3, coefs_5d4, coefs_5d5,
                                         new_coords[coords_limit * lpart:], k_r[coords_limit * lpart:]))
        chg_6d = np.append(chg_6d, d_chg(3, exps_6d, coefs_6d1, coefs_6d2, coefs_6d3, coefs_6d4, coefs_6d5,
                                         new_coords[coords_limit * lpart:], k_r[coords_limit * lpart:]))
        chg_4f = np.append(chg_4f, f_chg(0, exps_4f, coefs_4f1, coefs_4f2, coefs_4f3, coefs_4f4, coefs_4f5, coefs_4f6,
                                         coefs_4f7, new_coords[coords_limit * lpart:], k_r[coords_limit * lpart:]))
        chg_5f = np.append(chg_5f, f_chg(1, exps_5f, coefs_5f1, coefs_5f2, coefs_5f3, coefs_5f4, coefs_5f5, coefs_5f6,
                                         coefs_5f7, new_coords[coords_limit * lpart:], k_r[coords_limit * lpart:]))
        chg_5g = np.append(chg_5g, g_chg(0, exps_5g, coefs_5g1, coefs_5g2, coefs_5g3, coefs_5g4, coefs_5g5, coefs_5g6,
                                         coefs_5g7, coefs_5g8, coefs_5g9, new_coords[coords_limit * lpart:],
                                         k_r[coords_limit * lpart:]))

    else:
        if tot_chg:
            if iden == 1:
                c_coef_1s = np.array(
                    [116.578026855869, 112.510252969745, 41.165379246020, 56.773590524042, 86.402676630032,
                     63.577812378632, 0.637955734166, 44.707906728918, 71.946789359638, 105.619710421794,
                     38.996281296036, 88.894683580824, 9.202881074009])
                c_exp_1s = np.array(
                    [324.387996761133, 177.399068268376, 50.940492722610, 4073.371026571095, 95.863341632517,
                     110803.674121156117, 14.516932727395, 8458.147167515093, 2075.892101482822, 593.471652826499,
                     20211.367729746886, 1097.306310034251, 27.454141193838])
            elif iden == 2:
                c_coef_1s = np.array(
                    [162.741399115522, 1.000590152363, 120.717723038064, 65.800446492523, 176.972282692795,
                     70.279948577080, 97.856043902832, 127.188202750892, 15.158261318857, 78.581279593310,
                     147.872961853726, 121.986697165618, 169.430095505507])
                c_exp_1s = np.array(
                    [224.998268113168, 20.137330366020, 113070.822303298963, 70.515523980315, 392.151336096174,
                     20996.404792315752, 4456.523128073991, 128.364544225048, 38.128008571196, 9020.825868609380,
                     1255.100280831314, 2325.453330503726, 694.286492554629])
            elif iden == 3:
                c_coef_1s = np.array(
                    [3.894908509869, 142.249270911306, 50.657031837569, 126.375241189918, 218.051260068246,
                     192.172719533969, 0.408943286657, 115.196635211020, 210.000163764055, 242.782693209721,
                     250.696858077352, 121.091043858451, 8.165223094803, 44.126392088493, 216.928704406869,
                     114.664105986775])
                c_exp_1s = np.array(
                    [2664.774694202190, 5147.982644234844, 2747.036781808482, 115.306011217834, 998.024205222061,
                     1664.421679207663, 22.251540530064, 2920.109248055150, 115613.602533650555, 600.292587465404,
                     355.059222702842, 9850.273163919423, 39.286718666097, 67.119899157727, 204.056327537959,
                     21979.010168625922])
            core_chg = np.sum([s_chg(0, c_exp_1s, c_coef_1s, k_r)], axis=0)
        chg_1s = s_chg(0, exps_1s, coefs_1s, k_r)
        chg_2s = s_chg(1, exps_2s, coefs_2s, k_r)
        chg_3s = s_chg(2, exps_3s, coefs_3s, k_r)
        chg_4s = s_chg(3, exps_4s, coefs_4s, k_r)
        chg_5s = s_chg(4, exps_5s, coefs_5s, k_r)
        chg_6s = s_chg(5, exps_6s, coefs_6s, k_r)
        chg_7s = s_chg(6, exps_7s, coefs_7s, k_r)
        chg_2p = p_chg(0, exps_2p, coefs_2px, coefs_2py, coefs_2pz, new_coords, k_r)
        chg_3p = p_chg(1, exps_3p, coefs_3px, coefs_3py, coefs_3pz, new_coords, k_r)
        chg_4p = p_chg(2, exps_4p, coefs_4px, coefs_4py, coefs_4pz, new_coords, k_r)
        chg_5p = p_chg(3, exps_5p, coefs_5px, coefs_5py, coefs_5pz, new_coords, k_r)
        chg_6p = p_chg(4, exps_6p, coefs_6px, coefs_6py, coefs_6pz, new_coords, k_r)
        chg_3d = d_chg(0, exps_3d, coefs_3d1, coefs_3d2, coefs_3d3, coefs_3d4, coefs_3d5, new_coords, k_r)
        chg_4d = d_chg(1, exps_4d, coefs_4d1, coefs_4d2, coefs_4d3, coefs_4d4, coefs_4d5, new_coords, k_r)
        chg_5d = d_chg(2, exps_5d, coefs_5d1, coefs_5d2, coefs_5d3, coefs_5d4, coefs_5d5, new_coords, k_r)
        chg_6d = d_chg(3, exps_6d, coefs_6d1, coefs_6d2, coefs_6d3, coefs_6d4, coefs_6d5, new_coords, k_r)
        chg_4f = f_chg(0, exps_4f, coefs_4f1, coefs_4f2, coefs_4f3, coefs_4f4, coefs_4f5, coefs_4f6, coefs_4f7,
                       new_coords, k_r)
        chg_5f = f_chg(1, exps_5f, coefs_5f1, coefs_5f2, coefs_5f3, coefs_5f4, coefs_5f5, coefs_5f6, coefs_5f7,
                       new_coords, k_r)
        chg_5g = g_chg(0, exps_5g, coefs_5g1, coefs_5g2, coefs_5g3, coefs_5g4, coefs_5g5, coefs_5g6, coefs_5g7,
                       coefs_5g8, coefs_5g9, new_coords, k_r)

    tot_s = np.sum([chg_1s, chg_2s], axis=0)
    tot_s = np.sum([tot_s, chg_3s], axis=0)
    tot_s = np.sum([tot_s, chg_4s], axis=0)
    tot_s = np.sum([tot_s, chg_5s], axis=0)
    tot_s = np.sum([tot_s, chg_6s], axis=0)
    tot_s = np.sum([tot_s, chg_7s], axis=0)
    if count == 0:
        tot_C = tot_s
    else:
        tot_C = tot_C + tot_s

    del tot_s
    tot_p = np.sum([chg_2p, chg_3p], axis=0)
    tot_p = np.sum([tot_p, chg_4p], axis=0)
    tot_p = np.sum([tot_p, chg_5p], axis=0)
    tot_p = np.sum([tot_p, chg_6p], axis=0)
    tot_C = tot_C + tot_p
    del tot_p
    tot_d = np.sum([chg_3d, chg_4d], axis=0)
    tot_d = np.sum([tot_d, chg_5d], axis=0)
    tot_d = np.sum([tot_d, chg_6d], axis=0)
    tot_C = tot_C + tot_d
    del tot_d
    tot_f = np.sum([chg_4f, chg_5f], axis=0)
    tot_C = tot_C + tot_f
    del tot_f
    tot_C = tot_C + chg_5g
    del chg_1s, chg_2s, chg_3s, chg_4s, chg_5s, chg_6s, chg_7s, chg_2p, chg_3p, chg_4p, chg_5p, chg_6p, chg_3d, chg_4d, chg_5d, chg_6d, chg_4f, chg_5f, chg_5g
    if tot_chg:
        tot_C = tot_C + core_chg
    return tot_C


def H_chg_print(x, cutoff_distance, coefs, sites_elem, poscar_data, chg_coor, dim, vol, tot_H, count, jj):
    pos = x.coords
    pos_frac = x.frac_coords
    neighs_list = list(itertools.chain.from_iterable(poscar_data.structure.get_neighbors(x, cutoff_distance)))
    group_lst = [neighs_list[i:i + 4] for i in range(0, len(neighs_list), 4)]
    sorted_list = sorted(group_lst, key=itemgetter(1))
    v1 = sorted_list[0][0].coords - pos
    v2 = sorted_list[1][0].coords - pos
    u3 = np.cross(v1, v2)
    u2 = np.cross(v1, u3)
    u1 = v1 / np.linalg.norm(v1)
    u2 = u2 / np.linalg.norm(u2)
    u3 = u3 / np.linalg.norm(u3)

    matrx = np.transpose(np.array([u1, u2, u3]))

    chg_atoms = chg_coor - np.tile(pos_frac, (chg_coor.shape[0], 1))
    chg_atoms[chg_atoms > 0.5] -= 1.0
    chg_atoms[chg_atoms < -0.5] += 1.0

    chg_atoms = np.matrix(chg_atoms)

    dim = np.matrix(dim)

    matrx = np.matrix(matrx)
    new_coords = chg_atoms * dim * matrx
    squ_coords = np.square(new_coords)
    kr = np.sum(squ_coords, axis=1)
    k_r = np.sqrt(kr)
    exps_1s, exps_2s, exps_3s, exps_4s, exps_2p, exps_3p, exps_4p, exps_3d, exps_4d, exps_5d, exps_4f, exps_5f, exps_5g, coefs_1s, coefs_2s, coefs_3s, coefs_4s, coefs_2px, coefs_2py, coefs_2pz, coefs_3px, coefs_3py, coefs_3pz, coefs_4px, coefs_4py, coefs_4pz, coefs_3d1, coefs_3d2, coefs_3d3, coefs_3d4, coefs_3d5, coefs_4d1, coefs_4d2, coefs_4d3, coefs_4d4, coefs_4d5, coefs_5d1, coefs_5d2, coefs_5d3, coefs_5d4, coefs_5d5, coefs_4f1, coefs_4f2, coefs_4f3, coefs_4f4, coefs_4f5, coefs_4f6, coefs_4f7, coefs_5f1, coefs_5f2, coefs_5f3, coefs_5f4, coefs_5f5, coefs_5f6, coefs_5f7, coefs_5g1, coefs_5g2, coefs_5g3, coefs_5g4, coefs_5g5, coefs_5g6, coefs_5g7, coefs_5g8, coefs_5g9 = H_coef_split(
        coefs[0, jj, :])
    coords_limit = 1000000
    coords_shape = kr.shape[0]
    partsCHG = math.ceil(coords_shape / coords_limit)
    if coords_shape > coords_limit:
        chg_1s = s_chg(0, exps_1s, coefs_1s, k_r[:coords_limit])
        chg_2s = s_chg(1, exps_2s, coefs_2s, k_r[:coords_limit])
        chg_3s = s_chg(2, exps_3s, coefs_3s, k_r[:coords_limit])
        chg_4s = s_chg(3, exps_4s, coefs_4s, k_r[:coords_limit])
        chg_2p = p_chg(0, exps_2p, coefs_2px, coefs_2py, coefs_2pz, new_coords[:coords_limit], k_r[:coords_limit])
        chg_3p = p_chg(1, exps_3p, coefs_3px, coefs_3py, coefs_3pz, new_coords[:coords_limit], k_r[:coords_limit])
        chg_4p = p_chg(2, exps_4p, coefs_4px, coefs_4py, coefs_4pz, new_coords[:coords_limit], k_r[:coords_limit])
        chg_3d = d_chg(0, exps_3d, coefs_3d1, coefs_3d2, coefs_3d3, coefs_3d4, coefs_3d5, new_coords[:coords_limit],
                       k_r[:coords_limit])
        chg_4d = d_chg(1, exps_4d, coefs_4d1, coefs_4d2, coefs_4d3, coefs_4d4, coefs_4d5, new_coords[:coords_limit],
                       k_r[:coords_limit])
        chg_5d = d_chg(2, exps_5d, coefs_5d1, coefs_5d2, coefs_5d3, coefs_5d4, coefs_5d5, new_coords[:coords_limit],
                       k_r[:coords_limit])
        chg_4f = f_chg(0, exps_4f, coefs_4f1, coefs_4f2, coefs_4f3, coefs_4f4, coefs_4f5, coefs_4f6, coefs_4f7,
                       new_coords[:coords_limit], k_r[:coords_limit])
        chg_5f = f_chg(1, exps_5f, coefs_5f1, coefs_5f2, coefs_5f3, coefs_5f4, coefs_5f5, coefs_5f6, coefs_5f7,
                       new_coords[:coords_limit], k_r[:coords_limit])
        chg_5g = g_chg(0, exps_5g, coefs_5g1, coefs_5g2, coefs_5g3, coefs_5g4, coefs_5g5, coefs_5g6, coefs_5g7,
                       coefs_5g8, coefs_5g9, new_coords[:coords_limit], k_r[:coords_limit])
        lpart = 1
        for lpart in range(2, partsCHG):
            chg_1s = np.append(chg_1s,
                               s_chg(0, exps_1s, coefs_1s, k_r[(coords_limit) * (lpart - 1):coords_limit * lpart]))
            chg_2s = np.append(chg_2s,
                               s_chg(1, exps_2s, coefs_2s, k_r[(coords_limit) * (lpart - 1):coords_limit * lpart]))
            chg_3s = np.append(chg_3s,
                               s_chg(2, exps_3s, coefs_3s, k_r[(coords_limit) * (lpart - 1):coords_limit * lpart]))
            chg_4s = np.append(chg_4s,
                               s_chg(3, exps_4s, coefs_4s, k_r[(coords_limit) * (lpart - 1):coords_limit * lpart]))
            chg_2p = np.append(chg_2p, p_chg(0, exps_2p, coefs_2px, coefs_2py, coefs_2pz,
                                             new_coords[(coords_limit) * (lpart - 1):coords_limit * lpart],
                                             k_r[(coords_limit) * (lpart - 1):coords_limit * lpart]))
            chg_3p = np.append(chg_3p, p_chg(1, exps_3p, coefs_3px, coefs_3py, coefs_3pz,
                                             new_coords[(coords_limit) * (lpart - 1):coords_limit * lpart],
                                             k_r[(coords_limit) * (lpart - 1):coords_limit * lpart]))
            chg_4p = np.append(chg_4p, p_chg(2, exps_4p, coefs_4px, coefs_4py, coefs_4pz,
                                             new_coords[(coords_limit) * (lpart - 1):coords_limit * lpart],
                                             k_r[(coords_limit) * (lpart - 1):coords_limit * lpart]))
            chg_3d = np.append(chg_3d, d_chg(0, exps_3d, coefs_3d1, coefs_3d2, coefs_3d3, coefs_3d4, coefs_3d5,
                                             new_coords[(coords_limit) * (lpart - 1):coords_limit * lpart],
                                             k_r[(coords_limit) * (lpart - 1):coords_limit * lpart]))
            chg_4d = np.append(chg_4d, d_chg(1, exps_4d, coefs_4d1, coefs_4d2, coefs_4d3, coefs_4d4, coefs_4d5,
                                             new_coords[(coords_limit) * (lpart - 1):coords_limit * lpart],
                                             k_r[(coords_limit) * (lpart - 1):coords_limit * lpart]))
            chg_5d = np.append(chg_5d, d_chg(2, exps_5d, coefs_5d1, coefs_5d2, coefs_5d3, coefs_5d4, coefs_5d5,
                                             new_coords[(coords_limit) * (lpart - 1):coords_limit * lpart],
                                             k_r[(coords_limit) * (lpart - 1):coords_limit * lpart]))
            chg_4f = np.append(chg_4f,
                               f_chg(0, exps_4f, coefs_4f1, coefs_4f2, coefs_4f3, coefs_4f4, coefs_4f5, coefs_4f6,
                                     coefs_4f7, new_coords[(coords_limit) * (lpart - 1):coords_limit * lpart],
                                     k_r[(coords_limit) * (lpart - 1):coords_limit * lpart]))
            chg_5f = np.append(chg_5f,
                               f_chg(1, exps_5f, coefs_5f1, coefs_5f2, coefs_5f3, coefs_5f4, coefs_5f5, coefs_5f6,
                                     coefs_5f7, new_coords[(coords_limit) * (lpart - 1):coords_limit * lpart],
                                     k_r[(coords_limit) * (lpart - 1):coords_limit * lpart]))
            chg_5g = np.append(chg_5g,
                               g_chg(0, exps_5g, coefs_5g1, coefs_5g2, coefs_5g3, coefs_5g4, coefs_5g5, coefs_5g6,
                                     coefs_5g7, coefs_5g8, coefs_5g9,
                                     new_coords[(coords_limit) * (lpart - 1):coords_limit * lpart],
                                     k_r[(coords_limit) * (lpart - 1):coords_limit * lpart]))
        chg_1s = np.append(chg_1s, s_chg(0, exps_1s, coefs_1s, k_r[coords_limit * lpart:]))
        chg_2s = np.append(chg_2s, s_chg(1, exps_2s, coefs_2s, k_r[coords_limit * lpart:]))
        chg_3s = np.append(chg_3s, s_chg(2, exps_3s, coefs_3s, k_r[coords_limit * lpart:]))
        chg_4s = np.append(chg_4s, s_chg(3, exps_4s, coefs_4s, k_r[coords_limit * lpart:]))
        chg_2p = np.append(chg_2p, p_chg(0, exps_2p, coefs_2px, coefs_2py, coefs_2pz, new_coords[coords_limit * lpart:],
                                         k_r[coords_limit * lpart:]))
        chg_3p = np.append(chg_3p, p_chg(1, exps_3p, coefs_3px, coefs_3py, coefs_3pz, new_coords[coords_limit * lpart:],
                                         k_r[coords_limit * lpart:]))
        chg_4p = np.append(chg_4p, p_chg(2, exps_4p, coefs_4px, coefs_4py, coefs_4pz, new_coords[coords_limit * lpart:],
                                         k_r[coords_limit * lpart:]))
        chg_3d = np.append(chg_3d, d_chg(0, exps_3d, coefs_3d1, coefs_3d2, coefs_3d3, coefs_3d4, coefs_3d5,
                                         new_coords[coords_limit * lpart:], k_r[coords_limit * lpart:]))
        chg_4d = np.append(chg_4d, d_chg(1, exps_4d, coefs_4d1, coefs_4d2, coefs_4d3, coefs_4d4, coefs_4d5,
                                         new_coords[coords_limit * lpart:], k_r[coords_limit * lpart:]))
        chg_5d = np.append(chg_5d, d_chg(2, exps_5d, coefs_5d1, coefs_5d2, coefs_5d3, coefs_5d4, coefs_5d5,
                                         new_coords[coords_limit * lpart:], k_r[coords_limit * lpart:]))
        chg_4f = np.append(chg_4f, f_chg(0, exps_4f, coefs_4f1, coefs_4f2, coefs_4f3, coefs_4f4, coefs_4f5, coefs_4f6,
                                         coefs_4f7, new_coords[coords_limit * lpart:], k_r[coords_limit * lpart:]))
        chg_5f = np.append(chg_5f, f_chg(1, exps_5f, coefs_5f1, coefs_5f2, coefs_5f3, coefs_5f4, coefs_5f5, coefs_5f6,
                                         coefs_5f7, new_coords[coords_limit * lpart:], k_r[coords_limit * lpart:]))
        chg_5g = np.append(chg_5g, g_chg(0, exps_5g, coefs_5g1, coefs_5g2, coefs_5g3, coefs_5g4, coefs_5g5, coefs_5g6,
                                         coefs_5g7, coefs_5g8, coefs_5g9, new_coords[coords_limit * lpart:],
                                         k_r[coords_limit * lpart:]))
    else:
        chg_1s = s_chg(0, exps_1s, coefs_1s, k_r)
        chg_2s = s_chg(1, exps_2s, coefs_2s, k_r)
        chg_3s = s_chg(2, exps_3s, coefs_3s, k_r)
        chg_4s = s_chg(3, exps_4s, coefs_4s, k_r)
        chg_2p = p_chg(0, exps_2p, coefs_2px, coefs_2py, coefs_2pz, new_coords, k_r)
        chg_3p = p_chg(1, exps_3p, coefs_3px, coefs_3py, coefs_3pz, new_coords, k_r)
        chg_4p = p_chg(2, exps_4p, coefs_4px, coefs_4py, coefs_4pz, new_coords, k_r)
        chg_3d = d_chg(0, exps_3d, coefs_3d1, coefs_3d2, coefs_3d3, coefs_3d4, coefs_3d5, new_coords, k_r)
        chg_4d = d_chg(1, exps_4d, coefs_4d1, coefs_4d2, coefs_4d3, coefs_4d4, coefs_4d5, new_coords, k_r)
        chg_5d = d_chg(2, exps_5d, coefs_5d1, coefs_5d2, coefs_5d3, coefs_5d4, coefs_5d5, new_coords, k_r)
        chg_4f = f_chg(0, exps_4f, coefs_4f1, coefs_4f2, coefs_4f3, coefs_4f4, coefs_4f5, coefs_4f6, coefs_4f7,
                       new_coords, k_r)
        chg_5f = f_chg(1, exps_5f, coefs_5f1, coefs_5f2, coefs_5f3, coefs_5f4, coefs_5f5, coefs_5f6, coefs_5f7,
                       new_coords, k_r)
        chg_5g = g_chg(0, exps_5g, coefs_5g1, coefs_5g2, coefs_5g3, coefs_5g4, coefs_5g5, coefs_5g6, coefs_5g7,
                       coefs_5g8, coefs_5g9, new_coords, k_r)
    tot_s = np.sum([chg_1s, chg_2s], axis=0)
    tot_s = np.sum([tot_s, chg_3s], axis=0)
    tot_s = np.sum([tot_s, chg_4s], axis=0)
    if count == 0:
        tot_H = tot_s
    else:
        tot_H = tot_H + tot_s
    del tot_s
    tot_p = np.sum([chg_2p, chg_3p], axis=0)
    tot_p = np.sum([tot_p, chg_4p], axis=0)
    tot_H = tot_H + tot_p
    del tot_p
    tot_d = np.sum([chg_3d, chg_4d], axis=0)
    tot_d = np.sum([tot_d, chg_5d], axis=0)
    tot_H = tot_H + tot_d
    del tot_d
    tot_f = np.sum([chg_4f, chg_5f], axis=0)
    tot_H = tot_H + tot_f
    del tot_f
    tot_H = tot_H + chg_5g
    del chg_1s, chg_2s, chg_3s, chg_4s, chg_2p, chg_3p, chg_4p, chg_3d, chg_4d, chg_5d, chg_4f, chg_5f, chg_5g
    return tot_H


def chg_pred_data(poscar_data, at_elem, sites_elem, Coef_at1, Coef_at2, Coef_at3, Coef_at4, chg_coor, dim, vol,
                  tot_chg):
    tot_C = []
    tot_H = []
    tot_N = []
    tot_O = []
    cutoff_distance = 5.0
    for pp in range(0, 2):
        if pp == 0:
            C_at_charge = []
            coefs = Coef_at1
            jj = 0
            count = 0
            iden = 1
            for x in sites_elem[pp]:
                tot_C = C_chg_print(x, cutoff_distance, coefs, sites_elem, poscar_data, chg_coor, dim, vol, tot_C,
                                    count, jj, iden, tot_chg)
                #                if tot_chg:
                #                    core_chg=C_core()
                jj = jj + 1
                count = count + 1
        else:
            coefs = Coef_at2
            jj = 0
            count = 0
            H_at_charge = []
            for x in sites_elem[pp]:
                tot_H = H_chg_print(x, cutoff_distance, coefs, sites_elem, poscar_data, chg_coor, dim, vol, tot_H,
                                    count, jj)
                jj = jj + 1
                count = count + 1

    Pred_chg = tot_C + tot_H
    del tot_C, tot_H

    if at_elem[2] != 0:
        coefs = Coef_at3
        jj = 0
        count = 0
        iden = 2
        N_at_charge = []
        for x in sites_elem[2]:
            tot_N = C_chg_print(x, cutoff_distance, coefs, sites_elem, poscar_data, chg_coor, dim, vol, tot_N, count,
                                jj, iden, tot_chg)
            jj = jj + 1
            count = count + 1

        Pred_chg = Pred_chg + tot_N

        del tot_N

    if at_elem[3] != 0:
        coefs = Coef_at4
        jj = 0
        count = 0
        iden = 3
        O_at_charge = []
        for x in sites_elem[3]:
            tot_O = C_chg_print(x, cutoff_distance, coefs, sites_elem, poscar_data, chg_coor, dim, vol, tot_O, count,
                                jj, iden, tot_chg)
            jj = jj + 1
            count = count + 1

        Pred_chg = Pred_chg + tot_O

        del tot_O

    small_cell = vol / (Pred_chg.shape[0])

    pred_charge = np.sum(Pred_chg)
    print('Total pred_charge:', pred_charge * small_cell)
    return Pred_chg


def chg_print(Pred_chg, vol, localfile_loc, num_pts,out_dir):
    Pred_chg_row = np.array(Pred_chg * vol)
    h = math.ceil(Pred_chg.shape[0] / 5)
    mm = h * 5 - len(Pred_chg)
    Pred_chg_pad = np.pad(Pred_chg_row, (0, mm), 'constant')
    Pred_chg = np.reshape(Pred_chg_pad, (h, 5))
    bb = ' '
    with open(os.path.join(out_dir, "Pred_CHG_test" + localfile_loc + ".dat"), "a") as file_object:
        file_object.write('\n')
        file_object.write(str(num_pts[0]) + 10 * bb + str(num_pts[1]) + 10 * bb + str(num_pts[2]) + '\n')
        for i in range(0, h):
            file_object.write(
                str(Pred_chg[i][0]) + 3 * bb + str(Pred_chg[i][1]) + 3 * bb + str(Pred_chg[i][2]) + 3 * bb + str(
                    Pred_chg[i][3]) + 3 * bb + str(Pred_chg[i][4]) + '\n')



