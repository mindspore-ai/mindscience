# Copyright 2021-2022 @ Shenzhen Bay Laboratory &
#                       Peking University &
#                       Huawei Technologies Co., Ltd
#

# This code is a part of MindSPONGE:
# MindSpore Simulation Package tOwards Next Generation molecular modelling.
#
# MindSPONGE is open-source software based on the AI-framework:
# MindSpore (https://www.mindspore.cn/)
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
"""
Fill in water molecules in a box.
"""
import os
import numpy as np
from numpy import ndarray
from .hadder import read_pdb
from .pdb_generator import gen_pdb
from ...function import GLOBAL_UNITS, length_convert

# mol/L
_DENSITY = 55
# NW/A^3
DENSITY = _DENSITY * 6.022E23 / 1e03 ** 9
_AVGDIS = DENSITY ** (-1 / 3)
AVGDIS = _AVGDIS * 1.


def fill_water(pdb_in: str, pdb_out: str, gap: float = 4.0, box: ndarray = None, rebuild_hydrogen: bool = False,
               return_pdb_obj: bool = False, adaptive_length: float = 5.0):
    """ The function to fill in water.
    Args:
        pdb_in(str): The input molecule file.
        pdb_out(str): The output molecule file.
        gap(float): The minimum gap between water molecule and system molecule.
        box(np.float32): The box to fill water molecules.
        rebuild_hydrogen(bool): Decide to rebuild the hydrogen atoms in the molecule or not.
        return_pdb_obj(bool): If this option is on, the returned results would be a pdb object.
        adaptive_length(float): The water molecule width to add.
    """
    if box is None and adaptive_length is None:
        raise ValueError("Please input a box or adaptive_length for filling water molecules.")

    if os.path.exists(pdb_out):
        os.remove(pdb_out)

    pdb_obj = read_pdb(pdb_in, rebuild_hydrogen=rebuild_hydrogen, rebuild_suffix='_addH')
    atom_names = pdb_obj.flatten_atoms
    res_names = pdb_obj.init_res_names
    res_ids = pdb_obj.init_res_ids
    crds = pdb_obj.flatten_crds

    min_x = crds[:, 0].min()
    min_y = crds[:, 1].min()
    min_z = crds[:, 2].min()

    max_x = crds[:, 0].max()
    max_y = crds[:, 1].max()
    max_z = crds[:, 2].max()

    size_x = max_x - min_x
    size_y = max_y - min_y
    size_z = max_z - min_z

    if box is None:
        box = np.array([size_x + 2 * adaptive_length,
                        size_y + 2 * adaptive_length,
                        size_z + 2 * adaptive_length], np.float32)
    final_box = box + 0.5 * AVGDIS
    print('[MindSPONGE] The box size used when filling water is: {}'.format(final_box))

    if box[0] < size_x:
        raise ValueError("Please given a larger box in the X axis.")
    if box[1] < size_y:
        raise ValueError("Please given a larger box in the Y axis.")
    if box[2] < size_z:
        raise ValueError("Please given a larger box in the Z axis.")

    origin_x = min_x - adaptive_length * 0.9
    origin_y = min_y - adaptive_length * 0.9
    origin_z = min_z - adaptive_length * 0.9

    num_waters = np.ceil(box / AVGDIS)
    total_waters = np.product(num_waters)
    print('[MindSPONGE] The edge gap along x axis is {}.'.format((box[0] - size_x) / 2))
    print('[MindSPONGE] The edge gap along y axis is {}.'.format((box[1] - size_y) / 2))
    print('[MindSPONGE] The edge gap along z axis is {}.'.format((box[2] - size_z) / 2))

    o_x = origin_x + (np.arange(total_waters) % num_waters[0]) * AVGDIS
    o_y = origin_y + ((np.arange(total_waters) // num_waters[0]) % num_waters[1]) * AVGDIS
    o_z = origin_z + ((np.arange(total_waters) // np.product(num_waters[:2])) % num_waters[2]) * AVGDIS

    o_crd = np.concatenate((o_x[:, None], o_y[:, None], o_z[:, None]), axis=-1)

    dis = np.linalg.norm(crds - o_crd[:, None, :], axis=-1)
    filt = np.where((dis <= gap).sum(axis=-1) > 0)

    o_crd = np.delete(o_crd, filt, axis=0)
    print('[MindSPONGE] Totally {} waters is added to the system!'.format(int(o_crd.shape[0])))
    h1_crd = o_crd + np.array([0.079079641, 0.061207927, 0.0], np.float32) * 10
    h2_crd = o_crd + np.array([-0.079079641, 0.061207927, 0.0], np.float32) * 10
    water_crd = np.hstack((o_crd, h1_crd, h2_crd)).reshape((-1, 3))
    water_names = np.array(['O', 'H1', 'H2'] * o_crd.shape[0], np.str_)
    water_res = np.array(['WAT'] * water_crd.shape[0], np.str_)
    water_resid = np.concatenate((np.arange(o_crd.shape[0])[:, None],
                                  np.arange(o_crd.shape[0])[:, None],
                                  np.arange(o_crd.shape[0])[:, None]), axis=-1).reshape((-1))

    new_crd = np.vstack((crds, water_crd))
    atom_names = np.concatenate((atom_names, water_names))
    res_names = np.concatenate((res_names, water_res))
    res_ids = np.concatenate((res_ids, max(res_ids) + 1 + water_resid))

    gen_pdb(new_crd[None, :], atom_names, res_names, res_ids, pdb_name=pdb_out)
    print('[MindSPONGE] Adding water molecule task finished!')
    if return_pdb_obj:
        return read_pdb(pdb_out)

    global GLOBAL_UNITS
    length_unit = GLOBAL_UNITS.length_unit
    return final_box * length_convert('A', length_unit)
