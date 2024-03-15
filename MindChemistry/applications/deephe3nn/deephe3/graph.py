# Copyright 2024 Huawei Technologies Co., Ltd
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
graph
"""

# pylint: disable=W0123

import os
import json
import warnings
import numpy as np
import h5py

from .utils import flt2cplx, convert2numpyt


def load_orbital_types(path, return_orbital_types=False):
    """
    load_orbital_types of cell
    """
    orbital_types = []
    with open(path) as f:
        line = f.readline()
        while line:
            orbital_types.append(list(map(int, line.split())))
            line = f.readline()
    atom_num_orbital = [sum(map(lambda x: 2 * x + 1, atom_orbital_types)) for atom_orbital_types in orbital_types]
    if return_orbital_types:
        return atom_num_orbital, orbital_types
    return atom_num_orbital


def is_ij(edge_key):
    r''' determine whether the edge belongs to ij kind '''
    if isinstance(edge_key, str):
        edge_key = eval(edge_key)
    out = True
    is_same_cell = True
    for x in edge_key[0:3]:
        if x == 0:
            pass
        else:
            out = x > 0
            is_same_cell = False
            break
    if is_same_cell:
        if edge_key[3] > edge_key[4]:
            out = False
    return out


def convert_ijji(edge_key):
    r''' convert edge key between ij and ji '''
    if isinstance(edge_key, str):
        edge_key = eval(edge_key)
    out = [-edge_key[0], -edge_key[1], -edge_key[2], edge_key[4], edge_key[3]]
    return out


def get_edge_fea(cart_coords, lattice, default_dtype_np, edge_key):
    """
    get edge feature
    """
    cart_coords_i = cart_coords[edge_key[:, 3] - 1]
    cart_coords_j = cart_coords[edge_key[:, 4] - 1] + edge_key[:, :3].astype(convert2numpyt(default_dtype_np)) @ lattice
    dist = np.linalg.norm(cart_coords_j - cart_coords_i, axis=-1)
    edge_fea = np.concatenate([dist[:, None], (cart_coords_j - cart_coords_i)], axis=-1)

    return edge_fea.astype(np.float32)


def get_graph(cart_coords,
              numbers,
              stru_id,
              edge_aij,
              lattice,
              default_dtype_np,
              data_folder,
              target_file_name='hamiltonian',
              inference=False,
              only_ij=False,
              create_from_dft=True):
    """
    get graph process
    """
    if not isinstance(cart_coords, np.ndarray):
        cart_coords = np.array(cart_coords)

    if not isinstance(numbers, np.ndarray):
        numbers = np.array(numbers)

    if not isinstance(lattice, np.ndarray):
        lattice = np.array(lattice, dtype=np.float32)
    else:
        lattice = lattice.astype(np.float32)

    if data_folder is not None:
        atom_num_orbital = load_orbital_types(os.path.join(data_folder, 'orbital_types.dat'))

        if os.path.isfile(os.path.join(data_folder, 'info.json')):
            with open(os.path.join(data_folder, 'info.json'), 'r') as info_f:
                info_dict = json.load(info_f)
                spinful = info_dict["isspinful"]

        spinful = False

        if not inference:
            aij_dict = {}
            fid = h5py.File(os.path.join(data_folder, target_file_name), 'r')
            for k, v in fid.items():
                key = json.loads(k)
                if only_ij:
                    if not is_ij(key):
                        continue
                key = (key[0], key[1], key[2], key[3] - 1, key[4] - 1)
                v = np.array(v)
                if spinful:
                    aij_dict[key] = np.array(v, dtype=flt2cplx(default_dtype_np))
                else:
                    aij_dict[key] = np.array(v, dtype=convert2numpyt(default_dtype_np))
            fid.close()
        max_num_orbital = max(atom_num_orbital)

        ###create_from_dft True
    if create_from_dft:
        key_atom_list = [[] for _ in range(len(numbers))]
        edge_idx, edge_idx_first, key_list = [], [], []
        fid = h5py.File(os.path.join(data_folder, target_file_name), 'r')
        for k in fid.keys():
            key = eval(k)
            if only_ij:
                if not is_ij(key):
                    continue
            key_array = np.array(key)
            key_atom_list[key[3] - 1].append(key_array)
        fid.close()

        for index_first, (_, keys_tensor) in enumerate(zip(cart_coords, key_atom_list)):
            keys_array = np.stack(keys_tensor)
            len_nn = keys_array.shape[0]
            edge_idx_first.extend([index_first] * len_nn)
            edge_idx.extend((keys_array[:, 4] - 1).tolist())
            key_list.append(keys_array)

        edge_idx = np.stack([np.array(edge_idx_first, dtype="int64"), np.array(edge_idx, dtype="int64")])

        edge_key = np.concatenate(key_list)

        edge_fea = get_edge_fea(cart_coords, lattice, default_dtype_np, edge_key)

    if data_folder is not None:
        if inference:
            pass
        else:
            if edge_aij:
                if edge_fea.shape[0] < 0.9 * len(aij_dict):
                    warnings.warn("Too many aijs are not included within the radius")
                aij_mask = np.zeros(edge_fea.shape[0], dtype=np.bool_)
                new_type = convert2numpyt(default_dtype_np)
                aij = np.full([edge_fea.shape[0], max_num_orbital, max_num_orbital], np.nan, dtype=new_type)

                for index in range(edge_fea.shape[0]):
                    i, j = edge_idx[:, index]
                    key = edge_key[index].tolist()
                    key[3] -= 1  # h_{i0, jR} i and j is 0-based index
                    key[4] -= 1
                    key = tuple(key)
                    if key in aij_dict:
                        aij_mask[index] = True
                        aij[index, :atom_num_orbital[i], :atom_num_orbital[j]] = aij_dict[key]
                    else:
                        raise NotImplementedError(
                            "Not yet have support for graph radius including hopping without calculation")

                data = [
                    numbers, edge_idx, edge_fea, stru_id,
                    cart_coords.astype(convert2numpyt(default_dtype_np)),
                    np.expand_dims(lattice, 0), edge_key, aij, aij_mask,
                    np.array(atom_num_orbital), spinful
                ]

    return data
