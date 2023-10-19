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
Read information from a pdb format file.
"""
from collections import namedtuple
import numpy as np


def _read_pdb(pdb_name, rebuild_hydrogen=False, remove_hydrogen=False):
    """Read a pdb file and return atom information with numpy array format.
    Args:
        pdb_name(str): The pdb file name, absolute path is suggested.
    Returns:
        atom_names(list): 1-dimension list contain all atom names in each residue.
        res_names(list): 1-dimension list of all residue names.
        res_ids(numpy.int32): Unique id for each residue names.
        crds(list): The list format of coordinates.
        res_pointer(numpy.int32): The pointer where the residue starts.
        flatten_atoms(numpy.str_): The flatten atom names.
        flatten_crds(numpy.float32): The numpy array format of coordinates.
        init_res_names(list): The residue name information of each atom.
        init_res_ids(list): The residue id of each atom.
    """
    pdb_obj = namedtuple('PDBObject', ['atom_names', 'res_names', 'crds', 'res_pointer', 'flatten_atoms',
                                       'flatten_crds', 'init_res_names', 'init_res_ids', 'chain_id'])
    with open(pdb_name, 'r') as pdb:
        lines = pdb.readlines()
    atom_names = []
    atom_group = []
    res_names = []
    res_ids = []
    init_res_names = []
    init_res_ids = []
    crds = []
    crd_group = []
    res_pointer = []
    flatten_atoms = []
    flatten_crds = []
    chain_id = []
    c_id = 0
    for _, line in enumerate(lines):
        if line.startswith('END'):
            atom_names.append(atom_group)
            crds.append(crd_group)
            break
        if line.startswith('TER'):
            c_id += 1
            continue
        if not (line.startswith('ATOM') or line.startswith('HETATM')):
            continue

        atom_name = line[12:16].strip()
        if rebuild_hydrogen and atom_name.startswith('H'):
            continue
        if remove_hydrogen and atom_name.startswith('H'):
            continue
        res_name = line[17:20].strip()

        res_id = int(line[22:26].strip())
        crd = [float(line[30:38]),
               float(line[38:46]),
               float(line[46:54])]
        pointer = int(line[6:11].strip()) - 1
        flatten_atoms.append(atom_name)
        flatten_crds.append(crd)
        init_res_names.append(res_name)
        init_res_ids.append(res_id)
        if not res_ids:
            res_ids.append(res_id)
            res_names.append(res_name)
            atom_group.append(atom_name)
            crd_group.append(crd)
            res_pointer.append(0)
            chain_id.append(c_id)
        elif res_id != res_ids[-1]:
            atom_names.append(atom_group)
            crds.append(crd_group)
            atom_group = []
            crd_group = []
            res_ids.append(res_id)
            res_names.append(res_name)
            atom_group.append(atom_name)
            crd_group.append(crd)
            res_pointer.append(pointer)
            chain_id.append(c_id)
        else:
            atom_group.append(atom_name)
            crd_group.append(crd)

    atom_names.append(atom_group)
    crds.append(crd_group)

    flatten_atoms = np.array(flatten_atoms, np.str_)
    flatten_crds = np.array(flatten_crds, np.float32)
    init_res_names = np.array(init_res_names)
    init_res_ids = np.array(init_res_ids, np.int32)
    res_pointer = np.array(res_pointer, np.int32)
    return pdb_obj(atom_names, res_names, crds, res_pointer, flatten_atoms, flatten_crds, init_res_names, init_res_ids,
                   chain_id)
