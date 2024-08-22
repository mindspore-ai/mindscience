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
H-Adder Module.
"""
import os
import sys
import time
import yaml
import numpy as np
from .add_missing_atoms import add_h
from .pdb_generator import gen_pdb
from .pdb_parser import _read_pdb

RESIDUE_NAMES = np.array(['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HID', 'HIS',
                          'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'], np.str_)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
with open(CURRENT_DIR + '/hydrogen_names.yaml', 'r') as file:
    hnames = yaml.load(file.read(), Loader=yaml.SafeLoader)

with open(CURRENT_DIR + '/hydrogen_bond_types.yaml', 'r') as file:
    hbond_type = yaml.load(file.read(), Loader=yaml.SafeLoader)

for res in hbond_type:
    for key in hbond_type[res].keys():
        hbond_type[res][key] = np.array(hbond_type[res][key])

addhs = {'c6': 1,
         'dihedral': 1,
         'c2h4': 2,
         'ch3': 3,
         'cc3': 1,
         'c2h2': 2,
         'wat': 2}

sys.path.append('../')


def add_hydrogen(pdb_in, pdb_out):
    """ The API function for adding Hydrogen.
    Args:
        pdb_in(str): The input pdb file name, absolute file path is suggested.
        pdb_out(str): The output pdb file name, absolute file path is suggested.
    """
    # Record the time cost of Add Hydrogen.
    start_time = time.time()

    pdb_name = pdb_in
    new_pdb_name = pdb_out
    pdb_obj = _read_pdb(pdb_name, rebuild_hydrogen=True)
    atom_names = pdb_obj.atom_names
    res_names = pdb_obj.res_names

    crds = pdb_obj.crds
    chain_id = pdb_obj.chain_id
    is_amino = np.isin(res_names, RESIDUE_NAMES)

    for i, res_name in enumerate(res_names):
        if res_name == 'HIE':
            res_names[i] = 'HIS'
        if res_name == 'HOH':
            res_names[i] = 'WAT'
        if not is_amino[i]:
            continue
        if i == 0:
            res_names[i] = 'N' * (res_name != 'ACE') + res_name
            continue
        elif i == len(res_names) - 1:
            res_names[i] = 'C' * (res_name != 'NME') + res_name
            break
        if chain_id[i] < chain_id[i + 1]:
            res_names[i] = 'C' * (res_name != 'ACE') + res_name
        if chain_id[i] > chain_id[i - 1]:
            res_names[i] = 'N' * (res_name != 'ACE') + res_name

    for i, res_name in enumerate(res_names):
        h_names = []
        crds[i] = np.array(crds[i])

        if res_name == 'NME':
            c_index = np.where(np.array(atom_names[i - 1]) == 'C')
            atom_names[i].insert(0, 'C')
            crds[i] = np.append(crds[i - 1][c_index], crds[i], axis=-2)

        for atom in atom_names[i]:
            if (atom == 'C' and len(res_name) == 4 and res_name.startswith('C')
                    and np.isin(atom_names[i], 'OXT').sum() == 1):
                continue

            if atom in hbond_type[res_name].keys() and len(
                    hbond_type[res_name][atom].shape) == 1:
                addh_type = hbond_type[res_name][atom][0]
                h_names.extend(hnames[res_name][atom])
                m = np.where(np.array(atom_names[i]) == [atom])[0][0]
                n = np.where(
                    np.array(
                        atom_names[i]) == hbond_type[res_name][atom][1])[0][0]
                o = np.where(
                    np.array(
                        atom_names[i]) == hbond_type[res_name][atom][2])[0][0]
                new_crd = add_h(np.array(crds[i]),
                                atype=addh_type,
                                i=m,
                                j=n,
                                k=o)
                crds[i] = np.append(crds[i], new_crd, axis=0)

            elif atom in hbond_type[res_name].keys():
                for j, hbond in enumerate(hbond_type[res_name][atom]):
                    addh_type = hbond[0]
                    h_names.append(hnames[res_name][atom][j])
                    m = np.where(np.array(atom_names[i]) == [atom])[0][0]
                    n = np.where(np.array(atom_names[i]) == hbond[1])[0][0]
                    o = np.where(np.array(atom_names[i]) == hbond[2])[0][0]
                    new_crd = add_h(np.array(crds[i]),
                                    atype=addh_type,
                                    i=m,
                                    j=n,
                                    k=o)
                    crds[i] = np.append(crds[i], new_crd, axis=0)

            else:
                continue

        atom_names[i].extend(h_names)

        if res_name == 'NME':
            atom_names[i].pop(0)
            crds[i] = crds[i][1:]

    new_crds = crds[0]
    for crd in crds[1:]:
        new_crds = np.append(new_crds, crd, axis=0)

    new_atom_names = np.array(atom_names[0])
    for name in atom_names[1:]:
        new_atom_names = np.append(new_atom_names, name)

    new_res_names = []
    new_res_ids = []
    for i, crd in enumerate(crds):
        for _ in range(len(crd)):
            new_res_names.append(res_names[i])
            new_res_ids.append(i + 1)

    if new_crds.size == 0:
        print('[Error] Adding hydrogen atoms failed.')
        raise ValueError('The value of crd after adding hydrogen is empty!')

    # Clear old pdb files.
    if os.path.exists(new_pdb_name):
        os.remove(new_pdb_name)

    gen_pdb(new_crds, new_atom_names,
            new_res_names, new_res_ids, chain_id=chain_id, pdb_name=new_pdb_name)

    end_time = time.time()
    print(
        '[MindSPONGE] Adding {} hydrogen atoms for the protein molecule in {} seconds.'.format(
            new_crds.shape[-2] - len(crds[0]), round(end_time - start_time, 3)))


def read_pdb(pdb_name: str, rebuild_hydrogen: bool = False, rebuild_suffix: str = '_addH',
             remove_hydrogen: bool = False):
    """ Entry function for parse pdb files.
    Args:
        pdb_name(str): The pdb file name, absolute path is suggested.
        rebuild_hydrogen(Bool): Set to rebuild all hydrogen in pdb files or not.
        rebuild_suffix(str): If rebuild the hydrogen system, a new pdb file with suffix will be stored.
        remove_hydrogen(bool): Set to True if we don't want hydrogen in our system.
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
    if rebuild_hydrogen:
        out_name = pdb_name.replace('.pdb', '{}.pdb'.format(rebuild_suffix))
        add_hydrogen(pdb_name, out_name)
        return _read_pdb(out_name, remove_hydrogen=remove_hydrogen)

    return _read_pdb(pdb_name, remove_hydrogen=remove_hydrogen)
