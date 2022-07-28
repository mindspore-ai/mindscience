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
import numpy as np

restypes = [
    'A', 'R', 'N', 'D', 'C',
    'Q', 'E', 'G', 'H', 'I',
    'L', 'K', 'M', 'F', 'P',
    'S', 'T', 'W', 'Y', 'V'
]
resdict = {'ALA': 0, 'ARG': 1, 'ASN': 2, 'ASP': 3, 'CYS': 4,
           'GLN': 5, 'GLU': 6, 'GLY': 7, 'HIS': 8, 'ILE': 9,
           'LEU': 10, 'LYS': 11, 'MET': 12, 'PHE': 13, 'PRO': 14,
           'SER': 15, 'THR': 16, 'TRP': 17, 'TYR': 18, 'VAL': 19,
           'CALA': 0, 'CARG': 1, 'CASN': 2, 'CASP': 3, 'CCYS': 4,
           'CGLN': 5, 'CGLU': 6, 'CGLY': 7, 'CHIS': 8, 'CILE': 9,
           'CLEU': 10, 'CLYS': 11, 'CMET': 12, 'CPHE': 13, 'CPRO': 14,
           'CSER': 15, 'CTHR': 16, 'CTRP': 17, 'CTYR': 18, 'CVAL': 19,
           'NALA': 0, 'NARG': 1, 'NASN': 2, 'NASP': 3, 'NCYS': 4,
           'NGLN': 5, 'NGLU': 6, 'NGLY': 7, 'NHIS': 8, 'NILE': 9,
           'NLEU': 10, 'NLYS': 11, 'NMET': 12, 'NPHE': 13, 'NPRO': 14,
           'NSER': 15, 'NTHR': 16, 'NTRP': 17, 'NTYR': 18, 'NVAL': 19,
           'CHIE': 8, 'HIE': 8, 'NHIE': 8, 'WAT': 22
           }

atom_types = [
    'N', 'CA', 'C', 'CB', 'O', 'CG', 'CG1', 'CG2', 'OG', 'OG1', 'SG', 'CD',
    'CD1', 'CD2', 'ND1', 'ND2', 'OD1', 'OD2', 'SD', 'CE', 'CE1', 'CE2', 'CE3',
    'NE', 'NE1', 'NE2', 'OE1', 'OE2', 'CH2', 'NH1', 'NH2', 'OH', 'CZ', 'CZ2',
    'CZ3', 'NZ', 'OXT'
]
atom_order = {atom_type: i for i, atom_type in enumerate(atom_types)}
atom_type_num = len(atom_types)  # := 37.

restype_name_to_atom14_names = {
    'ALA': ['N', 'CA', 'C', 'O', 'CB', '', '', '', '', '', '', '', '', ''],
    'ARG': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'NE', 'CZ', 'NH1', 'NH2', '', '', ''],
    'ASN': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'OD1', 'ND2', '', '', '', '', '', ''],
    'ASP': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'OD1', 'OD2', '', '', '', '', '', ''],
    'CYS': ['N', 'CA', 'C', 'O', 'CB', 'SG', '', '', '', '', '', '', '', ''],
    'GLN': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OE1', 'NE2', '', '', '', '', ''],
    'GLU': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OE1', 'OE2', '', '', '', '', ''],
    'GLY': ['N', 'CA', 'C', 'O', '', '', '', '', '', '', '', '', '', ''],
    'HIS': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'ND1', 'CD2', 'CE1', 'NE2', '', '', '', ''],
    'ILE': ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', 'CD1', '', '', '', '', '', ''],
    'LEU': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', '', '', '', '', '', ''],
    'LYS': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'CE', 'NZ', '', '', '', '', ''],
    'MET': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'SD', 'CE', '', '', '', '', '', ''],
    'PHE': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', '', '', ''],
    'PRO': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', '', '', '', '', '', '', ''],
    'SER': ['N', 'CA', 'C', 'O', 'CB', 'OG', '', '', '', '', '', '', '', ''],
    'THR': ['N', 'CA', 'C', 'O', 'CB', 'OG1', 'CG2', '', '', '', '', '', '', ''],
    'TRP': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'NE1', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2'],
    'TYR': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'OH', '', ''],
    'VAL': ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', '', '', '', '', '', '', ''],
    'UNK': ['', '', '', '', '', '', '', '', '', '', '', '', '', ''],
    'WAT': ['OW', '', '', '', '', '', '', '', '', '', '', '', '', ''],
}
restype_name_to_atom14_masks = {
    'ALA': [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'ARG': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    'ASN': [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    'ASP': [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    'CYS': [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    'GLN': [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    'GLU': [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    'GLY': [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'HIS': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    'HIE': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    'ILE': [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    'LEU': [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    'LYS': [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    'MET': [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    'PHE': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    'PRO': [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    'SER': [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    'THR': [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    'TRP': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    'TYR': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
    'VAL': [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    'UNK': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'WAT': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}

atom14_order_dict = {'ALA': {'N': 0, 'CA': 1, 'C': 2, 'O': 3, 'CB': 4},
                     'ARG': {'N': 0,
                             'CA': 1,
                             'C': 2,
                             'O': 3,
                             'CB': 4,
                             'CG': 5,
                             'CD': 6,
                             'NE': 7,
                             'CZ': 8,
                             'NH1': 9,
                             'NH2': 10},
                     'ASN': {'N': 0,
                             'CA': 1,
                             'C': 2,
                             'O': 3,
                             'CB': 4,
                             'CG': 5,
                             'OD1': 6,
                             'ND2': 7},
                     'ASP': {'N': 0,
                             'CA': 1,
                             'C': 2,
                             'O': 3,
                             'CB': 4,
                             'CG': 5,
                             'OD1': 6,
                             'OD2': 7},
                     'CYS': {'N': 0, 'CA': 1, 'C': 2, 'O': 3, 'CB': 4, 'SG': 5},
                     'GLN': {'N': 0,
                             'CA': 1,
                             'C': 2,
                             'O': 3,
                             'CB': 4,
                             'CG': 5,
                             'CD': 6,
                             'OE1': 7,
                             'NE2': 8},
                     'GLU': {'N': 0,
                             'CA': 1,
                             'C': 2,
                             'O': 3,
                             'CB': 4,
                             'CG': 5,
                             'CD': 6,
                             'OE1': 7,
                             'OE2': 8},
                     'GLY': {'N': 0, 'CA': 1, 'C': 2, 'O': 3},
                     'HIS': {'N': 0,
                             'CA': 1,
                             'C': 2,
                             'O': 3,
                             'CB': 4,
                             'CG': 5,
                             'ND1': 6,
                             'CD2': 7,
                             'CE1': 8,
                             'NE2': 9},
                     'HIE': {'N': 0,
                             'CA': 1,
                             'C': 2,
                             'O': 3,
                             'CB': 4,
                             'CG': 5,
                             'ND1': 6,
                             'CD2': 7,
                             'CE1': 8,
                             'NE2': 9},
                     'ILE': {'N': 0,
                             'CA': 1,
                             'C': 2,
                             'O': 3,
                             'CB': 4,
                             'CG1': 5,
                             'CG2': 6,
                             'CD1': 7},
                     'LEU': {'N': 0,
                             'CA': 1,
                             'C': 2,
                             'O': 3,
                             'CB': 4,
                             'CG': 5,
                             'CD1': 6,
                             'CD2': 7},
                     'LYS': {'N': 0,
                             'CA': 1,
                             'C': 2,
                             'O': 3,
                             'CB': 4,
                             'CG': 5,
                             'CD': 6,
                             'CE': 7,
                             'NZ': 8},
                     'MET': {'N': 0, 'CA': 1, 'C': 2, 'O': 3, 'CB': 4, 'CG': 5, 'SD': 6, 'CE': 7},
                     'PHE': {'N': 0,
                             'CA': 1,
                             'C': 2,
                             'O': 3,
                             'CB': 4,
                             'CG': 5,
                             'CD1': 6,
                             'CD2': 7,
                             'CE1': 8,
                             'CE2': 9,
                             'CZ': 10},
                     'PRO': {'N': 0, 'CA': 1, 'C': 2, 'O': 3, 'CB': 4, 'CG': 5, 'CD': 6},
                     'SER': {'N': 0, 'CA': 1, 'C': 2, 'O': 3, 'CB': 4, 'OG': 5},
                     'THR': {'N': 0, 'CA': 1, 'C': 2, 'O': 3, 'CB': 4, 'OG1': 5, 'CG2': 6},
                     'TRP': {'N': 0,
                             'CA': 1,
                             'C': 2,
                             'O': 3,
                             'CB': 4,
                             'CG': 5,
                             'CD1': 6,
                             'CD2': 7,
                             'NE1': 8,
                             'CE2': 9,
                             'CE3': 10,
                             'CZ2': 11,
                             'CZ3': 12,
                             'CH2': 13},
                     'TYR': {'N': 0,
                             'CA': 1,
                             'C': 2,
                             'O': 3,
                             'CB': 4,
                             'CG': 5,
                             'CD1': 6,
                             'CD2': 7,
                             'CE1': 8,
                             'CE2': 9,
                             'CZ': 10,
                             'OH': 11},
                     'VAL': {'N': 0, 'CA': 1, 'C': 2, 'O': 3, 'CB': 4, 'CG1': 5, 'CG2': 6},
                     'UNK': {},
                     'WAT': {'OW': 0}}

atom14_to_atom37_dict = {'ALA': [0, 1, 2, 4, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         'ARG': [0, 1, 2, 4, 3, 5, 11, 23, 32, 29, 30, 0, 0, 0],
                         'ASN': [0, 1, 2, 4, 3, 5, 16, 15, 0, 0, 0, 0, 0, 0],
                         'ASP': [0, 1, 2, 4, 3, 5, 16, 17, 0, 0, 0, 0, 0, 0],
                         'CYS': [0, 1, 2, 4, 3, 10, 0, 0, 0, 0, 0, 0, 0, 0],
                         'GLN': [0, 1, 2, 4, 3, 5, 11, 26, 25, 0, 0, 0, 0, 0],
                         'GLU': [0, 1, 2, 4, 3, 5, 11, 26, 27, 0, 0, 0, 0, 0],
                         'GLY': [0, 1, 2, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         'HIS': [0, 1, 2, 4, 3, 5, 14, 13, 20, 25, 0, 0, 0, 0],
                         'HIE': [0, 1, 2, 4, 3, 5, 14, 13, 20, 25, 0, 0, 0, 0],
                         'ILE': [0, 1, 2, 4, 3, 6, 7, 12, 0, 0, 0, 0, 0, 0],
                         'LEU': [0, 1, 2, 4, 3, 5, 12, 13, 0, 0, 0, 0, 0, 0],
                         'LYS': [0, 1, 2, 4, 3, 5, 11, 19, 35, 0, 0, 0, 0, 0],
                         'MET': [0, 1, 2, 4, 3, 5, 18, 19, 0, 0, 0, 0, 0, 0],
                         'PHE': [0, 1, 2, 4, 3, 5, 12, 13, 20, 21, 32, 0, 0, 0],
                         'PRO': [0, 1, 2, 4, 3, 5, 11, 0, 0, 0, 0, 0, 0, 0],
                         'SER': [0, 1, 2, 4, 3, 8, 0, 0, 0, 0, 0, 0, 0, 0],
                         'THR': [0, 1, 2, 4, 3, 9, 7, 0, 0, 0, 0, 0, 0, 0],
                         'TRP': [0, 1, 2, 4, 3, 5, 12, 13, 24, 21, 22, 33, 34, 28],
                         'TYR': [0, 1, 2, 4, 3, 5, 12, 13, 20, 21, 32, 31, 0, 0],
                         'VAL': [0, 1, 2, 4, 3, 6, 7, 0, 0, 0, 0, 0, 0, 0],
                         'UNK': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         'WAT': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}


def read_pdb(pdb_name, ignoreh=False):
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
    with open(pdb_name, 'r', encoding="utf-8") as pdb:
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
    atom14_positions = []
    atom14_atom_exists = []
    residx_atom14_to_atom37 = []
    for index, line in enumerate(lines):
        if 'END' in line or 'TER' in line:
            atom_names.append(atom_group)
            crds.append(crd_group)
            atom14_positions.append(atom_pos)
            residx_atom14_to_atom37.append(atom14_to_atom37_dict[res_name])
            break
        if not line.startswith('ATOM'):
            continue
        atom_name = line[12:16].strip()
        if ignoreh and atom_name.startswith('H'):
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
            atom14_atom_exists.append(restype_name_to_atom14_masks[res_name])
            atom_group.append(atom_name)
            crd_group.append(crd)
            res_pointer.append(0)
            atom_pos = np.zeros((14, 3))
            if not atom_name.startswith('H') and atom_name != 'OXT':
                atom_pos[atom14_order_dict[res_name]
                         [atom_name]] = np.array(crd)
        elif res_id != res_ids[-1]:
            atom14_positions.append(atom_pos)
            residx_atom14_to_atom37.append(atom14_to_atom37_dict[res_name])
            atom_pos = np.zeros((14, 3))
            if not atom_name.startswith('H') and atom_name != 'OXT':
                atom_pos[atom14_order_dict[res_name]
                         [atom_name]] = np.array(crd)
            atom_names.append(atom_group)
            crds.append(crd_group)
            atom_group = []
            crd_group = []
            res_ids.append(res_id)
            res_names.append(res_name)
            atom14_atom_exists.append(restype_name_to_atom14_masks[res_name])
            atom_group.append(atom_name)
            crd_group.append(crd)
            res_pointer.append(pointer)
        else:
            atom_group.append(atom_name)
            crd_group.append(crd)
            if not atom_name.startswith('H') and atom_name != 'OXT':
                atom_pos[atom14_order_dict[res_name]
                         [atom_name]] = np.array(crd)
        if index == len(lines) - 1:
            atom_names.append(atom_group)
            crds.append(crd_group)
            atom14_positions.append(atom_pos)
            residx_atom14_to_atom37.append(atom14_to_atom37_dict[res_name])

    res_ids = np.array(res_ids, np.int32)
    flatten_atoms = np.array(flatten_atoms, np.str_)
    flatten_crds = np.array(flatten_crds, np.float32)
    init_res_names = np.array(init_res_names)
    init_res_ids = np.array(init_res_ids, np.int32)
    res_pointer = np.array(res_pointer, np.int32)
    # Violation loss parameters
    residue_index = np.arange(res_pointer.shape[0])
    aatype = np.zeros_like(residue_index)
    for i in range(res_pointer.shape[0]):
        aatype[i] = resdict[res_names[i]]
    atom14_atom_exists = np.array(atom14_atom_exists, np.float32)

    atom14_positions = np.array(atom14_positions, np.float32)
    residx_atom14_to_atom37 = np.array(residx_atom14_to_atom37, np.float32)

    return atom_names, res_names, res_ids, crds, res_pointer, flatten_atoms, flatten_crds, init_res_names,\
        init_res_ids,\
        residue_index, aatype, atom14_positions, atom14_atom_exists, residx_atom14_to_atom37
