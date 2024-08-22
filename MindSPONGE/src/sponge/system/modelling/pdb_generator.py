# Copyright 2021-2023 @ Shenzhen Bay Laboratory &
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
Module used to generate a pdb file via crd and res names.
"""

import os
import numpy as np


def remark_res_id(res_ids):
    """Remark the residue id to make it unique"""
    start_id = 1
    new_res_id = []
    for i, idx in enumerate(res_ids):
        if i == 0:
            new_res_id.append(start_id)
        elif idx == res_ids[i-1]:
            new_res_id.append(start_id)
        else:
            start_id += 1
            new_res_id.append(start_id)
    return np.array(new_res_id, np.int32)


def gen_pdb(crd, atom_names, res_names, res_ids, chain_id=None, pdb_name='temp.pdb', sequence_info=True, bonds=None):
    """Write protein crd information into pdb format files.
    Args:
        crd(numpy.float32): The coordinates of protein atoms.
        atom_names(numpy.str_): The atom names differ from aminos.
        res_names(numpy.str_): The residue names of amino names.
        res_ids(numpy.int32): A unique mask each same residue.
        pdb_name(str): The path to save the pdb file, absolute path is suggested.
        chain_id(numpy.int32): The chain index of each residue.
        sequence_info(bool): Decide to show the sequence in pdb file or not.
        bonds(numpy.int32): The bond index.
    """
    if os.path.exists(pdb_name):
        os.remove(pdb_name)

    res_ids = remark_res_id(res_ids)
    success = 1
    file = os.open(pdb_name, os.O_RDWR | os.O_CREAT)
    pdb = os.fdopen(file, "w")
    res_ids = np.array(res_ids, np.int32)
    chain_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    pdb.write('MODEL     1\n')

    # Write sequence information
    if sequence_info and chain_id is not None:
        chain_id = np.array(chain_id, np.int32)
        seq_label = 0
        start_label = 0
        for i, res in enumerate(res_names):
            if i > 0 and res_ids[i] == res_ids[i-1]:
                continue
            cri = res_ids[i] - 1
            if i == 0:
                seq_label += 1
                pdb.write('SEQRES'.ljust(6))
                pdb.write('{}'.format(seq_label).rjust(4))
                pdb.write('{}'.format(chain_labels[chain_id[cri]]).rjust(2))
                pdb.write('{}'.format((chain_id == chain_id[cri]).sum()).rjust(5))
                pdb.write(res[-3:].rjust(5))
            elif (cri-start_label) % 13 == 0 and chain_id[cri] == chain_id[cri-1]:
                seq_label += 1
                pdb.write('\n')
                pdb.write('SEQRES'.ljust(6))
                pdb.write('{}'.format(seq_label).rjust(4))
                pdb.write('{}'.format(chain_labels[chain_id[cri]]).rjust(2))
                pdb.write('{}'.format((chain_id == chain_id[cri]).sum()).rjust(5))
                pdb.write(res[-3:].rjust(5))
            elif chain_id[cri] != chain_id[cri-1]:
                pdb.write('\n')
                seq_label = 1
                start_label = cri
                pdb.write('SEQRES'.ljust(6))
                pdb.write('{}'.format(seq_label).rjust(4))
                pdb.write('{}'.format(chain_labels[chain_id[cri]]).rjust(2))
                pdb.write('{}'.format((chain_id == chain_id[cri]).sum()).rjust(5))
                pdb.write(res[-3:].rjust(5))
            elif (cri-start_label) % 13 != 0 and chain_id[cri] == chain_id[cri-1]:
                pdb.write(res[-3:].rjust(4))
        pdb.write('\n')

    # Write atom information
    write_atom(pdb, crd, res_ids, atom_names, res_names, chain_id, chain_labels)

    if bonds is not None:
        write_bonds(pdb, bonds, crd)

    pdb.write('TER\n')
    pdb.write('ENDMDL\n')
    pdb.write('END\n')

    pdb.close()
    return success

def write_bonds(pdb, bonds, crd):
    """Write bond information into pdb file"""
    num_atoms = crd.shape[-2]
    contact_map = np.zeros((num_atoms, num_atoms), dtype=np.int32)
    for bond in bonds:
        contact_map[bond[0]][bond[1]] = 1
        contact_map[bond[1]][bond[0]] = 1
    for i, contact in enumerate(contact_map):
        pdb.write('CONECT')
        neighs = np.where(contact > 0)[0]
        if neighs.size == 0:
            continue
        pdb.write('{}'.format((i + 1) % 100000).rjust(5))
        for atom in neighs:
            pdb.write('{}'.format((atom + 1) % 100000).rjust(5))
        pdb.write('\n')

def write_atom(pdb, crd, res_ids, atom_names, res_names, chain_id, chain_labels):
    """Write atom information into pdb file"""
    record_resids = res_ids.copy()
    for i, c in enumerate(crd):
        if (chain_id is not None and i > 0 and
                chain_id[res_ids[i] - 1] > chain_id[res_ids[i - 1] - 1]):
            pdb.write('TER\n')
            record_resids -= record_resids[i] - 1
        pdb.write('ATOM'.ljust(6))
        pdb.write('{}'.format((i + 1) % 100000).rjust(5))
        if len(atom_names[i]) < 4:
            pdb.write('  ')
            pdb.write(atom_names[i].ljust(3))
        else:
            pdb.write(' ')
            pdb.write(atom_names[i].ljust(4))
        pdb.write(res_names[i][-3:].rjust(4))
        if chain_id is None:
            pdb.write('A'.rjust(2))
        else:
            pdb.write('{}'.format(chain_labels[chain_id[res_ids[i] - 1]]).rjust(2))
        pdb.write('{}'.format(record_resids[i] % 10000).rjust(4))
        pdb.write('    ')
        pdb.write('{:.3f}'.format(c[0]).rjust(8))
        pdb.write('{:.3f}'.format(c[1]).rjust(8))
        pdb.write('{:.3f}'.format(c[2]).rjust(8))
        pdb.write('1.0'.rjust(6))
        pdb.write('0.0'.rjust(6))
        pdb.write('{}'.format(atom_names[i][0]).rjust(12))
        pdb.write('\n')
