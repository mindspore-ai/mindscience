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
Read information from a mol2 format file.
"""

from collections import namedtuple
import numpy as np


def mol2parser(file_name: str):
    """Read a mol2 file and return atom information with numpy array format.
    Args:
        file_name(str): The mol2 file name, absolute path is suggested.
    Returns:
        atom_names(np.str_): Numpy array contain all atom names.
        crds(np.float32): The numpy array format of coordinates.
        charges(np.float32): The charge of each atom.
        bond_indexes(np.int32): The bond information.
    """
    mol2_obj = namedtuple('Mol2Object', ['atom_names', 'atom_types', 'crds', 'charges', 'bond_indexes'])
    with open(file_name) as f:
        lines = f.readlines()

    atom_start_index = 0
    atom_end_index = 0
    atom_count = 0
    bond_start_index = 0
    bond_end_index = 0
    bond_count = 0

    for i, line in enumerate(lines):
        if line.startswith('@<TRIPOS>'):
            if atom_count == 1:
                atom_end_index = i
                atom_count += 1
            elif bond_count == 1:
                bond_end_index = i
                bond_count += 1
            if 'ATOM' in line:
                atom_start_index = i + 1
                atom_count += 1
            if 'BOND' in line:
                bond_start_index = i + 1
                bond_count += 1
        elif i == len(lines) - 1:
            if atom_count == 1:
                atom_end_index = i
                atom_count += 1
            elif bond_count == 1:
                bond_end_index = i
                bond_count += 1

    atoms = lines[atom_start_index: atom_end_index]
    bonds = lines[bond_start_index: bond_end_index]

    atom_names = np.array([[name for name in atom.split()[1:2]] for atom in atoms], np.str_).flatten()
    atom_types = np.array([[typ.upper() for typ in atom.split()[5:6]] for atom in atoms], np.str_).flatten()
    crds = np.array([[float(crd) for crd in atom.split()[2:5]] for atom in atoms], np.float32)
    charges = np.array([[float(charge) for charge in atom.split()[-1:]] for atom in atoms], np.float32).flatten()
    bond_index = np.array([[int(b) for b in bond.split()[1:3]] for bond in bonds], np.int32) - 1

    return mol2_obj(atom_names, atom_types, crds, charges, bond_index)
