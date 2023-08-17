
# Copyright 2021 The AIMM Group at Shenzhen Bay Laboratory & Peking University & Huawei Technologies Co., Ltd
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
"""data transform MSA TEMPLATE"""
import numpy as np
from mindsponge.common.residue_constants import chi_angles_mask, chi_pi_periodic, restype_1to3, chi_angles_atoms, \
    atom_order, residue_atom_renaming_swaps, restype_3to1, MAP_HHBLITS_AATYPE_TO_OUR_AATYPE, restype_order, \
    restypes, restype_name_to_atom14_names, atom_types, residue_atoms, STANDARD_ATOM_MASK, restypes_with_x_and_gap, \
    MSA_PAD_VALUES


def get_chi_atom_pos_indices():
    """get the atom indices for computing chi angles for all residue types"""
    chi_atom_pos_indices = []
    for residue_name in restypes:
        residue_name = restype_1to3[residue_name]
        residue_chi_angles = chi_angles_atoms[residue_name]
        atom_pos_indices = []
        for chi_angle in residue_chi_angles:
            atom_pos_indices.append([atom_order[atom] for atom in chi_angle])
        for _ in range(4 - len(atom_pos_indices)):
            atom_pos_indices.append([0, 0, 0, 0])  # For chi angles not defined on the AA.
        chi_atom_pos_indices.append(atom_pos_indices)

    chi_atom_pos_indices.append([[0, 0, 0, 0]] * 4)  # For UNKNOWN residue.

    return np.array(chi_atom_pos_indices)


def one_hot(depth, indices):
    """one hot compute"""
    res = np.eye(depth)[indices.reshape(-1)]
    return res.reshape(list(indices.shape) + [depth])


def make_atom14_masks(aatype):
    """create atom 14 position features from aatype"""
    rt_atom14_to_atom37 = []
    rt_atom37_to_atom14 = []
    rt_atom14_mask = []

    for restype in restypes:
        atom_names = restype_name_to_atom14_names.get(restype_1to3.get(restype))

        rt_atom14_to_atom37.append([(atom_order[name] if name else 0) for name in atom_names])

        atom_name_to_idx14 = {name: i for i, name in enumerate(atom_names)}
        rt_atom37_to_atom14.append([(atom_name_to_idx14[name] if name in atom_name_to_idx14 else 0)
                                    for name in atom_types])

        rt_atom14_mask.append([(1. if name else 0.) for name in atom_names])

    # Add dummy mapping for restype 'UNK'
    rt_atom14_to_atom37.append([0] * 14)
    rt_atom37_to_atom14.append([0] * 37)
    rt_atom14_mask.append([0.] * 14)

    rt_atom14_to_atom37 = np.array(rt_atom14_to_atom37, np.int32)
    rt_atom37_to_atom14 = np.array(rt_atom37_to_atom14, np.int32)
    rt_atom14_mask = np.array(rt_atom14_mask, np.float32)

    ri_atom14_to_atom37 = rt_atom14_to_atom37[aatype]
    ri_atom14_mask = rt_atom14_mask[aatype]

    atom14_atom_exists = ri_atom14_mask
    ri_atom14_to_atom37 = ri_atom14_to_atom37

    # create the gather indices for mapping back
    ri_atom37_to_atom14 = rt_atom37_to_atom14[aatype]
    ri_atom37_to_atom14 = ri_atom37_to_atom14

    # create the corresponding mask
    restype_atom37_mask = np.zeros([21, 37], np.float32)
    for restype, restype_letter in enumerate(restypes):
        restype_name = restype_1to3.get(restype_letter)
        atom_names = residue_atoms.get(restype_name)
        for atom_name in atom_names:
            atom_type = atom_order[atom_name]
            restype_atom37_mask[restype, atom_type] = 1

    atom37_atom_exists = restype_atom37_mask[aatype]
    res = [atom14_atom_exists, ri_atom14_to_atom37, ri_atom37_to_atom14, atom37_atom_exists]
    return res
    