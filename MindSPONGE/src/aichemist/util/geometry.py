# Copyright 2021-2023 @ Shenzhen Bay Laboratory &
#                       Peking University &
#                       Huawei Technologies Co., Ltd
#
# This code is a part of AIchemist package.
#
# The AIchemist is open-source software based on the AI-framework:
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
geometry
"""


import copy
import math

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolTransforms
from scipy.spatial.transform import Rotation


def random_rotation_translation(max_translation):
    rotation = Rotation.random(num=1)
    r = rotation.as_matrix().squeeze()

    t = np.random.randn(1, 3)
    t = t / np.linalg.norm(t)
    length = np.random.uniform(low=0, high=max_translation)
    t = t * length
    return r.astype(np.float32), t.astype(np.float32)


# R = 3x3 rotation matrix
# t = 3x1 column vector
# This already takes residue identity into account.
def rigid_transform_kabsch_3d(mat_a, mat_b):
    """rigid transform kabsch 3D"""
    assert mat_a.shape[1] == mat_b.shape[1]
    num_rows, num_cols = mat_a.shape
    if num_rows != 3:
        raise ValueError(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")
    num_rows, num_cols = mat_b.shape
    if num_rows != 3:
        raise ValueError(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise: 3 x 1
    centroid_a = np.mean(mat_a, axis=1, keepdims=True)
    centroid_b = np.mean(mat_b, axis=1, keepdims=True)

    # subtract mean
    mat_a_ = mat_a - centroid_a
    mat_b_ = mat_b - centroid_b

    mat_h = mat_a_ @ mat_b_.T

    # find rotation
    ut, _, vt = np.linalg.svd(mat_h)

    mat_r = vt.T @ ut.T

    # special reflection case
    if np.linalg.det(mat_r) < 0:
        ss = np.diag([1., 1., -1.])
        mat_r = (vt.T @ ss) @ ut.T
    assert math.fabs(np.linalg.det(mat_r) - 1) < 1e-5

    t = -mat_r @ centroid_a + centroid_b
    return mat_r, t


def get_torsions(mol_list):
    """_summary_

    Args:
        mol_list (_type_): _description_

    Returns:
        _type_: _description_
    """
    atom_counter = 0
    torsion_list = []
    for m in mol_list:
        torsion_smarts = '[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]'
        torsion_query = Chem.MolFromSmarts(torsion_smarts)
        matches = m.GetSubstructMatches(torsion_query)
        for match in matches:
            idx2 = match[0]
            idx3 = match[1]
            bond = m.GetBondBetweenAtoms(idx2, idx3)
            j_atom = m.GetAtomWithIdx(idx2)
            k_atom = m.GetAtomWithIdx(idx3)
            for b1 in j_atom.GetBonds():
                if b1.GetIdx() == bond.GetIdx():
                    continue
                idx1 = b1.GetOtherAtomIdx(idx2)
                for b2 in k_atom.GetBonds():
                    if (b2.GetIdx() == bond.GetIdx()) or (b2.GetIdx() == b1.GetIdx()):
                        continue
                    idx4 = b2.GetOtherAtomIdx(idx3)
                    # skip 3-membered rings
                    if idx4 == idx1:
                        continue
                    if m.GetAtomWithIdx(idx4).IsInRing():
                        torsion_list.append(
                            (idx4 + atom_counter, idx3 + atom_counter, idx2 + atom_counter, idx1 + atom_counter))
                    else:
                        torsion_list.append(
                            (idx1 + atom_counter, idx2 + atom_counter, idx3 + atom_counter, idx4 + atom_counter))
                    break
                break

        atom_counter += m.GetNumAtoms()
    return torsion_list


def mol_with_atom_index(mol):
    atoms = mol.GetNumAtoms()
    for idx in range(atoms):
        mol.GetAtomWithIdx(idx).SetProp('molAtomMapNumber', str(mol.GetAtomWithIdx(idx).GetIdx()))
    return mol


def set_dihedral(conf, atom_idx, new_vale):
    rdMolTransforms.SetDihedralDeg(conf, atom_idx[0], atom_idx[1], atom_idx[2], atom_idx[3], new_vale)


def get_dihedral(conf, atom_idx):
    return rdMolTransforms.GetDihedralDeg(conf, atom_idx[0], atom_idx[1], atom_idx[2], atom_idx[3])


def get_transformation_matrix(transformations):
    x, y, z, disp_x, disp_y, disp_z = transformations
    trans_mat = np.array([[np.cos(z) * np.cos(y), (np.cos(z) * np.sin(y) * np.sin(x)) - (np.sin(z) * np.cos(x)),
                           (np.cos(z) * np.sin(y) * np.cos(x)) + (np.sin(z) * np.sin(x)), disp_x],
                          [np.sin(z) * np.cos(y), (np.sin(z) * np.sin(y) * np.sin(x)) + (np.cos(z) * np.cos(x)),
                           (np.sin(z) * np.sin(y) * np.cos(x)) - (np.cos(z) * np.sin(x)), disp_y],
                          [-np.sin(y), np.cos(y) * np.sin(x), np.cos(y) * np.cos(x), disp_z],
                          [0, 0, 0, 1]
                         ], dtype=np.double)
    return trans_mat


def apply_changes(mol, values, rotable_bonds):
    """apply changes"""
    opt_mol = copy.deepcopy(mol)

    # apply rotations
    for r, rotable_bond in enumerate(rotable_bonds):
        set_dihedral(opt_mol.GetConformer(), rotable_bond, values[r])

    return opt_mol


# Clockwise dihedral2 from
# https://stackoverflow.com/questions/20305272/dihedral-torsion-angle-from-four-points-in-cartesian-coordinates-in-python
def get_dihedral_from_point_cloud(z, atom_idx):
    """_summary_

    Args:
        z (_type_): _description_
        atom_idx (_type_): _description_

    Returns:
        _type_: _description_
    """
    p = z[list(atom_idx)]
    b = p[:-1] - p[1:]
    b[0] *= -1
    v = np.array([v - (v.dot(b[1])/b[1].dot(b[1])) * b[1] for v in [b[0], b[2]]])
    # Normalize vectors
    v /= np.sqrt(np.einsum('...i,...i', v, v)).reshape(-1, 1)
    b1 = b[1] / np.linalg.norm(b[1])
    x = np.dot(v[0], v[1])
    m = np.cross(v[0], b1)
    y = np.dot(m, v[1])
    return np.degrees(np.arctan2(y, x))


def transpose_matrix(alpha):
    return np.array([[np.cos(np.radians(alpha)), np.sin(np.radians(alpha))],
                     [-np.sin(np.radians(alpha)), np.cos(np.radians(alpha))]], dtype=np.double)


def s_vec(alpha):
    return np.array([[np.cos(np.radians(alpha))],
                     [np.sin(np.radians(alpha))]], dtype=np.double)


def get_dihedral_von_mises(mol, conf, atom_idx, z):
    """_summary_

    Args:
        mol (_type_): _description_
        conf (_type_): _description_
        atom_idx (_type_): _description_
        z (_type_): _description_

    Returns:
        _type_: _description_
    """
    z = np.array(z)
    v = np.zeros((2, 1))
    i_atom = mol.GetAtomWithIdx(atom_idx[1])
    j_atom = mol.GetAtomWithIdx(atom_idx[2])
    k_0 = atom_idx[0]
    i = atom_idx[1]
    j = atom_idx[2]
    l_0 = atom_idx[3]
    for b1 in i_atom.GetBonds():
        k = b1.GetOtherAtomIdx(i)
        if k == j:
            continue
        for b2 in j_atom.GetBonds():
            other = b2.GetOtherAtomIdx(j)
            if other == i:
                continue
            assert k != other
            s_star = s_vec(get_dihedral_from_point_cloud(z, (k, i, j, other)))
            a_mat = transpose_matrix(get_dihedral(conf, (k, i, j, k_0)) + get_dihedral(conf, (l_0, i, j, other)))
            v = v + np.matmul(a_mat, s_star)
    v = v / np.linalg.norm(v)
    v = v.reshape(-1)
    return np.degrees(np.arctan2(v[1], v[0]))
