# Copyright 2022 Huawei Technologies Co., Ltd
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
The label generator for the pretraining.
"""
from collections import Counter


BOND_FEATURES = ['BondType', 'Stereo', 'BondDir']


def atom_to_vocab(mol, atom):
    """
    Convert atom to vocabulary. The convention is based on atom type and bond type.
    :param mol: the molecular.
    :param atom: the target atom.
    :return: the generated atom vocabulary with its contexts.
    """
    nei = Counter()
    for a in atom.GetNeighbors():
        bond = mol.GetBondBetweenAtoms(atom.GetIdx(), a.GetIdx())
        nei[str(a.GetSymbol()) + "-" + str(bond.GetBondType())] += 1
    keys = nei.keys()
    keys = list(keys)
    keys.sort()
    output = atom.GetSymbol()
    for k in keys:
        output = "%s_%s%d" % (output, k, nei[k])

    # The generated atom_vocab is too long?
    return output


def bond_to_vocab(mol, bond):
    """
    Convert bond to vocabulary. The convention is based on atom type and bond type.
    Considering one-hop neighbor atoms
    :param mol: the molecular.
    :param atom: the target atom.
    :return: the generated bond vocabulary with its contexts.
    """
    nei = Counter()
    two_neighbors = (bond.GetBeginAtom(), bond.GetEndAtom())
    two_indices = [a.GetIdx() for a in two_neighbors]
    for nei_atom in two_neighbors:
        for a in nei_atom.GetNeighbors():
            a_idx = a.GetIdx()
            if a_idx in two_indices:
                continue
            tmp_bond = mol.GetBondBetweenAtoms(nei_atom.GetIdx(), a_idx)
            nei[str(nei_atom.GetSymbol()) + '-' + get_bond_feature_name(tmp_bond)] += 1
    keys = list(nei.keys())
    keys.sort()
    output = get_bond_feature_name(bond)
    for k in keys:
        output = "%s_%s%d" % (output, k, nei[k])
    return output


def get_bond_feature_name(bond):
    """
    Return the string format of bond features.
    Bond features are surrounded with ()

    """
    ret = []
    for bond_feature in BOND_FEATURES:
        if bond_feature == 'BondType':
            fea = bond.GetBondType()
        elif bond_feature == 'Stereo':
            fea = bond.GetStereo()
        elif bond_feature == 'BondDir':
            fea = bond.GetBondDir()
        else:
            continue
        ret.append(str(fea))

    return '(' + '-'.join(ret) + ')'
