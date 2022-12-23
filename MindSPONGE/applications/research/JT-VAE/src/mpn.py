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
"""mpn"""
import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import rdkit.Chem as Chem
from .nnutils import index_select_nd
from .chemutils import get_mol
from .utils import for_stack


ELEM_LIST = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'Al', 'I', 'B', 'K', 'Se', 'Zn',
             'H', 'Cu', 'Mn', 'unknown']

ATOM_FDIM = len(ELEM_LIST) + 6 + 5 + 4 + 1
BOND_FDIM = 5 + 6
MAX_NB = 6


def onek_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def atom_features(atom):
    return ms.Tensor(onek_encoding_unk(atom.GetSymbol(), ELEM_LIST)
                     + onek_encoding_unk(atom.GetDegree(), [0, 1, 2, 3, 4, 5])
                     + onek_encoding_unk(atom.GetFormalCharge(), [-1, -2, 1, 2, 0])
                     + onek_encoding_unk(int(atom.GetChiralTag()), [0, 1, 2, 3])
                     + [atom.GetIsAromatic()], ms.float32)


def bond_features(bond):
    bt = bond.GetBondType()
    stereo = int(bond.GetStereo())
    fbond = [bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE, bt == Chem.rdchem.BondType.TRIPLE,
             bt == Chem.rdchem.BondType.AROMATIC, bond.IsInRing()]
    fstereo = onek_encoding_unk(stereo, [0, 1, 2, 3, 4, 5])
    return ms.Tensor(fbond + fstereo, ms.float32)


def mol2graph(mol_batch):
    """mol to graph"""
    padding = ops.zeros(ATOM_FDIM + BOND_FDIM, ms.float32)
    fatoms, fbonds = [], [padding]
    in_bonds, all_bonds = [], [(-1, -1)]
    scope = []
    total_atoms = 0

    for smiles in mol_batch:
        mol = get_mol(smiles)
        n_atoms = mol.GetNumAtoms()
        for atom in mol.GetAtoms():
            fatoms.append(atom_features(atom))
            in_bonds.append([])

        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtom()
            a2 = bond.GetEndAtom()
            x = a1.GetIdx() + total_atoms
            y = a2.GetIdx() + total_atoms

            b = len(all_bonds)
            all_bonds.append((x, y))
            fbonds.append(ops.concat([fatoms[x], bond_features(bond)], 0))
            in_bonds[y].append(b)

            b = len(all_bonds)
            all_bonds.append((y, x))
            fbonds.append(ops.concat([fatoms[y], bond_features(bond)], 0))
            in_bonds[x].append(b)

        scope.append((total_atoms, n_atoms))
        total_atoms += n_atoms

    total_bonds = len(all_bonds)
    fatoms = for_stack(fatoms, 0)
    fbonds = for_stack(fbonds, 0)
    agraph = np.zeros((total_atoms, MAX_NB), np.int32)
    bgraph = np.zeros((total_bonds, MAX_NB), np.int32)

    for a in range(total_atoms):
        for i, b in enumerate(in_bonds[a]):
            agraph[a, i] = b
    agraph = ms.Tensor(agraph, ms.int32)

    for b1 in range(1, total_bonds):
        x, y = all_bonds[b1]
        for i, b2 in enumerate(in_bonds[x]):
            if all_bonds[b2][0] != y:
                bgraph[b1, i] = b2
    bgraph = ms.Tensor(bgraph, ms.int32)
    return_value = fatoms, fbonds, agraph, bgraph, scope

    return return_value


class MPN(nn.Cell):
    """mpn"""

    def __init__(self, hidden_size, depth):
        super(MPN, self).__init__()
        self.hidden_size = hidden_size
        self.depth = depth

        self.w_i = nn.Dense(ATOM_FDIM + BOND_FDIM, hidden_size, has_bias=False)
        self.w_h = nn.Dense(hidden_size, hidden_size, has_bias=False)
        self.w_o = nn.Dense(ATOM_FDIM + hidden_size, hidden_size)

        self.relu = nn.ReLU()

    def construct(self, mol_graph):
        """construct"""
        fatoms, fbonds, agraph, bgraph, scope = mol_graph

        binput = self.w_i(fbonds)
        message = self.relu(binput)

        for _ in range(self.depth - 1):
            nei_message = index_select_nd(message, 0, bgraph)
            nei_message = nei_message.sum(axis=1)
            nei_message = self.w_h(nei_message)
            message = self.relu(binput + nei_message)

        nei_message = index_select_nd(message, 0, agraph)
        nei_message = nei_message.sum(axis=1)
        ainput = ops.concat([fatoms, nei_message], 1)
        atom_hiddens = self.relu(self.w_o(ainput))

        mol_vecs = []
        for st, le in scope:
            mol_vec = atom_hiddens.narrow(0, st, le).sum(axis=0) / le
            mol_vecs.append(mol_vec)

        mol_vecs = ops.stack(mol_vecs, 0)
        return mol_vecs
