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
"""jtmpn"""
import numpy as np
import rdkit.Chem as Chem
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from .nnutils import index_select_nd
from .utils import for_stack


ELEM_LIST = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'Al', 'I', 'B', 'K', 'Se', 'Zn',
             'H', 'Cu', 'Mn', 'unknown']

ATOM_FDIM = len(ELEM_LIST) + 6 + 5 + 1
BOND_FDIM = 5
MAX_NB = 10


def onek_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def atom_features(atom):
    return ms.Tensor(onek_encoding_unk(atom.GetSymbol(), ELEM_LIST)
                     + onek_encoding_unk(atom.GetDegree(), [0, 1, 2, 3, 4, 5])
                     + onek_encoding_unk(atom.GetFormalCharge(), [-1, -2, 1, 2, 0])
                     + [atom.GetIsAromatic()], ms.float32)


def bond_features(bond):
    bt = bond.GetBondType()
    return ms.Tensor([bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE,
                      bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC,
                      bond.IsInRing()], ms.float32)


class JTMPN(nn.Cell):
    """jtmpn"""

    def __init__(self, hidden_size, depth):
        super(JTMPN, self).__init__()
        self.hidden_size = hidden_size
        self.depth = depth

        self.w_i = nn.Dense(ATOM_FDIM + BOND_FDIM, hidden_size, has_bias=False)
        self.w_h = nn.Dense(hidden_size, hidden_size, has_bias=False)
        self.w_o = nn.Dense(ATOM_FDIM + hidden_size, hidden_size)

        self.relu = nn.ReLU()

    @staticmethod
    def update_mess(tree_mess, all_mess, mess_dict):
        for e, h in tree_mess:
            mess_dict[e] = len(all_mess)
            all_mess.append(h)
        return mess_dict, all_mess

    def construct(self, cand_batch, tree_mess):
        """construct"""
        fatoms, fbonds = [], []
        in_bonds, all_bonds = [], []
        mess_dict, all_mess = {}, [ops.zeros(self.hidden_size, ms.float32)]
        total_atoms = 0
        scope = []
        mess_dict, all_mess = self.update_mess(tree_mess, all_mess, mess_dict)

        for mol, all_nodes, _ in cand_batch:
            n_atoms = mol.GetNumAtoms()

            for atom in mol.GetAtoms():
                fatoms.append(atom_features(atom))
                in_bonds.append([])

            for bond in mol.GetBonds():
                a1 = bond.GetBeginAtom()
                a2 = bond.GetEndAtom()
                x = a1.GetIdx() + total_atoms
                y = a2.GetIdx() + total_atoms
                # Here x_nid,y_nid could be 0
                x_nid, y_nid = a1.GetAtomMapNum(), a2.GetAtomMapNum()
                x_bid = all_nodes[x_nid - 1].idx if x_nid > 0 else -1
                y_bid = all_nodes[y_nid - 1].idx if y_nid > 0 else -1

                bfeature = bond_features(bond)

                b = len(all_mess) + len(all_bonds)
                all_bonds.append((x, y))
                fbonds.append(ops.concat([fatoms[x], bfeature], 0))
                in_bonds[y].append(b)

                b = len(all_mess) + len(all_bonds)
                all_bonds.append((y, x))
                fbonds.append(ops.concat([fatoms[y], bfeature], 0))
                in_bonds[x].append(b)

                if x_bid >= 0 and y_bid >= 0 and x_bid != y_bid:
                    if (x_bid, y_bid) in mess_dict:
                        mess_idx = mess_dict[(x_bid, y_bid)]
                        in_bonds[y].append(mess_idx)
                    if (y_bid, x_bid) in mess_dict:
                        mess_idx = mess_dict[(y_bid, x_bid)]
                        in_bonds[x].append(mess_idx)

            scope.append((total_atoms, n_atoms))
            total_atoms += n_atoms

        total_bonds = len(all_bonds)
        total_mess = len(all_mess)
        fatoms = for_stack(fatoms, 0)
        fbonds = for_stack(fbonds, 0)
        tree_message = for_stack(all_mess, 0)
        agraph = np.zeros((total_atoms, MAX_NB), np.int32)
        bgraph = np.zeros((total_bonds, MAX_NB), np.int32)

        for a in range(total_atoms):
            for i, b in enumerate(in_bonds[a]):
                agraph[a, i] = b
        agraph = ms.Tensor(agraph, ms.int32)

        for b1 in range(total_bonds):
            x, y = all_bonds[b1]
            for i, b2 in enumerate(in_bonds[x]):
                if b2 < total_mess or all_bonds[b2 - total_mess][0] != y:
                    bgraph[b1, i] = b2
        bgraph = ms.Tensor(bgraph, ms.int32)

        binput = self.w_i(fbonds)
        graph_message = self.relu(binput)

        for i in range(self.depth - 1):
            message = ops.concat([tree_message, graph_message], 0)
            nei_message = index_select_nd(message, 0, bgraph)
            nei_message = nei_message.sum(axis=1)
            nei_message = self.w_h(nei_message)
            graph_message = self.relu(binput + nei_message)

        message = ops.concat([tree_message, graph_message], 0)
        nei_message = index_select_nd(message, 0, agraph)
        nei_message = nei_message.sum(axis=1)
        ainput = ops.concat([fatoms, nei_message], 1)
        atom_hiddens = self.relu(self.w_o(ainput))

        mol_vecs = []
        for st, le in scope:
            mol_vec = atom_hiddens.narrow(0, st, le).sum(axis=0) / le
            mol_vecs.append(mol_vec)

        mol_vecs = for_stack(mol_vecs, 0)
        return mol_vecs
