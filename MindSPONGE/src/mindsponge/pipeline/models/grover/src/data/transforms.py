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
Preprocess dataset.
"""
import math
import numpy as np
import mindspore as ms
from rdkit import Chem
from ..data.molgraph import mol2graph
from ..data.task_labels import atom_to_vocab, bond_to_vocab
from ..data.scaler import StandardScaler


class GroverCollator:
    """
        Collator for pretrain dataloader.
        :param shared_dict: a shared dict of multiprocess.
        :param args: Arguments.
        :param atom_vocab:
        :param bond_vocab:atom and bond vocabulary
    """

    def __init__(self, shared_dict, atom_vocab, bond_vocab, args):
        self.args = args
        self.shared_dict = shared_dict
        self.atom_vocab = atom_vocab
        self.bond_vocab = bond_vocab

    def atom_random_mask(self, smiles_batch):
        """
        Perform the random mask operation on atoms.
        :param smiles_batch:
        :return: The corresponding atom labels.
        """
        # There is a zero padding.
        vocab_label = [0]
        percent = 0.15
        for smi in smiles_batch:
            mol = Chem.MolFromSmiles(smi)
            mlabel = [0] * mol.GetNumAtoms()
            n_mask = math.ceil(mol.GetNumAtoms() * percent)
            perm = np.random.permutation(mol.GetNumAtoms())[:n_mask]
            for p in perm:
                atom = mol.GetAtomWithIdx(int(p))
                mlabel[p] = self.atom_vocab.stoi.get(atom_to_vocab(mol, atom), self.atom_vocab.other_index)

            vocab_label.extend(mlabel)
        return vocab_label

    def bond_random_mask(self, smiles_batch):
        """
        Perform the random mask operation on bonds.
        :param smiles_batch:
        :return: The corresponding bond labels.
        """
        # There is a zero padding.
        vocab_label = [0]
        percent = 0.15
        for smi in smiles_batch:
            mol = Chem.MolFromSmiles(smi)
            nm_atoms = mol.GetNumAtoms()
            nm_bonds = mol.GetNumBonds()
            mlabel = []
            n_mask = math.ceil(nm_bonds * percent)
            perm = np.random.permutation(nm_bonds)[:n_mask]
            virtual_bond_id = 0
            for a1 in range(nm_atoms):
                for a2 in range(a1 + 1, nm_atoms):
                    bond = mol.GetBondBetweenAtoms(a1, a2)

                    if bond is None:
                        continue
                    if virtual_bond_id in perm:
                        label = self.bond_vocab.stoi.get(bond_to_vocab(mol, bond), self.bond_vocab.other_index)
                        mlabel.extend([label])
                    else:
                        mlabel.extend([0])

                    virtual_bond_id += 1
            # todo: might need to consider bond_drop_rate
            # todo: double check reverse bond
            vocab_label.extend(mlabel)
        return vocab_label

    def per_batch_map(self, smiles_batch, features_batch):
        """
        Build the chem structure.
        """
        graph = mol2graph(smiles_batch, self.shared_dict, self.args)
        f_atoms, f_bonds, a2b, b2a, b2revb, a2a, a_scope, b_scope = graph.get_components()
        atom_vocab_label = np.array(self.atom_random_mask(smiles_batch), dtype=np.int32)
        bond_vocab_label = np.array(self.bond_random_mask(smiles_batch), dtype=np.int32)
        fgroup_label = np.array(features_batch, dtype=np.float32)

        outputs = f_atoms, f_bonds, a2b, b2a, b2revb, a2a, a_scope, b_scope, \
                  atom_vocab_label, bond_vocab_label, fgroup_label
        return outputs


class MolCollator:
    """
    Collator for train/eval dataloader
    :param shared_dict: a shared dict of multiprocess.
    :param args: Arguments.
    """

    def __init__(self, shared_dict, args):
        self.args = args
        self.shared_dict = shared_dict

    def per_batch_map(self, smiles_batch):
        graph = mol2graph(smiles_batch, self.shared_dict, self.args)
        f_atoms, f_bonds, a2b, b2a, b2revb, a2a, a_scope, b_scope = graph.get_components()
        outputs = f_atoms, f_bonds, a2b, b2a, b2revb, a2a, \
                  a_scope, b_scope, smiles_batch
        return outputs


def normalize_data(data_list, replace_nan_token, is_training, path):
    """
    Normalizes the data of the dataset using a StandardScaler (subtract mean, divide by standard deviation).

    If a scaler is provided, uses that scaler to perform the normalization. Otherwise fits a scaler to the
    features in the dataset and then performs the normalization.

    :param scaler: A fitted StandardScaler. Used if provided. Otherwise a StandardScaler is fit on
    this dataset and is then used.
    :param replace_nan_token: What to replace nans with.
    :return:
    """
    if not is_training:
        state = ms.load_checkpoint(path)
        scaler = StandardScaler(state['means'].asnumpy(),
                                state['stds'].asnumpy(),
                                replace_nan_token=replace_nan_token)

    else:
        scaler = StandardScaler(replace_nan_token=replace_nan_token)
        scaler = scaler.fit(data_list)

        state = [
            {'name': "means", 'data': ms.Tensor(scaler.means)},
            {'name': "stds", 'data': ms.Tensor(scaler.stds)}
        ]
        ms.save_checkpoint(state, path)

    normal_data = scaler.transform(data_list).tolist()
    return normal_data, scaler
