# Copyright 2023 Huawei Technologies Co., Ltd
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
"""docking data process script. generate data for training or inference"""

import os
import stat
from collections import OrderedDict
import json
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
import networkx as nx


ATOM_TYPE = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K',
             'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In',
             'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'Unknown']


def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(), ATOM_TYPE) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def smile_to_graph(input_smile):
    """extract features from smile to graph"""
    mol = Chem.MolFromSmiles(input_smile)

    c_size = mol.GetNumAtoms()

    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g_data = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g_data.edges:
        edge_index.append([e1, e2])

    return c_size, features, edge_index


def seq_cat(prot, max_seq_len, seq_dict):
    x = np.zeros(max_seq_len)
    for i, ch in enumerate(prot[:max_seq_len]):
        x[i] = seq_dict[ch]
    return x


def create_prot_csv(input_datasets):
    """extract data info from raw data and save to csv files"""

    for d_name in input_datasets:
        print('convert data from DeepDTA for ', d_name)
        fpath = 'data/' + d_name + '/'
        train_fold = json.load(open(fpath + "folds/train_fold_setting1.txt"))
        train_fold = [ee for e in train_fold for ee in e]
        valid_fold = json.load(open(fpath + "folds/test_fold_setting1.txt"))
        ligands = json.load(open(fpath + "ligands_can.txt"), object_pairs_hook=OrderedDict)
        proteins = json.load(open(fpath + "proteins.txt"), object_pairs_hook=OrderedDict)
        affinity = pickle.load(open(fpath + "Y", "rb"), encoding='latin1')
        drug_t = []
        prot_t = []
        for d in ligands.keys():
            lg = Chem.MolToSmiles(Chem.MolFromSmiles(ligands[d]), isomericSmiles=True)
            drug_t.append(lg)
        for t in proteins.keys():
            prot_t.append(proteins[t])
        if d_name == 'davis':
            affinity = [-np.log10(y / 1e9) for y in affinity]
        affinity = np.asarray(affinity)
        op = ['train', 'test']
        for op_t in op:
            rows, cols = np.where(np.isnan(affinity) is False)
            if op_t == 'train':
                rows, cols = rows[train_fold], cols[train_fold]
            elif op_t == 'test':
                rows, cols = rows[valid_fold], cols[valid_fold]
            with os.fdopen(os.open('data/{d_name}/{d_name}_{op_t}.csv', os.O_CREAT, stat.S_IWUSR), 'w') as f:
                f.write('compound_iso_smiles,target_sequence,affinity\n')
                row_list = list(range(len(rows)))
                for pair_ind in row_list:
                    ls = []
                    ls += [drug_t[rows[pair_ind]]]
                    ls += [prot_t[cols[pair_ind]]]
                    ls += [affinity[rows[pair_ind], cols[pair_ind]]]
                    f.write(','.join(map(str, ls)) + '\n')
        print('\ndataset:', d_name)
        print('train_fold:', len(train_fold))
        print('test_fold:', len(valid_fold))
        print('len(set(drugs)),len(set(prots)):', len(set(drug_t)), len(set(prot_t)))


def process_data(xd, xt, y, smile_graph_info):
    """save data into pickle file"""
    if not len(xd) == len(xt) == len(y):
        raise ValueError("The three lists must be the same length!")
    data_len = len(xd)
    res = []
    for i in tqdm(range(data_len), "generating features"):
        smiles = xd[i]
        target = xt[i]
        labels = y[i]

        # convert SMILES to molecular representation using rdkit
        c_size, features, edge_index = smile_graph_info[smiles]

        res_t = {"x": np.array(features),
                 "y": np.array([labels]),
                 "edge_index": np.array(edge_index).transpose(1, 0),
                 "target": np.array([target]),
                 "num_nodes": np.array([c_size])}
        res.append(res_t)
    return res


def generate_feature(data_path):
    """generate feature"""
    print(f"start preprocessing {data_path}:")

    seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
    max_seq_len = 1000
    seq_dict = {v: (i + 1) for i, v in enumerate(seq_voc)}

    df_data = pd.read_csv(data_path)
    drugs = list(df_data['compound_iso_smiles'])
    compound_iso_smiles_set = set(drugs)
    smile_graph = {}
    for smile in tqdm(compound_iso_smiles_set, "extracting smiles to graph"):
        g = smile_to_graph(smile)
        smile_graph[smile] = g

    prots = list(df_data['target_sequence'])
    if "affinity" not in df_data:
        y = np.zeros(len(drugs))
    else:
        y = list(df_data['affinity'])
    xt = [seq_cat(t, max_seq_len, seq_dict) for t in prots]
    drugs, prots, y = np.asarray(drugs), np.asarray(xt), np.asarray(y)

    feature = process_data(drugs, prots, y, smile_graph)
    return feature
