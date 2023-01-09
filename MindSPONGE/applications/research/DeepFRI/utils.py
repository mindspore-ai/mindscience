# Copyright 2023 Huawei Technologies Co., Ltd & CPL YiQin GAO Research Group
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""utils"""
import os
import csv
import collections

import numpy as np
import matplotlib.pyplot as plt
import mindspore as ms
import mindspore.nn as nn
import mindspore.dataset as ds
from mindspore import dtype

from Bio import SeqIO
from Bio.PDB.PDBParser import PDBParser

Annotation = collections.namedtuple('Annotation', ['a_1', 'a_2', 'a_3', 'a_4'])


class MyWithLossCell(nn.Cell):
    """
        Class for Calculating Loss.
    """

    def __init__(self, backbone):
        super(MyWithLossCell, self).__init__(auto_prefix=False)
        self._backbone = backbone
        self.loss_net = nn.SoftmaxCrossEntropyWithLogits(reduction='mean')

    def construct(self, cmap, seq, labels):
        """construct"""
        preds = self._backbone(cmap, seq)
        out = self.loss_net(preds.reshape((-1, 2)).astype(ms.float32), labels.reshape((-1, 2)).astype(ms.float32))
        return out

    def prediction(self, cmap, seq):
        return self._backbone(cmap, seq)


class DatasetGenerator:
    """
            Class for Generate Dataset.
    """

    def __init__(self, path, cmap_type='ca', ont='mf', n_go_terms=489, channels=26, cmap_thresh=10.0):
        schema = generate_gcn_schema(cmap_type, ont)
        dataset = ds.TFRecordDataset(path, schema=schema, shuffle=False)

        self.cmap = []
        self.seq = []
        self.labels = []

        self.length = dataset.get_dataset_size()
        for data in dataset.create_dict_iterator(output_numpy=True):
            dim = data['L'][0]
            adj = data[cmap_type + '_dist_matrix']
            adj = np.less_equal(adj, cmap_thresh).astype(np.float32)

            seq_1hot = data['seq_1hot'].astype(np.float32)

            labels = data[ont + '_labels'].astype(np.float32)
            inverse_labels = np.equal(labels, 0).astype(np.float32)
            y = np.stack((labels, inverse_labels), axis=-1)

            self.cmap.append(adj.reshape(dim, dim))
            self.seq.append(seq_1hot.reshape(dim, channels))
            self.labels.append(y.reshape(n_go_terms, 2))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.cmap[idx], self.seq[idx], self.labels[idx]


def datapipe(tfrecord_fn, cmap_type='ca', n_go_terms=489, cmap_thresh=10.0, channels=26, ont='mf', batch_size=64,
             pad_len=1000):
    """data pipe"""
    tfrecord = DatasetGenerator(tfrecord_fn, cmap_type, ont, n_go_terms, channels, cmap_thresh)
    dataset = ds.GeneratorDataset(tfrecord, column_names=['cmap', 'seq', 'labels'])

    dataset = dataset.shuffle(buffer_size=2000 + 3 * batch_size)
    dataset = dataset.padded_batch(batch_size, pad_info={'cmap': ([pad_len, pad_len], 0), 'seq': ([pad_len, 26], 0),
                                                         'labels': ([None, 2], 0)})
    return dataset


def plot_losses(history: dict, save_path: str):
    """plot losses"""
    save_img_path = os.path.join(save_path, 'test_model_loss.png')
    plt.switch_backend('agg')
    plt.figure()
    plt.plot(history['train_loss'], '-')
    plt.plot(history['val_loss'], '-')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(save_img_path, bbox_inches='tight')


def load_predicted_pdb(pdb_file):
    """Load predicted pdb"""
    # Generate (diagonalized) C_alpha distance matrix from a pdb file
    parser = PDBParser()
    structure = parser.get_structure(pdb_file.split('/')[-1].split('.')[0], pdb_file)
    residues = [r for r in structure.get_residues()]

    # sequence from atom lines
    records = SeqIO.parse(pdb_file, 'pdb-atom')
    seqs = [str(r.seq) for r in records]

    size = len(residues)
    distances = np.empty((size, size))
    for x in range(size):
        for y in range(size):
            one = residues[x]['CA'].get_coord()
            two = residues[y]['CA'].get_coord()
            distances[x, y] = np.linalg.norm(one - two)

    return distances, seqs[0]


def load_go_annot(filename):
    """Load GO annotations"""
    ontologies = ['mf', 'bp', 'cc']
    prot2annot = {}
    go_terms = {ontology: [] for ontology in ontologies}
    go_names = {ontology: [] for ontology in ontologies}
    with open(filename, mode='r') as tsv_file:
        reader = csv.reader(tsv_file, delimiter='\t')

        # molecular function
        next(reader, None)  # skip the headers
        go_terms[ontologies[0]] = next(reader)
        next(reader, None)  # skip the headers
        go_names[ontologies[0]] = next(reader)

        # biological process
        next(reader, None)  # skip the headers
        go_terms[ontologies[1]] = next(reader)
        next(reader, None)  # skip the headers
        go_names[ontologies[1]] = next(reader)

        # cellular component
        next(reader, None)  # skip the headers
        go_terms[ontologies[2]] = next(reader)
        next(reader, None)  # skip the headers
        go_names[ontologies[2]] = next(reader)

        next(reader, None)
        counts = {ontology: np.zeros(len(go_terms[ontology]), dtype=float) for ontology in ontologies}
        step = 0
        for row in reader:
            if step >= 10:
                break
            step += 1
            prot, prot_go_terms = row[0], row[1:]
            prot2annot[prot] = {ontology: [] for ontology in ontologies}
            for i in range(3):
                go_term_indices = [go_terms[ontologies[i]].index(go_term) for go_term in prot_go_terms[i].split(',') if
                                   go_term != '']
                try:
                    prot2annot[prot][ontologies[i]] = np.zeros(len(go_terms[ontologies[i]]))
                    prot2annot[prot][ontologies[i]][go_term_indices] = 1.0
                except KeyError:
                    print("KeyError")
                counts[ontologies[i]][go_term_indices] += 1.0

    go_annotation = Annotation(a_1=prot2annot, a_2=go_terms, a_3=go_names, a_4=counts)

    return go_annotation


def load_ec_annot(filename):
    """Load EC annotations"""
    prot2annot = {}
    with open(filename, mode='r') as tsv_file:
        reader = csv.reader(tsv_file, delimiter='\t')

        # molecular function
        next(reader, None)
        ec_numbers = {'ec': next(reader)}

        next(reader, None)
        try:
            counts = {'ec': np.zeros(len(ec_numbers['ec']), dtype=float)}
        except KeyError:
            print("KeyError")
        for row in reader:
            prot, prot_ec_numbers = row[0], row[1]
            try:
                ec_indices = [ec_numbers['ec'].index(ec_num) for ec_num in prot_ec_numbers.split(',')]
                prot2annot[prot] = {'ec': np.zeros(len(ec_numbers['ec']), dtype=np.int64)}
                prot2annot[prot]['ec'][ec_indices] = 1.0
                counts['ec'][ec_indices] += 1
            except KeyError:
                print("KeyError")

    ec_annotation = Annotation(a_1=prot2annot, a_2=ec_numbers, a_3=ec_numbers, a_4=counts)

    return ec_annotation


def norm_adj(adj, symm=True):
    """Normalize adj matrix"""
    adj += np.eye(adj.shape[1])
    if symm:
        temp = np.diag(1.0 / np.sqrt(adj.sum(axis=1)))
        adj = temp.dot(adj.dot(temp))
    else:
        adj /= adj.sum(axis=1)[:, np.newaxis]

    return adj


def seq2onehot(seq):
    """seq2onehot"""
    # Create 26-dim embedding
    chars = ['-', 'D', 'G', 'U', 'L', 'N', 'T', 'K', 'H', 'Y', 'W', 'C', 'P',
             'V', 'S', 'O', 'I', 'E', 'F', 'X', 'Q', 'A', 'B', 'Z', 'R', 'M']
    vocab_size = len(chars)
    vocab_embed = dict(zip(chars, range(vocab_size)))

    # Convert vocabulary to one-hot
    vocab_one_hot = np.zeros((vocab_size, vocab_size), dtype=int)
    for _, val in vocab_embed.items():
        vocab_one_hot[val, val] = 1

    embed_x = [vocab_embed[v] for v in seq]
    seqs_x = np.array([vocab_one_hot[j, :] for j in embed_x])

    return seqs_x


def generate_gcn_schema(cmap_type='ca', ont='mf'):
    """Generate gcn schema"""
    schema = ds.Schema()
    schema.add_column(name=cmap_type + '_dist_matrix', de_type=dtype.float32, shape=[-1])
    schema.add_column(name='seq_1hot', de_type=dtype.float32, shape=[-1])
    schema.add_column(name=ont + '_labels', de_type=dtype.int32, shape=[-1])
    schema.add_column(name='L', de_type=dtype.int32, shape=[1])

    return schema
