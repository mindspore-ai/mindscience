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
"""deepfridata"""
import numpy as np

from Bio import SeqIO
from Bio.PDB.PDBParser import PDBParser

from ...dataset import curry1
from ....common import residue_constants


@curry1
def dict_filter_key(feature, feature_list):
    feature = {k: v for k, v in feature.items() if k in feature_list}
    return feature


@curry1
def dict_replace_key(feature, replaced_key):
    if len(replaced_key) != 2:
        raise ValueError("'replaced_key' should equal to 2")
    origin_key, new_key = replaced_key
    if origin_key in feature:
        feature[new_key] = feature.pop(origin_key)
    return feature


@curry1
def dict_cast(feature, cast_type, filtered_list):
    """dict cast"""
    if len(cast_type) != 2:
        raise ValueError("'cast_type' should equal to 2")
    origin_type = cast_type[0]
    new_type = cast_type[1]
    for k, v in feature.items():
        if k not in filtered_list:
            if v.dtype == origin_type:
                feature[k] = v.astype(new_type)
    return feature


@curry1
def dict_suqeeze(feature, filter_list, axis):
    for k in filter_list:
        if k in feature:
            dims = feature[k].shape[axis]
            if isinstance(dims, int) and dims == 1:
                feature[k] = np.squeeze(feature[k], axis=axis)
    return feature


@curry1
def dict_take(feature, filter_list, axis):
    for k in filter_list:
        if k in feature:
            feature[k] = feature[k][axis]
    return feature


@curry1
def dict_del_key(feature, filter_list):
    for k in filter_list:
        if k in feature:
            del feature[k]
    return feature


@curry1
def one_hot_convert(feature, key, axis):
    if key in feature:
        feature[key] = np.argmax(feature[key], axis=axis)
    return feature


@curry1
def correct_restypes(feature, key):
    new_order_list = residue_constants.MAP_HHBLITS_AATYPE_TO_OUR_AATYPE
    new_order = np.array(new_order_list, dtype=feature[key].dtype)
    feature[key] = new_order[feature[key]]
    return feature


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


@curry1
def load_cmap(cmap=None):
    """load cmap"""

    dis, seq = load_predicted_pdb(cmap)
    adj = np.double(dis < 10.0)
    one_hot = seq2onehot(seq)
    one_hot = one_hot.reshape(1, *one_hot.shape)
    adj = adj.reshape(1, *adj.shape)
    temp = []
    temp.append(adj)
    temp.append(one_hot)
    return temp
