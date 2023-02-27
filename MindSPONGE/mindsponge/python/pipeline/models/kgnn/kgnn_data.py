# Copyright 2023 @ Shenzhen Bay Laboratory &
#                  Peking University &
#                  Huawei Technologies Co., Ltd
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
"""kgnn data"""
import os
import pickle
import stat
from collections import defaultdict

import numpy as np


def read_entity2id_file(file_path: str, drug_vocab: dict, entity_vocab: dict):
    """read_entity2id_file"""
    print(f'Logging Info - Reading entity2id file: {file_path}')
    assert not drug_vocab and not entity_vocab
    with open(file_path, encoding='utf8') as reader:
        count = 0
        for line in reader:
            if count == 0:
                count += 1
                continue
            _, entity = line.strip().split('\t')
            drug_vocab[entity] = len(drug_vocab)
            entity_vocab[entity] = len(entity_vocab)


def read_example_file(file_path: str, separator: str, drug_vocab: dict):
    """read_example_file"""
    print(f'Logging Info - Reading example file: {file_path}')
    assert drug_vocab
    examples = []
    with open(file_path, encoding='utf8') as reader:
        for _, line in enumerate(reader):
            d1, d2, flag = line.strip().split(separator)[:3]
            if d1 not in drug_vocab or d2 not in drug_vocab:
                continue
            if d1 in drug_vocab and d2 in drug_vocab:
                examples.append([drug_vocab[d1], drug_vocab[d2], int(flag)])

    examples_matrix = np.array(examples)
    print(f'size of example: {examples_matrix.shape}')
    return examples_matrix


def read_kg(file_path: str, entity_vocab: dict, relation_vocab: dict, neighbor_sample_size: int):
    """read_kg"""
    print(f'Logging Info - Reading kg file: {file_path}')

    kg = defaultdict(list)
    with open(file_path, encoding='utf8') as reader:
        count = 0
        for line in reader:
            if count == 0:
                count += 1
                continue
            head, tail, relation = line.strip().split(' ')

            if head not in entity_vocab:
                entity_vocab[head] = len(entity_vocab)
            if tail not in entity_vocab:
                entity_vocab[tail] = len(entity_vocab)
            if relation not in relation_vocab:
                relation_vocab[relation] = len(relation_vocab)

            # undirected graph
            kg[entity_vocab[head]].append((entity_vocab[tail], relation_vocab[relation]))
            kg[entity_vocab[tail]].append((entity_vocab[head], relation_vocab[relation]))
    print(f'Logging Info - num of entities: {len(entity_vocab)}, '
          f'num of relations: {len(relation_vocab)}')

    print('Logging Info - Constructing adjacency matrix...')
    n_entity = len(entity_vocab)
    adj_entity = np.zeros(shape=(n_entity, neighbor_sample_size), dtype=np.int64)
    adj_relation = np.zeros(shape=(n_entity, neighbor_sample_size), dtype=np.int64)

    np.random.seed(0)
    for entity_id in range(n_entity):
        all_neighbors = kg[entity_id]
        n_neighbor = len(all_neighbors)
        sample_indices = np.random.choice(
            n_neighbor,
            neighbor_sample_size,
            replace=n_neighbor < neighbor_sample_size
        )

        adj_entity[entity_id] = np.array([all_neighbors[i][0] for i in sample_indices])
        adj_relation[entity_id] = np.array([all_neighbors[i][1] for i in sample_indices])

    return adj_entity, adj_relation


def pickle_dump(filename: str, obj):
    """save python object to pickle file"""
    flags = os.O_WRONLY | os.O_CREAT
    modes = stat.S_IWUSR | stat.S_IRUSR
    with os.fdopen(os.open(filename, flags, modes), 'wb') as fout:
        pickle.dump(obj, fout)
    print(f'Logging Info - Saved: {filename}')


def format_filename(base_dir: str, filename_template: str, **kwargs):
    """Obtain the filename of data based on the provided template and parameters"""
    filename = os.path.join(base_dir, filename_template.format(**kwargs))
    return filename


def pickle_load(filename: str):
    """load pickle file"""
    try:
        flags = os.O_RDONLY
        modes = stat.S_IWUSR | stat.S_IRUSR
        with os.fdopen(os.open(filename, flags, modes), 'rb') as fout:
            obj = pickle.load(fout)
        print(f'Logging Info - Loaded: {filename}')
    except EOFError:
        print(f'Logging Error - Cannot load: {filename}')
        obj = None

    return obj
