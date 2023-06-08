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
"""kgnn dataset"""
import os
import numpy as np
from sklearn.model_selection import train_test_split
from mindspore.dataset import GeneratorDataset
from ...dataset import DataSet
from .kgnn_data import read_entity2id_file, read_example_file, read_kg, pickle_dump, format_filename

SEPARATOR = {'drug': '\t', 'kegg': '\t'}
DRUG_VOCAB_TEMPLATE = '{dataset}_drug_vocab.pkl'
ENTITY_VOCAB_TEMPLATE = '{dataset}_entity_vocab.pkl'
RELATION_VOCAB_TEMPLATE = '{dataset}_relation_vocab.pkl'
ADJ_ENTITY_TEMPLATE = '{dataset}_adj_entity.npy'
ADJ_RELATION_TEMPLATE = '{dataset}_adj_relation.npy'
DRUG_EXAMPLE = '{dataset}_{type}_examples.npy'


class KGNNDataSet(DataSet):
    """kgnn dataset"""
    def __init__(self, config):
        self.config = config
        self.dataset_url = "https://github.com/xzenglab/KGNN/tree/master/raw_data/kegg"
        self.train_data = None
        super().__init__()


    def __getitem__(self, idx):
        data = self.data_parse(idx)
        return data[:2], data[2]


    def __len__(self):
        return len(self.train_data)


    def download(self, path=None):
        if path is None:
            print(f"KGNN's dataset can be downloaded from {self.dataset_url}")
        else:
            print(f"KGNN's dataset can be downloaded from {self.dataset_url}")
            print(f"{path} can be used in method set_training_data_src to set the raw data path.")

    # pylint: disable=arguments-differ
    def process(self, data):
        return data


    def data_parse(self, idx):
        data = self.train_data[idx]
        return data


    def set_training_data_src(self, data_src=None):
        """set training data source path"""
        if not os.path.exists(os.path.join(data_src, self.config.dataset, 'train2id.txt')):
            raise IOError("train2id.txt doesn't exit!")
        if not os.path.exists(os.path.join(data_src, self.config.dataset, 'entity2id.txt')):
            raise IOError("train2id.txt doesn't exit!")
        if not os.path.exists(os.path.join(data_src, self.config.dataset, 'approved_example.txt')):
            raise IOError("train2id.txt doesn't exit!")

        entity2id_file_path = os.path.join(data_src, self.config.dataset, "entity2id.txt")
        kg_file_path = os.path.join(data_src, self.config.dataset, "train2id.txt")
        example_file_path = os.path.join(data_src, self.config.dataset, "approved_example.txt")

        drug_vocab = {}
        entity_vocab = {}
        relation_vocab = {}

        read_entity2id_file(entity2id_file_path, drug_vocab, entity_vocab)
        pickle_dump(format_filename(self.config.DATA_DIR, DRUG_VOCAB_TEMPLATE,
                                    dataset=self.config.dataset), drug_vocab)
        pickle_dump(format_filename(self.config.DATA_DIR, ENTITY_VOCAB_TEMPLATE,
                                    dataset=self.config.dataset), entity_vocab)

        train_examples_file = format_filename(self.config.DATA_DIR, DRUG_EXAMPLE,
                                              dataset=self.config.dataset, type="train")
        test_examples_file = format_filename(self.config.DATA_DIR, DRUG_EXAMPLE,
                                             dataset=self.config.dataset, type="test")
        examples = read_example_file(example_file_path, SEPARATOR.get(self.config.dataset), drug_vocab)
        x = examples[:, :2]
        y = examples[:, 2:3]
        train_data_x, test_data_x, train_y, test_y = train_test_split(x, y, test_size=0.2, stratify=y, random_state=0)
        train_data = np.c_[train_data_x, train_y]
        test_data = np.c_[test_data_x, test_y]
        np.save(train_examples_file, train_data)
        np.save(test_examples_file, test_data)
        if self.config.is_training:
            self.train_data = train_data
        else:
            self.train_data = test_data

        adj_entity, adj_relation = read_kg(kg_file_path, entity_vocab, relation_vocab,
                                           self.config.neighborsize)

        pickle_dump(format_filename(self.config.DATA_DIR, DRUG_VOCAB_TEMPLATE, dataset=self.config.dataset),
                    drug_vocab)
        pickle_dump(format_filename(self.config.DATA_DIR, ENTITY_VOCAB_TEMPLATE, dataset=self.config.dataset),
                    entity_vocab)
        pickle_dump(format_filename(self.config.DATA_DIR, RELATION_VOCAB_TEMPLATE, dataset=self.config.dataset),
                    relation_vocab)
        adj_entity_file = format_filename(self.config.DATA_DIR, ADJ_ENTITY_TEMPLATE, dataset=self.config.dataset)
        np.save(adj_entity_file, adj_entity)
        print('Logging Info - Saved:', adj_entity_file)

        adj_relation_file = format_filename(self.config.DATA_DIR, ADJ_RELATION_TEMPLATE, dataset=self.config.dataset)
        np.save(adj_relation_file, adj_relation)
        print('Logging Info - Saved:', adj_relation_file)

    # pylint: disable=arguments-differ
    def create_iterator(self, num_epochs):
        dataset = GeneratorDataset(source=self, column_names=["data", "label"],
                                   num_parallel_workers=self.config.num_parallel_workers,
                                   shuffle=self.config.is_training, num_shards=self.config.rank_size,
                                   shard_id=self.config.rank_id)
        dataset = dataset.batch(batch_size=self.config.batch_size, drop_remainder=True)
        iteration = dataset.create_dict_iterator(num_epochs=num_epochs, output_numpy=True)
        return iteration
