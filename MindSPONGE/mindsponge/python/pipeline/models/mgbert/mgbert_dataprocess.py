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
"""mgbert"""
import numpy as np

import pandas as pd
import mindspore.dataset as ds

from .utils import smiles2adjoin
from ...dataset import PSP

_msa_feature_names = ['msa', 'deletion_matrix', 'msa_mask', 'msa_row_mask', 'bert_mask', 'true_msa', 'msa_input']
str2num = {'<pad>': 0, 'H': 1, 'C': 2, 'N': 3, 'O': 4, 'F': 5, 'S': 6, 'Cl': 7, 'P': 8, 'Br': 9,
           'B': 10, 'I': 11, 'Si': 12, 'Se': 13, '<unk>': 14, '<mask>': 15, '<global>': 16}

num2str = {i: j for j, i in str2num.items()}


class MGBertDataSet(PSP):
    '''MEGAEvoGenDataSet'''

    def __init__(self, config):
        self.task_name = config.task_name
        if config.task_name == 'classification':
            self.smiles_field = config.smiles_field
            self.label_field = config.label_field
            self.vocab = str2num
            self.devocab = num2str
            self.max_len = config.max_len
            self.addh = config.addH
            self.normalize = config.normalize
            self.y_dtype = 'int64'
            self.batch = config.batch
            self.is_train = config.is_train
            self.is_test = config.is_test
            self.is_val = config.is_val
        elif config.task_name == 'regression':
            self.smiles_field = config.smiles_field
            self.label_field = config.label_field
            self.vocab = str2num
            self.devocab = num2str
            self.max_len = config.max_len
            self.addh = config.addH
            self.y_dtype = 'float32'
            self.normalize = config.normalize
            self.batch = config.batch
            self.is_train = config.is_train
            self.is_test = config.is_test
            self.is_val = config.is_val
        else:
            self.smiles_field = config.smiles_field
            self.vocab = str2num
            self.devocab = num2str
            self.addh = config.addH
            self.y_dtype = 'float32'
            self.normalize = config.normalize
            self.batch = config.batch
        self.df = None
        self.max = None
        self.min = None
        self.value_range = None
        self.dataset = None

        super().__init__()

    # pylint: disable=unnecessary-pass
    def __len__(self):
        pass

    # pylint: disable=arguments-differ, unnecessary-pass
    def __getitem__(self, idx):
        pass

    @staticmethod
    def numerical_smiles(smiles):
        """Numerical smiles"""
        atoms_list, adjoin_matrix = smiles2adjoin(smiles, explicit_hydrogens=True)
        atoms_list = ['<global>'] + atoms_list
        nums_list = [str2num.get(i, str2num.get('<unk>', "<unk>")) for i in atoms_list]
        temp = np.ones((len(nums_list), len(nums_list)))
        temp[1:, 1:] = adjoin_matrix
        adjoin_matrix = (1 - temp) * (-1e9)

        choices = np.random.permutation(len(nums_list) - 1)[:max(int(len(nums_list) * 0.15), 1)] + 1
        y = np.array(nums_list).astype('int64')
        weight = np.zeros(len(nums_list))
        for i in choices:
            rand = np.random.rand()
            weight[i] = 1
            if rand < 0.8:
                nums_list[i] = str2num.get('<mask>', "<mask>")
            elif rand < 0.9:
                nums_list[i] = int(np.random.rand() * 14 + 1)

        x = np.array(nums_list).astype('int64')
        weight = weight.astype('float32')
        output = [x, adjoin_matrix, y, weight]
        return output

    def set_training_data_src(self, path):
        "set_training_data_src"
        if path.endswith('.txt') or path.endswith('.csv'):
            self.df = pd.read_csv(path, sep='\t')
        else:
            self.df = pd.read_csv(path)
        if self.task_name != 'pretrain':
            self.df = self.df[self.df[self.smiles_field].str.len() <= self.max_len]
        if self.normalize:
            self.max = self.df[self.label_field].max()
            self.min = self.df[self.label_field].min()
            self.df[self.label_field] = (self.df[self.label_field] - self.min) / (self.max - self.min) - 0.5
            self.value_range = self.max - self.min

    # pylint: disable=arguments-differ
    def ms_numerical_smiles(self, smiles, label=None):
        """Mindspore numerical smiles for Graph Classification Dataset"""
        if label is not None:
            x, adjoin_matrix, y = [], [], []
            for item1, item2 in zip(smiles, label):
                atoms_list, adjoin_matrix_item = smiles2adjoin(item1, explicit_hydrogens=self.addh)
                atoms_list = ['<global>'] + atoms_list
                nums_list = [str2num.get(i, str2num.get('<unk>', '<unk>')) for i in atoms_list]
                temp = np.ones((len(nums_list), len(nums_list)))
                temp[1:, 1:] = adjoin_matrix_item
                adjoin_matrix_item = (1 - temp) * (-1e9)
                x_item = np.array(nums_list).astype('int64')
                y_item = np.array([item2]).astype('float32')

                x.append(x_item)
                adjoin_matrix.append(adjoin_matrix_item)
                y.append(y_item)
            max_len_x = max([len(i) for i in x])
            max_len_a = max([len(i) for i in adjoin_matrix])
            if max_len_x == max_len_a:
                x = np.array(
                    [np.pad(x[i], (0, max_len_x - len(x[i])), 'constant', constant_values=0) for i in
                     range(len(x))]).astype(np.int32)
                adjoin_matrix = np.array(
                    [np.pad(adjoin_matrix[i], (0, max_len_a - len(adjoin_matrix[i])), 'constant', constant_values=0) for
                     i
                     in range(len(adjoin_matrix))]).astype(np.float32)
                y = np.array(y, dtype=np.float32)
                y = y.reshape(y.shape[0] * y.shape[1])
                return x, adjoin_matrix, y
        else:
            x, adjoin_matrix, y, weight = [], [], [], []
            for item in smiles:
                x_item, adjoin_matrix_item, y_item, weight_item = self.numerical_smiles(item)
                x.append(x_item)
                adjoin_matrix.append(adjoin_matrix_item)
                y.append(y_item)
                weight.append(weight_item)
            max_len_x = max([len(i) for i in x])
            max_len_a = max([len(i) for i in adjoin_matrix])
            max_len_y = max([len(i) for i in y])
            max_len_w = max([len(i) for i in weight])
            if max_len_x == max_len_a == max_len_y == max_len_w:
                x = np.array(
                    [np.pad(x[i], (0, max_len_x - len(x[i])), 'constant', constant_values=0) for i in
                     range(len(x))]).astype(np.int32)
                adjoin_matrix = np.array(
                    [np.pad(adjoin_matrix[i], (0, max_len_a - len(adjoin_matrix[i])), 'constant', constant_values=0) for
                     i
                     in range(len(adjoin_matrix))]).astype(np.float32)
                y = np.array(
                    [np.pad(y[i], (0, max_len_y - len(y[i])), 'constant', constant_values=0) for i in range(len(y))])
                weight = np.array(
                    [np.pad(weight[i], (0, max_len_w - len(weight[i])), 'constant', constant_values=0) for i in
                     range(len(weight))])
                output = (x, adjoin_matrix, y, weight)
                return output
        raise ValueError("Inconsistent data shapes")

    # pylint: disable=arguments-differ
    def create_iterator(self, num_epochs):
        '''create_iterator'''
        data = self.df
        lengths = [0, 25, 50, 75, 100]
        if self.task_name == 'classification':
            data = data.dropna()
            data[self.label_field] = data[self.label_field].map(int)
            pdata = data[data[self.label_field] == 1]
            ndata = data[data[self.label_field] == 0]

            ptrain_idx = []
            for i in range(4):
                idx = pdata[(pdata[self.smiles_field].str.len() >= lengths[i]) & (
                    pdata[self.smiles_field].str.len() < lengths[i + 1])].sample(frac=0.8).index
                ptrain_idx.extend(idx)

            ntrain_idx = []
            for i in range(4):
                idx = ndata[(ndata[self.smiles_field].str.len() >= lengths[i]) & (
                    ndata[self.smiles_field].str.len() < lengths[i + 1])].sample(frac=0.8).index
                ntrain_idx.extend(idx)

            train_data = data[data.index.isin(ptrain_idx + ntrain_idx)]
            pdata = pdata[~pdata.index.isin(ptrain_idx)]
            ndata = ndata[~ndata.index.isin(ntrain_idx)]

            ptest_idx = []
            for i in range(4):
                idx = pdata[(pdata[self.smiles_field].str.len() >= lengths[i]) & (
                    pdata[self.smiles_field].str.len() < lengths[i + 1])].sample(frac=0.5).index
                ptest_idx.extend(idx)

            ntest_idx = []
            for i in range(4):
                idx = ndata[(ndata[self.smiles_field].str.len() >= lengths[i]) & (
                    ndata[self.smiles_field].str.len() < lengths[i + 1])].sample(frac=0.5).index
                ntest_idx.extend(idx)

            test_data = data[data.index.isin(ptest_idx + ntest_idx)]
            val_data = data[~data.index.isin(ptest_idx + ntest_idx + ptrain_idx + ntrain_idx)]

            if self.is_train:
                self.dataset = self.ms_numerical_smiles(train_data[self.smiles_field], train_data[self.label_field])
                self.dataset = ds.NumpySlicesDataset(self.dataset, column_names=["x", "adjoin_matrix", "y"],
                                                     shuffle=True)
                self.dataset = self.dataset.batch(self.batch, drop_remainder=True)

            elif self.is_test:
                self.dataset = self.ms_numerical_smiles(test_data[self.smiles_field], test_data[self.label_field])
                self.dataset = ds.NumpySlicesDataset(self.dataset, column_names=["x", "adjoin_matrix", "y"],
                                                     shuffle=False)
                self.dataset = self.dataset.batch(self.batch)

            elif self.is_val:
                self.dataset = self.ms_numerical_smiles(val_data[self.smiles_field], val_data[self.label_field])
                self.dataset = ds.NumpySlicesDataset(self.dataset, column_names=["x", "adjoin_matrix", "y"],
                                                     shuffle=False)
                self.dataset = self.dataset.batch(self.batch)


        elif self.task_name == 'regression':
            train_idx = []
            for i in range(4):
                idx = data[(data[self.smiles_field].str.len() >= lengths[i]) & (
                    data[self.smiles_field].str.len() < lengths[i + 1])].sample(frac=0.8).index
                train_idx.extend(idx)

            train_data = data[data.index.isin(train_idx)]
            data = data[~data.index.isin(train_idx)]

            test_idx = []
            for i in range(4):
                idx = data[(data[self.smiles_field].str.len() >= lengths[i]) & (
                    data[self.smiles_field].str.len() < lengths[i + 1])].sample(frac=0.5).index
                test_idx.extend(idx)

            test_data = data[data.index.isin(test_idx)]
            val_data = data[~data.index.isin(test_idx)]

            if self.is_train:
                self.dataset = self.ms_numerical_smiles(train_data[self.smiles_field], train_data[self.label_field])
                self.dataset = ds.NumpySlicesDataset(self.dataset, column_names=["x", "adjoin_matrix", "y"],
                                                     shuffle=True)
                self.dataset = self.dataset.batch(self.batch, drop_remainder=True)

            elif self.is_test:
                self.dataset = self.ms_numerical_smiles(test_data[self.smiles_field], test_data[self.label_field])
                self.dataset = ds.NumpySlicesDataset(self.dataset, column_names=["x", "adjoin_matrix", "y"],
                                                     shuffle=False)
                self.dataset = self.dataset.batch(self.batch)
                self.dataset = [self.dataset, self.value_range]

            elif self.is_val:
                self.dataset = self.ms_numerical_smiles(val_data[self.smiles_field], val_data[self.label_field])
                self.dataset = ds.NumpySlicesDataset(self.dataset, column_names=["x", "adjoin_matrix", "y"],
                                                     shuffle=False)
                self.dataset = self.dataset.batch(self.batch)
                self.dataset = [self.dataset, self.value_range]

        else:
            train_idx = []
            idx = data.sample(frac=0.9).index
            train_idx.extend(idx)
            data1 = data[data.index.isin(train_idx)]

            self.dataset = self.ms_numerical_smiles(data1[self.smiles_field].tolist())
            column_names = ["x", "adjoin_matrix", "y", "weight"]
            self.dataset = ds.NumpySlicesDataset(self.dataset, column_names=column_names,
                                                 shuffle=True)
            self.dataset = self.dataset.batch(self.batch, drop_remainder=True)

        self.dataset = self.dataset.repeat(num_epochs)

        ds.config.set_prefetch_size(100)
        return self.dataset

    # pylint: disable=unnecessary-pass
    def data_parse(self, idx):
        '''data_parse'''
        pass

    # pylint: disable=unnecessary-pass
    def process(self, data, **kwargs):
        '''process'''
        if data.endswith('.txt') or data.endswith('.csv'):
            self.df = pd.read_csv(data, sep='\t')
        else:
            self.df = pd.read_csv(data)
        self.max = self.df[self.label_field].max()
        self.min = self.df[self.label_field].min()
        self.value_range = self.max - self.min
        self.df = self.df[self.df[self.smiles_field].str.len() <= self.max_len]
        lengths = [0, 25, 50, 75, 100]
        data = self.df
        if self.task_name == 'classification':

            pdata = data[data[self.label_field] == 1]
            ndata = data[data[self.label_field] == 0]

            ptrain_idx = []
            for i in range(4):
                idx = pdata[(pdata[self.smiles_field].str.len() >= lengths[i]) & (
                    pdata[self.smiles_field].str.len() < lengths[i + 1])].sample(frac=0.8).index
                ptrain_idx.extend(idx)

            ntrain_idx = []
            for i in range(4):
                idx = ndata[(ndata[self.smiles_field].str.len() >= lengths[i]) & (
                    ndata[self.smiles_field].str.len() < lengths[i + 1])].sample(frac=0.8).index
                ntrain_idx.extend(idx)

            pdata = pdata[~pdata.index.isin(ptrain_idx)]
            ndata = ndata[~ndata.index.isin(ntrain_idx)]

            ptest_idx = []
            for i in range(4):
                idx = pdata[(pdata[self.smiles_field].str.len() >= lengths[i]) & (
                    pdata[self.smiles_field].str.len() < lengths[i + 1])].sample(frac=0.5).index
                ptest_idx.extend(idx)
            ntest_idx = []
            for i in range(4):
                idx = ndata[(ndata[self.smiles_field].str.len() >= lengths[i]) & (
                    ndata[self.smiles_field].str.len() < lengths[i + 1])].sample(frac=0.5).index
                ntest_idx.extend(idx)
            test_data = data[data.index.isin(ptest_idx + ntest_idx)]

        elif self.task_name == 'regression':
            train_idx = []
            for i in range(4):
                idx = data[(data[self.smiles_field].str.len() >= lengths[i]) & (
                    data[self.smiles_field].str.len() < lengths[i + 1])].sample(frac=0.8).index
                train_idx.extend(idx)

            data = data[~data.index.isin(train_idx)]

            test_idx = []
            for i in range(4):
                idx = data[(data[self.smiles_field].str.len() >= lengths[i]) & (
                    data[self.smiles_field].str.len() < lengths[i + 1])].sample(frac=0.5).index
                test_idx.extend(idx)

            test_data = data[data.index.isin(test_idx)]

        self.dataset = self.ms_numerical_smiles(test_data[self.smiles_field], test_data[self.label_field])
        self.dataset = ds.NumpySlicesDataset(self.dataset, column_names=["x", "adjoin_matrix", "y"],
                                             shuffle=False)

        self.dataset = self.dataset.batch(self.batch)
        if self.task_name == 'regression':
            self.dataset = [self.dataset, self.value_range]

        return self.dataset
