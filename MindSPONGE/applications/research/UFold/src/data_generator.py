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
"""Custom Data Set Classes"""
import os
from random import shuffle
from collections import Counter
from multiprocessing import Pool
from itertools import product
import _pickle as cPickle
from src.utils import gaussian, encoding2seq
import numpy as np

perm = list(product(np.arange(4), np.arange(4)))
perm2 = [[1, 3], [3, 1]]
perm_nc = [[0, 0], [0, 2], [0, 3], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2], [3, 0], [3, 3]]

char_dict = {
    0: 'A',
    1: 'U',
    2: 'C',
    3: 'G'
}


class RNASSDataGenerator():
    """RNA builder class"""
    def __init__(self, data_dir, split, upsampling=False):
        self.data_dir = data_dir
        self.split = split
        self.upsampling = upsampling
        # Load vocab explicitly when needed
        self.load_data()
        # Reset batch pointer to zero
        self.batch_pointer = 0

    def load_data(self):
        """Load the current split"""
        p = Pool()
        data_dir = self.data_dir
        with open(os.path.join(data_dir, '%s' % self.split), 'rb') as f:
            self.data = cPickle.load(f, encoding='iso-8859-1')
        if self.upsampling:
            self.data = self.upsampling_data_new()
        self.data_x = np.array([instance[0] for instance in self.data])
        self.data_y = np.array([instance[1] for instance in self.data])
        self.pairs = np.array([instance[-1] for instance in self.data])
        self.seq_length = np.array([instance[2] for instance in self.data])
        self.len = len(self.data)
        self.seq = list(p.map(encoding2seq, self.data_x))
        self.seq_max_len = len(self.data_x[0])
        self.data_name = np.array([instance[3] for instance in self.data])

    def upsampling_data(self):
        """upsample data"""
        name = [instance.name for instance in self.data]
        d_type = np.array(list(map(lambda x: x.split('/')[2], name)))
        data = np.array(self.data)
        max_num = max(Counter(list(d_type)).values())
        data_list = list()
        for t in sorted(list(np.unique(d_type))):
            index = np.where(d_type == t)[0]
            data_list.append(data[index])
        final_d_list = list()
        for i in [0, 1, 5, 7]:
            d = data_list[i]
            index = np.random.choice(d.shape[0], max_num)
            final_d_list += list(d[index])

        for i in [2, 3, 4]:
            d = data_list[i]
            index = np.random.choice(d.shape[0], max_num*2)
            final_d_list += list(d[index])

        d = data_list[6]
        index = np.random.choice(d.shape[0], int(max_num/2))
        final_d_list += list(d[index])

        shuffle(final_d_list)
        return final_d_list

    def upsampling_data_new(self):
        """upsample data"""
        name = [instance.name for instance in self.data]
        d_type = np.array(list(map(lambda x: x.split('_')[0], name)))
        data = np.array(self.data)
        data_list = list()
        for t in sorted(list(np.unique(d_type))):
            index = np.where(d_type == t)[0]
            data_list.append(data[index])
        final_d_list = list()
        for d in data_list:
            final_d_list += list(d)
            if d.shape[0] < 300:
                index = np.random.choice(d.shape[0], 300-d.shape[0])
                final_d_list += list(d[index])
            if d.shape[0] == 652:
                index = np.random.choice(d.shape[0], d.shape[0]*4)
                final_d_list += list(d[index])
        shuffle(final_d_list)
        return final_d_list

    def next_batch(self, batch_size):
        """get next batch"""
        bp = self.batch_pointer
        # This will return a smaller size if not sufficient
        # The user must pad the batch in an external API
        # Or write a TF module with variable batch size
        batch_x = self.data_x[bp:bp + batch_size]
        batch_y = self.data_y[bp:bp + batch_size]
        batch_seq_len = self.seq_length[bp:bp + batch_size]

        self.batch_pointer += batch_size
        if self.batch_pointer >= len(self.data_x):
            self.batch_pointer = 0

        yield batch_x, batch_y, batch_seq_len

    def pairs2map(self, pairs):
        seq_len = self.seq_max_len
        contact = np.zeros([seq_len, seq_len])
        for pair in pairs:
            contact[pair[0], pair[1]] = 1
        return contact

    def get_one_sample(self, index):
        """get sample"""
        data_seq = self.data_x[index]
        data_len = self.seq_length[index]
        data_pair = self.pairs[index]
        data_name = self.data_name[index]

        contact = self.pairs2map(data_pair)
        matrix_rep = np.zeros(contact.shape)
        output = [contact, data_seq, matrix_rep, data_len, data_name]
        return tuple(output)


class DatasetCutConcatNewCanonicle():
    """dataset processing class"""
    def __init__(self, data):
        # Initialization
        self.data = data

    def __len__(self):
        # Denotes the total number of samples
        return self.data.len

    def __getitem__(self, index):
        # Generates one sample of data
        # Select sample
        contact, data_seq, matrix_rep, data_len, data_name = self.data.get_one_sample(index)
        l = get_cut_len(data_len, 80)
        data_fcn = np.zeros((16, l, l))
        data_nc = np.zeros((10, l, l))
        if l >= 500:
            contact_adj = np.zeros((l, l))
            contact_adj[:data_len, :data_len] = contact[:data_len, :data_len]
            contact = contact_adj
            seq_adj = np.zeros((l, 4))
            seq_adj[:data_len] = data_seq[:data_len]
            data_seq = seq_adj
        for n, cord in enumerate(perm):
            i, j = cord
            data_fcn[n, :data_len, :data_len] = np.matmul(data_seq[:data_len, i].reshape(-1, 1),
                                                          data_seq[:data_len, j].reshape(1, -1))
        for n, cord in enumerate(perm_nc):
            i, j = cord
            data_nc[n, :data_len, :data_len] = np.matmul(data_seq[:data_len, i].reshape(-1, 1),
                                                         data_seq[:data_len, j].reshape(1, -1))
        data_nc = data_nc.sum(axis=0).astype(np.bool)
        data_fcn_1 = np.zeros((1, l, l))
        data_fcn_1[0, :data_len, :data_len] = creatmat(data_seq[:data_len,])
        data_fcn_2 = np.concatenate((data_fcn, data_fcn_1), axis=0)
        output = [contact[:l, :l], data_fcn_2, matrix_rep, data_len, data_seq[:l], data_name, data_nc, l]
        return tuple(output)


class DatasetCutConcatNewMergeTwo():
    """Characterizes a dataset"""
    def __init__(self, data1, data2):
        # Initialization
        self.data1 = data1
        self.data2 = data2
        self.merge_data()
        self.data = self.data2

    def __len__(self):
        # Denotes the total number of samples
        return self.data.len

    def __getitem__(self, index):
        # Generates one sample of data
        # Select sample
        contact, data_seq, matrix_rep, data_len, data_name = self.data.get_one_sample(index)
        l = get_cut_len(data_len, 80)
        data_fcn = np.zeros((16, l, l))
        feature = np.zeros((8, l, l))
        if l >= 500:
            contact_adj = np.zeros((l, l))
            contact_adj[:data_len, :data_len] = contact[:data_len, :data_len]
            contact = contact_adj
            seq_adj = np.zeros((l, 4))
            seq_adj[:data_len] = data_seq[:data_len]
            data_seq = seq_adj
        for n, cord in enumerate(perm):
            i, j = cord
            data_fcn[n, :data_len, :data_len] = np.matmul(data_seq[:data_len, i].reshape(-1, 1),
                                                          data_seq[:data_len, j].reshape(1, -1))
        data_fcn_1 = np.zeros((1, l, l))
        data_fcn_1[0, :data_len, :data_len] = creatmat(data_seq[:data_len,])
        temp = data_seq[None, :data_len, :data_len]
        temp = np.tile(temp, (temp.shape[1], 1, 1))
        temp_f = np.concatenate([temp, np.transpose(temp, [1, 0, 2])], 2).reshape((-1, data_len, data_len))
        feature[:, :data_len, :data_len] = temp_f
        feature = np.concatenate((data_fcn, feature), axis=0)
        data_fcn_2 = np.concatenate((data_fcn, data_fcn_1), axis=0)
        output = [contact[:l, :l], data_fcn_2, matrix_rep, data_len, data_seq[:l], data_name]
        return tuple(output)

    def merge_data(self):
        self.data2.data_x = np.concatenate((self.data1.data_x[:, :600, :], self.data2.data_x), axis=0)
        self.data2.data_y = np.concatenate((self.data1.data_y[:, :600, :], self.data2.data_y), axis=0)
        self.data2.seq_length = np.concatenate((self.data1.seq_length, self.data2.seq_length), axis=0)
        self.data2.pairs = np.concatenate((self.data1.pairs, self.data2.pairs), axis=0)
        self.data2.data_name = np.concatenate((self.data1.data_name, self.data2.data_name), axis=0)
        self.data2.len = len(self.data2.data_name)


class DatasetCutConcatNewMergeMulti():
    """dataset processing class"""
    def __init__(self, data_list):
        # Initialization
        self.data2 = data_list[0]
        if len(data_list) > 1:
            self.data = self.merge_data(data_list)
        else:
            self.data = self.data2

    def __len__(self):
        # Denotes the total number of samples
        return self.data.len

    def __getitem__(self, index):
        # Generates one sample of data
        # Select sample
        contact, data_seq, matrix_rep, data_len, data_name = self.data.get_one_sample(index)
        l = get_cut_len(data_len, 80)
        data_fcn = np.zeros((16, l, l))
        feature = np.zeros((8, l, l))
        if l >= 500:
            contact_adj = np.zeros((l, l))
            contact_adj[:data_len, :data_len] = contact[:data_len, :data_len]
            contact = contact_adj
            seq_adj = np.zeros((l, 4))
            seq_adj[:data_len] = data_seq[:data_len]
            data_seq = seq_adj
        for n, cord in enumerate(perm):
            i, j = cord
            data_fcn[n, :data_len, :data_len] = np.matmul(data_seq[:data_len, i].reshape(-1, 1),
                                                          data_seq[:data_len, j].reshape(1, -1))
        data_fcn_1 = np.zeros((1, l, l))
        data_fcn_1[0, :data_len, :data_len] = creatmat(data_seq[:data_len,])
        temp = data_seq[None, :data_len, :data_len]
        temp = np.tile(temp, (temp.shape[1], 1, 1))
        temp_f = np.concatenate([temp, np.transpose(temp, [1, 0, 2])], 2).reshape((-1, data_len, data_len))
        feature[:, :data_len, :data_len] = temp_f
        feature = np.concatenate((data_fcn, feature), axis=0)
        data_fcn_2 = np.concatenate((data_fcn, data_fcn_1), axis=0)
        output = [contact[:l, :l], data_fcn_2, matrix_rep, data_len, data_seq[:l], data_name]
        return tuple(output)

    def merge_data(self, data_list):
        """merge data"""
        self.data2.data_x = np.concatenate((data_list[0].data_x, data_list[1].data_x), axis=0)
        self.data2.data_y = np.concatenate((data_list[0].data_y, data_list[1].data_y), axis=0)
        self.data2.seq_length = np.concatenate((data_list[0].seq_length, data_list[1].seq_length), axis=0)
        self.data2.pairs = np.concatenate((data_list[0].pairs, data_list[1].pairs), axis=0)
        self.data2.data_name = np.concatenate((data_list[0].data_name, data_list[1].data_name), axis=0)
        for item in data_list[2:]:
            self.data2.data_x = np.concatenate((self.data2.data_x, item.data_x), axis=0)
            self.data2.data_y = np.concatenate((self.data2.data_y, item.data_y), axis=0)
            self.data2.seq_length = np.concatenate((self.data2.seq_length, item.seq_length), axis=0)
            self.data2.pairs = np.concatenate((self.data2.pairs, item.pairs), axis=0)
            self.data2.data_name = np.concatenate((self.data2.data_name, item.data_name), axis=0)

        self.data2.len = len(self.data2.data_name)
        return self.data2


def get_cut_len(data_len, set_len):
    l = data_len
    if l <= set_len:
        l = set_len
    else:
        l = (((l - 1) // 16) + 1) * 16
    return l


def paired(x, y):
    """get pair score"""
    if x == [1, 0, 0, 0] and y == [0, 1, 0, 0]:
        return 2
    if x == [0, 0, 0, 1] and y == [0, 0, 1, 0]:
        return 3
    if x == [0, 0, 0, 1] and y == [0, 1, 0, 0]:
        return 0.8
    if x == [0, 1, 0, 0] and y == [1, 0, 0, 0]:
        return 2
    if x == [0, 0, 1, 0] and y == [0, 0, 0, 1]:
        return 3
    if x == [0, 1, 0, 0] and y == [0, 0, 0, 1]:
        return 0.8
    return 0


def judgescore1(i, j, coefficient, data):
    """calculate coefficient 1"""
    for add in range(30):
        if i - add >= 0 and j + add < len(data):
            score = paired(list(data[i - add]), list(data[j + add]))
            if score == 0:
                break
            else:
                coefficient = coefficient + score * gaussian(add)
        else:
            break
    return coefficient


def judgescore2(i, j, coefficient, data):
    """calculate coefficient 2"""
    for add in range(1, 30):
        if i + add < len(data) and j - add >= 0:
            score = paired(list(data[i + add]), list(data[j - add]))
            if score == 0:
                break
            else:
                coefficient = coefficient + score * gaussian(add)
        else:
            break
    return coefficient


def creatmat(data):
    """create matrix"""
    mat = np.zeros([len(data), len(data)])
    for i in range(len(data)):
        for j in range(len(data)):
            coefficient = 0
            coefficient = judgescore1(i, j, coefficient, data)
            if coefficient > 0:
                coefficient = judgescore2(i, j, coefficient, data)
            mat[[i], [j]] = coefficient
    return mat
