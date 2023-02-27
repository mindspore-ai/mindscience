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
"""ufold data"""
import math
import os
from collections import Counter
from multiprocessing import Pool
from random import shuffle

import _pickle as cPickle
import numpy as np

from ...dataset import curry1

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


def gaussian(x):
    return math.exp(-0.5*(x*x))


def encoding2seq(arr):
    seq = list()
    for arr_row in list(arr):
        if sum(arr_row) == 0:
            seq.append('.')
        else:
            seq.append(char_dict.get(np.argmax(arr_row)))
    return ''.join(seq)


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


@curry1
def get_cut_len(data=None, set_len=80):
    "get cut length"
    l = data["data_len"]
    if l <= set_len:
        l = set_len
    else:
        l = (((l - 1) // 16) + 1) * 16
    data["l"] = l
    data_fcn = np.zeros((16, l, l))
    data_nc = np.zeros((10, l, l))
    data_fcn_1 = np.zeros((1, l, l))
    data["data_fcn"] = data_fcn
    data["data_nc"] = data_nc
    data["data_fcn_1"] = data_fcn_1
    if data["l"] > 500:
        data = cut_contact_and_data_seq(data)
    return data


def cut_contact_and_data_seq(data):
    contact_adj = np.zeros((data["l"], data["l"]))
    contact_adj[:data["data_len"], :data["data_len"]] = data["contact"][:data["data_len"], :data["data_len"]]
    data["contact"] = contact_adj
    seq_adj = np.zeros((data["l"], 4))
    seq_adj[:data["data_len"]] = data["data_seq"][:data["data_len"]]
    data["data_seq"] = seq_adj
    return data


@curry1
def permutation(data=None, perm=None):
    for n, cord in enumerate(perm):
        i, j = cord
        data["data_fcn"][n, :data["data_len"], :data["data_len"]] = \
            np.matmul(data["data_seq"][:data["data_len"], i].reshape(-1, 1),
                      data["data_seq"][:data["data_len"], j].reshape(1, -1))
    return data


@curry1
def permutation_nc(data=None, perm_nc=None):
    for n, cord in enumerate(perm_nc):
        i, j = cord
        data["data_nc"][n, :data["data_len"], :data["data_len"]] = \
            np.matmul(data["data_seq"][:data["data_len"], i].reshape(-1, 1),
                      data["data_seq"][:data["data_len"], j].reshape(1, -1))
    data["data_nc"] = data["data_nc"].sum(axis=0).astype(np.bool)
    return data


def creatematrix(data):
    """create matrix"""
    mat = np.zeros([len(data["data_seq"][:data["data_len"],]), len(data["data_seq"][:data["data_len"],])])
    for i in range(len(data["data_seq"][:data["data_len"],])):
        for j in range(len(data["data_seq"][:data["data_len"],])):
            coefficient = 0
            coefficient = judgescore1(i, j, coefficient, data["data_seq"][:data["data_len"],])
            if coefficient > 0:
                coefficient = judgescore2(i, j, coefficient, data["data_seq"][:data["data_len"],])
            mat[[i], [j]] = coefficient
    data["data_fcn_1"][0, :data["data_len"], :data["data_len"]] = mat
    data_fcn_2 = np.concatenate((data["data_fcn"], data["data_fcn_1"]), axis=0)
    data["data_fcn_2"] = data_fcn_2
    return data
