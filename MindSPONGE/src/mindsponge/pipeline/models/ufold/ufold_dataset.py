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
"""ufold dataset"""
import os
import subprocess
import collections
from itertools import product
import numpy as np
from mindspore.dataset import GeneratorDataset
from ...dataset import DataSet, data_process_run
from .ufold_data import RNASSDataGenerator
from .ufold_data import get_cut_len, permutation, permutation_nc, creatematrix, one_hot


class UFoldDataSet(DataSet):
    """UFold Dataset"""
    def __init__(self, config):
        self.config = config
        self.dataset_url = "https://pan.baidu.com/s/1y2EWQlZJhJfqi_UyUnEicw?pwd=o5k2"
        self.data = None
        self.data2 = None
        self.batch_size_1 = self.config.batch_size_stage_1
        self.train_column_list = ['contacts', 'seq_embeddings', 'matrix_reps', 'seq_lens', 'seq_ori', 'seq_name']
        self.perm = list(product(np.arange(4), np.arange(4)))
        self.perm_nc = [[0, 0], [0, 2], [0, 3], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2], [3, 0], [3, 3]]
        self.data_process = [
            get_cut_len(set_len=80),
            permutation(perm=self.perm)
        ]
        if not self.config.is_training:
            self.data_process.append(permutation_nc(perm_nc=self.perm_nc))
        self.data_process.append(creatematrix)
        super().__init__()


    def __getitem__(self, idx):
        data = self.data_parse(idx)
        output = self.train_process(data)
        return tuple(output)


    def __len__(self):
        return self.data.len


    def download(self, path=None):
        print(f"UFold's dataset can be downloaded from {self.dataset_url}")
        if path is not None:
            print(f"{path} can be used in method set_training_data_src to set the raw data path.")

    # pylint: disable=arguments-differ
    def train_process(self, data):
        """train process"""
        contact, data_seq, matrix_rep, data_len, data_name = data
        d = {}
        d["contact"] = contact
        d["data_seq"] = data_seq
        d["matrix_rep"] = matrix_rep
        d["data_len"] = data_len
        d["data_name"] = data_name
        features = data_process_run(d, self.data_process)
        if self.config.is_training:
            output = [features.get("contact")[:features.get("l"), :features.get("l")], features.get("data_fcn_2"),
                      features.get("matrix_rep"), features.get("data_len"),
                      features.get("data_seq")[:features.get("l")], features.get("data_name")]
        else:
            output = [features.get("contact")[:features.get("l"), :features.get("l")], features.get("data_fcn_2"),
                      features.get("matrix_rep"), features.get("data_len"),
                      features.get("data_seq")[:features.get("l")], features.get("data_name"),
                      features.get("data_nc"), features.get("l")]
        return output


    def process(self, data):
        if not data.endswith(".ct"):
            all_files = os.listdir(data)
        else:
            all_files = [data.split('/')[-1]]
        all_files.sort()
        all_files_list = []
        RNA_SS_data = collections.namedtuple('RNA_SS_data', 'seq ss_label length name pairs')

        for item in all_files:
            path = os.path.join(data.replace(data.split('/')[-1], ''), item)
            t0 = subprocess.getstatusoutput('awk \'{print $2}\' '+path)
            t0_1 = t0[1].split('\n')
            t0_1.pop(0)
            seq = ''.join(t0_1)

            one_hot_matrix = one_hot(seq.upper())

            t1 = subprocess.getstatusoutput('awk \'{print $1}\' '+path)
            t2 = subprocess.getstatusoutput('awk \'{print $3}\' '+path)

            if t1[0] == 0 and t2[0] == 0:
                t1_1 = t1[1].split('\n')
                t2_1 = t2[1].split('\n')
                t1_1.pop(0)
                t2_1.pop(0)
                pair_dict_all_list = [[int(item_tmp) - 1, int(t2_1[index_tmp]) - 1] for index_tmp, item_tmp
                                      in enumerate(t1_1) if int(t2_1[index_tmp]) != 0]
            else:
                pair_dict_all_list = []

            seq_name = data.split('/')[-1]
            seq_len = len(seq)
            # pylint: disable=consider-using-dict-comprehension
            pair_dict_all = dict([item for item in pair_dict_all_list if item[0] < item[1]])

            # pylint: disable=chained-comparison
            if seq_len > 0 and seq_len <= 600:
                ss_label = np.zeros((seq_len, 3), dtype=int)
                ss_label[[*pair_dict_all.keys()],] = [0, 1, 0]
                ss_label[[*pair_dict_all.values()],] = [0, 0, 1]
                ss_label[np.where(np.sum(ss_label, axis=1) <= 0)[0],] = [1, 0, 0]
                one_hot_matrix_600 = np.zeros((600, 4))
                one_hot_matrix_600[:seq_len,] = one_hot_matrix
                ss_label_600 = np.zeros((600, 3), dtype=int)
                ss_label_600[:seq_len,] = ss_label
                ss_label_600[np.where(np.sum(ss_label_600, axis=1) <= 0)[0],] = [1, 0, 0]
                sample_tmp = RNA_SS_data(seq=one_hot_matrix_600, ss_label=ss_label_600,
                                         length=seq_len, name=seq_name, pairs=pair_dict_all_list)
                all_files_list.append(sample_tmp)

        test_data = RNASSDataGenerator(data=all_files_list)
        all_d = []
        for i in range(test_data.len):
            d = test_data.get_one_sample(i)
            all_d.append(self.train_process(d))
        return all_d


    def data_parse(self, idx):
        data = self.data.get_one_sample(idx)
        return data


    def set_training_data_src(self, data_src=None):
        """set training data source path"""
        train_files = self.config.train_files
        train_data_list = []
        for file_item in train_files:
            print('Loading dataset: ', file_item)
            if file_item == 'ArchiveII':
                train_data_list.append(RNASSDataGenerator(data_src, file_item+'.pickle'))
            else:
                train_data_list.append(RNASSDataGenerator(data_src, file_item+'.cPickle'))
        print('Data Loading Done!!!')
        self.data2 = train_data_list[0]
        if len(train_data_list) > 1:
            self.data = self.merge_data(train_data_list)
        else:
            self.data = self.data2


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

    # pylint: disable=arguments-differ
    def create_iterator(self, num_epochs):
        dataset = GeneratorDataset(self, column_names=self.train_column_list, num_parallel_workers=3, shuffle=True)
        dataset = dataset.batch(batch_size=self.batch_size_1, drop_remainder=True)
        iteration = dataset.create_dict_iterator(num_epochs=num_epochs, output_numpy=False)
        return iteration
