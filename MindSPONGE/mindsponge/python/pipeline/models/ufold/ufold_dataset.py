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
from itertools import product
import numpy as np
from mindspore.dataset import GeneratorDataset
from ...dataset import DataSet, data_process_run
from .ufold_data import RNASSDataGenerator
from .ufold_data import get_cut_len, permutation, permutation_nc, creatematrix


class UFoldDataSet(DataSet):
    """UFold Dataset"""
    def __init__(self, config):
        self.config = config
        self.dataset_url = "https://pan.baidu.com/s/1y2EWQlZJhJfqi_UyUnEicw?pwd=o5k2"
        self.data = None
        self.data2 = None
        self.batch_size_1 = self.config.batch_size_stage_1
        self.train_column_list = ['contacts', 'seq_embeddings', 'matrix_reps', 'seq_lens', 'seq_ori', 'seq_name']
        self.test_column_list = ['contacts', 'seq_embeddings', 'matrix_reps',
                                 'seq_lens', 'seq_ori', 'seq_name', 'nc_map', 'l_len']
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
        contact, data_seq, matrix_rep, data_len, data_name = self.data.get_one_sample(idx)
        data = {}
        data["contact"] = contact
        data["data_seq"] = data_seq
        data["matrix_rep"] = matrix_rep
        data["data_len"] = data_len
        data["data_name"] = data_name
        output = self.process(data)
        return tuple(output)


    def __len__(self):
        return self.data.len


    def download(self, path=None):
        print(f"UFold's dataset can be downloaded from {self.dataset_url}")
        if path is not None:
            print(f"{path} can be used in method set_training_data_src to set the raw data path.")

    # pylint: disable=arguments-differ
    def process(self, data):
        features = data_process_run(data, self.data_process)
        if self.config.is_training:
            output = [features["contact"][:features["l"], :features["l"]], features["data_fcn_2"],
                      features["matrix_rep"], features["data_len"],
                      features["data_seq"][:features["l"]], features["data_name"]]
        else:
            output = [features["contact"][:features["l"], :features["l"]], features["data_fcn_2"],
                      features["matrix_rep"], features["data_len"], features["data_seq"][:features["l"]],
                      features["data_name"], features["data_name"], features["data_nc"], features["l"]]
        return output


    def data_parse(self, idx):
        pass


    def set_training_data_src(self, data_src=None):
        """set training data source path"""
        if self.config.is_training:
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
        else:
            test_file = self.config.test_file
            assert isinstance(test_file, str)
            print('Loading test file: ', test_file)
            if test_file == 'ArchiveII':
                test_data = RNASSDataGenerator(data_src, test_file+'.pickle')
            else:
                test_data = RNASSDataGenerator(data_src, test_file+'.cPickle')
            self.data = test_data


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
        if self.config.is_training:
            dataset = GeneratorDataset(self, column_names=self.train_column_list, num_parallel_workers=3, shuffle=True)
        else:
            dataset = GeneratorDataset(self, column_names=self.test_column_list, num_parallel_workers=3, shuffle=True)
        dataset = dataset.batch(batch_size=self.batch_size_1, drop_remainder=True)
        iteration = dataset.create_dict_iterator(num_epochs=num_epochs, output_numpy=False)
        return iteration
