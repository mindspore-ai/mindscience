# Copyright 2023 @ Huawei Technologies Co., Ltd
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
"""deepdr dataset"""
import numpy as np
import scipy.io as sio
from sklearn.preprocessing import minmax_scale
import mindspore as ms
from mindspore import Tensor
from mindspore.dataset import GeneratorDataset

from ...dataset import DataSet
from .deepdr_data import build_dataset


class DeepDRDataSet(DataSet):
    """DeepDR Dataset"""
    def __init__(self, config):
        self.config = config
        self.dataset_url = "https://github.com/ChengF-Lab/deepDR"
        if self.config.model == 'mda':
            self.org = self.config.ORG
            self.selectnets = self.config.mda_select_nets
            self.batch_size = self.config.batch_size_mda
            self.nets = []
            self.data_set = None
            self._data = None
            self._label = None
            self.nf = self.config.nf
            self.std = self.config.std
        elif self.config.model == 'cvae':
            self.batch_size = self.config.batch_size_cvae
            self.rtensor = None
        else:
            raise ValueError("Invalid model type!", self.config.model)
        super().__init__()


    def __getitem__(self, idx):
        if self.config.model == 'mda':
            data, label = self.data_parse(idx)
            return (data, label)
        data, label = self.data_parse(idx)
        return data, label


    def __len__(self):
        if self.config.model == 'mda':
            return len(self._data[0])
        return len(self.rtensor)


    def download(self, path=None):
        if path is None:
            print(f"DeepDR's dataset can be downloaded from {self.dataset_url}")
        else:
            print(f"DeepDR's dataset can be downloaded from {self.dataset_url}")
            print(f"{path} can be used in method set_training_data_src to set the raw data path.")

    # pylint: disable=arguments-differ
    def process(self, data, train_index=None):
        if self.config.model == 'mda':
            nets = []
            for i in self.selectnets:
                print("### [%d] Loading network..." % (i))
                n = sio.loadmat(data + self.org + '_net_' + str(i) + '.mat', squeeze_me=True)
                net = n['Net'].todense()
                print("Net %d, NNofile_keywords=%d \n" % (i, np.count_nonzero(net)))
                nets.append(minmax_scale(net))
            return nets
        whole_positive_index = []
        whole_negative_index = []
        for i in range(np.shape(data)[0]):
            for j in range(np.shape(data)[1]):
                if int(data[i][j]) == 1:
                    whole_positive_index.append([i, j])
                elif int(data[i][j]) == 0:
                    whole_negative_index.append([i, j])
        negative_sample_index = np.random.choice(np.arange(len(whole_negative_index)),
                                                 size=1 * len(whole_positive_index), replace=False)
        data_set = np.zeros((len(negative_sample_index) + len(whole_positive_index), 3), dtype=int)
        count = 0
        for i in whole_positive_index:
            data_set[count][0] = i[0]
            data_set[count][1] = i[1]
            data_set[count][2] = 1
            count += 1
        for i in negative_sample_index:
            data_set[count][0] = whole_negative_index[i][0]
            data_set[count][1] = whole_negative_index[i][1]
            data_set[count][2] = 0
            count += 1
        if train_index is None:
            dtitrain = data_set
        else:
            dtitrain = data_set[train_index]
        xtrain = np.zeros((np.shape(data)[0], np.shape(data)[1]))
        for ele in dtitrain:
            xtrain[ele[0], ele[1]] = ele[2]
        if self.config.is_training:
            rtensor = Tensor(xtrain.astype('float32')).asnumpy()
        else:
            rtensor = Tensor.from_numpy(xtrain.astype('float32'))
        return rtensor


    def data_parse(self, idx):
        if self.config.model == 'cvae':
            if self.config.rate:
                return self.rtensor[idx], self.rtensor[idx]
            self.rtensor[self.rtensor > 0] = 1
            self.rtensor = Tensor(self.rtensor.astype('float32'), ms.float32).asnumpy()
            return self.rtensor[idx], self.rtensor[idx]
        data = np.concatenate([self._data[i][idx] for i in range(9)], axis=0)
        label = np.concatenate([self._label[i][idx] for i in range(9)], axis=0)
        return data, label


    def set_training_data_src(self, data_src, train_index=None):
        """set training data source"""
        if self.config.model == 'mda':
            self.nets = self.process(data=data_src)
            train_data, train_label, test_data, test_label = build_dataset(self.nets, self.nf, self.std)
            if self.config.is_training:
                self._data = train_data
                self._label = train_label
            else:
                self._data = test_data
                self._label = test_label
        elif self.config.model == 'cvae':
            r = np.loadtxt(data_src)
            self.rtensor = r.transpose()
            if self.config.rate:
                assert train_index is not None
                self.rtensor = self.process(self.rtensor, train_index=train_index)


    def create_iterator(self, num_epochs, **kwargs):
        dataset = GeneratorDataset(source=self, column_names=['data', 'label'], shuffle=True)
        dataset = dataset.batch(batch_size=self.batch_size)
        iteration = dataset.create_dict_iterator(num_epochs=num_epochs)
        return iteration
