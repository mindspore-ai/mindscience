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
"""deepfridataset"""
import os
import pickle
from .deepfridata import load_cmap
from ...dataset import PSP, data_process_run


class DeepFriDataSet(PSP):
    '''deepfridataset'''

    def __init__(self, config, num_seq=1):
        self.config = config
        self.supported_models = ['DeepFri']
        self.in_memory = False
        self.deepfri_inputs()
        self.indx = 0

        self.data_process = [load_cmap()]

        self._num = num_seq
        super().__init__()

    # pylint: disable=arguments-differ
    def __getitem__(self, idx):
        if self.in_memory:
            data = self.inputs[idx]
        else:
            data = self.data_parse(idx)

        self.indx += 1
        features = self.process(data)
        return tuple(features)

    def __len__(self):
        data_len = len(os.listdir(self.training_pdb_path))
        return data_len

    def deepfri_inputs(self):
        feature_list = ['adj', 'seq_1hot']
        self.feature_list = feature_list

    # pylint: disable=arguments-differ
    def data_parse(self, idx):
        pkl_path = self.training_pkl_items[idx]
        f = open(pkl_path, "rb")
        data = pickle.load(f)
        return data

    def process(self, data):
        features = data_process_run(data, self.data_process)
        return features

    # pylint: disable=arguments-differ
    def create_iterator(self, num_epochs):
        pass
