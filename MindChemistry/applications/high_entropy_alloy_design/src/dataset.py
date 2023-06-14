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
"""create dataset"""
import pandas as pd
import numpy as np

from sklearn import preprocessing
from sklearn.model_selection import train_test_split


class HighEntropyAlloy():
    def __init__(self, root):
        super(HighEntropyAlloy, self).__init__()
        self.root = root
        self.raw_data = pd.read_excel(root + '/data/Data_base.xlsx', header=0)
        self.component_name = ['Fe', 'Ni', 'Co', 'Cr', 'V', 'Cu']
        self.property_name = {
            'stage1': ['VEC', 'AR1', 'AR2', 'PE', 'Density', 'TermalC', 'MP', 'FI', 'SI', 'TI', 'M'],
            'stage2': ['TC', 'MS', 'MagS']
        }
        self.bins = [18, 35, 48, 109, 202, 234, 525, 687]

    def process_train_gen_data(self):
        # load data
        gen_data = self.raw_data.iloc[:, 1:19].to_numpy().astype(np.float32)
        raw_x = gen_data[:, :6]
        raw_y = gen_data[:, 17].reshape(-1, 1)
        # generate label
        label_y = np.where(raw_y < 5, 1, 0).astype(np.float32)
        return raw_x, label_y

    def process_train_rank_data(self, stage_num, seed):
        # load data
        rank_data_train = self.raw_data[:696]
        df_all = rank_data_train.drop(columns=['alloy'])
        # filter adopted properties
        if stage_num == 1:
            feature_name = self.property_name['stage1']
        elif stage_num == 2:
            feature_name = self.property_name['stage1'] + self.property_name['stage2']
        # normalize properties
        min_max_scaler = preprocessing.MinMaxScaler()
        normalized_atomic_properties = min_max_scaler.fit_transform(df_all[feature_name])
        # define input and label
        composition = df_all[self.component_name]
        raw_x = np.concatenate([composition.values, normalized_atomic_properties], axis=1).astype(np.float32)
        y = df_all[['TEC']]
        label_y = y.values.astype(np.float32)
        # split train and test set with 7-fold stratify
        stratify_flag = np.digitize(y.index, self.bins, right=True)
        train_x, test_x, train_labels, test_labels = train_test_split(raw_x, label_y, test_size=0.15,
                                                                      random_state=seed,
                                                                      stratify=stratify_flag)
        return train_x, test_x, train_labels, test_labels

    def process_eval_data(self, stage_num):
        # load data
        rank_data_test = self.raw_data[696:]
        df_all = rank_data_test.drop(columns=['alloy'])
        # filter adopted properties
        if stage_num == 1:
            feature_name = self.property_name['stage1']
        elif stage_num == 2:
            feature_name = self.property_name['stage1'] + self.property_name['stage2']
        # define input and label
        min_max_scaler = preprocessing.MinMaxScaler()
        normalized_atomic_properties = min_max_scaler.fit_transform(df_all[feature_name])
        composition = df_all[self.component_name]
        raw_x = np.concatenate([composition.values, normalized_atomic_properties], axis=1).astype(np.float32)
        y = df_all[['TEC']]
        label_y = y.values.astype(np.float32)
        return raw_x, label_y
