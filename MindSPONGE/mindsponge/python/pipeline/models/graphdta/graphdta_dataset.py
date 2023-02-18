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
"""graphdta dataset processing script."""

import pickle
import numpy as np

from mindspore.dataset import GeneratorDataset


class GraphDTADataSet:
    """Class for Generate Dataset."""
    def __init__(self, config):
        self.batch_size = config.batch_size
        self.train_data = None
        self.train_index = None
        self.column_name = ["x_feature", "x_mask", "edge_feature", "edge_mask", "target_feature", "target_mask",
                            "label", "batch_info", "index_all"]

    def __getitem__(self, index):
        index_all = self.train_index[index*self.batch_size:(index+1)*self.batch_size]
        train_feature = self.process_data(self.train_data, self.batch_size, index_all)

        x_feature = train_feature.get('x_feature_batch')
        x_mask = train_feature.get('x_mask_batch')
        edge_feature = train_feature.get('edge_feature_batch')
        edge_mask = train_feature.get('edge_mask_batch')
        target_feature = train_feature.get('target_feature_batch')
        target_mask = train_feature.get('target_mask_batch')
        label = train_feature.get('label_batch')
        batch_info = train_feature.get('batch_info')
        res = x_feature, x_mask, edge_feature, edge_mask, target_feature, target_mask, label, batch_info, index_all

        return res

    def __len__(self):
        return int(len(self.train_index) / self.batch_size)

    @staticmethod
    def process_data(train_loader, batch_size, index_all):
        """process data"""
        max_edge = 320
        max_node = 144

        x_feature_batch = []
        edge_feature_batch = []
        label_batch = []
        batch_info = []
        target_feature_batch = []
        node_num_all = 0
        edge_num_all = 0

        for i_num, index in enumerate(index_all):

            data = train_loader[index]
            edge_index = data["edge_index"]
            target = data["target"].tolist()
            x = data["x"].tolist()
            label = data["y"].tolist()
            edge_num = len(edge_index[0])
            node_num = len(x)
            batch = [i_num] * node_num

            if i_num == 0:
                x_feature_batch = x
                edge_feature_batch = edge_index.tolist()
                target_feature_batch = target
                batch_info = batch
                label_batch = label
                node_num_all = node_num
                edge_num_all = edge_num

            else:
                x_feature_batch.extend(x)
                edge_feature_batch[0].extend(list(edge_index[0] + node_num_all))
                edge_feature_batch[1].extend(list(edge_index[1] + node_num_all))
                target_feature_batch.extend(target)
                batch_info.extend(batch)
                label_batch.extend(label)
                node_num_all += node_num
                edge_num_all += edge_num

        x_feature_batch = np.array(x_feature_batch)
        edge_feature_batch = np.array(edge_feature_batch)
        target_feature_batch = np.array(target_feature_batch)
        batch_info = np.array(batch_info)
        label_batch = np.array(label_batch)

        x_1 = np.zeros((max_node * batch_size, 78))
        x_mask1 = np.zeros((max_node * batch_size,))
        target_feat1 = target_feature_batch
        edge_feat1 = np.ones((2, max_edge * batch_size)) * node_num_all
        edge_mask1 = np.zeros((2, max_edge * batch_size))
        label1 = label_batch
        batch1 = np.zeros((max_node * batch_size,)).tolist()

        x_1[:node_num_all] = x_feature_batch
        batch1[:node_num_all] = batch_info[:]
        batch1 = np.array(batch1)
        x_mask1[:node_num_all] = 1
        edge_feat1[0][:edge_num_all] = edge_feature_batch[0]
        edge_feat1[1][:edge_num_all] = edge_feature_batch[1]
        edge_mask1[0][:edge_num_all] = 1
        edge_mask1[1][:edge_num_all] = 1

        new_train_data = {"x_feature_batch": x_1,
                          "x_mask_batch": x_mask1,
                          "edge_feature_batch": edge_feat1,
                          "edge_mask_batch": edge_mask1,
                          "target_feature_batch": target_feat1,
                          "target_mask_batch": np.zeros((batch_size, 1000)),
                          "label_batch": label1,
                          "batch_info": batch1,
                          }
        return new_train_data

    def set_training_data_src(self, data_src=None):
        """set training data src"""
        if data_src is None:
            raise FileNotFoundError
        with open(data_src, "rb") as f:
            input_data = pickle.load(f)
        self.train_data = input_data

    def create_iterator(self, num_epochs):
        """create data iterator"""
        index_all = list(range(len(self.train_data)))
        self.train_index = []
        for _ in range(num_epochs):
            np.random.shuffle(index_all)
            self.train_index.extend(index_all)

        dataset = GeneratorDataset(source=self, column_names=self.column_name,
                                   num_parallel_workers=4, shuffle=True, max_rowsize=16)
        iteration = dataset.create_dict_iterator(num_epochs=1, output_numpy=True)
        return iteration
