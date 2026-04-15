# Copyright 2025 Huawei Technologies Co., Ltd
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

"""prepare data script"""
import numpy as np


def convert_eval_data(data, max_len):
    """convert eval data"""
    np.set_printoptions(threshold=np.inf)
    mask_id = np.ones(len(data["input_ids"]), dtype=int)
    mask_id = np.pad(mask_id, (0, max_len - len(data["input_ids"])), mode="constant", constant_values=0)
    data["input_ids"] = np.pad(data["input_ids"], (0, max_len - len(data["input_ids"])),
                               mode="constant", constant_values=0)
    data["labels"] = np.pad(data["labels"], (0, max_len - len(data["labels"])), mode="constant", constant_values=-100)
    return data["input_ids"], data["labels"], mask_id


def generator_eval_data(eval_data, max_len):
    """generator eval data"""
    for data in eval_data:
        yield convert_eval_data(data, max_len)


def load_data(dataset):
    """load data"""
    input_ids = dataset['input_ids']
    length = dataset['length']
    labels = dataset['labels']
    return input_ids, length, labels


class GeneratorTrainData:
    """GeneratorTrainData"""
    def __init__(self, dataset):
        input_ids, length, labels = load_data(dataset)
        self._input_ids = input_ids
        self._length = length
        self._labels = labels

    def convert_data(self, index, max_len=2048):
        """convert_data"""
        input_ids = self._input_ids[index]
        labels = self._labels[index]
        mask_id = np.ones(len(input_ids), dtype=int)
        mask_id = np.pad(mask_id, (0, max_len - len(input_ids)), mode="constant", constant_values=0)
        mask_id = mask_id.astype(np.int32)
        data_input_ids = np.pad(input_ids, (0, max_len - len(input_ids)), mode="constant", constant_values=0)
        data_input_ids = data_input_ids.astype(np.int32)
        data_labels = np.pad(labels, (0, max_len - len(labels)), mode="constant", constant_values=-100)
        data_labels = data_labels.astype(np.int32)
        token_type_ids = np.zeros((max_len), dtype=np.int32)
        return data_input_ids, mask_id, token_type_ids, data_labels

    def __getitem__(self, index):
        data_input_ids, mask_id, token_type_ids, data_labels = self.convert_data(index)
        return data_input_ids, mask_id, token_type_ids, data_labels

    def __len__(self):
        return len(self._input_ids)
