# Copyright 2024 Huawei Technologies Co., Ltd
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
"""data loader of mindreord files; add mask, pad and bacth process."""
import os

import numpy as np
import mindspore.dataset as ds
from mindformers import T5Tokenizer

from ..utils.utils import seqs_tokenizer
from ....dataset import DataSet

MASK_TOKEN_ID = 33
PAD_INDEX = 0


class EncDecIds():
    """return `input_ids', 'masks', 'decode_ids'."""
    def __init__(self, mask_prob):
        super().__init__()
        self.mask_prob = mask_prob

    def __call__(self, raw_ids):
        decode_ids = np.array(raw_ids)
        mask_ids = np.ones_like(raw_ids)

        # determine which positions should be masked
        random_mask = np.random.rand(len(raw_ids)) < self.mask_prob
        random_mask[len(raw_ids) - 1] = False
        raw_ids[random_mask] = MASK_TOKEN_ID

        return (np.array(raw_ids), mask_ids, decode_ids)


def find_mindrecord_files(directory):
    """find mindrecord files"""
    files = os.listdir(directory)
    mindrecord_files = [os.path.join(directory, f) for f in files if f.endswith('.mindrecord')]
    return mindrecord_files


def create_pretrain_dataset(mr_files, batch_size, epochs, rank_size=0, rank_id=0):
    """create pretrain dataset"""
    if rank_size > 0:
        dataset = ds.MindDataset(dataset_files=mr_files, columns_list=["raw_ids"],
                                 num_shards=rank_size, shard_id=rank_id, shuffle=True)
    else:
        dataset = ds.MindDataset(dataset_files=mr_files, columns_list=["raw_ids"], shuffle=True)
    dataset = dataset.map(operations=EncDecIds(0.15), input_columns=["raw_ids"],
                          output_columns=["input_ids", "masks", "decode_ids"])

    # default: pad = 0
    padding_shape = ([512], 0)
    pad_info = {"input_ids": padding_shape, "masks": padding_shape, "decode_ids": padding_shape}
    dataset = dataset.padded_batch(batch_size=batch_size, drop_remainder=True, pad_info=pad_info)
    dataset = dataset.repeat(epochs)
    return dataset


class ProtT5TrainDataSet(DataSet):
    """ProtT5 downstream task dataSet"""
    def __init__(self, config):
        self.batch_size = config.train.batch_size
        self.data_path = None
        self.dataset = None
        self.phase = None
        self.t5_config_path = config.t5_config_path
        self.tokenizer = None

        super().__init__()

    # pylint: disable=E0302
    def __getitem__(self):
        raise NotImplementedError

    def __len__(self):
        if self.dataset:
            return self.dataset.get_dataset_size()
        return 0

    def set_phase(self, phase):
        self.phase = phase

    # pylint: disable=W0221
    def process(self, data, mode="embedding"):
        re_type = "ms"
        if mode == "generate":
            re_type = "np"

        if not self.tokenizer:
            self.tokenizer = T5Tokenizer.from_pretrained(self.t5_config_path)
        return seqs_tokenizer(data, self.tokenizer, return_tensors=re_type)

    def set_training_data_src(self, data_source):
        self.data_path = data_source

    def download(self, path=None):
        raise NotImplementedError

    def data_parse(self, idx):
        raise NotImplementedError

    # pylint: disable=W0221
    def create_iterator(self, num_epochs, rank_size=0, rank_id=0):
        mr_files = find_mindrecord_files(self.data_path)
        self.dataset = create_pretrain_dataset(mr_files, self.batch_size, num_epochs,
                                               rank_size=rank_size, rank_id=rank_id)
        data_loader = self.dataset.create_tuple_iterator()
        return data_loader
