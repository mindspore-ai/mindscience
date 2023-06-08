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
"""esm2 dataset"""
import mindspore as ms
from ...dataset import PSP
from ..esm_if1.module.util import Alphabet


class ESM2DataSet(PSP):
    """esm dataset"""
    def __init__(self, config):
        self.config = config
        self.alphabet = Alphabet.from_architecture(self.config.alphabet)
        self.batch_converter = self.alphabet.get_batch_converter()
        self.protein_data = []
        self.batch_size = config.batch_size
        self.data = []
        self.batch_lens = None
        self.batch_tokens = None
        self.batch_labels = None
        self.batch_strs = None
        for i in range(0, len(self.protein_data), self.batch_size):
            self.data.append(self.protein_data[i: i + self.batch_size])
        super().__init__()

    # pylint: disable=W0221
    def __getitem__(self, item):
        batch_tokens = self.process(self.data[item])
        return batch_tokens

    def __len__(self):
        return len(self.data)

    def set_data(self, protein):
        self.protein_data = protein

    def process(self, data, **kwargs):
        """pre-process data"""
        batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
        batch_lens = (batch_tokens != self.alphabet.padding_idx).sum(1)
        batch_tokens = ms.Tensor(batch_tokens, ms.int32)
        self.batch_lens = batch_lens
        self.batch_tokens = batch_tokens
        self.batch_labels = batch_labels
        self.batch_strs = batch_strs
        return batch_tokens

    def data_parse(self, idx):
        pass

    def create_iterator(self, num_epochs, **kwargs):
        pass
