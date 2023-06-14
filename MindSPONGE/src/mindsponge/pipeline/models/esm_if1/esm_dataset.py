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
"""esm dataset"""
import math
import json
import numpy as np
from mindspore.dataset import GeneratorDataset
# pylint: disable=relative-beyond-top-level
from .module.util import load_coords, CoordBatchConverter
from .module.util import Alphabet
from ...dataset import PSP


# pylint: disable=abstract-method
class ESMDataSet(PSP):
    """esm dataset"""
    def __init__(self, config):
        self.config = config
        self.alphabet = Alphabet.from_architecture(self.config.arch)
        self.feature_list = ['coords', 'confidence', 'padding_mask', 'prev_output_tokens', 'target']
        self.batch_size = self.config.batch_size
        self.traindata = None
        self.training_data_src = ""
        self.coords, self.confidence, self.padding_mask, self.prev_output_tokens, self.target = \
            None, None, None, None, None

        super().__init__()

    # pylint: disable=arguments-differ
    def __getitem__(self, item):
        output = [self.coords[item], self.confidence[item], self.padding_mask[item],
                  self.prev_output_tokens[item], self.target[item]]
        return output

    def __len__(self):
        return len(self.traindata)

    # pylint: disable=arguments-differ
    def process(self, pdbfile, chain="B"):
        coords, _ = load_coords(pdbfile, chain)
        return coords

    def data_generation(self, alphabet):
        """Data generation"""
        with open(self.training_data_src, "r") as f:
            traindata = json.load(f)
            f.close()
        self.traindata = traindata
        trainset = []
        for seq in self.traindata:
            trainset.append(self.mask(0.15, seq, p=0.05))
        batch = [(e["coords"], None, e["seq"]) for e in trainset[:]]
        batch_converter = CoordBatchConverter(alphabet)
        coords, confidence, _, tokens, padding_mask = (
            batch_converter(batch)
        )
        prev_output_tokens = tokens[:, :-1]
        target = tokens[:, 1:]
        prev_output_tokens = prev_output_tokens.astype(np.int32)
        target = target.astype(np.int32)
        output = [coords, confidence, padding_mask,
                  prev_output_tokens, target]
        return output

    def mask(self, mask_ratio, sentence, lower=1, upper=10, p=0.05):
        """Span masking"""

        sent_length = len(sentence['coords'])
        mask_num = math.ceil(sent_length * mask_ratio)
        mask = set()
        while len(mask) < mask_num:
            lens = list(range(lower, upper + 1))
            len_distrib = [p * (1 - p) ** (i - lower) for i in
                           range(lower, upper + 1)] if p >= 0 else None
            len_distrib = [x / (sum(len_distrib)) for x in len_distrib]
            span_len = np.random.choice(lens, p=len_distrib)
            anchor = np.random.choice(sent_length)
            if anchor in mask:
                continue
            for i in range(anchor, anchor + span_len):
                if len(mask) >= mask_num or i >= sent_length:
                    break
                mask.add(i)

        for num in mask:
            rand = np.random.random()
            if rand < 0.8:
                sentence['coords'][num - 1] = [[float('inf'), float('inf'), float('inf')],
                                               [float('inf'), float('inf'), float('inf')],
                                               [float('inf'), float('inf'), float('inf')]]
            elif rand < 0.9:
                # sample random token according to input distribution
                sentence['coords'][num - 1] = sentence['coords'][np.random.choice(sent_length)]
        return sentence

    def test_data(self, seq_length):
        pass

    # pylint: disable=arguments-differ
    def download(self):
        pass

    def set_training_data_src(self, data_src):
        self.training_data_src = data_src

    # pylint: disable=arguments-differ
    def create_iterator(self, num_epochs):
        self.coords, self.confidence, self.padding_mask, self.prev_output_tokens, self.target = \
            self.data_generation(self.alphabet)
        dataset = GeneratorDataset(source=self, column_names=self.feature_list, num_parallel_workers=1, shuffle=False)
        dataset = dataset.batch(self.batch_size)
        iteration = dataset.create_dict_iterator(num_epochs=num_epochs)
        return iteration
