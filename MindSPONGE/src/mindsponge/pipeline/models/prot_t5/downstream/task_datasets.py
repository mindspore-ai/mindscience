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
"""dataset loader of tasks."""
import random
import re

import numpy as np
import pandas as pd

import mindspore.dataset as ds
from mindformers import T5Tokenizer

from ..utils.utils import seqs_tokenizer
from ....dataset import DataSet

LABEL_MASKER = -100

# Category Definitions in Data
MEMBRANE_CATES = {'M': 0, 'S': 1, 'U': 1}
LOC_CATES = {
    'Cell.membrane': 0, 'Cytoplasm': 1, 'Endoplasmic.reticulum': 2,
    'Golgi.apparatus': 3, 'Lysosome/Vacuole': 4, 'Mitochondrion': 5,
    'Nucleus': 6, 'Peroxisome': 7, 'Plastid': 8, 'Extracellular': 9
    }
HHBLITS_D3_CATES = {'C': 0, 'E': 1, 'H': 2}
HHBLITS_D8_CATES = {'C': 0, 'E': 1, 'H': 2, 'B': 3, 'G': 4, 'I': 5, 'S': 6, 'T': 7}


def reverse_dict(original_dict):
    """reverse dict"""
    return {value: key for key, value in original_dict.items()}

MEMBRANE_LABEL_TO_CATE = {0: 'M', 1: 'S'}
LOC_LABEL_TO_CATE = reverse_dict(LOC_CATES)
HHBLITS_D3_LABEL_TO_CATE = reverse_dict(HHBLITS_D3_CATES)
HHBLITS_D8_LABEL_TO_CATE = reverse_dict(HHBLITS_D8_CATES)


def map_label_to_category(label_tensor, dct):
    """map label to category"""
    labels = label_tensor.asnumpy()
    vectorized_map = np.vectorize(lambda label: dct.get(label, ''))
    str_labels = vectorized_map(labels)
    return str_labels


def seq2array(seq):
    """sequence to numpy array"""
    return np.array(seq, dtype=np.int32)


def pad_trunc_addspecial(seq, max_length, sp=LABEL_MASKER, pad=0, add_special_tokens=True):
    """pad trunc addspecial"""
    if len(seq) > max_length:
        if add_special_tokens:
            seq = seq[:max_length-1]
        else:
            seq = seq[:max_length]

    if add_special_tokens:
        seq.append(sp)

    padded = seq + [pad] * max_length
    return padded[:max_length]


def get_task_info(cate_name):
    """get task info"""
    if cate_name == 'loc':
        return LOC_CATES
    if cate_name == 'membrane':
        return MEMBRANE_CATES
    return {}


def apply_tokenizer(text, tokenizer):
    """apply tokenizer"""
    text = re.sub(r"[UZOB]", "X", text)
    tokens = tokenizer(text, padding='max_length', truncation=True, add_special_tokens=True, max_length=512)
    ids, masks = tokens['input_ids'], tokens['attention_mask']
    return ids, masks


def create_deeploc_dataset(file_path, tokenizer, batch_size=32, cate_name=''):
    """create deeploc dataset"""
    df = pd.read_csv(file_path)
    res = []
    df = df.rename(columns=lambda x: x.strip())

    cate_dict = get_task_info(cate_name)

    for _, row in df.iterrows():
        text, cate = row['input'], row[cate_name]
        ids, masks = apply_tokenizer(text, tokenizer)
        eles = [seq2array(x) for x in [ids, masks, cate_dict.get(cate)]]
        res.append(tuple(eles))

    random.shuffle(res)
    dataset = ds.GeneratorDataset(res, column_names=["inputs", "masks", "labels"])
    dataset = dataset.shuffle(buffer_size=128)
    dataset = dataset.batch(batch_size=batch_size)
    return dataset


def load_hhblits_dataset(path):
    """load hhblits dataset"""
    df = pd.read_csv(path, names=['input', 'dssp3', 'dssp8', 'disorder', 'cb513_mask'], skiprows=1)
    df = df.rename(columns=lambda x: x.strip())

    input_fixed = ["".join(seq.split()) for seq in df['input']]
    input_fixed = [re.sub(r"[UZOB]", "X", seq) for seq in input_fixed]
    seqs = [" ".join(seq) for seq in input_fixed]

    label_fixed3 = ["".join(label.split()) for label in df['dssp3']]
    d3_labels = [list(label) for label in label_fixed3]

    label_fixed8 = ["".join(label.split()) for label in df['dssp8']]
    d8_labels = [list(label) for label in label_fixed8]

    return seqs, d3_labels, d8_labels


def create_hhblits_dataset(file_path, tokenizer, batch_size=32):
    """create hhblits dataset"""
    seqs, d3_labels, d8_labels = load_hhblits_dataset(file_path)
    res = []
    for seq, d3, d8 in zip(seqs, d3_labels, d8_labels):
        ids, masks = apply_tokenizer(seq, tokenizer)

        d3 = [HHBLITS_D3_CATES[x.strip()] for x in d3]
        d3 = pad_trunc_addspecial(d3, 512, pad=LABEL_MASKER)

        d8 = [HHBLITS_D8_CATES[x.strip()] for x in d8]
        d8 = pad_trunc_addspecial(d8, 512, pad=LABEL_MASKER)

        eles = [ids, masks, d3, d8]
        eles_tp = [seq2array(x) for x in eles]
        res.append(tuple(eles_tp))

    random.shuffle(res)
    dataset = ds.GeneratorDataset(res, column_names=["inputs", "masks", "d3labels", "d8labels"])
    dataset = dataset.shuffle(buffer_size=128)
    dataset = dataset.batch(batch_size=batch_size)
    return dataset


class ProtT5TaskDataSet(DataSet):
    """ProtT5 downstream task dataSet"""
    def __init__(self, config):
        self.task_name = config.task_name
        self.data_path = None
        self.dataset = None
        self.batch_size = config.train.batch_size
        self.t5_tokenizer = T5Tokenizer.from_pretrained(config.t5_config_path)
        self.phase = None

        super().__init__()

    # pylint: disable=E0302
    def __getitem__(self):
        pass

    def __len__(self):
        if self.dataset:
            return self.dataset.get_dataset_size()
        return 0

    def set_phase(self, phase):
        self.phase = phase

    def process(self, data, **kwargs):
        return seqs_tokenizer(data, self.t5_tokenizer, return_tensors="ms")

    def set_training_data_src(self, data_source):
        self.data_path = data_source

    def download(self, path=None):
        raise NotImplementedError

    def data_parse(self, idx):
        raise NotImplementedError

    # pylint: disable=W0221
    def create_iterator(self, num_epochs, cate_name=''):
        if self.task_name == "hhblits":
            self.dataset = create_hhblits_dataset(self.data_path, self.t5_tokenizer, self.batch_size)
        else:
            self.dataset = create_deeploc_dataset(self.data_path, self.t5_tokenizer, self.batch_size, \
                                                  cate_name=cate_name)

        return self.dataset
