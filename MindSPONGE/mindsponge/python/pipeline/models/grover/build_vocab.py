# Copyright 2022 Huawei Technologies Co., Ltd
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
"""
The vocabulary building scripts.
"""
import os
from .src.data.mindsporevocab import MolVocab


class BuildVocab:
    """BuildVocab"""
    def __init__(self):
        pass

    def build_vocab(self, data_path, vocab_save_folder, dataset_name, vocab_min_freq):
        """
        Build vocab(atom/bond) for unlabelled data training.
        """
        for vocab_type in ['atom', 'bond']:
            vocab_file = f"{vocab_type}_vocab.pkl"
            if dataset_name is not None:
                vocab_file = dataset_name + '_' + vocab_file
            vocab_save_path = os.path.join(vocab_save_folder, vocab_file)
            os.makedirs(os.path.dirname(vocab_save_path), exist_ok=True)
            vocab = MolVocab(file_path=data_path,
                             min_freq=vocab_min_freq,
                             num_workers=1,
                             vocab_type=vocab_type)
            print(f"{vocab_type} vocab size", len(vocab))
            vocab.save_vocab(vocab_save_path)
