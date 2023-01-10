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
import argparse
from src.data.torchvocab import MolVocab

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default="../../dataset/grover_new_dataset/druglike_merged_refine2.csv", type=str)
parser.add_argument('--vocab_save_folder', default="../../dataset/grover_new_dataset", type=str)
parser.add_argument('--dataset_name', type=str, default=None,
                    help="Will be the first part of the vocab file name. If it is None,"
                         "the vocab files will be: atom_vocab.pkl and bond_vocab.pkl")
parser.add_argument('--vocab_max_size', type=int, default=None)
parser.add_argument('--vocab_min_freq', type=int, default=1)
args = parser.parse_args()


def build_vocab():
    """
    Build vocab(atom/bond) for unlabelled data training.
    """
    for vocab_type in ['atom', 'bond']:
        vocab_file = f"{vocab_type}_vocab.pkl"
        if args.dataset_name is not None:
            vocab_file = args.dataset_name + '_' + vocab_file
        vocab_save_path = os.path.join(args.vocab_save_folder, vocab_file)
        os.makedirs(os.path.dirname(vocab_save_path), exist_ok=True)
        vocab = MolVocab(file_path=args.data_path,
                         min_freq=args.vocab_min_freq,
                         num_workers=1,
                         vocab_type=vocab_type)
        print(f"{vocab_type} vocab size", len(vocab))
        vocab.save_vocab(vocab_save_path)


if __name__ == '__main__':
    build_vocab()
