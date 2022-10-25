# Copyright 2021 The AIMM Group at Shenzhen Bay Laboratory & Peking University & Huawei Technologies Co., Ltd
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
utils module used for tmpdir generation.
"""
import time
import contextlib
import tempfile
import shutil
import pickle
import os

from absl import logging

from .parsers import parse_fasta


@contextlib.contextmanager
def tmpdir_manager(base_dir: str):
    tmpdir = tempfile.mkdtemp(dir=base_dir)
    try:
        yield tmpdir
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


@contextlib.contextmanager
def timing(msg: str):
    logging.info('Started %s', msg)
    tic = time.time()
    yield
    toc = time.time()
    logging.info('Finished %s in %.3f seconds', msg, toc - tic)


def get_raw_feature(input_path, feature_generator, use_pkl):
    '''get raw feature of protein by loading pkl file or searching from database'''
    if use_pkl:
        f = open(input_path, "rb")
        data = pickle.load(f)
        f.close()
        return data
    return feature_generator.monomer_feature_generate(input_path)


def get_crop_size(input_path, use_pkl):
    '''get crop size of sequence by comparing all input sequences\' length'''
    filenames = os.listdir(input_path)
    max_length = 0
    for filename in filenames:
        file_full_path = os.path.join(input_path, filename)
        if use_pkl:
            with open(file_full_path, "rb") as f:
                data = pickle.load(f)
            current_crop_size = (data["msa"].shape[1] // 256 + 1) * 256
            max_length = max(max_length, current_crop_size)
        else:
            with open(file_full_path, "r") as f:
                input_fasta_str = f.read()
            input_seqs, _ = parse_fasta(input_fasta_str)
            current_crop_size = (len(input_seqs[0]) // 256 + 1) * 256
            max_length = max(max_length, current_crop_size)

    return max_length


def multimer_get_raw_feature(input_paths: list, feature_generator, use_pkl):
    '''get raw feature of protein by loading pkl file or searching from database'''
    if use_pkl:
        f = open(input_paths[0], "rb")
        data = pickle.load(f)
        f.close()
        return data
    return feature_generator.multimer_feature_generate(input_paths)


def multimer_get_crop_size(input_paths: list, use_pkl):
    ''' multimer get crop size'''
    current_crop_size = 0
    if use_pkl:
        with open(input_paths[0], "rb") as f:
            data = pickle.load(f)
        current_crop_size = (data["msa"].shape[1] // 256 + 1) * 256
    else:
        total_input_seqs = 0
        for _, fasta_path_ in enumerate(input_paths):
            with open(fasta_path_, "r") as f:
                input_fasta_str = f.read()
            input_seqs, _ = parse_fasta(input_fasta_str)
            total_input_seqs += len(input_seqs[0])
        current_crop_size = (total_input_seqs // 256 + 1) * 256
    return current_crop_size
        