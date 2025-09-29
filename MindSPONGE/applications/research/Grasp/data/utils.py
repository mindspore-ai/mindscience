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
import gzip
import numpy as np
from absl import logging
from scipy import sparse as sp

from .parsers import parse_fasta

truncated_normal_stddev_factor = np.asarray(.87962566103423978, dtype=np.float32)


@contextlib.contextmanager
def tmpdir_manager(base_dir: str):
    """Context manager that deletes a temporary directory on exit.
    for example:
        with tmpdir_manager(base_dir='/tmp') as tmp_dir:
            test_file = os.path.join(tmp_dir, 'input.fasta')
            with open(test_file, "w") as f:
               f.write("this is a test. \n")
            print("exit")
    this would create a tmp data directory and when finished the main process of writing "this is a test. \n" into
    the tmp file,(after print "exit"), the system would destroy the previous tmp dir
    """
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


# def load_pickle(path):
#     def load(path):
#         assert path.endswith(".pkl") or path.endswith(
#             ".pkl.gz"
#         ), f"bad suffix in {path} as pickle file."
#         open_fn = gzip.open if path.endswith(".gz") else open
#         with open_fn(path, "rb") as f:
#             return pickle.load(f)

#     ret = load(path)
#     ret = uncompress_features(ret)
#     return ret


# def uncompress_features(feats):
#     if "sparse_deletion_matrix_int" in feats:
#         v = feats.pop("sparse_deletion_matrix_int")
#         v = to_dense_matrix(v)
#         feats["deletion_matrix"] = v
#     return feats


# def to_dense_matrix(spmat_dict):
#     spmat = sp.coo_matrix(
#         (spmat_dict["data"], (spmat_dict["row"], spmat_dict["col"])),
#         shape=spmat_dict["shape"],
#         dtype=np.float32,
#     )
#     return spmat.toarray()


def str_hash(text):
  hash=0
  for ch in text:
    hash = ( hash*281  ^ ord(ch)*997) & 0xFFFFFFFF
  return hash


@contextlib.contextmanager
def numpy_seed(seed, *addl_seeds, key=None):
    """Context manager which seeds the NumPy PRNG with the specified seed and
    restores the state afterward"""
    if seed is None:
        yield
        return
    def check_seed(s):
        assert type(s) == int or type(s) == np.int32 or type(s) == np.int64
    check_seed(seed)
    if len(addl_seeds) > 0:
        for s in addl_seeds:
            check_seed(s)
        seed = int(hash((seed, *addl_seeds)) % 1e8)
        if key is not None:
            seed = int(hash((seed, str_hash(key))) % 1e8)
    state = np.random.get_state()
    np.random.seed(seed)
    # np.random.seed(123)
    try:
        yield
    finally:
        np.random.set_state(state)


# def batch_by_size(
#     indices,
#     batch_size=None,
#     required_batch_size_multiple=1,
# ):
#     """
#     Yield mini-batches of indices bucketed by size. Batches may contain
#     sequences of different lengths.

#     Args:
#         indices (List[int]): ordered list of dataset indices
#         batch_size (int, optional): max number of sentences in each
#             batch (default: None).
#         required_batch_size_multiple (int, optional): require batch size to
#             be less than N or a multiple of N (default: 1).
#     """

#     batch_size = batch_size if batch_size is not None else 1
#     bsz_mult = required_batch_size_multiple

#     step = ((batch_size + bsz_mult - 1) // bsz_mult) * bsz_mult

#     if not isinstance(indices, np.ndarray):
#         indices = np.fromiter(indices, dtype=np.int64, count=-1)

#     num_batches = (len(indices) + step - 1) // step
#     steps = np.arange(num_batches - 1) + 1
#     steps *= step
#     batch_indices = np.split(indices, steps)
#     assert len(batch_indices) == num_batches
#     # validation or test data size is smaller than a mini-batch size in some downstream tasks.
#     assert batch_indices[0].shape[0] <= step
#     return batch_indices
