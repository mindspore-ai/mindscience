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
"""Fragmented functions"""
import math
import argparse
import numpy as np
import mindspore as ms
import mindspore.ops as ops


sign = ops.Sign()

label_dict = {
    '.': np.array([1, 0, 0]),
    '(': np.array([0, 1, 0]),
    ')': np.array([0, 0, 1])
}
seq_dict = {
    'A': np.array([1, 0, 0, 0]),
    'U': np.array([0, 1, 0, 0]),
    'C': np.array([0, 0, 1, 0]),
    'G': np.array([0, 0, 0, 1]),
    'N': np.array([0, 0, 0, 0]),
    'M': np.array([1, 0, 1, 0]),
    'Y': np.array([0, 1, 1, 0]),
    'W': np.array([1, 0, 0, 0]),
    'V': np.array([1, 0, 1, 1]),
    'K': np.array([0, 1, 0, 1]),
    'R': np.array([1, 0, 0, 1]),
    'I': np.array([0, 0, 0, 0]),
    'X': np.array([0, 0, 0, 0]),
    'S': np.array([0, 0, 1, 1]),
    'D': np.array([1, 1, 0, 1]),
    'P': np.array([0, 0, 0, 0]),
    'B': np.array([0, 1, 1, 1]),
    'H': np.array([1, 1, 1, 0])
}

char_dict = {
    0: 'A',
    1: 'U',
    2: 'C',
    3: 'G'
}


def get_args():
    """get user input"""
    argparser = argparse.ArgumentParser(description="diff through pp")
    argparser.add_argument(
        '-c', '--config',
        metavar='C',
        default='src/config.json',
        help='The Configuration file'
    )
    argparser.add_argument('--test', type=bool, default=False,
                           help='skip training to test directly.')
    argparser.add_argument('--nc', type=int, default=0, choices=[0, 1],
                           help='whether predict non-canonical pairs.')
    argparser.add_argument('--train_files', type=str, required=False, nargs='+',
                           default=['ArchiveII', 'TS0', 'bpnew', 'TS1', 'TS2', 'TS3'],
                           help='training file name list.')
    argparser.add_argument('--test_files', required=False, nargs='?',
                           default='ArchiveII', choices=['ArchiveII', 'TS0', 'bpnew', 'TS1', 'TS2', 'TS3'],
                           help='test file name')
    argparser.add_argument("--ckpt_file", type=str, help="Checkpoint file path.")
    argparser.add_argument('--device_target', type=str, default='GPU', choices=['GPU', 'Ascend'],
                           help='device where the code will be implemented. (Default: GPU)')
    argparser.add_argument("--device_id", type=int, default=0, help="Device id")
    args = argparser.parse_args()
    return args


def gaussian(x):
    return math.exp(-0.5*(x*x))


def paired(x, y):
    """get pair score"""
    if x == 'A' and y == 'U':
        return 2
    if x == 'G' and y == 'C':
        return 3
    if x == 'G' and y == 'U':
        return 0.8
    if x == 'U' and y == 'A':
        return 2
    if x == 'C' and y == 'G':
        return 3
    if x == 'U' and y == 'G':
        return 0.8
    return 0


def encoding2seq(arr):
    seq = list()
    for arr_row in list(arr):
        if sum(arr_row) == 0:
            seq.append('.')
        else:
            seq.append(char_dict.get(np.argmax(arr_row)))
    return ''.join(seq)


def evaluate_exact_new(pred_a, true_a, eps=1e-11):
    """get pred, recall and f1_score"""
    tp_map = sign(ms.Tensor(pred_a) * ms.Tensor(true_a))
    tp = tp_map.sum()
    pred_p = sign(ms.Tensor(pred_a)).sum()
    true_p = true_a.sum()
    fp = pred_p - tp
    fn = true_p - tp
    recall = (tp + eps)/(tp + fn + eps)
    precision = (tp + eps)/(tp + fp + eps)
    f1_score_ms = (2 * tp + eps)/(2 * tp + fp + fn + eps)
    return precision, recall, f1_score_ms
