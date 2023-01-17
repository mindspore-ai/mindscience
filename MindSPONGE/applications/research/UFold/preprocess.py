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
"""preprocess"""
import os
import argparse
import collections
import numpy as np
import mindspore as ms
from mindspore.dataset import GeneratorDataset
from src.data_generator import RNASSDataGenerator
from src.data_generator import DatasetCutConcatNewCanonicle as Dataset_FCN


parser = argparse.ArgumentParser('preprocess')
parser.add_argument('--test_files', required=False, nargs='?',
                    default='TS2', choices=['ArchiveII', 'TS0', 'bpnew', 'TS1', 'TS2', 'TS3'],
                    help='test file name')
parser.add_argument('--target_path', help='target directory', default='./ascend310_infer/preprocess_Result')
parser.add_argument("--ori", type=str, default="ascend310_infer/ori", help="result ori path.")
parser.add_argument("--contact", type=str, default="ascend310_infer/contacts", help="result ori path.")
args = parser.parse_args()


if __name__ == "__main__":
    ms.set_seed(1)
    RNA_SS_data = collections.namedtuple('RNA_SS_data', 'seq ss_label length name pairs')
    test_file = args.test_files
    print('Loading test file: ', test_file)
    if test_file == 'ArchiveII':
        test_data = RNASSDataGenerator('data/', test_file+'.pickle')
    else:
        test_data = RNASSDataGenerator('data/', test_file+'.cPickle')

    test_set = Dataset_FCN(test_data)
    test_generator = GeneratorDataset(test_set, column_names=['contacts', 'seq_embeddings', 'matrix_reps',
                                                              'seq_lens', 'seq_ori', 'seq_name', 'nc_map', 'l_len'],
                                      num_parallel_workers=1,
                                      shuffle=True).batch(batch_size=1, drop_remainder=True)
    data_path = args.target_path
    ori_path = args.ori
    con_path = args.contact
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    if not os.path.exists(ori_path):
        os.makedirs(ori_path)
    if not os.path.exists(con_path):
        os.makedirs(con_path)

    for i, (contacts, seq_embeddings, _, _, seq_ori, _, _, _) in enumerate(test_generator):
        file_name = "ufold_" + test_file + "_" + str(i) + ".bin"
        file_ori = "ufold_" + test_file + "_ori_" + str(i) + ".bin"
        file_con = "ufold_" + test_file + "_con_" + str(i) + ".bin"
        image_file_path = os.path.join(data_path, file_name)
        image_ori_path = os.path.join(ori_path, file_ori)
        image_con_path = os.path.join(con_path, file_con)
        # 利用numpy中数组的成员函数tofile将数据集存储为二进制流文件
        seq_embeddings.asnumpy().astype(np.float32).tofile(image_file_path)
        seq_ori.asnumpy().astype(np.float32).tofile(image_ori_path)
        contacts.asnumpy().astype(np.float32).tofile(image_con_path)
    print("Export bin files finished!")
