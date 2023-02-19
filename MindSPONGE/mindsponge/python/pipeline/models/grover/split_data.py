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
"""Split dataset"""
import random
import csv
import os
import stat
from .src.util.utils import load_smiles

TRAIN_PERCENT = 0.8
VAL_PERCENT = 0.2

random.seed(3)


class SplitData:
    """SplitData"""
    def __init__(self):
        pass

    def split_data(self, data_dir, file_name):
        """
        Split data for training and evaluating.
        """
        data_path = os.path.join(data_dir, file_name + ".csv")
        train_data_path = os.path.join(data_dir, file_name + "_train.csv")
        val_data_path = os.path.join(data_dir, file_name + "_val.csv")

        smiles = load_smiles(data_path)
        num_smiles = len(smiles)

        list_smiles = range(num_smiles)

        num_train = int(num_smiles * TRAIN_PERCENT)
        num_val = int(num_smiles * VAL_PERCENT)

        train = random.sample(list_smiles, num_train)
        val_test = [i for i in list_smiles if not i in train]
        val = random.sample(val_test, num_val)
        print("train: {}, val: {}".format(len(train), len(val)))

        flags = os.O_WRONLY | os.O_CREAT
        modes = stat.S_IWUSR | stat.S_IRUSR
        with os.fdopen(os.open(train_data_path, flags, modes), 'w', newline='') as train_file:
            train_writer = csv.writer(train_file)
            train_writer.writerow(["smiles"])
            for i in train:
                train_writer.writerow(smiles[i])

        with os.fdopen(os.open(val_data_path, flags, modes), 'w', newline='') as val_file:
            val_writer = csv.writer(val_file)
            val_writer.writerow(["smiles"])
            for i in val:
                val_writer.writerow(smiles[i])
