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
Preprocess the data for 310 inferring.
"""
import argparse
import os
import numpy as np
from src.data.dataset import create_grover_dataset


def generate_bin():
    """Generate bin files."""

    parser = argparse.ArgumentParser(description='postprocess')
    parser.add_argument('--data_dir', type=str, default='./exampledata/finetune', help='input path')
    parser.add_argument('--dataset', type=str, default='bbbp', help='Dataset name')
    parser.add_argument('--dataset_type', type=str, default='classification', help='Dataset type')
    parser.add_argument('--result_path', type=str, default='./preprocess_Result', help='Result path')
    parser.add_argument('--scaler_path', type=str, default='../ckpt', help='scaler dir')
    parser.add_argument('--batch_size', type=int, default=2, help='batch size')
    parser.add_argument('--features_scaling', type=bool, default=False, help='Normalize feature')
    parser.add_argument('--device_num', type=int, default=1, help='The num of device')
    parser.add_argument('--bond_drop_rate', type=int, default=0)
    parser.add_argument('--no_cache', type=bool, default=True)
    config = parser.parse_args()

    config.result_path = os.path.join(config.result_path, config.dataset)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)

    features_batch_path = os.path.join(config.result_path, "00_features_batch")
    if not os.path.exists(features_batch_path):
        os.makedirs(features_batch_path)

    f_atoms_path = os.path.join(config.result_path, "01_f_atoms")
    if not os.path.exists(f_atoms_path):
        os.makedirs(f_atoms_path)

    f_bonds_path = os.path.join(config.result_path, "02_f_bonds")
    if not os.path.exists(f_bonds_path):
        os.makedirs(f_bonds_path)

    a2b_path = os.path.join(config.result_path, "03_a2b")
    if not os.path.exists(a2b_path):
        os.makedirs(a2b_path)

    b2a_path = os.path.join(config.result_path, "04_b2a")
    if not os.path.exists(b2a_path):
        os.makedirs(b2a_path)

    b2revb_path = os.path.join(config.result_path, "05_b2revb")
    if not os.path.exists(b2revb_path):
        os.makedirs(b2revb_path)

    a2a_path = os.path.join(config.result_path, "06_a2a")
    if not os.path.exists(a2a_path):
        os.makedirs(a2a_path)

    a_scope_path = os.path.join(config.result_path, "07_a_scope")
    if not os.path.exists(a_scope_path):
        os.makedirs(a_scope_path)

    b_scope_path = os.path.join(config.result_path, "08_b_scope")
    if not os.path.exists(b_scope_path):
        os.makedirs(b_scope_path)

    targets_path = os.path.join(config.result_path, "09_targets")
    if not os.path.exists(targets_path):
        os.makedirs(targets_path)

    preds_path = os.path.join(config.result_path, "10_preds")
    if not os.path.exists(preds_path):
        os.makedirs(preds_path)

    smiles_path = os.path.join(config.data_dir, config.dataset + "_val.csv")
    feature_path = os.path.join(config.data_dir, config.dataset + "_val.npz")
    config.scaler_path = os.path.join(config.scaler_path, config.dataset + "_scaler")

    dataset, _ = create_grover_dataset(config, smiles_path=smiles_path, feature_path=feature_path,
                                       is_training=False)
    data_loader = dataset.create_dict_iterator(output_numpy=True)

    for step_idx, data in enumerate(data_loader):
        file_name = "GROVER_bs" + str(config.batch_size) + "_" + str(step_idx) + ".bin"
        features_batch = data["features"]
        targets = data["labels"].astype(np.float32)
        f_atoms = data["f_atoms"]
        f_bonds = data["f_bonds"]
        a2b = data["a2b"]
        b2a = data["b2a"]
        b2revb = data["b2revb"]
        a2a = data["a2a"]
        a_scope = data["a_scope"].astype(np.int64)
        b_scope = data["b_scope"].astype(np.int64)

        features_batch.tofile(os.path.join(features_batch_path, file_name))
        f_atoms.tofile(os.path.join(f_atoms_path, file_name))
        f_bonds.tofile(os.path.join(f_bonds_path, file_name))
        a2b.tofile(os.path.join(a2b_path, file_name))
        b2a.tofile(os.path.join(b2a_path, file_name))
        b2revb.tofile(os.path.join(b2revb_path, file_name))
        a2a.tofile(os.path.join(a2a_path, file_name))
        a_scope.tofile(os.path.join(a_scope_path, file_name))
        b_scope.tofile(os.path.join(b_scope_path, file_name))
        targets_name = "targets.bin"
        targets.tofile(os.path.join(targets_path, targets_name))
        preds_name = config.dataset + ".bin"
        preds = np.zeros(targets.shape, np.float32)
        preds.tofile(os.path.join(preds_path, preds_name))

        if step_idx == 0:
            break

    print("=" * 20, 'export files finished', "=" * 20)


if __name__ == '__main__':
    generate_bin()
