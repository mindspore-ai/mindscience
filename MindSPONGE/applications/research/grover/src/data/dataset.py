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
"""GROVER dataset."""
import multiprocessing
import os
import numpy as np
from mindspore import dataset as ds
from src.util.utils import load_features, load_smiles_labels
from src.data.mindsporevocab import MolVocab
from src.data.transforms import MolCollator, GroverCollator, normalize_data


def get_smiles_labels(args, smiles_file, is_training, mode):
    """
    Load and Process smiles and labels.
    """
    smiles_list, labels_list = load_smiles_labels(smiles_file)

    if mode == "finetune" and args.dataset_type == "regression":
        labels_scaler_path = os.path.join(args.scaler_path, "labels_scaler.ckpt")
        labels_list, labels_scaler = normalize_data(labels_list, replace_nan_token=None, is_training=is_training,
                                                    path=labels_scaler_path)
    else:
        labels_scaler = None

    return smiles_list, labels_list, labels_scaler


def get_features(args, features_file, is_training, mode):
    """
    Load and Process features.
    """
    lines = load_features(features_file)
    features_list = []
    for features in lines:
        if features is not None:
            # Fix nans in features
            replace_token = 0
            features = np.where(np.isnan(features), replace_token, features).astype(np.float32)
            features_list.append(features)

    # normalize features
    if mode == "finetune" and args.features_scaling:
        features_scaler_path = os.path.join(args.scaler_path, "features_scaler.ckpt")
        features_list, features_scaler = normalize_data(features_list, replace_nan_token=0, is_training=is_training,
                                                        path=features_scaler_path)
    else:
        features_scaler = None

    return features_list, features_scaler


class GroverDataset:
    """
    GROVER Dataset.
    """

    def __init__(self, args, smiles_file, feature_file, is_training, mode):
        self.is_training = is_training
        self.mode = mode
        self.smiles_file = smiles_file
        self.feature_file = feature_file
        self.smiles_list, self.labels_list, self.labels_scaler = get_smiles_labels(args, self.smiles_file,
                                                                                   self.is_training, self.mode)
        self.features_list, self.features_scaler = get_features(args, self.feature_file, self.is_training, self.mode)

    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        features = self.features_list[idx]
        labels = self.labels_list[idx]
        return smiles, features, labels

    def __len__(self):
        assert len(self.smiles_list) == len(self.features_list)
        return len(self.smiles_list)

    def get_features_dim(self):
        features_dim = len(self.features_list[0]) if self.features_list[0] is not None else 0
        return features_dim

    def get_num_tasks(self):
        num_tasks = len(self.labels_list[0]) if self.labels_list[0] is not None else 0
        return num_tasks


def create_pretrain_dataset(args, smiles_path, feature_path,
                            num_shards=1, shard_id=0, is_training=True):
    """
    Create dataset for pretrain model.
    """
    grover_dataset = GroverDataset(args, smiles_path, feature_path, is_training, mode="pretrain")

    cores = multiprocessing.cpu_count()
    num_parallel_workers = int(cores / args.device_num)

    # load atom and bond vocabulary and the semantic motif labels.
    atom_vocab = MolVocab.load_vocab(args.atom_vocab_path)
    bond_vocab = MolVocab.load_vocab(args.bond_vocab_path)
    atom_vocab_size, bond_vocab_size = len(atom_vocab), len(bond_vocab)
    fg_size = 85

    mol_collator = GroverCollator(shared_dict={}, atom_vocab=atom_vocab, bond_vocab=bond_vocab, args=args)
    per_batch_match_op = mol_collator.per_batch_map

    dataset_column_names = ["smiles", "features", "none"]
    output_columns = ["f_atoms", "f_bonds", "a2b", "b2a", "b2revb", "a2a", "a_scope", "b_scope", "atom_vocab_label",
                      "bond_vocab_label", "fgroup_label"]
    dataset = ds.GeneratorDataset(grover_dataset, column_names=dataset_column_names,
                                  shuffle=False, num_shards=num_shards, shard_id=shard_id)
    dataset = dataset.batch(batch_size=args.batch_size, num_parallel_workers=min(8, num_parallel_workers))
    dataset = dataset.map(operations=per_batch_match_op, input_columns=["smiles", "features"],
                          output_columns=output_columns,
                          num_parallel_workers=min(8, num_parallel_workers))
    dataset = dataset.project(output_columns)
    return dataset, (atom_vocab_size, bond_vocab_size, fg_size)


def create_grover_dataset(args, smiles_path, feature_path,
                          num_shards=1, shard_id=0, is_training=True):
    """
    Create dataset for train/eval model.
    """
    grover_dataset = GroverDataset(args, smiles_path, feature_path, is_training, mode="finetune")
    labels_scaler = grover_dataset.labels_scaler
    args.num_tasks = grover_dataset.get_num_tasks()
    args.output_size = args.num_tasks
    args.features_dim = grover_dataset.get_features_dim()

    cores = multiprocessing.cpu_count()
    num_parallel_workers = int(cores / args.device_num)

    mol_collator = MolCollator({}, args)
    per_batch_match_op = mol_collator.per_batch_map

    dataset_column_names = ["smiles", "features", "labels"]
    output_columns = ["f_atoms", "f_bonds", "a2b", "b2a", "b2revb", "a2a", "a_scope", "b_scope", "smiles"]
    columns = ["f_atoms", "f_bonds", "a2b", "b2a", "b2revb", "a2a", "a_scope", "b_scope", "smiles", "features",
               "labels"]
    dataset = ds.GeneratorDataset(grover_dataset, column_names=dataset_column_names,
                                  shuffle=False, num_shards=num_shards, shard_id=shard_id)
    dataset = dataset.batch(batch_size=args.batch_size, num_parallel_workers=min(8, num_parallel_workers))
    dataset = dataset.map(operations=per_batch_match_op, input_columns=["smiles"], output_columns=output_columns,
                          num_parallel_workers=min(8, num_parallel_workers))
    dataset = dataset.project(columns)

    return dataset, labels_scaler
