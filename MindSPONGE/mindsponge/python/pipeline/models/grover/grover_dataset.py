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
"""grover_dataset"""
import os
import multiprocessing
import numpy as np
from mindspore import dataset as ds
from .src.util.utils import load_features, load_smiles_labels
from .src.data.mindsporevocab import MolVocab
from .src.data.transforms import MolCollator, GroverCollator, normalize_data
from ...dataset import DataSet


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


class GroverDataSet(DataSet):
    """
    GROVER Dataset.
    """
    def __init__(self, config):
        self.config = config
        self.in_memory = False
        self.is_training = False
        self.mode = None
        self.smiles_path = None
        self.feature_path = None
        self.smiles_list = None
        self.labels_list = None
        self.labels_scaler = None
        self.features_list = None
        self.features_scaler = None
        self.atom_vocab_path = None
        self.bond_vocab_path = None
        super().__init__()

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

    def process(self, data, **kwargs):
        return data

    def download(self, path=None):
        pass

    def data_parse(self, idx):
        pass

    def set_training_data_src(self, data_src):
        """set_training_data_src"""
        if self.config.parser_name == "eval":
            self.smiles_path = os.path.join(data_src, "bbbp_val.csv")
            self.feature_path = os.path.join(data_src, "bbbp_val.npz")
            self.config.scaler_path = os.path.join(data_src, "bbbp_scaler")
            self.is_training = False
            self.mode = "finetune"
        elif self.config.parser_name == "gen":
            self.smiles_path = os.path.join(data_src, "bbbp_val.csv")
            self.feature_path = os.path.join(data_src, "bbbp_val.npz")
            self.is_training = False
            self.mode = "finetune"
        elif self.config.parser_name == "pretrain":
            self.smiles_path = os.path.join(data_src, "tryout_train.csv")
            self.feature_path = os.path.join(data_src, "tryout_train.npz")
            self.atom_vocab_path = os.path.join(data_src, "tryout_atom_vocab.pkl")
            self.bond_vocab_path = os.path.join(data_src, "tryout_bond_vocab.pkl")
            self.mode = "pretrain"
            self.is_training = True
        else:
            self.smiles_path = os.path.join(data_src, "bbbp_train.csv")
            self.feature_path = os.path.join(data_src, "bbbp_train.npz")
            self.config.scaler_path = os.path.join(data_src, "bbbp_scaler")
            self.mode = "finetune"
            if not os.path.exists(self.config.scaler_path):
                os.makedirs(self.config.scaler_path)
            self.is_training = True
        self.smiles_list, self.labels_list, self.labels_scaler = get_smiles_labels(self.config, self.smiles_path,
                                                                                   self.is_training, self.mode)
        self.features_list, self.features_scaler = get_features(self.config, self.feature_path, self.is_training,
                                                                self.mode)

    def create_iterator(self, num_epochs, **kwargs):
        if self.config.parser_name == "pretrain":
            dataset = self.create_pretrain_dataset()
            iteration = dataset.create_dict_iterator(output_numpy=False)
        else:
            dataset = self.create_grover_dataset()
            iteration = dataset.create_dict_iterator(output_numpy=False)

        return iteration

    def create_pretrain_dataset(self):
        """
        Create dataset for pretrain model.
        """
        cores = multiprocessing.cpu_count()
        num_parallel_workers = int(cores / self.config.device_num)

        # load atom and bond vocabulary and the semantic motif labels.
        atom_vocab = MolVocab.load_vocab(self.atom_vocab_path)
        bond_vocab = MolVocab.load_vocab(self.bond_vocab_path)
        self.config.atom_vocab_size, self.config.bond_vocab_size = len(atom_vocab), len(bond_vocab)
        self.config.fg_size = 85

        mol_collator = GroverCollator(shared_dict={}, atom_vocab=atom_vocab, bond_vocab=bond_vocab, args=self.config)
        per_batch_match_op = mol_collator.per_batch_map

        dataset_column_names = ["smiles", "features", "none"]
        output_columns = ["f_atoms", "f_bonds", "a2b", "b2a", "b2revb", "a2a", "a_scope", "b_scope", "atom_vocab_label",
                          "bond_vocab_label", "fgroup_label"]
        dataset = ds.GeneratorDataset(self, column_names=dataset_column_names,
                                      shuffle=False, num_shards=self.config.device_num, shard_id=self.config.rank)
        dataset = dataset.batch(batch_size=self.config.batch_size, num_parallel_workers=min(8, num_parallel_workers))
        dataset = dataset.map(operations=per_batch_match_op, input_columns=["smiles", "features"],
                              output_columns=output_columns,
                              num_parallel_workers=min(8, num_parallel_workers))
        dataset = dataset.project(output_columns)
        return dataset

    def create_grover_dataset(self):
        """
        Create dataset for train/eval model.
        """
        labels_scaler = self.labels_scaler
        self.config.num_tasks = self.get_num_tasks()
        self.config.output_size = self.config.num_tasks
        self.config.features_dim = self.get_features_dim()

        cores = multiprocessing.cpu_count()
        num_parallel_workers = int(cores / self.config.device_num)

        mol_collator = MolCollator({}, self.config)
        per_batch_match_op = mol_collator.per_batch_map

        dataset_column_names = ["smiles", "features", "labels"]
        output_columns = ["f_atoms", "f_bonds", "a2b", "b2a", "b2revb", "a2a", "a_scope", "b_scope", "smiles"]
        columns = ["f_atoms", "f_bonds", "a2b", "b2a", "b2revb", "a2a", "a_scope", "b_scope", "smiles", "features",
                   "labels"]
        dataset = ds.GeneratorDataset(self, column_names=dataset_column_names,
                                      shuffle=False, num_shards=self.config.device_num, shard_id=self.config.rank)
        dataset = dataset.batch(batch_size=self.config.batch_size, num_parallel_workers=min(8, num_parallel_workers))
        dataset = dataset.map(operations=per_batch_match_op, input_columns=["smiles"], output_columns=output_columns,
                              num_parallel_workers=min(8, num_parallel_workers))
        dataset = dataset.project(columns)
        self.config.labels_scaler = labels_scaler
        return dataset
