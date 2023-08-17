# Copyright 2023 Huawei Technologies Co., Ltd
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
"proteinmpnndataset"
import os
import pickle

from ...dataset import PSP, data_process_run

from .proteinmpnndata import pre_process, tied_featurize, featurize
from .dataset import StructureDatasetPDB, Definebatch, parse_pdb


class ProteinMpnnDataset(PSP):
    """proteinmpnndataset"""

    def __init__(self, config):
        self.config = config
        self.supported_models = ['Proteinmpnn']
        self.in_memory = False
        self.phase = None
        self.is_training = self.config.is_training
        self.proteinmpnn_inputs()
        if self.is_training:
            self.data_process = [featurize()]
        else:
            self.data_process = [pre_process(), tied_featurize()]
        super().__init__()

    # pylint: disable=arguments-differ
    def __getitem__(self, idx):
        if self.in_memory:
            data = self.inputs[idx]
        else:
            data = self.data_parse(idx)

        self.indx += 1
        features = self.process(data)
        return tuple(features)

    def __len__(self):
        data_len = len(os.listdir(self.training_pdb_path))
        return data_len

    def proteinmpnn_inputs(self):
        if self.is_training:
            feature_list = ["x", "s", "mask", "chain_m", "residue_idx", "chain_encoding_all", "mask_for_loss"]
        else:
            feature_list = ["x", "s", "mask", "chain_m", "chain_m_pos", "residue_idx", "chain_encoding_all", "randn_1",
                            "omit_aas_np", "bias_aas_np", "omit_aa_mask", "pssm_coef", "pssm_bias",
                            "masked_chain_length_list_list", "masked_list_list",
                            "bias_by_res_all", "pssm_log_odds_mask"]
        self.feature_list = feature_list

    def data_parse(self, idx):
        pkl_path = self.training_pkl_items[idx]
        f = open(pkl_path, "rb")
        data = pickle.load(f)
        return data

    # pylint: disable=arguments-differ
    def process(self, data):
        pdb_dict_list = parse_pdb(data)
        all_chain_list = [item[-1:] for item in list(pdb_dict_list[0]) if item[:9] == 'seq_chain']
        designed_chain_list = all_chain_list
        fixed_chain_list = [letter for letter in all_chain_list if letter not in designed_chain_list]
        chain_id_dict = {}
        chain_id_dict[pdb_dict_list[0]['name']] = (designed_chain_list, fixed_chain_list)
        features = data_process_run(pdb_dict_list.copy(), self.data_process)
        return features

    def set_training_data_src(self, data_src):
        self.pkl_path = data_src

    # pylint: disable=arguments-differ
    def create_iterator(self, num_epochs):
        with open(self.pkl_path, 'rb') as f_read:
            pdb_dict_train = pickle.load(f_read)
        dataset_train = StructureDatasetPDB(pdb_dict_train, truncate=None, max_length=100)
        loader_train = Definebatch(dataset_train, num_epochs, batch_size=10000)
        return loader_train
