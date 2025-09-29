# Copyright 2023 @ Shenzhen Bay Laboratory &
#                  Peking University &
#                  Huawei Technologies Co., Ltd
#
# This code is a part of MindSPONGE:
# MindSpore Simulation Package tOwards Next Generation molecular modelling.
#
# MindSPONGE is open-source software based on the AI-framework:
# MindSpore (https://www.mindspore.cn/)
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
"""colabdesign dataset"""
import os
import pickle

from mindspore.dataset import GeneratorDataset

from ...dataset import PSP, data_process_run
from .colabdesign_data import prep, get_weights


class ColabDesignDataSet(PSP):
    """ColabDesignDataSet"""

    def __init__(self, config, num_seq=1):
        self.config = config
        self.supported_models = ['ColabDesign']
        self.in_memory = False
        self.colabdesign_inputs()
        self.indx = 0
        self.training_data_src = ""
        self.training_pkl_path = ""
        self.training_pdb_path = ""
        self.training_pdb_items = ""
        self.training_pkl_items = ""
        self.data_process = [get_weights(self.indx, cfg=config), prep(cfg=config)]

        self._num = num_seq
        super().__init__()

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

    def colabdesign_inputs(self):
        feature_list = ["msa_feat", "msa_mask", "seq_mask_batch", \
                        "template_aatype", "template_all_atom_masks", "template_all_atom_positions", "template_mask", \
                        "template_pseudo_beta_mask", "template_pseudo_beta", \
                        "extra_msa", "extra_has_deletion", "extra_deletion_value", "extra_msa_mask", \
                        "residx_atom37_to_atom14", "atom37_atom_exists_batch", \
                        "residue_index_batch", "batch_aatype", "batch_all_atom_positions", "batch_all_atom_mask",
                        "opt_temp", \
                        "opt_soft", "opt_hard", "prev_pos", "prev_msa_first_row", "prev_pair"]
        self.feature_list = feature_list

    # pylint: disable=arguments-differ
    def data_parse(self, idx):
        pkl_path = self.training_pkl_items[idx]
        f = open(pkl_path, "rb")
        data = pickle.load(f)
        return data

    # pylint: disable=arguments-differ
    def process(self, data):
        features = data_process_run(data.copy(), self.data_process)
        return features

    def set_training_data_src(self, data_src):
        self.training_data_src = data_src
        self.training_pkl_path = self.training_data_src + "/pkl/"
        self.training_pdb_path = self.training_data_src + "/pdb/"
        self.training_pdb_items = [self.training_pdb_path + key for key in sorted(os.listdir(self.training_pdb_path))]
        self.training_pkl_items = [self.training_pkl_path + key for key in sorted(os.listdir(self.training_pkl_path))]

    # pylint: disable=arguments-differ
    def create_iterator(self, num_epochs):
        dataset = GeneratorDataset(source=self, column_names=self.feature_list, num_parallel_workers=4, shuffle=False,
                                   max_rowsize=16)
        iteration = dataset.create_dict_iterator(num_epochs=num_epochs, output_numpy=True)
        return iteration
