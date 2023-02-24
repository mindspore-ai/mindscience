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
"""evogen"""
import os
import pickle
import numpy as np

from mindsponge.common.protein import from_pdb_string
from mindspore.dataset import GeneratorDataset

from .evogen_datafunction import dict_replace_key, correct_restypes, dict_concatenate, one_hot_convert, \
    dict_expand_dims, dict_take, dict_del_key, make_mask, initialize_hhblits_profile, msa_sample, msa_bert_mask, \
    generate_random_sample, random_crop_to_size, dict_suqeeze, dict_cast, dict_filter_key
from ...dataset import PSP, data_process_run

NUM_RES = 'num residues placeholder'
NUM_MSA_SEQ = 'msa placeholder'
NUM_EXTRA_SEQ = 'extra msa placeholder'
NUM_TEMPLATES = 'num templates placeholder'
NUM_SEQ = "length msa placeholder"
NUM_NOISE = 'num noise placeholder'
NUM_LATENT_DIM = "num latent placeholder"

FEATURE_LIST = {
    'msa_feat': [NUM_MSA_SEQ, NUM_RES, None],
    'msa_mask': [NUM_MSA_SEQ, NUM_RES],
    'seq_mask': [NUM_RES],
    'msa_input': [NUM_MSA_SEQ, NUM_RES, 2],
    'query_input': [NUM_RES, 2],
    'additional_input': [NUM_RES, 4],
    'evogen_random_data': [NUM_NOISE, NUM_MSA_SEQ, NUM_RES, NUM_LATENT_DIM],
    'evogen_context_mask': [NUM_MSA_SEQ, 2],
    'seq_length': []
}

_msa_feature_names = ['msa', 'deletion_matrix', 'msa_mask', 'msa_row_mask', 'bert_mask', 'true_msa', 'msa_input']


class MEGAEvoGenDataSet(PSP):
    '''MEGAEvoGenDataSet'''
    def __init__(self, config, data_src, seed=0):
        self.config = config
        self.in_memory = False
        self.phase = None
        self.training_data_src = data_src
        self.training_pkl_path = self.training_data_src + "/pkl/"
        self.training_pdb_path = self.training_data_src + "/pdb/"
        self.training_pdb_items = [self.training_pdb_path + key for key in sorted(os.listdir(self.training_pdb_path))]
        self.training_pkl_items = [self.training_pkl_path + key for key in sorted(os.listdir(self.training_pkl_path))]
        self.data_process = [
            dict_replace_key(['deletion_matrix_int', 'deletion_matrix']),
            dict_expand_dims(keys=["deletion_matrix", "msa"], axis=-1),
            correct_restypes(key="msa"),
            dict_concatenate(keys=["msa", "deletion_matrix"], result_key="msa_input"),
            one_hot_convert(key="aatype", axis=-1),
            dict_expand_dims(keys=["aatype"], result_key=["aatype_tmp"], axis=-1),
            dict_take(filter_list=["deletion_matrix"], result_key=["deletion_matrix_tmp"], axis=0),
            dict_concatenate(keys=["aatype_tmp", "deletion_matrix_tmp"], result_key="query_input"),
            dict_del_key(filter_list=["aatype_tmp", "deletion_matrix_tmp"]),
            make_mask(key="msa", result_key="msa_mask"),
            make_mask(key="aatype", result_key="seq_mask"),
            initialize_hhblits_profile,
            msa_sample(msa_feature_list=_msa_feature_names, keep_extra=False,
                       max_msa_clusters=self.config.max_msa_clusters, seed=seed),
            msa_bert_mask(uniform_prob=self.config.data.masked_msa.uniform_prob,
                          profile_prob=self.config.data.masked_msa.profile_prob,
                          same_prob=self.config.data.masked_msa.same_prob,
                          replace_fraction=self.config.data.masked_msa.replace_fraction, seed=seed),
            dict_take(filter_list=["msa", "msa_mask", "bert_mask"],
                      result_key=["bert_msa_tmp", "msa_mask_tmp", "bert_mask_tmp"], axis=0),
            dict_expand_dims(keys=["residue_index"], result_key=["residue_index_tmp"], axis=-1),
            dict_suqeeze(filter_list=["msa_mask"], axis=-1),
            dict_concatenate(keys=["bert_msa_tmp", "residue_index_tmp", "msa_mask_tmp", "bert_mask_tmp"],
                             result_key="additional_input"),
            random_crop_to_size(feature_list=FEATURE_LIST, crop_size=self.config.crop_size,
                                max_msa_clusters=self.config.max_msa_clusters, seed=seed),
            generate_random_sample(num_noise=self.config.data.random_sample.num_noise,
                                   latent_dim=self.config.data.random_sample.latent_dim,
                                   context_true_prob=self.config.data.random_sample.context_true_prob,
                                   keep_prob=self.config.data.random_sample.keep_prob,
                                   available_msa_fraction=self.config.data.random_sample.available_msa_fraction,
                                   max_msa_clusters=self.config.max_msa_clusters, crop_size=self.config.crop_size,
                                   seed=seed),
            dict_filter_key(feature_list=FEATURE_LIST),
            dict_cast([np.int64, np.float32], filtered_list=[]),
            dict_cast([np.int32, np.float32], filtered_list=[]),
            dict_cast([np.float64, np.float32], filtered_list=[]),
        ]
        super().__init__()

    def __getitem__(self, idx):
        if self.in_memory:
            data, _ = self.inputs[idx]
        else:
            data, _ = self.data_parse(idx)
        features = self.process(data)
        tuple_feature = tuple([features[key] for key in self.feature_list])
        return tuple_feature

    def __len__(self):
        data_len = len(os.listdir(self.training_pkl_path))
        assert data_len == len(os.listdir(self.training_pdb_path))
        return data_len

    # pylint: disable=arguments-differ
    def process(self, data):
        '''process'''
        features = data_process_run(data, self.data_process)
        return features

    # pylint: disable=arguments-differ
    def data_parse(self, idx):
        '''data_parse'''
        pkl_path = self.training_pkl_items[idx]
        f = open(pkl_path, "rb")
        data = pickle.load(f)
        f.close()
        pdb_path = self.training_pdb_items[idx]
        with open(pdb_path, 'r') as f:
            prot_pdb = from_pdb_string(f.read())
            f.close()
        aatype = prot_pdb.aatype
        atom37_positions = prot_pdb.atom_positions.astype(np.float32)
        atom37_mask = prot_pdb.atom_mask.astype(np.float32)

        # get ground truth of atom14
        label = {'aatype': aatype,
                 'all_atom_positions': atom37_positions,
                 'all_atom_mask': atom37_mask}
        return data, label

    # pylint: disable=arguments-differ
    def create_iterator(self, num_epochs):
        '''create_iterator'''
        dataset = GeneratorDataset(source=self, column_names=self.feature_list, num_parallel_workers=4, shuffle=False,
                                   max_rowsize=16)
        iteration = dataset.create_dict_iterator(num_epochs=num_epochs, output_numpy=True)
        return iteration
