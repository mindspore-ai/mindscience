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
"""multimer dataset"""
import numpy as np
from mindspore import context
from .multimer_data import make_msa_profile, sample_msa, make_masked_msa, nearest_neighbor_clusters, \
    create_msa_feat, random_crop_to_size, \
    dict_cast, dict_filter_key, prev_initial, make_atom14_mask
from .multimer_feature import _inference_feature, _msa_feature_names
from ...dataset import data_process_run, DataSet


class MultimerDataSet(DataSet):
    """MultimerDataSet"""
    def __init__(self, config, seed=0):
        self.config = config
        self.in_memory = False
        self.phase = None
        self.feature_list = None
        self.feature_names = _inference_feature
        self.multimer_inputs()

        self.data_process = [
            make_msa_profile(axis=0),
            sample_msa(msa_feature_list=_msa_feature_names, max_seq=self.config.data.num_msa, seed=seed),
            make_masked_msa(config=self.config.data.masked_msa, seed=seed),
            nearest_neighbor_clusters,
            create_msa_feat,
            make_atom14_mask,
            random_crop_to_size(feature_list=self.feature_names, crop_size=self.config.seq_length,
                                max_templates=self.config.data.max_templates,
                                max_msa_clusters=self.config.max_msa_clusters,
                                max_extra_msa=self.config.max_extra_msa,
                                seed=seed, random_recycle=self.config.data.random_recycle),
            ]

        self.tail_fns = []
        if context.get_context("device_target") == "GPU":
            self.mixed_precision = False
        else:
            self.mixed_precision = True

        if self.mixed_precision:
            data_cast_fns = [dict_cast([np.float64, np.float16], []),
                             dict_cast([np.float32, np.float16], []),
                             dict_cast([np.int64, np.int32], [])]
        else:
            data_cast_fns = [dict_cast([np.float64, np.float32], []), dict_cast([np.int64, np.int32], [])]

        self.tail_fns.extend([dict_filter_key(feature_list=self.feature_names),
                              prev_initial])
        self.tail_fns.extend(data_cast_fns)
        super().__init__()

    def __getitem__(self, idx):
        pass

    def __len__(self):
        pass

    def multimer_inputs(self):
        feature_list = ['aatype', 'residue_index', 'template_aatype', 'template_all_atom_mask',
                        'template_all_atom_positions', 'asym_id', 'sym_id', 'entity_id', 'seq_mask', 'msa_mask',
                        'target_feat', 'msa_feat', 'extra_msa', 'extra_deletion_matrix', 'extra_msa_mask',
                        'residx_atom37_to_atom14', 'atom37_atom_exists',
                        'prev_pos', 'prev_msa_first_row', 'prev_pair']
        self.feature_list = feature_list

    # pylint: disable=arguments-differ
    def process(self, data):
        """process"""
        res = {}
        for _ in range(4):
            features = data_process_run(data.copy(), self.data_process)
            if res == {}:
                res = {x: () for x in features.keys()}
            for key in features.keys():
                if key == "num_residues":
                    res[key] = features[key]
                else:
                    res[key] += (features[key][None],)
        for key in res.keys():
            if key != 'num_residues':
                res[key] = np.concatenate(res[key], axis=0)
        features = res
        features = data_process_run(features, self.tail_fns)
        return features

    def download(self, path=None):
        pass

    def data_parse(self, input_data, idx):
        pass

    # pylint: disable=arguments-differ
    def create_iterator(self, num_epochs):
        pass
