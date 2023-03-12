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

# pylint: disable=W0221

"""megafold_dataset"""
import os
import pickle
import numpy as np

from mindspore import context
from mindspore.dataset import GeneratorDataset
from mindspore.communication import get_rank, get_group_size
from mindsponge.common.protein import from_pdb_string

from .megafold_data import correct_restypes, make_atom14_mask, tail_process, \
    dict_replace_key, dict_cast, dict_take, one_hot_convert, initial_hhblits_profile, \
    template_pseudo_beta, msa_block_deletion, msa_sample, msa_bert_mask, \
    msa_nearest_neighbor_clusters, msa_summarize_clusters, dict_del_key, \
    extra_msa_crop, msa_feature_concatenate, dict_filter_key, random_crop_to_size, \
    template_feature_crop, label_pseudo_beta, initial_template_mask, prev_initial, \
    label_make_atom14_positions, label_atom37_to_frames, label_atom37_to_torsion_angles
from .megafold_feature import _msa_feature_names, _inference_feature, _training_feature
from ...dataset import PSP, data_process_run


class MEGAFoldDataSet(PSP):
    "MEGAFoldDataSet"
    def __init__(self, config, seed=0):
        self.config = config
        self.supported_models = ['MEGAFold']
        self.in_memory = False
        self.phase = None

        self.is_training = self.config.is_training
        if self.is_training:
            self.is_parallel = self.config.train.is_parallel

        self.feature_names = _inference_feature
        if self.is_training:
            self.feature_names.update(_training_feature)

        self.megafold_inputs()

        self.data_process = [
            dict_replace_key(['deletion_matrix_int', 'deletion_matrix']),
            initial_template_mask,
            correct_restypes(key="msa"),
            one_hot_convert(key="aatype", axis=-1),
            initial_hhblits_profile,
            make_atom14_mask]
        template_process = [
            one_hot_convert(key="template_aatype", axis=-1),
            correct_restypes(key="template_aatype"),
            template_pseudo_beta
        ]

        data_config = self.config.data
        if data_config.use_templates:
            self.data_process.extend(template_process)
        max_msa_clusters = data_config.max_msa_clusters
        if data_config.reduce_msa_clusters_by_max_templates:
            max_msa_clusters = data_config.max_msa_clusters - data_config.max_templates

        self.ensemble = []
        if self.is_training:
            self.ensemble.append(
                msa_block_deletion(msa_feature_list=_msa_feature_names,
                                   msa_fraction_per_block=data_config.block_deletion.msa_fraction_per_block,
                                   randomize_num_blocks=data_config.block_deletion.randomize_num_blocks,
                                   num_blocks=data_config.block_deletion.num_blocks,
                                   seed=seed))
        self.ensemble.append(
            msa_sample(msa_feature_list=_msa_feature_names, keep_extra=data_config.keep_extra,
                       max_msa_clusters=max_msa_clusters, seed=seed))

        if data_config.masked_msa.use_masked_msa:
            self.ensemble.append(
                msa_bert_mask(uniform_prob=data_config.masked_msa.uniform_prob,
                              profile_prob=data_config.masked_msa.profile_prob,
                              same_prob=data_config.masked_msa.same_prob,
                              replace_fraction=data_config.masked_msa_replace_fraction,
                              seed=seed))

        if data_config.msa_cluster_features:
            self.ensemble.extend([msa_nearest_neighbor_clusters(), msa_summarize_clusters])

        extra_msa_feature_names = ['extra_' + x for x in _msa_feature_names]
        if data_config.max_extra_msa:
            self.ensemble.append(extra_msa_crop(feature_list=extra_msa_feature_names,
                                                max_extra_msa=data_config.max_extra_msa))
        else:
            self.ensemble.append(dict_del_key(filter_list=extra_msa_feature_names))

        self.ensemble.append(msa_feature_concatenate)
        if self.config.fixed_size:
            self.ensemble.append(
                random_crop_to_size(
                    feature_list=self.feature_names, crop_size=self.config.seq_length,
                    max_templates=data_config.max_templates, max_msa_clusters=max_msa_clusters,
                    max_extra_msa=data_config.max_extra_msa,
                    subsample_templates=data_config.subsample_templates, seed=seed,
                    random_recycle=data_config.random_recycle))
        else:
            self.ensemble.append(template_feature_crop(max_templates=data_config.max_templates))

        if self.is_training:
            self.label_fns = [label_make_atom14_positions,
                              label_atom37_to_frames(is_affine=True),
                              label_atom37_to_torsion_angles(alt_torsions=True),
                              label_pseudo_beta]
        self.tail_fns = []
        if self.is_training:
            self.tail_fns.append(dict_take(filter_list=_training_feature, axis=-1))

        if context.get_context("device_target") == "GPU":
            self.mixed_precision = False
        else:
            self.mixed_precision = True

        if self.mixed_precision:
            data_cast_fns = [dict_cast([np.float64, np.float16], []),
                             dict_cast([np.float32, np.float16], []),
                             dict_cast([np.int64, np.int32], [])]
        else:
            data_cast_fns = [dict_cast([np.float64, np.float32], []), \
                             dict_cast([np.int64, np.int32], [])]

        self.tail_fns.extend([dict_filter_key(feature_list=self.feature_names),
                              prev_initial,
                              tail_process])
        self.tail_fns.extend(data_cast_fns)
        self.training_data_src = None
        self.training_pkl_path = None
        self.training_pdb_path = None
        self.training_pdb_items = None
        self.training_pkl_items = None

        super().__init__()

    def __len__(self):
        return len(self.training_pkl_items)

    def __getitem__(self, idx):
        if self.in_memory:
            data, label = self.inputs[idx]
        else:
            data, label = self.data_parse(idx)
        features = self.process(data, label, 4)
        tuple_feature = tuple([features.get(key, np.array([])) for key in self.feature_list])
        return tuple_feature

    def process(self, data, label=None, ensemble_num=4):
        if self.is_training:
            labels = data_process_run(label, self.label_fns)
            data.update(labels)
        features = data_process_run(data.copy(), self.data_process)
        if self.ensemble is not None:
            res = {}
            for _ in range(ensemble_num):
                ensemble_features = data_process_run(features.copy(), self.ensemble)
                if not res:
                    res = {x: () for x in ensemble_features.keys()}
                for key in ensemble_features.keys():
                    if key == "num_residues":
                        res[key] = ensemble_features[key]
                    else:
                        res[key] += (ensemble_features[key][None],)
            for key in res.keys():
                if key != 'num_residues':
                    res[key] = np.concatenate(res.get(key, np.array([])), axis=0)
            features = res
        features = data_process_run(features, self.tail_fns)
        return features

    def data_parse(self, idx):
        "data_parse"
        pkl_path = self.training_pkl_items[idx]
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)

        pdb_path = self.training_pdb_items[idx]
        with open(pdb_path, 'r') as f:
            prot_pdb = from_pdb_string(f.read())
        aatype = prot_pdb.aatype
        atom37_positions = prot_pdb.atom_positions.astype(np.float32)
        atom37_mask = prot_pdb.atom_mask.astype(np.float32)

        # get ground truth of atom14
        label = {'aatype': aatype,
                 'all_atom_positions': atom37_positions,
                 'all_atom_mask': atom37_mask}
        seq_len = aatype.shape[0]
        if data["template_aatype"].shape[0] == 0:
            data["template_aatype"] = np.zeros((1, seq_len, 22)).astype(np.float32)
            data["template_all_atom_masks"] = np.zeros((1, seq_len, 37)).astype(np.float32)
            data["template_all_atom_positions"] = np.zeros((1, seq_len, 37, 3)).astype(np.float32)

        return data, label

    def set_training_data_src(self, data_src):
        "set_training_data_src"
        self.training_data_src = data_src
        self.training_pkl_path = self.training_data_src + "/pkl/"
        self.training_pdb_path = self.training_data_src + "/pdb/"

        pkl_names = os.listdir(self.training_pkl_path)
        pkl_names = [name.split(".")[0] for name in pkl_names if name[-4:] == ".pkl"]
        pdb_names = os.listdir(self.training_pdb_path)
        pdb_names = [name.split(".")[0] for name in pdb_names if name[-4:] == ".pdb"]
        name_list = list(set(pkl_names).intersection(set(pdb_names)))

        self.training_pdb_items = [self.training_pdb_path + key + ".pdb" for  key in name_list]
        self.training_pkl_items = [self.training_pkl_path + key + ".pkl" for key in name_list]

    def create_iterator(self, num_epochs, **kwargs):
        "create_iterator"
        if self.is_parallel:
            rank_id = get_rank()
            rank_size = get_group_size()
            dataset = GeneratorDataset(source=self, column_names=self.feature_list,
                                       num_parallel_workers=4, shuffle=True,
                                       num_shards=rank_size,
                                       shard_id=rank_id, max_rowsize=16)
        else:
            dataset = GeneratorDataset(source=self, column_names=self.feature_list,
                                       num_parallel_workers=4, shuffle=True, max_rowsize=16)

        iteration = dataset.create_dict_iterator(num_epochs=num_epochs, output_numpy=True)
        return iteration

    def megafold_inputs(self):
        "megafold_inputs"
        feature_list = ['target_feat', 'msa_feat', 'msa_mask', 'seq_mask', 'aatype',
                        'template_aatype', 'template_all_atom_masks', 'template_all_atom_positions',
                        'template_mask', 'template_pseudo_beta_mask', 'template_pseudo_beta',
                        'extra_msa', 'extra_has_deletion', 'extra_deletion_value', 'extra_msa_mask',
                        'residx_atom37_to_atom14', 'atom37_atom_exists', 'residue_index',
                        'prev_pos', 'prev_msa_first_row', 'prev_pair']

        label_list = ["pseudo_beta", "pseudo_beta_mask", "all_atom_mask", "true_msa",
                      "bert_mask", "residx_atom14_to_atom37", "restype_atom14_bond_lower_bound",
                      "restype_atom14_bond_upper_bound", "atomtype_radius",
                      "backbone_affine_tensor", "backbone_affine_mask", "atom14_gt_positions",
                      "atom14_alt_gt_positions", "atom14_atom_is_ambiguous", "atom14_gt_exists",
                      "atom14_atom_exists", "atom14_alt_gt_exists", "all_atom_positions",
                      "rigidgroups_gt_frames", "rigidgroups_gt_exists",
                      "rigidgroups_alt_gt_frames", "torsion_angles_sin_cos",
                      "use_clamped_fape", "filter_by_solution", "chi_mask"]
        self.feature_list = feature_list
        if self.is_training:
            self.feature_list += label_list
