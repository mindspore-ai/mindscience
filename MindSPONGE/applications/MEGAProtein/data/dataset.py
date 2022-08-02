# Copyright 2022 Huawei Technologies Co., Ltd & CPL YiQin GAO Research Group
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
"""train dataset"""
import datetime
import os
import pickle
import time
import numpy as np
from mindspore import dataset as ds
from mindspore.communication import get_rank

from mindsponge.common.residue_constants import make_atom14_dists_bounds
from mindsponge.common.protein import from_pdb_string
from mindsponge.common.utils import make_atom14_positions
from mindsponge.data.data_transform import pseudo_beta_fn, atom37_to_frames, atom37_to_torsion_angles
from .preprocess import Feature


def create_dataset(train_data_dir, raw_feature_dir, names, data_cfg, center_name_path, shuffle=False,
                   num_parallel_worker=4,
                   is_parallel=False, mixed_precision=False):
    """create train dataset"""
    column_name = ["target_feat", "msa_feat", "msa_mask", "seq_mask_batch", "aatype_batch",
                   "template_aatype", "template_all_atom_masks",
                   "template_all_atom_positions", "template_mask",
                   "template_pseudo_beta_mask", "template_pseudo_beta", "extra_msa", "extra_has_deletion",
                   "extra_deletion_value", "extra_msa_mask", "residx_atom37_to_atom14",
                   "atom37_atom_exists_batch", "residue_index_batch", "prev_pos",
                   "prev_msa_first_row", "prev_pair", "pseudo_beta_gt",
                   "pseudo_beta_mask_gt", "all_atom_mask_gt",
                   "true_msa", "bert_mask", "residue_index", "seq_mask",
                   "atom37_atom_exists", "aatype", "residx_atom14_to_atom37",
                   "atom14_atom_exists", "backbone_affine_tensor", "backbone_affine_mask",
                   "atom14_gt_positions", "atom14_alt_gt_positions",
                   "atom14_atom_is_ambiguous", "atom14_gt_exists", "atom14_alt_gt_exists",
                   "all_atom_positions", "rigidgroups_gt_frames", "rigidgroups_gt_exists",
                   "rigidgroups_alt_gt_frames", "torsion_angles_sin_cos_gt", "chi_mask", "atomtype_radius",
                   "restype_atom14_bond_lower_bound", "restype_atom14_bond_upper_bound", "use_clamped_fape",
                   "filter_by_solution", "prot_name_index"]

    dataset_generator = DatasetGenerator(train_data_dir, raw_feature_dir, names, data_cfg, center_name_path,
                                         mixed_precision)
    ds.config.set_prefetch_size(1)

    if is_parallel:
        rank_id = get_rank() % 8
        rank_size = 8
        train_dataset = ds.GeneratorDataset(source=dataset_generator, column_names=column_name,
                                            num_parallel_workers=num_parallel_worker, shuffle=shuffle,
                                            num_shards=rank_size,
                                            shard_id=rank_id, max_rowsize=16)
    else:
        train_dataset = ds.GeneratorDataset(source=dataset_generator, column_names=column_name,
                                            num_parallel_workers=num_parallel_worker, shuffle=shuffle, max_rowsize=16)
    return train_dataset


class DatasetGenerator:
    """dataset generator"""
    def __init__(self, train_data_dir, raw_feature_dir, names, data_cfg, resolution_data, mixed_precision):
        self.t1 = time.time()
        print("start dataset init: ", str(datetime.datetime.now()))
        self.data_cfg = data_cfg
        self.num_residues = data_cfg.eval.crop_size
        self.train_data_dir = train_data_dir
        self.raw_feature_dir = raw_feature_dir
        self.names = [name.replace("\n", "") for name in names]
        self.mixed_precision = mixed_precision

        self.resolution_info = resolution_data
        print("end dataset init: ", time.time() - self.t1)

    def __getitem__(self, index):
        prot_name = self.names[index]
        prot_name_index = np.asarray([index]).astype(np.int32)
        arrays, prev_pos, prev_msa_first_row, prev_pair, label_arrays = self._get_train_data(prot_name)
        atomtype_radius = np.array(
            [1.55, 1.7, 1.7, 1.7, 1.52, 1.7, 1.7, 1.7, 1.52, 1.52, 1.8, 1.7, 1.7, 1.7, 1.55, 1.55,
             1.52, 1.52, 1.8, 1.7, 1.7, 1.7, 1.7, 1.55, 1.55, 1.55, 1.52, 1.52, 1.7, 1.55, 1.55,
             1.52, 1.7, 1.7, 1.7, 1.55, 1.52])
        restype_atom14_bond_lower_bound, restype_atom14_bond_upper_bound, _ = \
            make_atom14_dists_bounds(overlap_tolerance=1.5, bond_length_tolerance_factor=12.0)
        use_clamped_fape = np.random.binomial(1, 0.9, size=1)
        filter_by_solution = np.array(1.0)
        extra_feats = [atomtype_radius, restype_atom14_bond_lower_bound,
                       restype_atom14_bond_upper_bound, use_clamped_fape, filter_by_solution, prot_name_index]
        dtype = np.float32
        if self.mixed_precision:
            dtype = np.float16
        extra_feats = [array.astype(dtype) for array in extra_feats]
        all_feats = arrays + [prev_pos, prev_msa_first_row, prev_pair] + label_arrays + extra_feats

        return tuple(all_feats)

    def __len__(self):
        return len(self.names)

    def _get_solution_flag(self, prot_name):
        """get resolution data"""
        prot_new_name = prot_name.rsplit('_', 1)[0]
        if prot_new_name not in self.resolution_info:
            return np.array(1.0).astype(np.float32)
        resolution = float(self.resolution_info[prot_new_name]['resolution'])
        nmr = self.resolution_info[prot_new_name]['method']
        if resolution < 3 and nmr != 'NMR':
            return np.array(1.0).astype(np.float32)
        return np.array(0.0).astype(np.float32)

    def _get_train_labels(self, prot_pdb):
        """get train labels"""
        aatype = prot_pdb.aatype
        seq_len = len(aatype)
        atom37_positions = prot_pdb.atom_positions.astype(np.float32)
        atom37_mask = prot_pdb.atom_mask.astype(np.float32)

        # get ground truth of atom14
        label_features = {'aatype': aatype,
                          'all_atom_positions': atom37_positions,
                          'all_atom_mask': atom37_mask}

        atom14_features = make_atom14_positions(aatype, atom37_mask, atom37_positions)
        atom14_keys = ["atom14_atom_exists", "atom14_gt_exists", "atom14_gt_positions", "residx_atom14_to_atom37",
                       "residx_atom37_to_atom14", "atom37_atom_exists", "atom14_alt_gt_positions",
                       "atom14_alt_gt_exists", "atom14_atom_is_ambiguous"]
        for index, array in enumerate(atom14_features):
            label_features[atom14_keys[index]] = array

        # get ground truth of rigid groups
        rigidgroups_label_feature = atom37_to_frames(aatype, atom37_positions, atom37_mask, is_affine=True)
        label_features.update(rigidgroups_label_feature)

        # get ground truth of angle
        angle_label_feature = atom37_to_torsion_angles(aatype.reshape((1, -1)),
                                                       atom37_positions.reshape((1, seq_len, 37, 3)),
                                                       atom37_mask.reshape((1, seq_len, 37)), True)
        label_features.update(angle_label_feature)

        # get pseudo_beta, pseudo_beta_mask
        pseudo_beta, pseudo_beta_mask = pseudo_beta_fn(aatype, atom37_positions, atom37_mask)
        label_features["pseudo_beta"] = pseudo_beta
        label_features["pseudo_beta_mask"] = pseudo_beta_mask
        label_features["chi_mask"] = label_features.get("torsion_angles_mask")[:, 3:]
        label_features['torsion_angles_sin_cos'] = label_features.get('torsion_angles_sin_cos')[:, 3:, :]
        label_features['backbone_affine_mask'] = pseudo_beta_mask
        label_features.pop("aatype")

        return label_features

    def _get_train_data(self, prot_name):
        """get train data"""
        pdb_path = os.path.join(self.train_data_dir, prot_name + '.pdb')
        with open(pdb_path, 'r') as f:
            prot_pdb = from_pdb_string(f.read())
            f.close()
        with open(os.path.join(self.raw_feature_dir, prot_name + '.pkl'), "rb") as f:
            raw_feature = pickle.load(f)
            f.close()
        label_features = self._get_train_labels(prot_pdb)
        seed = global_seed()
        raw_feature.update(label_features)
        processed_feature = Feature(self.data_cfg, raw_feature, is_training=True)
        processed_feat = processed_feature.pipeline(self.data_cfg, self.mixed_precision, seed=seed)
        return processed_feat


class SeedMaker:
    """Return unique seeds."""

    def __init__(self, initial_seed=0):
        self.next_seed = initial_seed

    def __call__(self):
        i = self.next_seed
        self.next_seed += 1
        return i


global_seed = SeedMaker()
