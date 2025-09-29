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
"""data process"""
import numpy as np
import pickle
from mindsponge1.data.data_transform import one_hot, correct_msa_restypes, randomly_replace_msa_with_unknown, \
    fix_templates_aatype, pseudo_beta_fn, make_atom14_masks, make_msa_feat_v2, make_extra_msa_feat, \
    block_delete_msa_indices, sample_msa, sample_msa_v2, make_masked_msa, make_masked_msa_v2, \
    nearest_neighbor_clusters, nearest_neighbor_clusters_v2, summarize_clusters, crop_extra_msa, \
    make_msa_feat, random_crop_to_size, generate_random_sample, atom37_to_torsion_angles
from mindsponge1.common.residue_constants import atom_type_num

from .utils import numpy_seed
from .multimer_process import get_spatial_crop_idx_v2, get_spatial_crop_idx, get_contiguous_crop_idx, \
    apply_crop_idx, select_feat, make_fixed_size, map_fn, make_pseudo_beta
from utils_xyh import show_npdict
import pickle

NUM_RES = 'num residues placeholder'
NUM_MSA_SEQ = 'msa placeholder'
NUM_EXTRA_SEQ = 'extra msa placeholder'
NUM_TEMPLATES = 'num templates placeholder'
NUM_SEQ = "length msa placeholder"
NUM_NOISE = 'num noise placeholder'
NUM_LATENT_DIM = "num latent placeholder"
_MSA_FEATURE_NAMES = ['msa', 'deletion_matrix', 'msa_mask', 'msa_row_mask', 'bert_mask', 'true_msa', 'msa_input']

FEATURES = {
    # Static features of a protein sequence
    "aatype": (np.float32, [NUM_RES, 21]),
    "between_segment_residues": (np.int64, [NUM_RES, 1]),
    "deletion_matrix": (np.float32, [NUM_SEQ, NUM_RES, 1]),
    "msa": (np.int64, [NUM_SEQ, NUM_RES, 1]),
    "num_alignments": (np.int64, [NUM_RES, 1]),
    "residue_index": (np.int64, [NUM_RES, 1]),
    "seq_length": (np.int64, [NUM_RES, 1]),
    "all_atom_positions": (np.float32, [NUM_RES, atom_type_num, 3]),
    "all_atom_mask": (np.int64, [NUM_RES, atom_type_num]),
    "resolution": (np.float32, [1]),
    "template_domain_names": (str, [NUM_TEMPLATES]),
    "template_sum_probs": (np.float32, [NUM_TEMPLATES, 1]),
    "template_aatype": (np.float32, [NUM_TEMPLATES, NUM_RES, 22]),
    "template_all_atom_positions": (np.float32, [NUM_TEMPLATES, NUM_RES, atom_type_num, 3]),
    "template_all_atom_masks": (np.float32, [NUM_TEMPLATES, NUM_RES, atom_type_num, 1]),
    "atom14_atom_exists": (np.float32, [NUM_RES, 14]),
    "atom14_gt_exists": (np.float32, [NUM_RES, 14]),
    "atom14_gt_positions": (np.float32, [NUM_RES, 14, 3]),
    "residx_atom14_to_atom37": (np.float32, [NUM_RES, 14]),
    "residx_atom37_to_atom14": (np.float32, [NUM_RES, 37]),
    "atom37_atom_exists": (np.float32, [NUM_RES, 37]),
    "atom14_alt_gt_positions": (np.float32, [NUM_RES, 14, 3]),
    "atom14_alt_gt_exists": (np.float32, [NUM_RES, 14]),
    "atom14_atom_is_ambiguous": (np.float32, [NUM_RES, 14]),
    "rigidgroups_gt_frames": (np.float32, [NUM_RES, 8, 12]),
    "rigidgroups_gt_exists": (np.float32, [NUM_RES, 8]),
    "rigidgroups_group_exists": (np.float32, [NUM_RES, 8]),
    "rigidgroups_group_is_ambiguous": (np.float32, [NUM_RES, 8]),
    "rigidgroups_alt_gt_frames": (np.float32, [NUM_RES, 8, 12]),
    "backbone_affine_tensor": (np.float32, [NUM_RES, 7]),
    "torsion_angles_sin_cos": (np.float32, [NUM_RES, 4, 2]),
    "torsion_angles_mask": (np.float32, [NUM_RES, 7]),
    "pseudo_beta": (np.float32, [NUM_RES, 3]),
    "pseudo_beta_mask": (np.float32, [NUM_RES]),
    "chi_mask": (np.float32, [NUM_RES, 4]),
    "backbone_affine_mask": (np.float32, [NUM_RES]),
}

feature_list = {
    'aatype': [NUM_RES],
    'all_atom_mask': [NUM_RES, None],
    'all_atom_positions': [NUM_RES, None, None],
    'alt_chi_angles': [NUM_RES, None],
    'atom14_alt_gt_exists': [NUM_RES, None],
    'atom14_alt_gt_positions': [NUM_RES, None, None],
    'atom14_atom_exists': [NUM_RES, None],
    'atom14_atom_is_ambiguous': [NUM_RES, None],
    'atom14_gt_exists': [NUM_RES, None],
    'atom14_gt_positions': [NUM_RES, None, None],
    'atom37_atom_exists': [NUM_RES, None],
    'backbone_affine_mask': [NUM_RES],
    'backbone_affine_tensor': [NUM_RES, None],
    'bert_mask': [NUM_MSA_SEQ, NUM_RES],
    'chi_angles': [NUM_RES, None],
    'chi_mask': [NUM_RES, None],
    'extra_deletion_value': [NUM_EXTRA_SEQ, NUM_RES],
    'extra_has_deletion': [NUM_EXTRA_SEQ, NUM_RES],
    'extra_msa': [NUM_EXTRA_SEQ, NUM_RES],
    'extra_msa_mask': [NUM_EXTRA_SEQ, NUM_RES],
    'extra_msa_row_mask': [NUM_EXTRA_SEQ],
    'is_distillation': [],
    'msa_feat': [NUM_MSA_SEQ, NUM_RES, None],
    'msa_mask': [NUM_MSA_SEQ, NUM_RES],
    'msa_row_mask': [NUM_MSA_SEQ],
    'pseudo_beta': [NUM_RES, None],
    'pseudo_beta_mask': [NUM_RES],
    'random_crop_to_size_seed': [None],
    'residue_index': [NUM_RES],
    'residx_atom14_to_atom37': [NUM_RES, None],
    'residx_atom37_to_atom14': [NUM_RES, None],
    'resolution': [],
    'rigidgroups_alt_gt_frames': [NUM_RES, None, None],
    'rigidgroups_group_exists': [NUM_RES, None],
    'rigidgroups_group_is_ambiguous': [NUM_RES, None],
    'rigidgroups_gt_exists': [NUM_RES, None],
    'rigidgroups_gt_frames': [NUM_RES, None, None],
    'seq_length': [],
    'seq_mask': [NUM_RES],
    'target_feat': [NUM_RES, None],
    'template_aatype': [NUM_TEMPLATES, NUM_RES],
    'template_all_atom_masks': [NUM_TEMPLATES, NUM_RES, None],
    'template_all_atom_positions': [
        NUM_TEMPLATES, NUM_RES, None, None],
    'template_backbone_affine_mask': [NUM_TEMPLATES, NUM_RES],
    'template_backbone_affine_tensor': [
        NUM_TEMPLATES, NUM_RES, None],
    'template_mask': [NUM_TEMPLATES],
    'template_pseudo_beta': [NUM_TEMPLATES, NUM_RES, None],
    'template_pseudo_beta_mask': [NUM_TEMPLATES, NUM_RES],
    'template_sum_probs': [NUM_TEMPLATES, None],
    'true_msa': [NUM_MSA_SEQ, NUM_RES],
    'torsion_angles_sin_cos': [NUM_RES, None, None],
    'msa_input': [NUM_MSA_SEQ, NUM_RES, 2],
    'query_input': [NUM_RES, 2],
    'additional_input': [NUM_RES, 4],
    'random_data': [NUM_NOISE, NUM_MSA_SEQ, NUM_RES, NUM_LATENT_DIM],
    'context_mask': [NUM_MSA_SEQ, 2]
}

multimer_feature_list = {
    "aatype": [NUM_RES],
    "all_atom_mask": [NUM_RES, None],
    "all_atom_positions": [NUM_RES, None, None],
    "alt_chi_angles": [NUM_RES, None],
    "atom14_alt_gt_exists": [NUM_RES, None],
    "atom14_alt_gt_positions": [NUM_RES, None, None],
    "atom14_atom_exists": [NUM_RES, None],
    "atom14_atom_is_ambiguous": [NUM_RES, None],
    "atom14_gt_exists": [NUM_RES, None],
    "atom14_gt_positions": [NUM_RES, None, None],
    "atom37_atom_exists": [NUM_RES, None],
    "frame_mask": [NUM_RES],
    "true_frame_tensor": [NUM_RES, None, None],
    "bert_mask": [NUM_MSA_SEQ, NUM_RES],
    "chi_angles_sin_cos": [NUM_RES, None, None],
    "chi_mask": [NUM_RES, None],
    "crop_and_fix_size_seed":[],
    "deletion_matrix": [NUM_MSA_SEQ, NUM_RES],
    "extra_msa_deletion_value": [NUM_EXTRA_SEQ, NUM_RES],
    "extra_msa_has_deletion": [NUM_EXTRA_SEQ, NUM_RES],
    "extra_msa": [NUM_EXTRA_SEQ, NUM_RES],
    "extra_msa_mask": [NUM_EXTRA_SEQ, NUM_RES],
    "extra_msa_row_mask": [NUM_EXTRA_SEQ],
    "hhblits_profile": [NUM_RES, None],
    "is_distillation": [],
    "msa": [NUM_MSA_SEQ, NUM_RES],
    "msa_feat": [NUM_MSA_SEQ, NUM_RES, None],
    "msa_mask": [NUM_MSA_SEQ, NUM_RES],
    "msa_chains": [NUM_MSA_SEQ, None],
    "msa_row_mask": [NUM_MSA_SEQ],
    "num_alignments": [],
    "pseudo_beta": [NUM_RES, None],
    "pseudo_beta_mask": [NUM_RES],
    "residue_index": [NUM_RES],
    "residx_atom14_to_atom37": [NUM_RES, None],
    "residx_atom37_to_atom14": [NUM_RES, None],
    "resolution": [],
    "rigidgroups_alt_gt_frames": [NUM_RES, None, None, None],
    "rigidgroups_group_exists": [NUM_RES, None],
    "rigidgroups_group_is_ambiguous": [NUM_RES, None],
    "rigidgroups_gt_exists": [NUM_RES, None],
    "rigidgroups_gt_frames": [NUM_RES, None, None, None],
    "seq_length": [],
    "seq_mask": [NUM_RES],
    "target_feat": [NUM_RES, None],
    "template_aatype": [NUM_TEMPLATES, NUM_RES],
    "template_all_atom_masks": [NUM_TEMPLATES, NUM_RES, None],
    "template_all_atom_positions": [NUM_TEMPLATES, NUM_RES, None, None],
    "template_alt_torsion_angles_sin_cos": [NUM_TEMPLATES, NUM_RES, None, None],
    "template_frame_mask": [NUM_TEMPLATES, NUM_RES],
    "template_frame_tensor": [NUM_TEMPLATES, NUM_RES, None, None],
    "template_mask": [NUM_TEMPLATES],
    "template_pseudo_beta": [NUM_TEMPLATES, NUM_RES, None],
    "template_pseudo_beta_mask": [NUM_TEMPLATES, NUM_RES],
    "template_sum_probs": [NUM_TEMPLATES, None],
    "template_torsion_angles_mask": [NUM_TEMPLATES, NUM_RES, None],
    "template_torsion_angles_sin_cos": [NUM_TEMPLATES, NUM_RES, None, None],
    "true_msa": [NUM_MSA_SEQ, NUM_RES],
    "use_clamped_fape": [],
    "assembly_num_chains": [1],
    "asym_id": [NUM_RES],
    "sym_id": [NUM_RES],
    "entity_id": [NUM_RES],
    "num_sym": [NUM_RES],
    "asym_len": [None],
    "cluster_bias_mask": [NUM_MSA_SEQ],
}


def feature_shape(feature_name, num_residues, msa_length, num_templates, features=None):
    """Get the shape for the given feature name."""
    features = features or FEATURES
    if feature_name.endswith("_unnormalized"):
        feature_name = feature_name[:-13]
    unused_dtype, raw_sizes = features.get(feature_name, (None, None))
    replacements = {NUM_RES: num_residues,
                    NUM_SEQ: msa_length}

    if num_templates is not None:
        replacements[NUM_TEMPLATES] = num_templates

    sizes = [replacements.get(dimension, dimension) for dimension in raw_sizes]
    for dimension in sizes:
        if isinstance(dimension, str):
            raise ValueError("Could not parse %s (shape: %s) with values: %s" % (
                feature_name, raw_sizes, replacements))
    size_r = [int(x) for x in sizes]
    return size_r


def parse_reshape_logic(parsed_features, features, num_template, key=None):
    """Transforms parsed serial features to the correct shape."""
    # Find out what is the number of sequences and the number of alignments.
    num_residues = np.reshape(parsed_features['seq_length'].astype(np.int32), (-1,))[0]

    if "num_alignments" in parsed_features:
        num_msa = np.reshape(parsed_features["num_alignments"].astype(np.int32), (-1,))[0]
    else:
        num_msa = 0

    if key is not None and "key" in features:
        parsed_features["key"] = [key]  # Expand dims from () to (1,).

    # Reshape the arrays according to the sequence length and num alignments.
    for k, v in parsed_features.items():
        new_shape = feature_shape(
            feature_name=k,
            num_residues=num_residues,
            msa_length=num_msa,
            num_templates=num_template,
            features=features)
        new_shape_size = 1
        for dim in new_shape:
            new_shape_size *= dim

        if np.size(v) != new_shape_size:
            raise ValueError("the size of feature {} ({}) could not be reshaped into {}"
                             "".format(k, np.size(v), new_shape))

        if "template" not in k:
            # Make sure the feature we are reshaping is not empty.
            if np.size(v) <= 0:
                raise ValueError("The feature {} is not empty.".format(k))
        parsed_features[k] = np.reshape(v, new_shape)

    return parsed_features


def _make_features_metadata(feature_names):
    """Makes a feature name to type and shape mapping from a list of names."""
    # Make sure these features are always read.
    required_features = ["sequence", "domain_name", "template_domain_names"]
    feature_names = list(set(feature_names) - set(required_features))

    features_metadata = {name: FEATURES.get(name) for name in feature_names}
    return features_metadata


def np_to_array_dict(np_example, features):
    """Creates dict of arrays.

    Args:
      np_example: A dict of NumPy feature arrays.
      features: A list of strings of feature names to be returned in the dataset.

    Returns:
      A dictionary of features mapping feature names to features. Only the given
      features are returned, all other ones are filtered out.
    """
    features_metadata = _make_features_metadata(features)
    array_dict = {k: v for k, v in np_example.items() if k in features_metadata}
    if "template_domain_names" in np_example:
        num_template = len(np_example["template_domain_names"])
    else:
        num_template = 0

    # Ensures shapes are as expected. Needed for setting size of empty features
    # e.g. when no template hits were found.
    array_dict = parse_reshape_logic(array_dict, features_metadata, num_template)
    array_dict['template_mask'] = np.ones([num_template], np.float32)
    return array_dict


class Feature:
    """feature process"""

    def __init__(self, cfg, raw_feature=None, is_training=False, model_cfg=None, is_evogen=False, is_multimer=False):
        if raw_feature and isinstance(raw_feature, dict):
            self.ensemble_num = 0
            self.cfg = cfg
            self.model_cfg = model_cfg
            if 'deletion_matrix_int' in raw_feature:
                raw_feature['deletion_matrix'] = (raw_feature.pop('deletion_matrix_int').astype(np.float32))
            feature_names = cfg.common.unsupervised_features
            if cfg.common.use_templates:
                feature_names += cfg.common.template_features
            self.is_training = is_training
            self.is_evogen = is_evogen
            self.is_multimer = is_multimer
            if self.is_training:
                feature_names += cfg.common.supervised_features
            if self.is_multimer:
                feature_names += cfg.common.multimer_features
                feature_names += cfg.common.recycling_features
                raw_feature = {k: v for k, v in raw_feature.items() if k in feature_names}
                raw_feature['template_all_atom_masks'] = (raw_feature.pop('template_all_atom_mask'))
            if not self.is_multimer:
                raw_feature = np_to_array_dict(np_example=raw_feature, features=feature_names)
            # with open("/data6/yhding/1228/compare/myinit_feat.pkl", "wb") as f:
            #     pickle.dump(raw_feature, f)
            for key in raw_feature:
                setattr(self, key, raw_feature[key])

    def non_ensemble(self, distillation=False, replace_proportion=0.0, use_templates=True):
        """non ensemble"""
        if self.is_multimer:
            data = vars(self)
            num_seq = data["msa"].shape[0]
            seq_len = data["msa"].shape[1]
            max_seq = self.cfg.common.max_msa_entry // seq_len
            if num_seq > max_seq:
                keep_index = (np.random.choice(num_seq - 1, max_seq - 1, replace=False) + 1)
                keep_index = np.sort(keep_index)
                keep_index = np.concatenate((np.array([0]), keep_index), axis=0)
                for k in ["msa", "deletion_matrix", "msa_mask", "msa_row_mask",
                          "bert_mask", "true_msa", "msa_chains"]:
                    if k in data:
                        setattr(self, k, data[k][keep_index])
        if self.is_evogen:
            msa, msa_input = correct_msa_restypes(self.msa, self.deletion_matrix, self.is_evogen)
            setattr(self, "msa", msa)
            setattr(self, "msa_input", msa_input.astype(np.float32))
        else:
            setattr(self, "msa", correct_msa_restypes(self.msa))
        setattr(self, "is_distillation", np.array(float(distillation), dtype=np.float32))
        # convert int64 to int32
        for k, v in vars(self).items():
            if k not in ("ensemble_num", "is_training", "is_evogen", "cfg", "model_cfg", "is_multimer"):
                if k.endswith("_mask"):
                    setattr(self, k, v.astype(np.float32))
                elif v.dtype in (np.int64, np.uint8, np.int8):
                    setattr(self, k, v.astype(np.int32))
        if len(self.aatype.shape) == 2:
            aatype = np.argmax(self.aatype, axis=-1)
            setattr(self, "aatype", aatype.astype(np.int32))
        if self.is_evogen:
            query_input = np.concatenate((aatype[:, None], self.deletion_matrix[0]),
                                         axis=-1).astype(np.int32)
            setattr(self, "query_input", query_input.astype(np.float32))
        data = vars(self)
        if "resolution" in data and len(data["resolution"].shape) == 1:
            setattr(self, "resolution", data["resolution"][0])
        namelist = ['msa', 'num_alignments', 'seq_length', 'sequence', 'superfamily', 'deletion_matrix',
                    'resolution', 'between_segment_residues', 'residue_index', 'template_all_atom_masks']
        if self.is_multimer:
            namelist.append('domain_name')
            namelist.remove('resolution')
        for k in namelist:
            if k in data:
                final_dim = data[k].shape[-1]
                if isinstance(final_dim, int) and final_dim == 1:
                    setattr(self, k, np.squeeze(data[k], axis=-1))
        # Remove fake sequence dimension
        for k in ['seq_length', 'num_alignments']:
            if k in data and len(data[k].shape):
                setattr(self, k, data[k][0])
        msa, aatype = randomly_replace_msa_with_unknown(self.msa, self.aatype, replace_proportion)
        setattr(self, "msa", msa)
        setattr(self, "aatype", aatype)
        # seq_mask
        seq_mask = np.ones(self.aatype.shape, dtype=np.float32)
        setattr(self, "seq_mask", seq_mask)
        # msa_mask and msa_row_mask
        msa_mask = np.ones(self.msa.shape, dtype=np.float32)
        msa_row_mask = np.ones(self.msa.shape[0], dtype=np.float32)
        setattr(self, "msa_mask", msa_mask)
        setattr(self, "msa_row_mask", msa_row_mask)
        if 'hhblits_profile' not in data:
            # Compute the profile for every residue (over all MSA sequences).
            if self.is_multimer:
                setattr(self, 'hhblits_profile', np.mean(one_hot(22, self.msa) * self.msa_mask[:, :, None], axis=0))
            else:
                setattr(self, 'hhblits_profile', np.mean(one_hot(22, self.msa), axis=0))
        if use_templates:
            if not self.is_multimer:
                template_aatype = fix_templates_aatype(self.template_aatype)
                setattr(self, "template_aatype", template_aatype)
            else:
                setattr(self, "template_mask", np.ones(self.template_aatype.shape[0], dtype=np.float32))
            template_pseudo_beta, template_pseudo_beta_mask = pseudo_beta_fn(self.template_aatype,
                                                                             self.template_all_atom_positions,
                                                                             self.template_all_atom_masks)
            setattr(self, "template_pseudo_beta", template_pseudo_beta)
            setattr(self, "template_pseudo_beta_mask", template_pseudo_beta_mask)
            if self.is_multimer:
                num_templates = self.template_mask.shape[-1]
                max_templates = self.cfg.common.max_templates
                if num_templates > 0:
                    if self.cfg.common.subsample_templates:
                        max_templates = min(max_templates, np.random.randint(0, num_templates + 1))
                        template_idx = np.random.choice(num_templates, max_templates, replace=False)
                    else:
                        # use top templates
                        template_idx = np.arange(min(num_templates, max_templates), dtype=np.int64)
                    for k, v in vars(self).items():
                        if k.startswith("template"):
                            try:
                               v = v[template_idx]
                            except Exception as ex:
                                print(ex.__class__, ex)
                                print("num_templates", num_templates)
                                print(k, v.shape)
                                print("protein_shape:", {k: v.shape for k, v in vars(self).items() if "shape" in dir(v)})
                        setattr(self, k, v)
                if self.cfg.common.use_template_torsion_angles:
                    aatype = self.template_aatype
                    all_atom_positions = self.template_all_atom_positions
                    all_atom_mask = self.template_all_atom_masks
                    angle_arrays_feature = atom37_to_torsion_angles(aatype, all_atom_positions, all_atom_mask, alt_torsions=False, is_multimer=self.is_multimer)
                    setattr(self, "template_torsion_angles_sin_cos", angle_arrays_feature["torsion_angles_sin_cos"])
                    setattr(self, "template_alt_torsion_angles_sin_cos", angle_arrays_feature["alt_torsion_angles_sin_cos"])
                    setattr(self, "template_torsion_angles_mask", angle_arrays_feature["torsion_angles_mask"])

        atom14_atom_exists, residx_atom14_to_atom37, residx_atom37_to_atom14, atom37_atom_exists = \
            make_atom14_masks(self.aatype)
        setattr(self, "atom14_atom_exists", atom14_atom_exists)
        setattr(self, "residx_atom14_to_atom37", residx_atom14_to_atom37)
        setattr(self, "residx_atom37_to_atom14", residx_atom37_to_atom14)
        setattr(self, "atom37_atom_exists", atom37_atom_exists)

        if self.is_multimer:
            if "between_segment_residues" in vars(self).keys():
                has_break = np.clip(self.between_segment_residues.astype(np.float32), 0, 1)
            else:
                has_break = np.zeros_like(self.aatype, dtype=np.float32)
                if "asym_len" in vars(self):
                    asym_len = self.asym_len
                    entity_ends = np.cumsum(asym_len, axis=-1)[:-1]
                    has_break[entity_ends] = 1.0
                has_break = has_break.astype(np.float32)
            aatype_1hot = one_hot(21, self.aatype)
            if self.cfg.common.target_feat_dim == 22:
                target_feat = [np.expand_dims(has_break, axis=-1), aatype_1hot]
            else:
                target_feat = [aatype_1hot]
            setattr(self, "target_feat", np.concatenate(target_feat, axis=-1))

    def ensemble(self, data, msa_fraction_per_block=0.3, randomize_num_blocks=True, num_blocks=5, keep_extra=True,
                 max_msa_clusters=124, masked_msa=None, uniform_prob=0.1, profile_prob=0.1, same_prob=0.1,
                 replace_fraction=0.15, msa_cluster_features=True, max_extra_msa=1024, crop_size=256, max_templates=4,
                 subsample_templates=True, fixed_size=True, seed=0, random_recycle=False):
        """ensemble"""
        if not self.is_multimer:
            self.ensemble_num += 1
            if self.is_training:
                keep_indices = block_delete_msa_indices(data["msa"], msa_fraction_per_block, randomize_num_blocks,
                                                        num_blocks)
                for k in _MSA_FEATURE_NAMES:
                    if k in data:
                        data[k] = data[k][keep_indices]
            is_sel, not_sel_seq, sel_seq = sample_msa(data["msa"], max_msa_clusters)

        # ensure first row of msa is input sequence
        data["msa"] = np.concatenate([data["aatype"][None,:], data["msa"]], axis=0)
        zero_deletion = np.zeros((data["deletion_matrix"].shape[-1])).astype(data["deletion_matrix"].dtype)
        data["deletion_matrix"] = np.concatenate([zero_deletion[None,:], data["deletion_matrix"]], axis=0)

        # exist numpy random op
        if self.is_multimer:
            # print(data["is_distillation"])
            is_sel, not_sel_seq, sel_seq = sample_msa_v2(data["msa"], data["msa_chains"], data["msa_mask"],
                                                         max_msa_clusters, biased_msa_by_chain=self.cfg.common.biased_msa_by_chain) # True
            # print(is_sel, not_sel_seq, sel_seq) # 正确
            if "msa_input" in _MSA_FEATURE_NAMES:
                _MSA_FEATURE_NAMES.remove("msa_input")
                _MSA_FEATURE_NAMES.append("msa_chains")
        
        for k in _MSA_FEATURE_NAMES:
            if k in data:
                if keep_extra and not is_sel:
                    new_shape = list(data[k].shape)
                    new_shape[0] = 1
                    data['extra_' + k] = np.zeros(new_shape)
                elif keep_extra and is_sel:
                    data['extra_' + k] = data[k][not_sel_seq]
                if k == 'msa' and not self.is_multimer:
                    data['extra_msa'] = data['extra_msa'].astype(np.int32)
                data[k] = data[k][sel_seq]
        if masked_msa:
            if self.is_evogen:
                make_masked_msa_result = make_masked_msa(
                    data["msa"], data["hhblits_profile"],
                    uniform_prob, profile_prob,
                    same_prob,
                    replace_fraction,
                    data['residue_index'], data['msa_mask'], self.is_evogen)
                data["bert_mask"], data["true_msa"], data["msa"], data["additional_input"] = make_masked_msa_result
                data["additional_input"] = data["additional_input"].astype(np.float32)
            elif self.is_multimer:
                    
                data["bert_mask"], data["true_msa"], data["msa"] = make_masked_msa_v2(data["msa"],
                                                                                     data["hhblits_profile"],
                                                                                     data['msa_mask'],
                                                                                     data["entity_id"],
                                                                                     data["sym_id"],
                                                                                     data["num_sym"],
                                                                                     uniform_prob,
                                                                                     profile_prob,
                                                                                     same_prob,
                                                                                     replace_fraction,
                                                                                     share_mask=self.cfg.common.share_mask, #True
                                                                                     bert_mask=data["bert_mask"])
            else:
                data["bert_mask"], data["true_msa"], data["msa"] = make_masked_msa(data["msa"], data["hhblits_profile"],
                                                                                   uniform_prob, profile_prob,
                                                                                   same_prob,
                                                                                   replace_fraction)
            
        if msa_cluster_features:
            if self.is_multimer:
                data["cluster_profile"], data["cluster_deletion_mean"] = nearest_neighbor_clusters_v2(data["msa"],
                                                                                                      data["msa_mask"],
                                                                                                      data["extra_msa"],
                                                                                                      data["extra_msa_mask"],
                                                                                                      data["deletion_matrix"],
                                                                                                      data["extra_deletion_matrix"])
            else:
                data["extra_cluster_assignment"] = nearest_neighbor_clusters(data["msa_mask"], data["msa"],
                                                                            data["extra_msa_mask"], data["extra_msa"])
                data["cluster_profile"], data["cluster_deletion_mean"] = summarize_clusters(data["msa"], data["msa_mask"],
                                                                                            data[
                                                                                                "extra_cluster_assignment"],
                                                                                            data["extra_msa_mask"],
                                                                                            data["extra_msa"],
                                                                                            data["extra_deletion_matrix"],
                                                                                            data["deletion_matrix"])
            
        if self.is_multimer:
            data["msa_feat"] = make_msa_feat_v2(data["msa"], data["deletion_matrix"], data["cluster_deletion_mean"], data["cluster_profile"])
            # with open("/data6/yhding/1228/ensemble_compare/my_make_msa_feat.pkl", "wb") as f:
            #     pickle.dump(data, f)
            extra_feats = make_extra_msa_feat(data["extra_msa"], data["extra_deletion_matrix"], data["extra_msa_mask"], self.cfg.common.max_extra_msa)
            data["extra_msa"] = extra_feats["extra_msa"]
            data["extra_msa_mask"] = extra_feats["extra_msa_mask"]
            data["extra_msa_has_deletion"] = extra_feats["extra_msa_has_deletion"]
            data["extra_msa_deletion_value"] = extra_feats["extra_msa_deletion_value"]
            
        else:
            if max_extra_msa:
                select_indices = crop_extra_msa(data["extra_msa"], max_extra_msa)
                if select_indices:
                    for k in _MSA_FEATURE_NAMES:
                        if 'extra_' + k in data:
                            data['extra_' + k] = data['extra_' + k][select_indices]
            else:
                for k in _MSA_FEATURE_NAMES:
                    if 'extra_' + k in data:
                        del data['extra_' + k]
            data["extra_has_deletion"], data["extra_deletion_value"], data["msa_feat"], data["target_feat"] = make_msa_feat(
                data["between_segment_residues"], data["aatype"], data["msa"], data["deletion_matrix"],
                data["cluster_deletion_mean"], data["cluster_profile"], data["extra_deletion_matrix"])

            if fixed_size:
                data = {k: v for k, v in data.items() if k in feature_list}

                num_res_crop_size, num_templates_crop_size_int, num_res_crop_start, num_res_crop_size_int, \
                templates_crop_start, templates_select_indices = random_crop_to_size(
                    data["seq_length"], data["template_mask"], crop_size, max_templates,
                    subsample_templates, seed, random_recycle)
                for k, v in data.items():
                    if k not in feature_list or ('template' not in k and NUM_RES not in feature_list.get(k)):
                        continue

                    # randomly permute the templates before cropping them.
                    if k.startswith('template') and subsample_templates:
                        v = v[templates_select_indices]

                    crop_sizes = []
                    crop_starts = []
                    for i, (dim_size, dim) in enumerate(zip(feature_list.get(k), v.shape)):
                        is_num_res = (dim_size == NUM_RES)
                        if i == 0 and k.startswith('template'):
                            crop_size_ = num_templates_crop_size_int
                            crop_start = templates_crop_start
                        else:
                            crop_start = num_res_crop_start if is_num_res else 0
                            crop_size_ = (num_res_crop_size_int if is_num_res else (-1 if dim is None else dim))
                        crop_sizes.append(crop_size_)
                        crop_starts.append(crop_start)
                    if len(v.shape) == 1:
                        data[k] = v[crop_starts[0]:crop_starts[0] + crop_sizes[0]]
                    elif len(v.shape) == 2:
                        data[k] = v[crop_starts[0]:crop_starts[0] + crop_sizes[0],
                                    crop_starts[1]:crop_starts[1] + crop_sizes[1]]
                    elif len(v.shape) == 3:
                        data[k] = v[crop_starts[0]:crop_starts[0] + crop_sizes[0],
                                    crop_starts[1]:crop_starts[1] + crop_sizes[1],
                                    crop_starts[2]:crop_starts[2] + crop_sizes[2]]
                    else:
                        data[k] = v[crop_starts[0]:crop_starts[0] + crop_sizes[0],
                                    crop_starts[1]:crop_starts[1] + crop_sizes[1],
                                    crop_starts[2]:crop_starts[2] + crop_sizes[2],
                                    crop_starts[3]:crop_starts[3] + crop_sizes[3]]

                data["seq_length"] = num_res_crop_size

                pad_size_map = {
                    NUM_RES: crop_size,
                    NUM_MSA_SEQ: max_msa_clusters,
                    NUM_EXTRA_SEQ: max_extra_msa,
                    NUM_TEMPLATES: max_templates,
                }

                for k, v in data.items():
                    if k == 'extra_cluster_assignment':
                        continue
                    shape = list(v.shape)
                    schema = feature_list.get(k)
                    assert len(shape) == len(
                        schema), f'Rank mismatch between shape and shape schema for {k}: {shape} vs {schema}'

                    pad_size = [pad_size_map.get(s2, None) or s1 for (s1, s2) in zip(shape, schema)]
                    padding = [(0, p - v.shape[i]) for i, p in enumerate(pad_size)]
                    if padding:
                        data[k] = np.pad(v, padding)
                        data[k].reshape(pad_size)
            else:
                for k, v in data.items():
                    if k.startswith('template_'):
                        data[k] = v[:max_templates]
            if self.is_evogen:
                data["random_data"], data["context_mask"] = generate_random_sample(self.cfg, self.model_cfg)
                data["context_mask"] = data["context_mask"].astype(np.float32)
        return data

    def process_res(self, features, res, dtype):
        """process result"""
        arrays, prev_pos, prev_msa_first_row, prev_pair = res
        if self.is_evogen:
            evogen_keys = ["target_feat", "seq_mask", "aatype", "residx_atom37_to_atom14", "atom37_atom_exists",
                           "residue_index", "msa_mask", "msa_input", "query_input", "additional_input", "random_data",
                           "context_mask"]
            arrays = [features[key] for key in evogen_keys]
            arrays = [array.astype(dtype) if array.dtype == "float64" else array for array in arrays]
            arrays = [array.astype(dtype) if array.dtype == "float32" else array for array in arrays]
            res = [arrays, prev_pos, prev_msa_first_row, prev_pair]
            return res
        if self.is_training:
            label_keys = ["pseudo_beta", "pseudo_beta_mask", "all_atom_mask",
                          "true_msa", "bert_mask", "residue_index", "seq_mask",
                          "atom37_atom_exists", "aatype", "residx_atom14_to_atom37",
                          "atom14_atom_exists", "backbone_affine_tensor", "backbone_affine_mask",
                          "atom14_gt_positions", "atom14_alt_gt_positions",
                          "atom14_atom_is_ambiguous", "atom14_gt_exists", "atom14_alt_gt_exists",
                          "all_atom_positions", "rigidgroups_gt_frames", "rigidgroups_gt_exists",
                          "rigidgroups_alt_gt_frames", "torsion_angles_sin_cos", "chi_mask"]
            label_arrays = [features[key] for key in label_keys]
            label_arrays = [array[0] for array in label_arrays]
            label_arrays = [array.astype(dtype) if array.dtype == "float64" else array for array in label_arrays]
            label_arrays = [array.astype(dtype) if array.dtype == "float32" else array for array in label_arrays]
            res = [arrays, prev_pos, prev_msa_first_row, prev_pair, label_arrays]
            return res
        return res


    def crop_and_fix_size(self, features, crop_and_fix_size_seed):
        crop_feats = dict(multimer_feature_list)
        crop_and_fix_size_seed = int(crop_and_fix_size_seed)
        with numpy_seed(crop_and_fix_size_seed, key="multimer_crop"):
            use_spatial_crop = np.random.rand() < self.cfg.common.spatial_crop_prob # 0.5
        if use_spatial_crop:
            crop_idx = get_spatial_crop_idx(features, crop_size=self.cfg.common.crop_size, random_seed=crop_and_fix_size_seed, ca_ca_threshold=self.cfg.common.ca_ca_threshold)
            # crop_idx = get_spatial_crop_idx_v2(features, crop_size=self.cfg.common.crop_size, random_seed=crop_and_fix_size_seed, ca_ca_threshold=self.cfg.common.ca_ca_threshold)
        else:
            crop_idx = get_contiguous_crop_idx(features, crop_size=self.cfg.common.crop_size, random_seed=crop_and_fix_size_seed)
        # print(len(crop_idx), features["msa"].shape)

        features = apply_crop_idx(features, shape_schema=crop_feats, crop_idx=crop_idx)

        # show_npdict(features, "crop but not pad")

        return features

    def pipeline(self, cfg, mixed_precision=True, seed=0):
        """feature process pipeline"""
        self.non_ensemble(cfg.common.distillation, cfg.common.replace_proportion, cfg.common.use_templates)
        non_ensemble_data = vars(self).copy()

        crop_and_fix_size_seed = seed
        num_recycling = self.cfg.common.num_recycle + 1 # 3 + 1
        num_ensembles = self.cfg.common.num_ensembles # 1
        max_msa_clusters = self.cfg.common.max_msa_clusters - self.cfg.common.max_templates #256-4
        max_extra_msa = self.cfg.common.max_extra_msa #1024
        def wrap_ensemble(data, i):
            d = data.copy()
            
            d = self.ensemble(d, max_msa_clusters=max_msa_clusters, #252
                                max_extra_msa=max_extra_msa, #1024
                                masked_msa=self.cfg.common.use_masked_msa, # True
                                profile_prob=self.cfg.common.profile_prob, # 0.1
                                same_prob=self.cfg.common.same_prob, # 0.1
                                uniform_prob=self.cfg.common.uniform_prob, # 0.1
                                replace_fraction=self.cfg.common.replace_fraction, # 0.15
                                msa_cluster_features=self.cfg.common.msa_cluster_features) #True

            # d = self.crop_and_fix_size(d, crop_and_fix_size_seed)

            if self.cfg.common.reduce_msa_clusters_by_max_templates: # True
                pad_msa_clusters = self.cfg.common.max_msa_clusters - self.cfg.common.max_templates
            else:
                pad_msa_clusters = self.cfg.common.max_msa_clusters
            crop_feats = dict(multimer_feature_list)
            d = select_feat(d, crop_feats)
            d = make_fixed_size(d, crop_feats, 
                                pad_msa_clusters, # 252
                                self.cfg.common.max_extra_msa, # 1024
                                self.cfg.common.crop_size, # 384
                                self.cfg.common.max_templates) # 4
                    
            return d
        
        features = non_ensemble_data.copy()

        features.pop("cfg")
        features_new = self.crop_and_fix_size(features, crop_and_fix_size_seed)
        for key in list(set(list(features.keys())) - set(list(features_new.keys()))):
            features_new[key] = features[key]
        features = features_new
        features["seq_length"] = np.array(features["msa"].shape[1])
        # print('\n\n====================== features after crop ===================')
        # show_npdict(features)
        ensemble_features = map_fn(
            lambda x: wrap_ensemble(features, x),
            np.arange(num_recycling * num_ensembles)
        )
        
        if self.cfg.common.reduce_msa_clusters_by_max_templates:
            pad_msa_clusters = self.cfg.common.max_msa_clusters - self.cfg.common.max_templates
        else:
            pad_msa_clusters = self.cfg.common.max_msa_clusters
        crop_feats = dict(multimer_feature_list)
        processed_features = select_feat(features, crop_feats)
        processed_features = make_fixed_size(processed_features, crop_feats, 
                            pad_msa_clusters,
                            self.cfg.common.max_extra_msa,
                            self.cfg.common.crop_size,
                            self.cfg.common.max_templates)
        processed_features = {k: np.stack([processed_features[k]], axis=0) for k in processed_features}
        
        np.set_printoptions(threshold=np.inf)
        processed_features.update(ensemble_features)
        # show_npdict(processed_features, "feats after ensemble")
        # print(processed_features["num_sym"].shape, flush=True)

        # print(f"\n\n==========================ori processed_feat before duplicating")
        # # for key, value in all_labels[0].items():
        # #     print(key, value.shape, value.dtype, flush=True)
        # keys = list(processed_features.keys())
        # keys.sort()
        # for key in keys:
        #     value = processed_features[key]
        #     print(key, value.shape, value.dtype, flush=True)

        # for key, value in processed_features.items():
        #     if value.shape[0] == 1:
        #         processed_features[key] = np.concatenate([value] * num_recycling, axis=0)

        # print(f"\n\n==========================ori processed_feat")
        # # for key, value in all_labels[0].items():
        # #     print(key, value.shape, value.dtype, flush=True)
        # keys = list(processed_features.keys())
        # keys.sort()
        # for key in keys:
        #     value = processed_features[key]
        #     print(key, value.shape, value.dtype, flush=True)

        def custom_padding(seq_length, array, dim, res_length):
            """Pad array to fixed size."""
            padding_size = seq_length - res_length
            extra_array_shape = list(array.shape)
            extra_array_shape[dim] = padding_size
            extra_array = np.zeros(extra_array_shape, dtype=array.dtype)
            array = np.concatenate((array, extra_array), axis=dim)
            return array


        crop_1_dim_key = ['aatype', 'target_feat', 'residx_atom37_to_atom14', 'atom37_atom_exists',
                'residue_index', 'asym_id', 'sym_id', 'entity_id', 'seq_mask', "num_sym"]
        crop_2_dim_key = ['msa_feat', 'template_aatype', 'template_all_atom_masks', 'template_all_atom_positions',
                        'extra_msa', 'extra_msa_deletion_value', 'extra_msa_mask', 'msa_mask', "bert_mask", "true_msa"]
        
        res_length = processed_features["msa_feat"].shape[2]
        for key in crop_1_dim_key:
            processed_features[key] = custom_padding(self.cfg.common.crop_size, processed_features[key], 1, res_length)
        for key in crop_2_dim_key:
            processed_features[key] = custom_padding(self.cfg.common.crop_size, processed_features[key], 2, res_length)

        num_extra_seq = processed_features['extra_msa'].shape[1]
        if num_extra_seq < self.cfg.common.max_extra_msa:
            for key in ["extra_msa", "extra_msa_mask", "extra_msa_deletion_value"]:
                processed_features[key] = custom_padding(self.cfg.common.max_extra_msa, processed_features[key], 1, num_extra_seq)
        else:
            for key in ["extra_msa", "extra_msa_mask", "extra_msa_deletion_value"]:
                processed_features[key] = processed_features[key][:, :self.cfg.common.max_extra_msa, :]
    
        processed_features["extra_msa_deletion_value"] = processed_features["extra_msa_deletion_value"]
        dtype = np.float16
        for key, value in processed_features.items():
            if value.dtype == "float64":
                # print(key, "hello, float64")
                processed_features[key] = value.astype(dtype)
                # print(processed_features[key].dtype)
            if value.dtype == "float32":
                processed_features[key] = value.astype(dtype)



        # print(f"\n\n==========================processed_feat after padding")
        # # for key, value in all_labels[0].items():
        # #     print(key, value.shape, value.dtype, flush=True)
        # keys = list(processed_features.keys())
        # keys.sort()
        # for key in keys:
        #     value = processed_features[key]
        #     print(key, value.shape, value.dtype, flush=True)
        # show_npdict(processed_features, 'processed_feat after padding')


        input_keys = ['aatype', 'residue_index', 'template_aatype', 'template_all_atom_masks',
                    'template_all_atom_positions', 'asym_id', 'sym_id', 'entity_id', 'seq_mask', 'msa_mask',
                    'target_feat', 'msa_feat', 'extra_msa', 'extra_msa_deletion_value', 'extra_msa_mask',
                    'residx_atom37_to_atom14', 'atom37_atom_exists']

        # input_keys.sort()
        # print(f"\n\n==========================infer input")
        # print(processed_features["asym_id"][0])
        # print(processed_features["sym_id"][0])
        # # import time
        # # time.sleep(10)
        # print(processed_features["entity_id"][0])
        # print(processed_features["residue_index"][0])
        res_arrays = []
        for key in input_keys:
            value = processed_features[key]
            res_arrays.append(value)
            # print(key, value.shape, value.dtype)
        # print(np.sum(np.abs(processed_features["msa_feat"][1] - processed_features["msa_feat"][0])))
        # print(np.sum(np.abs(processed_features["msa_feat"][2] - processed_features["msa_feat"][1])))
        
        prev_pos = np.zeros([self.cfg.common.crop_size, 37, 3]).astype(dtype)
        prev_msa_first_row = np.zeros([self.cfg.common.crop_size, 256]).astype(dtype)
        prev_pair = np.zeros([self.cfg.common.crop_size, self.cfg.common.crop_size, 128]).astype(dtype)
        num_sym = processed_features["num_sym"][0]
        bert_mask = processed_features["bert_mask"]
        true_msa = processed_features["true_msa"]
        res = [res_arrays, prev_pos, prev_msa_first_row, prev_pair, num_sym, bert_mask, true_msa]

        return res



class MultimerFeature:
    """multimer feature process"""

    def __init__(self, mixed_precision=True):
        self.mixed_precision = mixed_precision

    def np_mask_mean(self, mask, value, axis=None, drop_mask_channel=False, eps=1e-10):
        """Numpy masked mean."""
        if drop_mask_channel:
            mask = mask[..., 0]
        mask_shape = mask.shape
        value_shape = value.shape
        broadcast_factor = 1.
        value_size = value_shape[axis]
        mask_size = mask_shape[axis]
        if mask_size == 1:
            broadcast_factor *= value_size
        return np.sum(mask * value, axis=axis) / (np.sum(mask, axis=axis) * broadcast_factor + eps)

    def sample_msa(self, raw_features, max_seq):
        """Sample MSA randomly."""
        logits = (np.clip(np.sum(raw_features['msa_mask'], axis=-1), 0., 1.) - 1.) * 1e6
        if 'cluster_bias_mask' not in raw_features:
            cluster_bias_mask = np.pad(
                np.zeros(raw_features['msa'].shape[0] - 1), (1, 0), constant_values=1.)
        else:
            cluster_bias_mask = raw_features['cluster_bias_mask']
        logits += cluster_bias_mask * 1e6
        z = np.random.gumbel(loc=0.0, scale=1.0, size=logits.shape)
        index_order = np.argsort(-(logits + z), axis=-1, kind='quicksort', order=None)
        sel_idx = index_order[:max_seq]
        extra_idx = index_order[max_seq:]
        for k in ['msa', 'deletion_matrix', 'msa_mask', 'bert_mask']:
            if k in raw_features:
                raw_features['extra_' + k] = raw_features[k][extra_idx]
                raw_features[k] = raw_features[k][sel_idx]
        return raw_features

    def make_masked_msa(self, raw_features, config, epsilon=1e-6):
        """create data for BERT on raw MSA."""
        random_aa = np.array([0.05] * 20 + [0., 0.], dtype=np.float32)
        categorical_probs = (
            config.uniform_prob * random_aa +
            config.profile_prob * raw_features['msa_profile'] +
            config.same_prob * np.eye(22)[raw_features['msa']])
        pad_shapes = [[0, 0] for _ in range(len(categorical_probs.shape))]
        pad_shapes[-1][1] = 1
        mask_prob = 1. - config.profile_prob - config.same_prob - config.uniform_prob
        categorical_probs = np.pad(categorical_probs, pad_shapes, constant_values=mask_prob)
        sh = raw_features['msa'].shape
        mask_position = (np.random.uniform(0., 1., sh) < config.replace_fraction).astype(np.float32)
        mask_position *= raw_features['msa_mask']
        logits = np.log(categorical_probs + epsilon)
        z = np.random.gumbel(loc=0.0, scale=1.0, size=logits.shape)
        bert_msa = np.eye(logits.shape[-1], dtype=logits.dtype)[np.argmax(logits + z, axis=-1)]
        bert_msa = (np.where(mask_position,
                             np.argmax(bert_msa, axis=-1), raw_features['msa']))
        bert_msa *= (raw_features['msa_mask'].astype(np.int64))
        if 'bert_mask' in raw_features:
            raw_features['bert_mask'] *= mask_position.astype(np.float32)
        else:
            raw_features['bert_mask'] = mask_position.astype(np.float32)
        raw_features['true_msa'] = raw_features['msa']
        raw_features['msa'] = bert_msa
        return raw_features

    def softmax(self, x, axis):
        """ Softmax func"""
        x -= np.max(x, axis=axis, keepdims=True)
        x = np.exp(x) / np.sum(np.exp(x), axis=axis, keepdims=True)
        return x

    def nearest_neighbor_clusters(self, raw_features, gap_agreement_weight=0.):
        """Assign each extra MSA sequence to its nearest neighbor in sampled MSA."""
        weights = np.array(
            [1.] * 21 + [gap_agreement_weight] + [0.], dtype=np.float32)
        msa_mask = raw_features['msa_mask']
        msa_one_hot = np.eye(23)[raw_features['msa']]
        extra_mask = raw_features['extra_msa_mask']
        extra_one_hot = np.eye(23)[raw_features['extra_msa']]
        msa_one_hot_masked = msa_mask[:, :, None] * msa_one_hot
        extra_one_hot_masked = extra_mask[:, :, None] * extra_one_hot
        agreement = np.einsum('mrc, nrc->nm', extra_one_hot_masked,
                              weights * msa_one_hot_masked)
        cluster_assignment = self.softmax(1e3 * agreement, axis=0)
        cluster_assignment *= np.einsum('mr, nr->mn', msa_mask, extra_mask)
        cluster_count = np.sum(cluster_assignment, axis=-1)
        cluster_count += 1.
        msa_sum = np.einsum('nm, mrc->nrc', cluster_assignment, extra_one_hot_masked)
        msa_sum += msa_one_hot_masked
        cluster_profile = msa_sum / cluster_count[:, None, None]
        extra_deletion_matrix = raw_features['extra_deletion_matrix']
        deletion_matrix = raw_features['deletion_matrix']
        del_sum = np.einsum('nm, mc->nc', cluster_assignment,
                            extra_mask * extra_deletion_matrix)
        del_sum += deletion_matrix
        cluster_deletion_mean = del_sum / cluster_count[:, None]
        return cluster_profile, cluster_deletion_mean

    def create_msa_feat(self, raw_features):
        """Create and concatenate MSA features."""
        msa_1hot = np.eye(23)[raw_features['msa']]
        deletion_matrix = raw_features['deletion_matrix']
        has_deletion = np.clip(deletion_matrix, 0., 1.)[..., None]
        deletion_value = (np.arctan(deletion_matrix / 3.) * (2. / np.pi))[..., None]
        deletion_mean_value = (np.arctan(raw_features['cluster_deletion_mean'] / 3.) *
                               (2. / np.pi))[..., None]
        msa_feat = [
            msa_1hot,
            has_deletion,
            deletion_value,
            raw_features['cluster_profile'],
            deletion_mean_value
        ]
        return np.concatenate(msa_feat, axis=-1)

    def custom_padding(self, seq_length, array, dim, res_length):
        """Pad array to fixed size."""
        padding_size = seq_length - res_length
        extra_array_shape = list(array.shape)
        extra_array_shape[dim] = padding_size
        extra_array = np.zeros(extra_array_shape, dtype=array.dtype)
        array = np.concatenate((array, extra_array), axis=dim)
        return array

    def pipeline(self, model_cfg, data_cfg, raw_feature):
        """Preprocesses Numpy feature dict in multimer model"""
        if not data_cfg.common.random_recycle:
            np.random.seed(0)

        features = raw_feature.copy()
        features['msa_profile'] = self.np_mask_mean(features['msa_mask'][:, :, None],
                                                    np.eye(22)[features['msa']], axis=0)

        features['target_feat'] = np.eye(21)[features['aatype']]

        # if data_cfg.common.target_feat_dim == 22:
        #     bsr = np.zeros_like(features["aatype"], dtype=np.float32)
        #     has_break = np.clip(bsr, 0, 1)
        #     features["target_feat"] = np.concatenate([np.expand_dims(has_break, axis=-1), features['target_feat']], axis=-1)

        # print(features["target_feat"].shape)


        features = self.sample_msa(features, model_cfg.multimer.embeddings_and_evoformer.num_msa)
        features = self.make_masked_msa(features, model_cfg.multimer.embeddings_and_evoformer.masked_msa)
        (features['cluster_profile'], features['cluster_deletion_mean']) = self.nearest_neighbor_clusters(features)
        features['msa_feat'] = self.create_msa_feat(features)
        res_length = features['aatype'].shape[0]
        _, _, features['residx_atom37_to_atom14'], features['atom37_atom_exists'] = \
            make_atom14_masks(features['aatype'])
        crop_0_dim_key = ['aatype', 'target_feat', 'residx_atom37_to_atom14', 'atom37_atom_exists',
                          'residue_index', 'asym_id', 'sym_id', 'entity_id', 'seq_mask']
        crop_1_dim_key = ['msa_feat', 'template_aatype', 'template_all_atom_mask', 'template_all_atom_positions',
                          'extra_msa', 'extra_deletion_matrix', 'extra_msa_mask', 'msa_mask']
        for key in crop_0_dim_key:
            features[key] = self.custom_padding(model_cfg.seq_length, features[key], 0, res_length)
        for key in crop_1_dim_key:
            features[key] = self.custom_padding(model_cfg.seq_length, features[key], 1, res_length)
        num_extra_seq = features['extra_msa'].shape[0]
        if num_extra_seq < data_cfg.common.max_extra_msa:
            for key in ["extra_msa", "extra_msa_mask", "extra_deletion_matrix"]:
                features[key] = self.custom_padding(data_cfg.common.max_extra_msa, features[key], 0, num_extra_seq)
        else:
            for key in ["extra_msa", "extra_msa_mask", "extra_deletion_matrix"]:
                features[key] = features[key][:data_cfg.common.max_extra_msa, :]

        features['extra_deletion_matrix'] =  np.arctan(features['extra_deletion_matrix'] / 3.) * (2. / np.pi)
        input_keys = ['aatype', 'residue_index', 'template_aatype', 'template_all_atom_mask',
                      'template_all_atom_positions', 'asym_id', 'sym_id', 'entity_id', 'seq_mask', 'msa_mask',
                      'target_feat', 'msa_feat', 'extra_msa', 'extra_deletion_matrix', 'extra_msa_mask',
                      'residx_atom37_to_atom14', 'atom37_atom_exists']
        dtype = np.float32
        if self.mixed_precision:
            dtype = np.float16
        print("msa_feat_sum", np.sum(features["msa_feat"]), flush=True)
        arrays = [features[key] for key in input_keys]
        arrays = [array.astype(dtype) if array.dtype == "float64" else array for array in arrays]
        arrays = [array.astype(dtype) if array.dtype == "float32" else array for array in arrays]
        return arrays

