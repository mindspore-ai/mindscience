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
"""megafold_feature"""
NUM_RES = 'num residues placeholder'
NUM_MSA_SEQ = 'msa placeholder'
NUM_EXTRA_SEQ = 'extra msa placeholder'
NUM_TEMPLATES = 'num templates placeholder'
_msa_feature_names = ['msa', 'deletion_matrix', 'msa_mask', 'msa_row_mask', 'bert_mask', 'true_msa']

_inference_feature = {
    'aatype': [NUM_RES],
    'atom37_atom_exists': [NUM_RES, None],
    'extra_deletion_value': [NUM_EXTRA_SEQ, NUM_RES],
    'extra_has_deletion': [NUM_EXTRA_SEQ, NUM_RES],
    'extra_msa': [NUM_EXTRA_SEQ, NUM_RES],
    'extra_msa_mask': [NUM_EXTRA_SEQ, NUM_RES],
    'msa_feat': [NUM_MSA_SEQ, NUM_RES, None],
    'msa_mask': [NUM_MSA_SEQ, NUM_RES],
    'residue_index': [NUM_RES],
    'residx_atom37_to_atom14': [NUM_RES, None],
    'seq_mask': [NUM_RES],
    'target_feat': [NUM_RES, None],
    'template_aatype': [NUM_TEMPLATES, NUM_RES],
    'template_all_atom_masks': [NUM_TEMPLATES, NUM_RES, None],
    'template_all_atom_positions': [NUM_TEMPLATES, NUM_RES, None, None],
    'template_mask': [NUM_TEMPLATES],
    'template_pseudo_beta': [NUM_TEMPLATES, NUM_RES, None],
    'template_pseudo_beta_mask': [NUM_TEMPLATES, NUM_RES],
    'num_residues': [None]
}

_training_feature = {
    'all_atom_mask': [NUM_RES, None],
    'all_atom_positions': [NUM_RES, None, None],
    'atom14_alt_gt_exists': [NUM_RES, None],
    'atom14_alt_gt_positions': [NUM_RES, None, None],
    'atom14_atom_exists': [NUM_RES, None],
    'atom14_atom_is_ambiguous': [NUM_RES, None],
    'atom14_gt_exists': [NUM_RES, None],
    'atom14_gt_positions': [NUM_RES, None, None],
    'backbone_affine_mask': [NUM_RES],
    'backbone_affine_tensor': [NUM_RES, None],
    'bert_mask': [NUM_MSA_SEQ, NUM_RES],
    'chi_mask': [NUM_RES, None],
    'pseudo_beta': [NUM_RES, None],
    'pseudo_beta_mask': [NUM_RES],
    'residx_atom14_to_atom37': [NUM_RES, None],
    'rigidgroups_alt_gt_frames': [NUM_RES, None, None],
    'rigidgroups_gt_exists': [NUM_RES, None],
    'rigidgroups_gt_frames': [NUM_RES, None, None],
    'true_msa': [NUM_MSA_SEQ, NUM_RES],
    'torsion_angles_sin_cos': [NUM_RES, None, None]
}
