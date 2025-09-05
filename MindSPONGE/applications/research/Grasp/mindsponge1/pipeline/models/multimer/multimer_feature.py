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
"""multimer feature"""
NUM_RES = 'num residues placeholder'
NUM_MSA_SEQ = 'msa placeholder'
NUM_EXTRA_SEQ = 'extra msa placeholder'
NUM_TEMPLATES = 'num templates placeholder'
_msa_feature_names = ['msa', 'deletion_matrix', 'msa_mask', 'bert_mask']

_inference_feature = {
    'aatype': [NUM_RES],
    'residue_index': [NUM_RES],
    'template_aatype': [NUM_TEMPLATES, NUM_RES],
    'template_all_atom_mask': [NUM_TEMPLATES, NUM_RES, None],
    'template_all_atom_positions': [NUM_TEMPLATES, NUM_RES, None, None],
    'asym_id': [NUM_RES],
    'sym_id': [NUM_RES],
    'entity_id': [NUM_RES],
    'seq_mask': [NUM_RES],
    'msa_mask': [NUM_MSA_SEQ, NUM_RES],
    'target_feat': [NUM_RES, None],
    'msa_feat': [NUM_MSA_SEQ, NUM_RES, None],
    'extra_msa': [NUM_EXTRA_SEQ, NUM_RES],
    'extra_deletion_matrix': [NUM_EXTRA_SEQ, NUM_RES],
    'extra_msa_mask': [NUM_EXTRA_SEQ, NUM_RES],
    'residx_atom37_to_atom14': [NUM_RES, None],
    'atom37_atom_exists': [NUM_RES, None],
    'num_residues': [None]
}
