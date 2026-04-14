# Copyright 2023 Huawei Technologies Co., Ltd & CPL YiQin GAO Research Group
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
"""megaassessment dataset"""
import numpy as np
from mindsponge.common.residue_constants import order_restype_with_x
from mindsponge.common.utils import get_aligned_seq

from ..megafold.megafold_dataset import MEGAFoldDataSet


class MEGAAssessmentDataSet(MEGAFoldDataSet):
    """megasssessment dataset"""
    def __init__(self, config, seed=0):
        self.config = config
        self.supported_models = ['MEGAAssessment']
        super().__init__(self.config, seed)

    def process(self, data, label=None, ensemble_num=4):
        features = super().process(data, label, ensemble_num)
        if not label:
            features = self.process_pdb(features, data)
        return features

    def align_with_aatype(self, true_aatype, aatype, atom37_positions, atom37_mask):
        """align pdb with aatype"""
        if len(true_aatype) == len(aatype):
            out = aatype, atom37_positions, atom37_mask, np.ones((aatype.shape[0])).astype(np.float32)
            return out
        seq1 = [order_restype_with_x.get(x) for x in aatype]
        seq2 = [order_restype_with_x.get(x) for x in true_aatype]
        seq1 = ''.join(seq1)
        seq2 = ''.join(seq2)
        _, align_relationship, _ = get_aligned_seq(seq1, seq2)
        pdb_index = 0
        seq_len = len(true_aatype)
        new_aatype = np.zeros((seq_len,)).astype(np.int32)
        new_atom37_positions = np.zeros((seq_len, 37, 3)).astype(np.float32)
        new_atom37_mask = np.zeros((seq_len, 37)).astype(np.float32)
        align_mask = np.zeros((seq_len,)).astype(np.float32)
        for i in range(len(true_aatype)):
            if align_relationship[i] == "-":
                new_aatype[i] = 20
                new_atom37_positions[i] = np.zeros((37, 3)).astype(np.float32)
                new_atom37_mask[i] = np.zeros((37,)).astype(np.float32)
                align_mask[i] = 0
            else:
                new_aatype[i] = aatype[pdb_index]
                new_atom37_positions[i] = atom37_positions[pdb_index]
                new_atom37_mask[i] = atom37_mask[pdb_index]
                align_mask[i] = 1
                pdb_index += 1
        out = new_aatype, new_atom37_positions, new_atom37_mask, align_mask
        return out

    def process_pdb(self, features, data):
        """get atom information from pdb"""
        decoy_aatype = data["decoy_aatype"]
        decoy_atom37_positions = data["decoy_atom_positions"].astype(np.float32)
        decoy_atom37_mask = data["decoy_atom_mask"].astype(np.float32)
        ori_res_length = data['msa'].shape[1]
        padding_val = features["aatype"][0].shape[0] - ori_res_length
        true_aatype = features["aatype"][0][:ori_res_length]
        decoy_aatype, decoy_atom37_positions, decoy_atom37_mask, align_mask = \
            self.align_with_aatype(true_aatype, decoy_aatype, decoy_atom37_positions, decoy_atom37_mask)
        decoy_atom37_positions = np.pad(decoy_atom37_positions, ((0, padding_val), (0, 0), (0, 0)))
        decoy_atom37_mask = np.pad(decoy_atom37_mask, ((0, padding_val), (0, 0)))
        align_mask = np.pad(align_mask, (0, padding_val))

        features["decoy_atom_positions"] = decoy_atom37_positions
        features["decoy_atom_mask"] = decoy_atom37_mask
        features["align_mask"] = align_mask

        return features
