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
"""design_fold"""
import mindspore.common.dtype as mstype
import mindspore.nn as nn
import mindspore.numpy as mnp
from mindspore.ops import operations as P
from mindspore.common.tensor import Tensor
from mindspore import ops

from fold import MegaFold  # need to import the package of Megafold like sys.path.append('../MEGAProtein/model/')
from model.loss_design import LossNet


class Colabdesign(nn.Cell):
    """Colabdesign"""

    def __init__(self, config, mixed_precision, seq_vector, ori_seq_len, design_cfg, protocol):
        super(Colabdesign, self).__init__()
        self.megafold = MegaFold(config, mixed_precision)
        self.megafold.add_flags_recursive(train_backward=True)
        self.cfg = config
        self.seq_vector = seq_vector
        self.ori_seq_len = ori_seq_len
        self.crop_size = config.seq_length
        self.opt_alpha = Tensor(design_cfg.opt_alpha, mstype.float32)
        self.opt_bias = Tensor(design_cfg.opt_bias, mstype.float32)
        self.opt_use_pssm = design_cfg.opt_use_pssm
        self.loss_net = LossNet(design_cfg, protocol)

    def soft_seq(self, x, ori_seq_len, opt_temp_num, opt_soft_num, opt_hard_num):
        """soft_seq"""
        seq_input = x[:, :ori_seq_len, :]
        seq_logits = seq_input * self.opt_alpha + self.opt_bias
        seq_pssm = P.Softmax()(seq_logits)
        seq_soft = P.Softmax()(seq_logits / opt_temp_num)
        seq_hard = P.OneHot()(seq_soft.argmax(-1), 20, Tensor(1.0, mstype.float32), Tensor(0.0, mstype.float32))
        seq_hard = ops.stop_gradient(seq_hard - seq_soft) + seq_soft

        seq_pseudo = opt_soft_num * seq_soft + (1 - opt_soft_num) * seq_input

        hard_mask = opt_hard_num
        seq_pseudo = hard_mask * seq_hard + (1 - hard_mask) * seq_pseudo
        seqs_res = (seq_logits, seq_pssm, seq_pseudo, seq_hard)
        return seqs_res

    def update_seq(self, seq, msa_feat, ori_seq_len, seq_1hot=None, seq_pssm=None):
        """update the sequence features"""

        if seq_1hot is None:
            seq_1hot = seq
        if seq_pssm is None:
            seq_pssm = seq

        seq_1hot = mnp.pad(seq_1hot, [[0, 0], [0, self.crop_size - ori_seq_len], [0, 22 - seq_1hot.shape[-1]]])
        seq_pssm = mnp.pad(seq_pssm, [[0, 0], [0, self.crop_size - ori_seq_len], [0, 22 - seq_pssm.shape[-1]]])
        msa_feat = mnp.zeros_like(msa_feat, dtype=mstype.float32)
        msa_feat[:seq_1hot.shape[0], :, 0:22] = seq_1hot
        msa_feat[:seq_1hot.shape[0], :, 25:47] = seq_pssm
        return msa_feat

    def construct(self, msa_feat, msa_mask, seq_mask,
                  template_aatype, template_all_atom_masks, template_all_atom_positions,
                  template_mask, template_pseudo_beta_mask, template_pseudo_beta, extra_msa, extra_has_deletion,
                  extra_deletion_value, extra_msa_mask,
                  residx_atom37_to_atom14, atom37_atom_exists, residue_index, true_aatype, true_all_atom_positions,
                  true_all_atom_mask, opt_temp_num, opt_soft_num, opt_hard_num,
                  prev_pos, prev_msa_first_row, prev_pair):
        """construct"""
        seqs_res = self.soft_seq(self.seq_vector, self.ori_seq_len,
                                 opt_temp_num, opt_soft_num, opt_hard_num)
        seq_logits, seq_pssm, seq_pseudo, seq_hard = seqs_res[0], seqs_res[1], seqs_res[2], seqs_res[3]
        if self.opt_use_pssm:
            pssm = seq_pssm
        else:
            pssm = seq_pseudo
        msa_feat = self.update_seq(seq_pseudo, msa_feat, self.ori_seq_len, seq_pssm=pssm)
        target_feat = msa_feat[0, :, :21]
        target_feat = mnp.pad(target_feat, [[0, 0], [1, 0]])
        aatype = seq_pseudo[0].argmax(-1)
        aatype = mnp.pad(aatype, [[0, self.crop_size - self.ori_seq_len]])

        dist_logits, bin_edges, experimentally_logits, _, aligned_error_logits, \
        _, _, _, _, predicted_lddt_logits, _, _, _, \
        _, final_atom_positions = self.megafold(target_feat, msa_feat, msa_mask, seq_mask, aatype,
                                                template_aatype, template_all_atom_masks,
                                                template_all_atom_positions,
                                                template_mask, template_pseudo_beta_mask,
                                                template_pseudo_beta, extra_msa, extra_has_deletion,
                                                extra_deletion_value, extra_msa_mask,
                                                residx_atom37_to_atom14, atom37_atom_exists,
                                                residue_index,
                                                prev_pos, prev_msa_first_row, prev_pair)
        if self.save_best:
            pdb_result = (
                final_atom_positions, atom37_atom_exists, predicted_lddt_logits, aatype, residue_index, seq_hard)
            return pdb_result

        loss_all = \
            self.loss_net(true_aatype, true_all_atom_positions, true_all_atom_mask, final_atom_positions,
                          self.ori_seq_len, dist_logits, bin_edges, experimentally_logits, predicted_lddt_logits,
                          aligned_error_logits, residue_index, seq_logits)
        return loss_all
