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
"""multimer model"""
import numpy as np
from scipy.special import softmax
import mindspore.common.dtype as mstype
import mindspore.nn as nn
import mindspore.numpy as mnp
from mindspore.ops import operations as P
from mindspore.common.tensor import Tensor
from mindspore import Parameter
from mindsponge.data_transform import get_chi_atom_pos_indices
from mindsponge.common import residue_constants
from mindsponge.common.utils import dgram_from_positions, pseudo_beta_fn
from mindsponge.cell.initializer import lecun_init
from .module.multimer_block import compute_chi_angles
from .module.multimer_template_embedding import MultimerTemplateEmbedding
from .module.multimer_evoformer import MultimerEvoformer
from .module.multimer_structure import MultimerStructureModule
from .module.multimer_head import PredictedLDDTHead


def caculate_constant_array(seq_length):
    '''constant array'''
    chi_atom_indices = np.array(get_chi_atom_pos_indices()).astype(np.int32)
    chi_angles_mask = list(residue_constants.chi_angles_mask)
    chi_angles_mask.append([0.0, 0.0, 0.0, 0.0])
    chi_angles_mask = np.array(chi_angles_mask).astype(np.float32)
    mirror_psi_mask = np.float32(np.asarray([1., 1., -1., 1., 1., 1., 1.])[None, None, :, None])
    chi_pi_periodic = np.float32(np.array(residue_constants.chi_pi_periodic))

    indices0 = np.arange(4).reshape((-1, 1, 1, 1, 1)).astype("int32")  # 4 batch
    indices0 = indices0.repeat(seq_length, axis=1)  # seq_length sequence length
    indices0 = indices0.repeat(4, axis=2)  # 4 chis
    indices0 = indices0.repeat(4, axis=3)  # 4 atoms

    indices1 = np.arange(seq_length).reshape((1, -1, 1, 1, 1)).astype("int32")
    indices1 = indices1.repeat(4, axis=0)
    indices1 = indices1.repeat(4, axis=2)
    indices1 = indices1.repeat(4, axis=3)

    constant_array = [chi_atom_indices, chi_angles_mask, mirror_psi_mask, chi_pi_periodic, indices0, indices1]
    constant_array = [Tensor(val) for val in constant_array]
    return constant_array


def compute_confidence(predicted_lddt_logits, return_lddt=False):
    """compute confidence"""

    num_bins = predicted_lddt_logits.shape[-1]
    bin_width = 1 / num_bins
    start_n = bin_width / 2
    plddt = compute_plddt(predicted_lddt_logits, start_n, bin_width)
    confidence = np.mean(plddt)
    if return_lddt:
        return confidence, plddt

    return confidence


def compute_plddt(logits, start_n, bin_width):
    """Computes per-residue pLDDT from logits.

    Args:
      logits: [num_res, num_bins] output from the PredictedLDDTHead.

    Returns:
      plddt: [num_res] per-residue pLDDT.
    """
    bin_centers = np.arange(start=start_n, stop=1.0, step=bin_width)
    probs = softmax(logits, axis=-1)
    predicted_lddt_ca = np.sum(probs * bin_centers[None, :], axis=-1)
    return predicted_lddt_ca * 100


class MultimerArch(nn.Cell):
    """MultimerArch"""

    def __init__(self, config, mixed_precision):
        super(MultimerArch, self).__init__()

        self.cfg = config

        if mixed_precision:
            self._type = mstype.float16
        else:
            self._type = mstype.float32
        self.recycle_pos = self.cfg.model.recycle_pos
        self.recycle_features = self.cfg.model.recycle_features
        self.max_relative_feature = self.cfg.model.max_relative_feature
        self.num_bins = self.cfg.model.prev_pos.num_bins
        self.min_bin = self.cfg.model.prev_pos.min_bin
        self.max_bin = self.cfg.model.prev_pos.max_bin
        self.use_chain_relative = self.cfg.model.use_chain_relative
        self.max_relative_chain = self.cfg.model.max_relative_chain
        self.template_enabled = self.cfg.model.template.enabled
        self.num_extra_msa = self.cfg.model.num_extra_msa
        self.extra_msa_stack_num = self.cfg.model.evoformer.extra_msa_stack_num
        self.msa_stack_num = self.cfg.model.evoformer.msa_stack_num
        self.chi_atom_indices, self.chi_angles_mask, _, _, \
            self.indices0, self.indices1 = caculate_constant_array(self.cfg.seq_length)
        self.pi = np.pi
        self.batch_block = 4
        self.preprocess_1d = nn.Dense(21, self.cfg.model.msa_channel,
                                      weight_init=lecun_init(21))
        self.preprocess_msa = nn.Dense(self.cfg.model.common.msa_feat_dim, self.cfg.model.msa_channel,
                                       weight_init=lecun_init(self.cfg.model.common.msa_feat_dim))
        self.left_single = nn.Dense(21, self.cfg.model.pair_channel,
                                    21)
        self.right_single = nn.Dense(21, self.cfg.model.pair_channel,
                                     weight_init=lecun_init(21))
        self.prev_pos_linear = nn.Dense(self.cfg.model.common.dgram_dim, self.cfg.model.pair_channel,
                                        weight_init=lecun_init(self.cfg.model.common.dgram_dim))
        self.extra_msa_one_hot = nn.OneHot(depth=23, axis=-1)
        self.template_aatype_one_hot = nn.OneHot(depth=22, axis=-1)
        self.prev_msa_first_row_norm = nn.LayerNorm([256,], epsilon=1e-5)
        self.prev_pair_norm = nn.LayerNorm([128,], epsilon=1e-5)
        if self.use_chain_relative:
            self.rel_pos_one_hot = nn.OneHot(depth=self.cfg.model.max_relative_feature * 2 + 2, axis=-1)
            self.rel_chain_one_hot = nn.OneHot(depth=self.max_relative_chain * 2 + 2, axis=-1)
            self.position_activations = nn.Dense(self.cfg.model.pair_in_dim, self.cfg.model.pair_channel,
                                                 weight_init=lecun_init(self.cfg.model.common.pair_in_dim))
        else:
            self.one_hot = nn.OneHot(depth=self.cfg.model.max_relative_feature * 2 + 1, axis=-1)
            self.position_activations = nn.Dense(self.cfg.model.common.pair_in_dim, self.cfg.model.pair_channel,
                                                 weight_init=lecun_init(self.cfg.model.common.pair_in_dim))
        self.extra_msa_activations = nn.Dense(25, self.cfg.model.extra_msa_channel, weight_init=lecun_init(25))
        self.template_embedding = MultimerTemplateEmbedding(self.cfg.model, mixed_precision)

        self.batch_matmul_trans_b = P.BatchMatMul(transpose_b=True)
        self.template_single_embedding = nn.Dense(34, self.cfg.model.msa_channel,
                                                  weight_init=
                                                  lecun_init(34, initializer_name='relu'))
        self.template_projection = nn.Dense(self.cfg.model.msa_channel, self.cfg.model.msa_channel,
                                            weight_init=lecun_init(self.cfg.model.msa_channel,
                                                                   initializer_name='relu'))
        self.relu = nn.ReLU()
        self.single_activations = nn.Dense(self.cfg.model.msa_channel, self.cfg.model.seq_channel,
                                           weight_init=lecun_init(self.cfg.model.msa_channel))
        extra_msa_stack = nn.CellList()
        for _ in range(self.extra_msa_stack_num):
            extra_msa_block = MultimerEvoformer(self.cfg.model,
                                                msa_act_dim=64,
                                                pair_act_dim=128,
                                                is_extra_msa=True,
                                                batch_size=None)
            extra_msa_stack.append(extra_msa_block)
        self.extra_msa_stack = extra_msa_stack
        self.msa_stack = MultimerEvoformer(self.cfg.model,
                                           msa_act_dim=256,
                                           pair_act_dim=128,
                                           is_extra_msa=False,
                                           batch_size=self.msa_stack_num)
        self.idx_evoformer_block = Parameter(Tensor(0, mstype.int32), requires_grad=False)
        self.evoformer_num_block_eval = Tensor(self.msa_stack_num, mstype.int32)

        self.structure_module = MultimerStructureModule(self.cfg,
                                                        self.cfg.model.seq_channel,
                                                        self.cfg.model.pair_channel)

        self.module_lddt = PredictedLDDTHead(self.cfg.model.heads.predicted_lddt,
                                             self.cfg.model.seq_channel)

    def construct(self, aatype, residue_index, template_aatype, template_all_atom_mask, template_all_atom_positions,
                  asym_id, sym_id, entity_id, seq_mask, msa_mask, target_feat, msa_feat,
                  extra_msa, extra_deletion_matrix, extra_msa_mask,
                  residx_atom37_to_atom14, atom37_atom_exists, prev_pos, prev_msa_first_row, prev_pair):
        """construct"""
        preprocess_1d = self.preprocess_1d(target_feat)
        preprocess_msa = self.preprocess_msa(msa_feat)
        msa_activations = mnp.expand_dims(preprocess_1d, axis=0) + preprocess_msa
        left_single = self.left_single(target_feat)
        right_single = self.right_single(target_feat)
        pair_activations = P.ExpandDims()(left_single, 1) + P.ExpandDims()(right_single, 0)
        mask_2d = P.ExpandDims()(seq_mask, 1) * P.ExpandDims()(seq_mask, 0)
        if self.recycle_pos:
            prev_pseudo_beta = pseudo_beta_fn(aatype, prev_pos, None)
            dgram = dgram_from_positions(prev_pseudo_beta, self.num_bins, self.min_bin, self.max_bin, self._type)
            pair_activations += self.prev_pos_linear(dgram)
        if self.recycle_features:
            prev_msa_first_row = self.prev_msa_first_row_norm(prev_msa_first_row)
            msa_activations = mnp.concatenate(
                (mnp.expand_dims(prev_msa_first_row + msa_activations[0, ...], 0), msa_activations[1:, ...]), 0)
            pair_activations += self.prev_pair_norm(prev_pair)
        if self.max_relative_feature:
            pair_activations += self._relative_encoding(residue_index, asym_id, sym_id, entity_id)

        if self.template_enabled:
            multichain_mask = asym_id[:, None] == asym_id[None, :]
            template_pair_representation = self.template_embedding(pair_activations, template_aatype,
                                                                   template_all_atom_mask, template_all_atom_positions,
                                                                   mask_2d, multichain_mask)
            pair_activations += template_pair_representation
        msa_1hot = self.extra_msa_one_hot(extra_msa)
        has_deletion = mnp.clip(extra_deletion_matrix, 0., 1.)
        deletion_value = (mnp.arctan(extra_deletion_matrix / 3.) * (2. / self.pi))
        extra_msa_feat = mnp.concatenate((msa_1hot, has_deletion[..., None], deletion_value[..., None]), axis=-1)
        extra_msa_activations = self.extra_msa_activations(extra_msa_feat)
        extra_msa_mask_tmp = P.Transpose()(P.ExpandDims()(extra_msa_mask, -1), (2, 1, 0))
        extra_msa_norm = P.Transpose()(self.batch_matmul_trans_b(extra_msa_mask_tmp, extra_msa_mask_tmp), (1, 2, 0))

        for i in range(self.extra_msa_stack_num):
            extra_msa_activations, pair_activations = \
               self.extra_msa_stack[i](extra_msa_activations, pair_activations, extra_msa_mask, extra_msa_norm,
                                       mask_2d)
        if self.template_enabled:
            aatype_one_hot = self.template_aatype_one_hot(template_aatype)
            chi_angles, chi_mask = compute_chi_angles(template_aatype,
                                                      template_all_atom_positions,
                                                      template_all_atom_mask,
                                                      self.chi_atom_indices,
                                                      self.chi_angles_mask,
                                                      self.indices0,
                                                      self.indices1,
                                                      self.batch_block)
            template_features = mnp.concatenate([aatype_one_hot,
                                                 mnp.sin(chi_angles) * chi_mask,
                                                 mnp.cos(chi_angles) * chi_mask,
                                                 chi_mask], axis=-1)
            template_mask = chi_mask[:, :, 0]
            template_activations = self.template_single_embedding(template_features)
            template_activations = self.relu(template_activations)
            template_activations = self.template_projection(template_activations)
            msa_activations = mnp.concatenate([msa_activations, template_activations], axis=0)
            msa_mask = mnp.concatenate([msa_mask, template_mask], axis=0)
        msa_mask_tmp = P.Transpose()(P.ExpandDims()(msa_mask, -1), (2, 1, 0))
        msa_mask_norm = P.Transpose()(self.batch_matmul_trans_b(msa_mask_tmp, msa_mask_tmp), (1, 2, 0))
        self.idx_evoformer_block = self.idx_evoformer_block * 0
        while self.idx_evoformer_block < self.evoformer_num_block_eval:
            msa_activations, pair_activations = self.msa_stack(msa_activations,
                                                               pair_activations,
                                                               msa_mask,
                                                               msa_mask_norm,
                                                               mask_2d,
                                                               self.idx_evoformer_block)
            self.idx_evoformer_block += 1
        single_activations = self.single_activations(msa_activations[0])
        msa_first_row = msa_activations[0]
        final_atom_positions, _, rp_structure_module, _, _, \
        _, _, _, _, _ = \
            self.structure_module(single_activations,
                                  pair_activations,
                                  seq_mask,
                                  aatype,
                                  residx_atom37_to_atom14,
                                  atom37_atom_exists)
        predicted_lddt_logits = self.module_lddt(rp_structure_module)
        final_atom_positions = P.Cast()(final_atom_positions, self._type)
        prev_pos = final_atom_positions
        prev_msa_first_row = msa_first_row
        prev_pair = pair_activations
        return prev_pos, prev_msa_first_row, prev_pair, predicted_lddt_logits

    def _relative_encoding(self, residue_index, asym_id, sym_id, entity_id):
        """Add relative position encoding"""
        rel_feats = []
        asym_id_same = mnp.equal(asym_id[:, None], asym_id[None, :])
        offset = residue_index[:, None] - residue_index[None, :]
        clipped_offset = mnp.clip(
            offset + self.max_relative_feature, xmin=0, xmax=2 * self.max_relative_feature)

        if self.use_chain_relative:
            final_offset = mnp.where(asym_id_same, clipped_offset,
                                     (2 * self.max_relative_feature + 1) *
                                     mnp.ones_like(clipped_offset))
            rel_pos = self.rel_pos_one_hot(final_offset)
            rel_feats.append(rel_pos)
            entity_id_same = mnp.equal(entity_id[:, None], entity_id[None, :])
            rel_feats.append(entity_id_same.astype(rel_pos.dtype)[..., None])
            rel_sym_id = sym_id[:, None] - sym_id[None, :]
            max_rel_chain = self.max_relative_chain
            clipped_rel_chain = mnp.clip(
                rel_sym_id + max_rel_chain, xmin=0, xmax=2 * max_rel_chain)
            final_rel_chain = mnp.where(entity_id_same, clipped_rel_chain,
                                        (2 * max_rel_chain + 1) *
                                        mnp.ones_like(clipped_rel_chain))
            rel_chain = self.rel_chain_one_hot(final_rel_chain.astype(mstype.int32))
            rel_feats.append(rel_chain)
        else:
            rel_pos = self.one_hot(clipped_offset)
            rel_feats.append(rel_pos)
        rel_feat = mnp.concatenate(rel_feats, axis=-1)
        return self.position_activations(rel_feat)
