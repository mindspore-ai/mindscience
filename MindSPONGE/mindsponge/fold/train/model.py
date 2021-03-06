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
"""model"""
import numpy as np
import mindspore.common.dtype as mstype
import mindspore.nn as nn
import mindspore.numpy as mnp
from mindspore.common.tensor import Tensor
from mindspore.ops import functional as F
from mindspore.ops import operations as P

from common import residue_constants
from common.utils import get_chi_atom_indices, lecun_init
from common.utils import pseudo_beta_fn, dgram_from_positions, atom37_to_torsion_angles

from module.evoformer_module import TemplateEmbedding, EvoformerIteration, EvoformerIterationMSA
from module.loss_module import LossNet
from module.structure_module import StructureModule, DistogramHead, ExperimentallyResolvedHead, \
    MaskedMsaHead, PredictedLDDTHead, PredictedAlignedErrorHead


class AlphaFold(nn.Cell):
    """alphafold model"""
    def __init__(self, config, global_config, extra_num, evo_num):
        super(AlphaFold, self).__init__()
        self.config = config.model.embeddings_and_evoformer
        self.pair_dim = config.model.embeddings_and_evoformer.pair_channel
        self.seq_channel = config.model.embeddings_and_evoformer.seq_channel
        self.msa_channel = config.model.embeddings_and_evoformer.msa_channel
        self.pair_channel = config.model.embeddings_and_evoformer.pair_channel
        self.preprocess_1d = nn.Dense(22, self.config.msa_channel, weight_init=lecun_init(22)).to_float(mstype.float16)
        self.preprocess_msa = nn.Dense(49, self.config.msa_channel, weight_init=lecun_init(49)).to_float(mstype.float16)
        self.left_single = nn.Dense(22, self.config.pair_channel, weight_init=lecun_init(22)).to_float(mstype.float16)
        self.right_single = nn.Dense(22, self.config.pair_channel, weight_init=lecun_init(22)).to_float(mstype.float16)
        self.prev_pos_linear = nn.Dense(15, self.config.pair_channel, weight_init=lecun_init(15)).to_float(
            mstype.float16)
        self.pair_activations = nn.Dense(65, self.config.pair_channel, weight_init=lecun_init(65)).to_float(
            mstype.float16)
        self.prev_msa_first_row_norm = nn.LayerNorm([256,], epsilon=1e-5)
        self.prev_pair_norm = nn.LayerNorm([128,], epsilon=1e-5)
        self.one_hot = nn.OneHot(depth=self.config.max_relative_feature * 2 + 1, axis=-1)
        self.extra_msa_activations = nn.Dense(25, self.config.extra_msa_channel, weight_init=lecun_init(25)).to_float(
            mstype.float16)
        self.template_single_embedding = nn.Dense(57, self.config.msa_channel,
                                                  weight_init=
                                                  lecun_init(57, initializer_name='relu')).to_float(mstype.float16)
        self.template_projection = nn.Dense(self.config.msa_channel, self.config.msa_channel,
                                            weight_init=lecun_init(self.config.msa_channel,
                                                                   initializer_name='relu')).to_float(mstype.float16)
        self.single_activations = nn.Dense(self.config.msa_channel, self.config.seq_channel,
                                           weight_init=lecun_init(self.config.msa_channel)).to_float(mstype.float16)
        self.relu = nn.ReLU()
        self.recycle_pos = self.config.recycle_pos
        self.recycle_features = self.config.recycle_features
        self.template_enable = self.config.template.enabled
        self.max_relative_feature = self.config.max_relative_feature
        self.template_enabled = self.config.template.enabled
        self.template_embed_torsion_angles = self.config.template.embed_torsion_angles
        self.num_bins = self.config.prev_pos.num_bins
        self.min_bin = self.config.prev_pos.min_bin
        self.max_bin = self.config.prev_pos.max_bin
        self.extra_msa_one_hot = nn.OneHot(depth=23, axis=-1)
        self.template_aatype_one_hot = nn.OneHot(depth=22, axis=-1)
        self.template_embedding = TemplateEmbedding(self.config.template, global_config.template_embedding.slice_num,
                                                    global_config=global_config)
        self.template_embedding.recompute()
        self.extra_msa_stack_num_block = extra_num
        msa_stack_layers = nn.CellList()
        for _ in range(self.extra_msa_stack_num_block):
            msa_stack_block = EvoformerIterationMSA(self.config.evoformer,
                                                    msa_act_dim=64,
                                                    pair_act_dim=128,
                                                    is_extra_msa=True,
                                                    global_config=global_config)

            msa_stack_layers.append(msa_stack_block)
        self.extra_msa_stack_iteration = msa_stack_layers

        self.evoformer_num_block = evo_num
        evoformer_layers = nn.CellList()
        for _ in range(self.evoformer_num_block):
            evoformer_block = EvoformerIteration(self.config.evoformer,
                                                 msa_act_dim=256,
                                                 pair_act_dim=128,
                                                 is_extra_msa=False,
                                                 global_config=global_config)
            evoformer_block.recompute()
            evoformer_layers.append(evoformer_block)
        self.evoformer_iteration = evoformer_layers

        self.loss_net = LossNet(config)
        self.structure_module = StructureModule(config.model.heads.structure_module,
                                                self.config.seq_channel,
                                                self.config.pair_channel,
                                                global_config=global_config)

        self.module_lddt = PredictedLDDTHead(config.model.heads.predicted_lddt,
                                             global_config,
                                             self.config.seq_channel)
        self.module_distogram = DistogramHead(config.model.heads.distogram,
                                              global_config,
                                              self.pair_dim)
        self.module_exp_resolved = ExperimentallyResolvedHead(config.model.heads.experimentally_resolved,
                                                              global_config,
                                                              self.seq_channel)
        self.module_mask = MaskedMsaHead(config.model.heads.masked_msa,
                                         global_config,
                                         self.msa_channel)
        self.module_lddt = PredictedLDDTHead(config.model.heads.predicted_lddt,
                                             global_config,
                                             self.seq_channel)
        self.aligned_error = PredictedAlignedErrorHead(config.model.heads.predicted_aligned_error,
                                                       self.pair_dim,
                                                       global_config)
        self._init_tensor(global_config)
        self.matmul_trans_b = P.MatMul(transpose_b=True)
        self.batch_matmul_trans_b = P.BatchMatMul(transpose_b=True)

    def _init_tensor(self, global_config):
        """parameter and tensor init"""
        self.chi_atom_indices = Tensor(get_chi_atom_indices(), mstype.int32)
        chi_angles_mask = list(residue_constants.chi_angles_mask)
        chi_angles_mask.append([0.0, 0.0, 0.0, 0.0])
        self.chi_angles_mask = Tensor(chi_angles_mask, mstype.float32)
        self.mirror_psi_mask = Tensor(np.asarray([1., 1., -1., 1., 1., 1., 1.])[None, None, :, None], mstype.float32)
        self.chi_pi_periodic = Tensor(residue_constants.chi_pi_periodic, mstype.float32)

        indices0 = np.arange(4).reshape((-1, 1, 1, 1, 1)).astype("int64")  # 4 batch
        indices0 = indices0.repeat(global_config.seq_length, axis=1)  # seq_length sequence length
        indices0 = indices0.repeat(4, axis=2)  # 4 chis
        self.indices0 = Tensor(indices0.repeat(4, axis=3))  # 4 atoms

        indices1 = np.arange(global_config.seq_length).reshape((1, -1, 1, 1, 1)).astype("int64")
        indices1 = indices1.repeat(4, axis=0)
        indices1 = indices1.repeat(4, axis=2)
        self.indices1 = Tensor(indices1.repeat(4, axis=3))
        self.zeros = Tensor(0, mstype.int32)

    def construct(self, target_feat, msa_feat, msa_mask, seq_mask, aatype,
                  template_aatype, template_all_atom_masks, template_all_atom_positions,
                  template_mask, template_pseudo_beta_mask, template_pseudo_beta, extra_msa, extra_has_deletion,
                  extra_deletion_value, extra_msa_mask,
                  residx_atom37_to_atom14, atom37_atom_exists, residue_index,
                  prev_pos, prev_msa_first_row, prev_pair, pseudo_beta_gt, pseudo_beta_mask_gt,
                  all_atom_mask_gt, true_msa, bert_mask,
                  residx_atom14_to_atom37, restype_atom14_bond_lower_bound, restype_atom14_bond_upper_bound,
                  atomtype_radius, backbone_affine_tensor, backbone_affine_mask,
                  atom14_gt_positions, atom14_alt_gt_positions, atom14_atom_is_ambiguous, atom14_gt_exists,
                  atom14_atom_exists, atom14_alt_gt_exists, all_atom_positions, rigidgroups_gt_frames,
                  rigidgroups_gt_exists, rigidgroups_alt_gt_frames, torsion_angles_sin_cos_gt, use_clamped_fape,
                  filter_by_solution, chi_mask):
        """construct"""

        preprocess_1d = self.preprocess_1d(target_feat)
        preprocess_msa = self.preprocess_msa(msa_feat)
        msa_activations1 = mnp.expand_dims(preprocess_1d, axis=0) + preprocess_msa

        left_single = self.left_single(target_feat)
        right_single = self.right_single(target_feat)

        pair_activations = P.ExpandDims()(left_single, 1) + P.ExpandDims()(right_single, 0)
        seq_mask = P.Cast()(seq_mask, mstype.float32)
        mask_2d = P.ExpandDims()(seq_mask, 1) * P.ExpandDims()(seq_mask, 0)

        if self.recycle_pos:
            prev_pseudo_beta = pseudo_beta_fn(aatype, prev_pos, None)
            dgram = dgram_from_positions(prev_pseudo_beta, self.num_bins, self.min_bin, self.max_bin)
            pair_activations += self.prev_pos_linear(dgram)

        prev_msa_first_row = F.depend(prev_msa_first_row, pair_activations)
        if self.recycle_features:
            prev_msa_first_row = self.prev_msa_first_row_norm(prev_msa_first_row.astype(mstype.float32)).astype(
                mstype.float16)
            msa_activations1 = mnp.concatenate(
                (mnp.expand_dims(prev_msa_first_row + msa_activations1[0, ...], 0), msa_activations1[1:, ...]), 0)
            pair_activations += self.prev_pair_norm(prev_pair.astype(mstype.float32))

        if self.max_relative_feature:
            offset = P.ExpandDims()(residue_index, 1) - P.ExpandDims()(residue_index, 0)
            rel_pos = self.one_hot(mnp.clip(offset + self.max_relative_feature, 0, 2 * self.max_relative_feature))
            pair_activations += self.pair_activations(rel_pos)

        template_pair_representation = 0
        if self.template_enable:
            template_pair_representation = self.template_embedding(pair_activations, template_aatype,
                                                                   template_all_atom_masks, template_all_atom_positions,
                                                                   template_mask, template_pseudo_beta_mask,
                                                                   template_pseudo_beta, mask_2d)
            pair_activations += template_pair_representation

        msa_1hot = self.extra_msa_one_hot(extra_msa)
        extra_msa_feat = mnp.concatenate((msa_1hot, extra_has_deletion[..., None], extra_deletion_value[..., None]),
                                         axis=-1)
        extra_msa_activations = self.extra_msa_activations(extra_msa_feat)
        msa_act = extra_msa_activations
        pair_act = pair_activations

        msa_act = msa_act.astype(mstype.float32)
        pair_act = pair_act.astype(mstype.float32)
        extra_msa_mask = extra_msa_mask.astype(mstype.float32)
        extra_msa_mask_tmp = P.Transpose()(P.ExpandDims()(extra_msa_mask, -1), (2, 1, 0)).astype(mstype.float16)
        extra_msa_norm = P.Transpose()(self.batch_matmul_trans_b(extra_msa_mask_tmp, extra_msa_mask_tmp), (1, 2, 0))

        for i in range(self.extra_msa_stack_num_block):
            msa_act, pair_act = \
                self.extra_msa_stack_iteration[i](msa_act, pair_act, extra_msa_mask, extra_msa_norm, mask_2d)

        num_templ, num_res = template_aatype.shape
        aatype_one_hot = self.template_aatype_one_hot(template_aatype)
        torsion_angles_sin_cos, alt_torsion_angles_sin_cos, torsion_angles_mask = atom37_to_torsion_angles(
            template_aatype, template_all_atom_positions, template_all_atom_masks, self.chi_atom_indices,
            self.chi_angles_mask, self.mirror_psi_mask, self.chi_pi_periodic, self.indices0, self.indices1)
        template_features = mnp.concatenate([aatype_one_hot,
                                             mnp.reshape(torsion_angles_sin_cos, [num_templ, num_res, 14]),
                                             mnp.reshape(alt_torsion_angles_sin_cos, [num_templ, num_res, 14]),
                                             torsion_angles_mask], axis=-1)
        template_activations = self.template_single_embedding(template_features)
        template_activations = self.relu(template_activations.astype(mstype.float32))
        template_activations = self.template_projection(template_activations)
        msa_activations2 = mnp.concatenate([msa_activations1, template_activations], axis=0)
        torsion_angle_mask = torsion_angles_mask[:, :, 2]
        torsion_angle_mask = torsion_angle_mask.astype(msa_mask.dtype)
        msa_mask = mnp.concatenate([msa_mask, torsion_angle_mask], axis=0)

        msa_activations2 = msa_activations2.astype(mstype.float16)
        pair_activations = pair_act.astype(mstype.float16)
        msa_mask = msa_mask.astype(mstype.float16)
        mask_2d = mask_2d.astype(mstype.float16)
        # return msa_activations2, pair_activations, msa_mask, mask_2d

        msa_mask_tmp = P.Transpose()(P.ExpandDims()(msa_mask, -1), (2, 1, 0)).astype(mstype.float16)
        msa_mask_norm = P.Transpose()(self.batch_matmul_trans_b(msa_mask_tmp, msa_mask_tmp), (1, 2, 0))

        for i in range(self.evoformer_num_block):
            msa_activations2, pair_activations = self.evoformer_iteration[i](msa_activations2, pair_activations,
                                                                             msa_mask, msa_mask_norm, mask_2d)

        single_activations = self.single_activations(msa_activations2[0])
        num_sequences = msa_feat.shape[0]
        msa = msa_activations2[:num_sequences, :, :]
        msa_first_row = msa_activations2[0]

        final_atom_positions, _, rp_structure_module, atom14_pred_positions, final_affines, \
        angles_sin_cos_new, um_angles_sin_cos_new, sidechain_frames, sidechain_atom_pos, structure_traj = \
            self.structure_module(single_activations,
                                  pair_activations,
                                  seq_mask,
                                  aatype,
                                  residx_atom37_to_atom14,
                                  atom37_atom_exists)
        if self.train_backward:
            predicted_lddt_logits = self.module_lddt(rp_structure_module)
            dist_logits, bin_edges = self.module_distogram(pair_activations)
            experimentally_logits = self.module_exp_resolved(single_activations)
            masked_logits = self.module_mask(msa)
            aligned_error_logits, aligned_error_breaks = self.aligned_error(pair_activations.astype(mstype.float32))

            final_atom_positions = final_atom_positions.astype(mstype.float32)
            dist_logits = dist_logits.astype(mstype.float32)
            bin_edges = bin_edges.astype(mstype.float32)
            experimentally_logits = experimentally_logits.astype(mstype.float32)
            masked_logits = masked_logits.astype(mstype.float32)
            aligned_error_logits = aligned_error_logits.astype(mstype.float32)
            aligned_error_breaks = aligned_error_breaks.astype(mstype.float32)
            atom14_pred_positions = atom14_pred_positions.astype(mstype.float32)
            final_affines = final_affines.astype(mstype.float32)
            angles_sin_cos_new = angles_sin_cos_new.astype(mstype.float32)
            predicted_lddt_logits = predicted_lddt_logits.astype(mstype.float32)
            structure_traj = structure_traj.astype(mstype.float32)
            sidechain_frames = sidechain_frames.astype(mstype.float32)
            sidechain_atom_pos = sidechain_atom_pos.astype(mstype.float32)

            loss_all = self.loss_net(dist_logits, bin_edges, pseudo_beta_gt, pseudo_beta_mask_gt,
                                     experimentally_logits, atom37_atom_exists, all_atom_mask_gt, true_msa,
                                     masked_logits, bert_mask, atom14_pred_positions, residue_index, aatype,
                                     residx_atom14_to_atom37, restype_atom14_bond_lower_bound,
                                     restype_atom14_bond_upper_bound, seq_mask, atomtype_radius, final_affines,
                                     aligned_error_breaks, aligned_error_logits, angles_sin_cos_new,
                                     um_angles_sin_cos_new, backbone_affine_tensor, backbone_affine_mask,
                                     atom14_gt_positions, atom14_alt_gt_positions, atom14_atom_is_ambiguous,
                                     atom14_gt_exists, atom14_atom_exists, atom14_alt_gt_exists,
                                     final_atom_positions, all_atom_positions, predicted_lddt_logits,
                                     structure_traj, rigidgroups_gt_frames, rigidgroups_gt_exists,
                                     rigidgroups_alt_gt_frames,
                                     sidechain_frames, sidechain_atom_pos, torsion_angles_sin_cos_gt,
                                     chi_mask, use_clamped_fape, filter_by_solution)
            return loss_all

        prev_pos = final_atom_positions.astype(mstype.float32)
        prev_msa_first_row = msa_first_row.astype(mstype.float32)
        prev_pair = pair_activations.astype(mstype.float32)
        return prev_pos, prev_msa_first_row, prev_pair
