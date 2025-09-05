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
from mindspore.ops import operations as P
from mindspore.common.tensor import Tensor
from mindspore import Parameter
from mindspore import ops
import mindsponge.common.residue_constants as residue_constants
from mindsponge1.common.utils import dgram_from_positions, pseudo_beta_fn, atom37_to_torsion_angles
from mindsponge1.data.data_transform import get_chi_atom_pos_indices
from mindsponge1.cell.initializer import lecun_init
from module.template_embedding import  MultimerTemplateEmbedding #TemplateEmbedding
from module.evoformer import MultimerEvoformer #Evoformer
# from module.structure import StructureModule
from module.structure_multimer import MultimerStructureModule
from module.head import DistogramHead, ExperimentallyResolvedHead, MaskedMsaHead, \
    PredictedLDDTHead, PredictedAlignedErrorHead

from common.utils import compute_chi_angles, ComputeChiAngles
from scipy.special import softmax
from restraint_sample import BINS
# from mindsponge1.cell.dense import ProcessSBR
from mindspore.communication import get_rank

from typing import Dict, Optional, Tuple
import numpy as np
import scipy.special

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


def _calculate_bin_centers(breaks: np.ndarray):
  """Gets the bin centers from the bin edges.

  Args:
    breaks: [num_bins - 1] the error bin edges.

  Returns:
    bin_centers: [num_bins] the error bin centers.
  """
  step = (breaks[1] - breaks[0])

  # Add half-step to get the center 
  bin_centers = breaks + step / 2
  # Add a catch-all bin at the end.
  bin_centers = np.concatenate([bin_centers, [bin_centers[-1] + step]],
                               axis=0)
  return bin_centers


def _calculate_expected_aligned_error(
    alignment_confidence_breaks: np.ndarray,
    aligned_distance_error_probs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Calculates expected aligned distance errors for every pair of residues.

    Args:
      alignment_confidence_breaks: [num_bins - 1] the error bin edges.
      aligned_distance_error_probs: [num_res, num_res, num_bins] the predicted
        probs for each error bin, for each pair of residues.

    Returns:
      predicted_aligned_error: [num_res, num_res] the expected aligned distance
        error for each pair of residues.
      max_predicted_aligned_error: The maximum predicted error possible.
    """
    bin_centers = _calculate_bin_centers(alignment_confidence_breaks)

    # Tuple of expected aligned distance error and max possible error.
    return (np.sum(aligned_distance_error_probs * bin_centers, axis=-1),
          np.asarray(bin_centers[-1]))


def compute_predicted_aligned_error(
    logits: np.ndarray,
    breaks: np.ndarray) -> Dict[str, np.ndarray]:
    """Computes aligned confidence metrics from logits.

    Args:
      logits: [num_res, num_res, num_bins] the logits output from
        PredictedAlignedErrorHead.
      breaks: [num_bins - 1] the error bin edges.

    Returns:
      aligned_confidence_probs: [num_res, num_res, num_bins] the predicted
        aligned error probabilities over bins for each residue pair.
      predicted_aligned_error: [num_res, num_res] the expected aligned distance
        error for each pair of residues.
      max_predicted_aligned_error: The maximum predicted error possible.
    """
    aligned_confidence_probs = scipy.special.softmax(
        logits,
        axis=-1)
    predicted_aligned_error, max_predicted_aligned_error = (
        _calculate_expected_aligned_error(
            alignment_confidence_breaks=breaks,
            aligned_distance_error_probs=aligned_confidence_probs))
    return {
        'aligned_confidence_probs': aligned_confidence_probs,
        'predicted_aligned_error': predicted_aligned_error,
        'max_predicted_aligned_error': max_predicted_aligned_error,
    }


def predicted_tm_score(
    logits: np.ndarray,
    breaks: np.ndarray,
    residue_weights: Optional[np.ndarray] = None,
    asym_id: Optional[np.ndarray] = None,
    interface: bool = False) -> np.ndarray:
    """Computes predicted TM alignment or predicted interface TM alignment score.

    Args:
      logits: [num_res, num_res, num_bins] the logits output from
        PredictedAlignedErrorHead.
      breaks: [num_bins] the error bins.
      residue_weights: [num_res] the per residue weights to use for the
        expectation.
      asym_id: [num_res] the asymmetric unit ID - the chain ID. Only needed for
        ipTM calculation, i.e. when interface=True.
      interface: If True, interface predicted TM score is computed.

    Returns:
      ptm_score: The predicted TM alignment or the predicted iTM score.
    """

    # residue_weights has to be in [0, 1], but can be floating-point, i.e. the
    # exp. resolved head's probability.
    if residue_weights is None:
        residue_weights = np.ones(logits.shape[0])

    bin_centers = _calculate_bin_centers(breaks)

    num_res = int(np.sum(residue_weights))
    # Clip num_res to avoid negative/undefined d0.
    clipped_num_res = max(num_res, 19)

    # Compute d_0(num_res) as defined by TM-score, eqn. (5) in Yang & Skolnick
    # "Scoring function for automated assessment of protein structure template
    # quality", 2004: http://zhanglab.ccmb.med.umich.edu/papers/2004_3.pdf 
    
    d0 = 1.24 * (clipped_num_res - 15) ** (1./3) - 1.8

    # Convert logits to probs.
    probs = scipy.special.softmax(logits, axis=-1)

    # TM-Score term for every bin.
    tm_per_bin = 1. / (1 + np.square(bin_centers) / np.square(d0))
    # E_distances tm(distance).
    predicted_tm_term = np.sum(probs * tm_per_bin, axis=-1)

    pair_mask = np.ones(shape=(num_res, num_res), dtype=bool)
    if interface:
        pair_mask *= asym_id[:, None] != asym_id[None, :]

    predicted_tm_term *= pair_mask

    pair_residue_weights = pair_mask * (
        residue_weights[None, :] * residue_weights[:, None])
    normed_residue_mask = pair_residue_weights / (1e-8 + np.sum(
        pair_residue_weights, axis=-1, keepdims=True))
    per_alignment = np.sum(predicted_tm_term * normed_residue_mask, axis=-1)
    return np.asarray(per_alignment[(per_alignment * residue_weights).argmax()])


def compute_ranking_score(logits, breaks, asym_id):
    # print(logits.shape, breaks.shape, asym_id.shape)
    iptm = predicted_tm_score(logits, breaks, asym_id=asym_id, interface=True)
    ptm = predicted_tm_score(logits, breaks)
    return 0.8*iptm + 0.2*ptm


class MegaFold(nn.Cell):
    """MegaFold"""

    def __init__(self, config, mixed_precision, device_num):
        super(MegaFold, self).__init__()
        self.dump = ops.TensorDump()
        self.cfg = config

        if mixed_precision:
            self._type = mstype.float16
        else:
            self._type = mstype.float32

        self.is_training = self.cfg.is_training
        self.recycle_pos = self.cfg.recycle_pos
        self.recycle_features = self.cfg.recycle_features
        self.max_relative_feature = self.cfg.max_relative_feature
        self.use_chain_relative = self.cfg.multimer.embeddings_and_evoformer.use_chain_relative
        self.max_relative_chain = self.cfg.multimer.embeddings_and_evoformer.max_relative_chain

        self.num_bins = self.cfg.prev_pos.num_bins
        self.min_bin = self.cfg.prev_pos.min_bin
        self.max_bin = self.cfg.prev_pos.max_bin
        self.template_enabled = self.cfg.template.enabled
        self.extra_msa_stack_num = self.cfg.evoformer.extra_msa_stack_num
        self.msa_stack_num = self.cfg.evoformer.msa_stack_num
        self.chi_atom_indices, self.chi_angles_mask, self.mirror_psi_mask, self.chi_pi_periodic, \
        self.indices0, self.indices1 = caculate_constant_array(self.cfg.seq_length)

        # self.contact_one_hot = nn.OneHot(depth=2, axis=-1)
        self.sbr_act_dim = 128
        self.sbr_act1 = nn.Dense(len(BINS)+1, self.sbr_act_dim, weight_init=lecun_init(len(BINS)+1), activation='relu')
        self.sbr_act2 = nn.Dense(self.sbr_act_dim, self.sbr_act_dim, weight_init=lecun_init(self.sbr_act_dim))
        # self.sbr_gate = nn.Dense(self.sbr_act_dim+self.cfg.pair_channel, self.sbr_act_dim, weight_init='zeros', bias_init='ones')
        self.sigmoid = nn.Sigmoid()
        # self.preprocess_contact = nn.Dense(1, 128, lecun_init(15)).to_float(mstype.float16)
        # self.process_sbr = ProcessSBR(len(BINS)+1, 32, gate=True, pair_input_dim=self.cfg.pair_channel)
        # self.process_sbr = ProcessSBR(len(BINS)+1, 32)

        # print("debug self.cfg.common.target_feat_dim", self.cfg.common.target_feat_dim)
        self.preprocess_1d = nn.Dense(self.cfg.common.target_feat_dim, self.cfg.msa_channel,
                                      weight_init=lecun_init(self.cfg.common.target_feat_dim))
        self.preprocess_msa = nn.Dense(self.cfg.common.msa_feat_dim, self.cfg.msa_channel,
                                       weight_init=lecun_init(self.cfg.common.msa_feat_dim))
        # self.preprocess_msa
        self.left_single = nn.Dense(self.cfg.common.target_feat_dim, self.cfg.pair_channel,
                                    weight_init=lecun_init(self.cfg.common.target_feat_dim))
        self.right_single = nn.Dense(self.cfg.common.target_feat_dim, self.cfg.pair_channel,
                                     weight_init=lecun_init(self.cfg.common.target_feat_dim))
        self.prev_pos_linear = nn.Dense(self.cfg.common.dgram_dim, self.cfg.pair_channel,
                                        weight_init=lecun_init(self.cfg.common.dgram_dim))

        self.extra_msa_one_hot = nn.OneHot(depth=23, axis=-1)
        # self.extra_msa_one_hot.onehot.shard(((1,2,1),))
        # self.extra_msa_one_hot = P.OneHot(-1).shard(((1,2,1),(),(),()))
        self.template_aatype_one_hot = nn.OneHot(depth=22, axis=-1)
        self.prev_msa_first_row_norm = nn.LayerNorm([256,], epsilon=1e-5)
        self.prev_pair_norm = nn.LayerNorm([128,], epsilon=1e-5)
        # self.prev_pair_norm.layer_norm.shard(((1, device_num, 1), (1,), (1,)))
        if self.use_chain_relative:
            self.rel_pos_one_hot = nn.OneHot(depth=self.cfg.max_relative_feature * 2 + 2, axis=-1) # 32 * 2 + 2 = 66
            self.rel_chain_one_hot = nn.OneHot(depth=self.max_relative_chain * 2 + 2, axis=-1) # 2 * 2 + 2 = 6
            self.position_activations = nn.Dense(self.cfg.multimer.pair_in_dim, self.cfg.pair_channel, #73
                                                 weight_init=lecun_init(self.cfg.multimer.pair_in_dim))
            self.interface_activations = nn.Dense(2, self.cfg.pair_channel, #2
                                                 weight_init='zeros',
                                                 has_bias=False)
        else:
            self.one_hot = nn.OneHot(depth=self.cfg.max_relative_feature * 2 + 1, axis=-1) # 65
            self.position_activations = nn.Dense(self.cfg.common.pair_in_dim, self.cfg.pair_channel,
                                                 weight_init=lecun_init(self.cfg.common.pair_in_dim))
        self.extra_msa_activations = nn.Dense(25, self.cfg.extra_msa_channel, weight_init=lecun_init(25))
        self.template_embedding = MultimerTemplateEmbedding(self.cfg, device_num, mixed_precision)

        self.matmul_trans_b = P.MatMul(transpose_b=True)
        self.batch_matmul_trans_b = P.BatchMatMul(transpose_b=True)
        self.template_single_embedding = nn.Dense(34, self.cfg.msa_channel,
                                                  weight_init=
                                                  lecun_init(34, initializer_name='relu'))
        self.template_projection = nn.Dense(self.cfg.msa_channel, self.cfg.msa_channel,
                                            weight_init=lecun_init(self.cfg.msa_channel,
                                                                   initializer_name='relu'))
        self.relu = nn.ReLU()
        self.single_activations = nn.Dense(self.cfg.msa_channel, self.cfg.seq_channel,
                                           weight_init=lecun_init(self.cfg.msa_channel))
        extra_msa_stack = nn.CellList()
        for _ in range(self.extra_msa_stack_num):
            extra_msa_block = MultimerEvoformer(self.cfg,
                                        msa_act_dim=64,
                                        pair_act_dim=128,
                                        is_extra_msa=True,
                                        batch_size=None,
                                        device_num=device_num)
            extra_msa_stack.append(extra_msa_block)
        self.extra_msa_stack = extra_msa_stack
        self.aligned_error = PredictedAlignedErrorHead(self.cfg.heads.predicted_aligned_error,
                                                        self.cfg.pair_channel)
        if self.is_training:
            msa_stack = nn.CellList()
            for _ in range(self.msa_stack_num):
                msa_block = MultimerEvoformer(self.cfg,
                                      msa_act_dim=256,
                                      pair_act_dim=128,
                                      is_extra_msa=False,
                                      batch_size=None,
                                      device_num=device_num)
                msa_stack.append(msa_block)
            self.msa_stack = msa_stack
            self.module_distogram = DistogramHead(self.cfg.heads.distogram,
                                                  self.cfg.pair_channel)
            self.module_exp_resolved = ExperimentallyResolvedHead(self.cfg.seq_channel)
            self.module_mask = MaskedMsaHead(self.cfg.heads.masked_msa,
                                             self.cfg.msa_channel)
        else:
            # print("debug", self.cfg, self.msa_stack_num)
            self.msa_stack = MultimerEvoformer(self.cfg,
                                       msa_act_dim=256,
                                       pair_act_dim=128,
                                       is_extra_msa=False,
                                       batch_size=self.msa_stack_num,
                                       device_num=device_num)
        self.idx_evoformer_block = Parameter(Tensor(0, mstype.int32), requires_grad=False)
        self.evoformer_num_block_eval = Tensor(self.msa_stack_num, mstype.int32)

        self.structure_module = MultimerStructureModule(self.cfg,
                                                        self.cfg.seq_channel,
                                                        self.cfg.pair_channel,
                                                        device_num)
        # raw notion
        # self.structure_module = StructureModule(self.cfg,
        #                                         self.cfg.seq_channel,
        #                                         self.cfg.pair_channel)

        self.module_lddt = PredictedLDDTHead(self.cfg.heads.predicted_lddt,
                                             self.cfg.seq_channel)
        self.add_2 = P.Add().shard(((1, device_num, 1), (1, device_num, 1)))
        self.concat_0_2 = P.Concat(0).shard(((1, device_num, 1), (1, device_num, 1)))
        self.concat_e_3 = P.Concat(-1).shard(((1, device_num, 1), (1, device_num, 1), (1, device_num, 1)))
        self.cast3 = P.Cast().shard(((1, device_num, 1),))
        self.cast2 = P.Cast().shard(((1, device_num),))
        self.expand2 = P.ExpandDims().shard(((1, device_num),))
        self.concat_e_4 = P.Concat(-1).shard(((1, device_num, 1),(1, device_num, 1), (1, device_num, 1), (1, device_num, 1)))
        self.concat_0_2_2 = P.Concat(0).shard(((1, device_num), (1, device_num)))
        self.compute_chi_angles = ComputeChiAngles(device_num)
        self.squeeze2 = P.Squeeze().shard(((1, device_num, 1),))
        self.slice_ops2 = P.Slice().shard(((1, device_num, 1),))
        self.allgather3 = P.StridedSlice().shard(((1, 1, 1),))
        self.allgather2 = P.StridedSlice().shard(((1, 1),))


    def _relative_encoding(self, residue_index, asym_id, sym_id, entity_id, interface_mask):
        """Add relative position encoding"""
        rel_feats = []
        asym_id_same = mnp.equal(P.ExpandDims()(asym_id, 1), P.ExpandDims()(asym_id, 0)).astype(mstype.int32) # seq_len * seq_len
        offset = P.ExpandDims()(residue_index, 1) - P.ExpandDims()(residue_index, 0) # seq_len * seq_len
        clipped_offset = mnp.clip(
            offset + self.max_relative_feature, xmin=0, xmax= 2 * self.max_relative_feature)
        interface_feat = None
        if self.use_chain_relative:
            final_offset = mnp.where(asym_id_same, clipped_offset,
                                     (2 * self.max_relative_feature + 1) *
                                     mnp.ones_like(clipped_offset))
            rel_pos = self.rel_pos_one_hot(final_offset) # seq_len * seq_len * 66
            rel_feats.append(rel_pos)
            # entity_id_same = mnp.equal(entity_id[:, None], entity_id[None, :])  # seq_len * seq_len * 1
            entity_id_same = mnp.equal(P.ExpandDims()(entity_id, 1), P.ExpandDims()(entity_id, 0)).astype(mstype.int32)  # seq_len * seq_len * 1
            rel_feats.append(entity_id_same.astype(rel_pos.dtype)[..., None])
            rel_sym_id = P.ExpandDims()(sym_id, 1) - P.ExpandDims()(sym_id, 0)
            max_rel_chain = self.max_relative_chain
            clipped_rel_chain = mnp.clip(
                rel_sym_id + max_rel_chain, xmin=0, xmax=2 * max_rel_chain)
            # entity_id_same = entity_id_same.astype(mstype.int32)
            final_rel_chain = mnp.where(entity_id_same, clipped_rel_chain,
                                        (2 * max_rel_chain + 1) *
                                        mnp.ones_like(clipped_rel_chain))
            rel_chain = self.rel_chain_one_hot(final_rel_chain.astype(mstype.int32))  # seq_len * seq_len * 6
            rel_feats.append(rel_chain)
            interface_feat = mnp.concatenate([mnp.tile(interface_mask[:, None, None], (1, len(interface_mask), 1)), mnp.tile(interface_mask[None, :, None], (len(interface_mask), 1, 1))], axis=-1)
        else:
            rel_pos = self.one_hot(clipped_offset)
            rel_feats.append(rel_pos)
        rel_feat = mnp.concatenate(rel_feats, axis=-1)  # seq_len * seq_len * 73 for multimer
        return self.position_activations(rel_feat)+self.interface_activations(interface_feat)#, rel_feat, interface_feat

    def construct(self, aatype, residue_index, template_aatype, template_all_atom_masks, template_all_atom_positions,
                  asym_id, sym_id, entity_id, seq_mask, msa_mask, target_feat, msa_feat,
                  extra_msa, extra_msa_deletion_value, extra_msa_mask,
                  residx_atom37_to_atom14, atom37_atom_exists, 
                  sbr, sbr_mask, interface_mask, prev_pos, prev_msa_first_row, prev_pair):
    # def construct(self, extra_msa_activations, pair_activations, extra_msa_mask, extra_msa_norm, mask_2d, rank):
    # def construct(self, msa_feat):
        """construct"""
        # # print("debug target_feat", target_feat, type(target_feat[0][0]), target_feat[0][0].dtype)
        preprocess_1d = self.preprocess_1d(target_feat) # raw target_feat 256 21 1, 128 256 (1,2,1)
        # self.dump(f"msa_feat_23_rank{rank}", msa_feat)
        preprocess_msa = self.preprocess_msa(msa_feat) # raw (508 128 49) (49 256)
        # self.dump(f"preprocess_msa_23_rank{rank}", preprocess_msa)

        # msa_activations = mnp.expand_dims(preprocess_1d, axis=0) + preprocess_msa
        msa_activations = self.add_2(mnp.expand_dims(preprocess_1d, axis=0), preprocess_msa)
        # print("debug megafold msa_activations", preprocess_msa)
        left_single = self.left_single(target_feat)
        right_single = self.right_single(target_feat)
        # # print("debug megafold left_single", left_single)
        # # print("debug megafold right_single", right_single)
        
        pair_activations = P.ExpandDims()(left_single, 1) + P.ExpandDims()(right_single, 0)
        
        # # print("pair_activations 410", pair_activations)
        mask_2d = P.ExpandDims()(seq_mask, 1) * P.ExpandDims()(seq_mask, 0)
        if self.recycle_pos:
            prev_pseudo_beta, _ = pseudo_beta_fn(aatype, prev_pos, atom37_atom_exists)
            dgram = dgram_from_positions(prev_pseudo_beta, self.num_bins, self.min_bin, self.max_bin, self._type)
            pair_activations += self.prev_pos_linear(dgram)
            # print("pair_activations 418", pair_activations)
        # self.dump(f"pair_activations_437_{rank}_copy", pair_activations)
        if self.recycle_features:
            # print("debug prev_msa_first_row1", prev_msa_first_row)
            prev_msa_first_row = self.prev_msa_first_row_norm(prev_msa_first_row)
            # print("debug prev_msa_first_row2", prev_msa_first_row)
            # print("debug mnp.expand_dims(prev_msa_first_row + msa_activations[0, ...], 0): ", mnp.expand_dims(prev_msa_first_row + msa_activations[0, ...], 0))
            # msa_activations = mnp.concatenate(
            #     (mnp.expand_dims(prev_msa_first_row + msa_activations[0, ...], 0), msa_activations[1:, ...]), 0)
            # msa_activations = self.concat_0_2((mnp.expand_dims(prev_msa_first_row + msa_activations[0, ...], 0), 
            #                                   P.Cast()(msa_activations[1:, ...], mstype.float32)))
            msa_activations = self.concat_0_2((mnp.expand_dims(prev_msa_first_row + msa_activations[0, ...], 0), 
                                              msa_activations[1:, ...]))
            pair_activations += self.prev_pair_norm(prev_pair)
            # print("msa_activations: ", msa_activations)
            # print("pair_activations: ", pair_activations)

        if self.max_relative_feature:
            pair_activations += self._relative_encoding(residue_index, asym_id, sym_id, entity_id, interface_mask)
            # print("pair_activations 429", pair_activations)
        # return pair_activations
        # template_pair_representation = 0
        if self.template_enabled:
            multichain_mask = mnp.equal(P.ExpandDims()(asym_id, 1), P.ExpandDims()(asym_id, 0))
            # print("debug template_embedding", "template_aatype", template_aatype, "template_all_atom_masks", template_all_atom_masks,
            # "template_all_atom_positions", template_all_atom_positions, "mask_2d", mask_2d, "multichain_mask", multichain_mask)
            template_pair_representation = self.template_embedding(pair_activations, template_aatype,
                                                                   template_all_atom_masks,
                                                                   template_all_atom_positions, mask_2d,
                                                                   multichain_mask)
            # print("pair_activations 438", pair_activations)
            # print("template_pair_representation, self.template_embedding", template_pair_representation)                                                    
            pair_activations += template_pair_representation
            # print("pair_activations", pair_activations)
        msa_1hot = self.extra_msa_one_hot(extra_msa)
        # msa_1hot = self.extra_msa_one_hot(extra_msa, 23, Tensor(1.0, mstype.float32), Tensor(0.0, mstype.float32))
        extra_has_deletion = self.cast2(extra_msa_deletion_value > 0, extra_msa_deletion_value.dtype)
        # extra_msa_feat = mnp.concatenate((msa_1hot, extra_has_deletion[..., None], extra_msa_deletion_value[..., None]),
        #                                  axis=-1)
        extra_msa_feat = self.concat_e_3((msa_1hot, self.cast3(self.expand2(extra_has_deletion, -1), mstype.float32), self.cast3(self.expand2(extra_msa_deletion_value, -1), mstype.float32)))
        # print("extra_msa_feat: ", extra_msa_feat)
        extra_msa_activations = self.extra_msa_activations(extra_msa_feat)
        # print("extra_msa_activations_raw: ", extra_msa_activations)
        extra_msa_norm = P.ExpandDims()(P.MatMul(transpose_a=True)(extra_msa_mask, extra_msa_mask), -1)
        # # print("extra_msa_norm: ", extra_msa_norm)
        # # print(extra_msa_stack_num)
        # self.dump(f"extra_msa_activations_{rank}", extra_msa_activations)
        # self.dump(f"pair_activations_{rank}", pair_activations)
        # self.dump(f"extra_msa_mask_{rank}", extra_msa_mask)
        # self.dump(f"extra_msa_norm_{rank}", extra_msa_norm)
        # self.dump(f"mask_2d_{rank}", mask_2d)
        # return self.extra_msa_stack_num, extra_msa_activations, pair_activations, extra_msa_mask, extra_msa_norm, mask_2d
        for i in range(self.extra_msa_stack_num):
            # print("pair_activations 449, round", i, pair_activations)
            extra_msa_activations, pair_activations = \
                self.extra_msa_stack[i](extra_msa_activations, pair_activations, extra_msa_mask, extra_msa_norm,
                                        mask_2d)
        # print("extra_msa_activations: ", extra_msa_activations)
        # print("pair_activations: ", pair_activations)
        # return pair_activations
        if self.template_enabled:
            aatype_one_hot = self.template_aatype_one_hot(template_aatype)
            chi_angles, chi_mask = self.compute_chi_angles(template_aatype,
                                                      template_all_atom_positions,
                                                      template_all_atom_masks,
                                                      self.chi_atom_indices,
                                                      self.chi_angles_mask,
                                                      self.indices0,
                                                      self.indices1)

            
            template_features = self.concat_e_4([aatype_one_hot,
                                                 mnp.sin(chi_angles) * chi_mask,
                                                 mnp.cos(chi_angles) * chi_mask,
                                                 chi_mask])
            # template_mask = chi_mask[:, :, 0]
            template_mask = self.squeeze2(self.slice_ops2(chi_mask, (0,0,0), (chi_mask.shape[0], chi_mask.shape[1], 1)))
            template_activations = self.template_single_embedding(template_features)
            template_activations = self.relu(template_activations)
            template_activations = self.template_projection(template_activations)
            # msa_activations = mnp.concatenate([msa_activations, template_activations], axis=0)
            # print("msa_activations:", msa_activations)
            # msa_activations = self.concat_0_2([msa_activations, self.cast3(template_activations, mstype.float32)])
            msa_activations = self.concat_0_2([msa_activations, template_activations])

            # print("msa_mask's type: ", msa_mask)
            # print("template_mask's type: ", template_mask)
            msa_mask = self.concat_0_2_2([self.cast2(msa_mask, mstype.float32), template_mask])
            
        # print("msa_mask: ", msa_mask)
        # print("msa_activations:", msa_activations)
        msa_mask_norm = P.ExpandDims()(P.MatMul(transpose_a=True)(msa_mask, msa_mask), -1)
        # print("msa_mask_norm: ", msa_mask_norm)

        # # raw notion
        # # contact info
        # # contact_info_input = contact_mask_input.astype(mstype.float16)
        # # contact_feature = contact_info_input[..., None] * 10.0 # increase signal
        # # contact_act = self.preprocess_contact(contact_feature)
        # # pair_activations += contact_act

        sbr_act = self.sbr_act1(sbr*100)
        sbr_act = self.sbr_act2(sbr_act)
        
        # # raw notion
        # # sbr_act = self.sbr_act2(sbr_act) * self.sigmoid(self.sbr_gate(P.Concat(-1)((pair_activations, sbr_act))))
        # # sbr_act = self.process_sbr(sbr, sbr_mask)
        # # print("debug msa_activations 477", msa_activations)

        # print("self.is_training:", self.is_training)
        # print("msa_activations", msa_activations)
        # print("pair_activations", pair_activations)
        # print("msa_mask", msa_mask)
        # print("msa_mask_norm", msa_mask_norm)
        # print("mask_2d", mask_2d)
        # print("sbr_act", sbr_act)
        # print("sbr_mask", sbr_mask)
        # print("interface_mask", interface_mask)
        # print("self.idx_evoformer_block", self.idx_evoformer_block)

        # if self.is_training:
        #   for i in range(self.msa_stack_num):
        #       msa_activations, pair_activations = self.msa_stack[i](msa_activations, pair_activations, msa_mask,
        #                                                             msa_mask_norm, mask_2d, sbr_act, sbr_mask, interface_mask)
        # else:
        self.idx_evoformer_block = self.idx_evoformer_block * 0
        while self.idx_evoformer_block < self.evoformer_num_block_eval:
            msa_activations, pair_activations = self.msa_stack(msa_activations,
                                                                pair_activations,
                                                                msa_mask,
                                                                msa_mask_norm,
                                                                mask_2d,
                                                                sbr_act,
                                                                sbr_mask,
                                                                interface_mask,
                                                                self.idx_evoformer_block)
            self.idx_evoformer_block += 1
        # print("msa_activations: ", msa_activations)
        # print("pair_activations: ", pair_activations)

        # print("debug msa_activations 496", msa_activations)
        # print("msa_activations:", msa_activations)
        single_activations = self.single_activations(msa_activations[0])
        # print("single_activations:", single_activations)
        num_sequences = msa_feat.shape[0]
        # print("num_sequences:", num_sequences)
        msa = msa_activations[:num_sequences, :, :]
        # print("msa:", msa)
        # msa_first_row = msa_activations[0]
        msa_first_row = self.squeeze2(
            self.slice_ops2(msa_activations, (0,0,0), (1, msa_activations.shape[1], msa_activations.shape[2])))
        # print("msa_first_row:", msa_first_row)

        # print("debug megafold single_activations", single_activations)
        final_atom_positions, _, rp_structure_module, _, _, \
        _, _, _, _, _ = \
          self.structure_module(single_activations,
                                pair_activations,
                                seq_mask,
                                aatype,
                                sbr_act,
                                sbr_mask,
                                interface_mask,
                                residx_atom37_to_atom14,
                                atom37_atom_exists)
        predicted_lddt_logits = self.module_lddt(rp_structure_module)
        aligned_error_logits, aligned_error_breaks = self.aligned_error(pair_activations)
        # if self.is_training and self.train_backward:
        #     predicted_lddt_logits = self.module_lddt(rp_structure_module)
        #     dist_logits, bin_edges = self.module_distogram(pair_activations)
        #     experimentally_logits = self.module_exp_resolved(single_activations)
        #     masked_logits = self.module_mask(msa)
        #     return dist_logits, bin_edges, experimentally_logits, masked_logits, aligned_error_logits, \
        #            aligned_error_breaks, atom14_pred_positions, final_affines, angles_sin_cos_new, \
        #            predicted_lddt_logits, structure_traj, sidechain_frames, sidechain_atom_pos, \
        #            um_angles_sin_cos_new, final_atom_positions
        final_atom_positions = P.Cast()(final_atom_positions, self._type)
        # print("final_atom_positions: ", final_atom_positions)
        prev_pos = final_atom_positions
        prev_msa_first_row = msa_first_row
        prev_pair = pair_activations
        # if self.is_training:
        #     return prev_pos, prev_msa_first_row, prev_pair, predicted_lddt_logits
        # print("debug see confg diff", self.cfg)
        # self.dump(f"256prev_pos_{rank}", prev_pos)
        # self.dump(f"256prev_msa_first_row_{rank}", prev_msa_first_row)
        # self.dump(f"256prev_pair_{rank}", prev_pair)
        # self.dump(f"256predicted_lddt_logits_{rank}", predicted_lddt_logits)
        # self.dump(f"256aligned_error_logits_{rank}", aligned_error_logits)

        prev_pos = self.allgather3(prev_pos, (0, 0, 0), (self.cfg.seq_length, 37, 3), (1, 1, 1))
        # prev_msa_first_row = self.allgather2(prev_msa_first_row, (0, 0), (self.cfg.seq_length, 256), (1, 1))
        # prev_pair = self.allgather3(prev_pair, (0, 0, 0), (self.cfg.seq_length, self.cfg.seq_length, 128), (1, 1, 1))
        predicted_lddt_logits = self.allgather2(predicted_lddt_logits, (0, 0), (self.cfg.seq_length, 50), (1, 1))
        aligned_error_logits = self.allgather3(aligned_error_logits, (0, 0, 0), (self.cfg.seq_length, self.cfg.seq_length, 64), (1, 1, 1))

        # prev_msa_first_row = self.allgather(prev_msa_first_row)
        # prev_pair = self.allgather(prev_pair)
        # predicted_lddt_logits = self.allgather(predicted_lddt_logits)
        # aligned_error_logits = self.allgather(aligned_error_logits)
        # aligned_error_breaks = self.allgather(aligned_error_breaks)
        # print("prev_pos: ", prev_pos)
        # print("prev_msa_first_row: ", prev_msa_first_row)
        # print("prev_pair: ", prev_pair)
        # print("predicted_lddt_logits: ", predicted_lddt_logits)
        # print("aligned_error_logits: ", aligned_error_logits)
        # print("aligned_error_breaks: ", aligned_error_breaks)

        return prev_pos, prev_msa_first_row, prev_pair, predicted_lddt_logits, aligned_error_logits, aligned_error_breaks