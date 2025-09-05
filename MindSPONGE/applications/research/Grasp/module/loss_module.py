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
"""loss module"""

import mindspore as ms
import mindspore.communication.management as D
import mindspore.nn as nn
import mindspore.numpy as mnp
from mindspore import Tensor
from mindspore.context import ParallelMode
from mindspore.parallel._utils import _get_parallel_mode
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindsponge1.common import residue_constants
from mindsponge1.common.utils import pseudo_beta_fn
from mindsponge1.common.geometry import invert_point, quaternion_from_tensor, vecs_expand_dims
from mindsponge1.metrics.structure_violations import get_structural_violations, compute_renamed_ground_truth, backbone, \
    sidechain, supervised_chi, local_distance_difference_test
from mindsponge1.metrics import BalancedMSE, BinaryFocal, MultiClassFocal
# from restraint_sample import BINS
from restraint_sample import BINS


class LossNet(nn.Cell):
    """loss net"""

    def __init__(self, config, train_fold=True):
        super(LossNet, self).__init__()
        self.config = config
        self.num_res = config.seq_length
        self.num_bins = config.heads.distogram.num_bins
        self.resolution = config.heads.resolution
        self.distogram_weight = config.heads.distogram.weight
        self.distogram_one_hot = nn.OneHot(depth=self.num_bins, axis=-1)
        self.distogram_one_hot_sbr = nn.OneHot(depth=len(BINS)+1, axis=-1)
        self.exp_min_resolution = config.heads.experimentally_resolved.min_resolution
        self.exp_max_resolution = config.heads.experimentally_resolved.max_resolution
        self.exp_res_filter_by_resolution = config.heads.experimentally_resolved.filter_by_resolution
        self.experimentally_weight = config.heads.experimentally_resolved.weight
        self.exp_res_mask = Tensor(1, ms.float32) \
            if not self.exp_res_filter_by_resolution or \
               (self.exp_min_resolution <= self.resolution <= self.exp_max_resolution) else Tensor(0, ms.float32)

        self.ael_min_resolution = config.heads.predicted_aligned_error.min_resolution
        self.ael_max_resolution = config.heads.predicted_aligned_error.max_resolution
        self.ael_res_filter_by_resolution = config.heads.predicted_aligned_error.filter_by_resolution
        self.ael_res_mask = Tensor(1, ms.float32) \
            if not self.ael_res_filter_by_resolution or \
               (self.ael_min_resolution <= self.resolution <= self.ael_max_resolution) else Tensor(0, ms.float32)
        self.aligned_one_hot = nn.OneHot(depth=config.heads.predicted_aligned_error.num_bins)

        self.plddt_min_resolution = config.heads.predicted_lddt.min_resolution
        self.plddt_max_resolution = config.heads.predicted_lddt.max_resolution
        self.plddt_res_filter_by_resolution = config.heads.predicted_lddt.filter_by_resolution
        self.plddt_res_mask = Tensor(1, ms.float32) \
            if not self.plddt_res_filter_by_resolution or \
               (self.plddt_min_resolution <= self.resolution <= self.plddt_max_resolution) else Tensor(0, ms.float32)
        self.plddt_weight = config.heads.predicted_lddt.weight

        self.masked_one_hot = nn.OneHot(depth=config.heads.masked_msa.num_output, axis=-1)
        self.masked_weight = config.heads.masked_msa.weight
        self.sidechain_weight_frac = config.structure_module.sidechain.weight_frac
        self.angle_norm_weight = config.structure_module.angle_norm_weight
        self.chi_weight = config.structure_module.chi_weight
        self.chi_pi_periodic = mnp.asarray(residue_constants.chi_pi_periodic, ms.float32)

        self.violation_tolerance_factor = config.structure_module.violation_tolerance_factor
        self.clash_overlap_tolerance = config.structure_module.clash_overlap_tolerance
        self.sidechain_atom_clamp_distance = config.structure_module.sidechain.atom_clamp_distance
        # self.sidechain_atom_clamp_distance = self.sidechain_atom_clamp_distance * 1000
        self.sidechain_length_scale = config.structure_module.sidechain.length_scale
        self.fape_clamp_distance = config.structure_module.fape.clamp_distance
        self.fape_loss_unit_distance = config.structure_module.fape.loss_unit_distance
        self.predicted_lddt_num_bins = config.heads.predicted_lddt.num_bins
        self.c_one_hot = nn.OneHot(depth=14)
        self.n_one_hot = nn.OneHot(depth=14)
        self.zeros = Tensor(0, ms.int32)
        self.twos = Tensor(2, ms.int32)
        self.dists_mask_i = mnp.eye(14, 14)
        self.cys_sg_idx = Tensor(5, ms.int32)
        self.train_fold = train_fold
        self.sigmoid_cross_entropy = P.SigmoidCrossEntropyWithLogits()

    def softmax_cross_entropy(self, logits, labels):
        """Computes softmax cross entropy given logits and one-hot class labels."""
        loss = -mnp.sum(labels * nn.LogSoftmax()(logits), axis=-1)
        return mnp.asarray(loss)
    
    def softmax_cross_entropy_binary(self, logits, labels, binary_mask):
        """Computes softmax cross entropy given logits and one-hot class labels."""
        labels_positive  = mnp.sum(labels * binary_mask, axis=-1)
        pred_positive = mnp.sum(nn.Softmax()(logits) * binary_mask, axis=-1)
        loss = -((labels_positive * P.Log()(pred_positive + 1e-10)) + (1 - labels_positive) * P.Log()(1 - pred_positive + 1e-10))
        return mnp.asarray(loss)

    def distogram_loss(self, logits, bin_edges, pseudo_beta, pseudo_beta_mask, sbr_intra_mask, sbr_inter_mask):
        """Log loss of a distogram."""
        positions = pseudo_beta
        mask = pseudo_beta_mask

        sq_breaks = mnp.square(bin_edges)
        dist_t = mnp.square(mnp.expand_dims(positions, axis=-2) - mnp.expand_dims(positions, axis=-3))
        dist2 = P.ReduceSum(True)(dist_t.astype(ms.float32), -1)
        aa = (dist2 > sq_breaks).astype(ms.float32)

        true_bins = P.ReduceSum()(aa, -1)
        true_bins = true_bins.astype(ms.int32)
        errors = self.softmax_cross_entropy(labels=self.distogram_one_hot(true_bins), logits=logits)
        square_mask = mnp.expand_dims(mask, axis=-2) * mnp.expand_dims(mask, axis=-1)
        
        sbr_inter_mask *= square_mask
        sbr_intra_mask *= square_mask
        avg_error = (P.ReduceSum()(errors * square_mask, (-2, -1)) /
                     (1e-6 + P.ReduceSum()(square_mask.astype(ms.float32), (-2, -1))))
        # sbr_inter_disto_loss = (P.ReduceSum()(errors * sbr_inter_mask, (-2, -1)) /
        #              (1e-6 + P.ReduceSum()(sbr_inter_mask.astype(ms.float32), (-2, -1))))
        # sbr_intra_disto_loss = (P.ReduceSum()(errors * sbr_intra_mask, (-2, -1)) /
        #              (1e-6 + P.ReduceSum()(sbr_intra_mask.astype(ms.float32), (-2, -1))))

        dist2 = dist2[..., 0]
        loss = avg_error
        true_dist = mnp.sqrt(1e-6 + dist2)
        return loss, true_dist #, sbr_intra_disto_loss, sbr_inter_disto_loss

    def get_mask(self, sbr_mask, asym_id):
        sbr_mask = P.Cast()(sbr_mask, ms.float32)
        intra_chain_mask = P.Cast()(asym_id[:, None] == asym_id[None, :], ms.float32)
        sbr_intra_mask = intra_chain_mask * sbr_mask
        sbr_inter_mask = P.Cast()((1 - intra_chain_mask) * sbr_mask, ms.float32)
        return sbr_intra_mask, sbr_inter_mask


    def experimentally_loss(self, experimentally_logits, atom37_atom_exists, all_atom_mask, filter_by_solution):
        """experimentally_loss"""
        logits = experimentally_logits

        # Does the atom appear in the amino acid?
        atom_exists = atom37_atom_exists
        # Is the atom resolved in the experiment? Subset of atom_exists,
        # *except for OXT*
        all_atom_mask = all_atom_mask.astype(mnp.float32)

        xent = self.sigmoid_cross_entropy(logits, all_atom_mask)
        loss = P.ReduceSum()(xent * atom_exists) / (1e-8 + P.ReduceSum()(atom_exists.astype(ms.float32)))
        loss = loss * filter_by_solution
        loss *= self.exp_res_mask
        return loss

    def masked_head_loss(self, true_msa, logits, bert_mask):
        """masked_head_loss"""
        errors = self.softmax_cross_entropy(logits=logits, labels=self.masked_one_hot(true_msa))
        loss = (P.ReduceSum()(errors * bert_mask, (-2, -1)) /
                (1e-8 + P.ReduceSum()(bert_mask.astype(ms.float32), (-2, -1))))
        return loss



    # todo
    def structure_loss(self, atom14_gt_positions, atom14_alt_gt_positions, atom14_atom_is_ambiguous,
                       atom14_gt_exists, atom14_atom_exists, final_atom14_positions, atom14_alt_gt_exists,
                       residue_index, aatype, residx_atom14_to_atom37, lower_bound, upper_bound, seq_mask,
                       atomtype_radius, angles_sin_cos, um_angles_sin_cos, traj, backbone_affine_tensor,
                       backbone_affine_mask, rigidgroups_gt_frames, rigidgroups_gt_exists, rigidgroups_alt_gt_frames,
                       pred_frames, pred_positions, sin_cos_true_chi, torsion_angle_mask, use_clamped_fape, asym_id, 
                       sbr_mask):
        """structure_loss"""
        atom14_pred_positions = final_atom14_positions
        # Compute renaming and violations.
        alt_naming_is_better, renamed_atom14_gt_positions, renamed_atom14_gt_exists = \
            compute_renamed_ground_truth(atom14_gt_positions,
                                         atom14_alt_gt_positions,
                                         atom14_atom_is_ambiguous,
                                         atom14_gt_exists,
                                         atom14_pred_positions,
                                         atom14_alt_gt_exists)
        (bonds_c_n_loss_mean, angles_ca_c_n_loss_mean, angles_c_n_ca_loss_mean, _,
         _, _, clashes_per_atom_loss_sum, _, per_atom_loss_sum, _, _, _,
         clashes_per_atom_clash_count, per_atom_clash_count) = \
            get_structural_violations(atom14_atom_exists, residue_index, aatype, residx_atom14_to_atom37,
                                      atom14_pred_positions, asym_id, self.violation_tolerance_factor,
                                      self.clash_overlap_tolerance, lower_bound, upper_bound, atomtype_radius,
                                      self.c_one_hot(self.twos), self.n_one_hot(self.zeros), self.dists_mask_i,
                                      self.cys_sg_idx)

        bond_loss = bonds_c_n_loss_mean + angles_ca_c_n_loss_mean  * 0.3 + angles_c_n_ca_loss_mean * 0.3

        #num_atoms = P.ReduceSum()(atom14_atom_exists.astype(ms.float32))
        num_atoms = P.ReduceSum()(clashes_per_atom_clash_count + per_atom_clash_count)
        clash_loss = P.ReduceSum()(clashes_per_atom_loss_sum + per_atom_loss_sum) / (1e-6 + num_atoms)

        structure_violation_loss = bond_loss + clash_loss

        # from structure module result
        _, fape_loss, no_clamp, fape_nc_intra, fape_nc_inter, sbr_intra_fape_loss, sbr_inter_fape_loss = \
            backbone(traj, backbone_affine_tensor, backbone_affine_mask, \
                     self.fape_clamp_distance, self.fape_loss_unit_distance, use_clamped_fape, asym_id, sbr_mask)

        loss_sidechain = sidechain(alt_naming_is_better, rigidgroups_gt_frames, rigidgroups_alt_gt_frames,
                                   rigidgroups_gt_exists, renamed_atom14_gt_positions, renamed_atom14_gt_exists,
                                   self.sidechain_atom_clamp_distance, self.sidechain_length_scale, pred_frames,
                                   pred_positions)
        angle_norm_loss = supervised_chi(seq_mask, aatype, sin_cos_true_chi, torsion_angle_mask,
                                         angles_sin_cos, um_angles_sin_cos, self.chi_weight,
                                         self.angle_norm_weight, self.chi_pi_periodic)
        return fape_loss, loss_sidechain, angle_norm_loss, structure_violation_loss, no_clamp, bond_loss, \
            clash_loss, fape_nc_intra, fape_nc_inter, sbr_intra_fape_loss, sbr_inter_fape_loss

    def predicted_lddt_loss(self, final_atom_positions, all_atom_positions, all_atom_mask, predicted_lddt_logits,
                            filter_by_solution):
        """predicted_lddt_loss"""
        pred_all_atom_pos = final_atom_positions
        true_all_atom_pos = all_atom_positions
        lddt_ca = local_distance_difference_test(
            predicted_points=pred_all_atom_pos[None, :, 1, :],
            true_points=true_all_atom_pos[None, :, 1, :],
            true_points_mask=all_atom_mask[None, :, 1:2].astype(mnp.float32),
            cutoff=15.,
            per_residue=True)[0]

        lddt_ca = F.stop_gradient(lddt_ca)

        bin_index = mnp.floor(lddt_ca * self.predicted_lddt_num_bins).astype(ms.int32)

        # protect against out of range for lddt_ca == 1
        bin_index = mnp.minimum(bin_index, self.predicted_lddt_num_bins - 1)
        lddt_ca_one_hot = nn.OneHot(depth=self.predicted_lddt_num_bins)(bin_index)

        logits = predicted_lddt_logits
        errors = self.softmax_cross_entropy(labels=lddt_ca_one_hot, logits=logits)

        mask_ca = all_atom_mask[:, 1]
        mask_ca = mask_ca.astype(mnp.float32)
        loss = P.ReduceSum()(errors * mask_ca) / P.ReduceSum()(P.ReduceSum()(mask_ca) + 1e-8)
        loss = loss * filter_by_solution
        loss *= self.plddt_res_mask

        return loss

    def aligned_error_loss(self, final_affines, backbone_affine_tensor, backbone_affine_mask, pae_breaks, pae_logits,
                           filter_by_solution):
        """aligned_error_loss"""
        # Shape (num_res, 7) predict affine
        _, rotation_pd, translation_pd = quaternion_from_tensor(final_affines)
        translation_point_pd = vecs_expand_dims(translation_pd, -2)
        rotation_pd_tensor = rotation_pd
        # Shape (num_res, 7) true affine
        _, rotation_gt, translation_gt = quaternion_from_tensor(backbone_affine_tensor)
        translation_point_tr = vecs_expand_dims(translation_gt, -2)
        rotation_gt_tensor = rotation_gt
        mask = backbone_affine_mask
        square_mask = (mask[:, None] * mask[None, :]).astype(ms.float32)
        breaks = pae_breaks
        logits = pae_logits

        local_frames_pd = invert_point(translation_point_pd, rotation_pd_tensor, translation_pd, extra_dims=1)
        local_frames_gt = invert_point(translation_point_tr, rotation_gt_tensor, translation_gt, extra_dims=1)
        # todo to be checked
        error_dist2 = mnp.square(local_frames_pd[0] - local_frames_gt[0]) + \
                      mnp.square(local_frames_pd[1] - local_frames_gt[1]) + \
                      mnp.square(local_frames_pd[2] - local_frames_gt[2])
        error_dist2 = F.stop_gradient(error_dist2)
        # # Compute the squared error for each alignment.
        sq_breaks = mnp.square(breaks)
        true_bins = P.ReduceSum()((error_dist2[..., None] > sq_breaks).astype(mnp.float32), -1)

        errors = self.softmax_cross_entropy(labels=self.aligned_one_hot(true_bins.astype(ms.int32)), logits=logits)

        loss = (P.ReduceSum()(errors * square_mask, (-2, -1)) /
                (1e-8 + P.ReduceSum()(square_mask, (-2, -1))))
        loss = loss * filter_by_solution
        loss *= self.ael_res_mask

        return loss

    def distance_rmsd_loss(self, predicted_atom_positions, label_atom_positions, rmsd_mask):
        dist1 = P.Sqrt()(P.ReduceSum()(P.Square()(predicted_atom_positions[None]-predicted_atom_positions[:,None]), -1) + 1e-8)
        dist2 = P.Sqrt()(P.ReduceSum()(P.Square()(label_atom_positions[None]    -label_atom_positions[:,None])    , -1) + 1e-8)
        error = P.Square()(dist1 - dist2)
        loss = P.Sqrt()(P.ReduceSum()(error * rmsd_mask) / (P.ReduceSum()(rmsd_mask) + 1e-8) + 1e-8)/ self.fape_loss_unit_distance
        return loss
    
    def backbone_drmsd_loss(self, pseudo_beta_pred, pseudo_beta_gt, final_atom_positions, all_atom_positions, mask):
        rmsd_loss_cb = self.distance_rmsd_loss(pseudo_beta_pred, pseudo_beta_gt, mask.astype(ms.float32))
        rmsd_loss_ca = self.distance_rmsd_loss(final_atom_positions[:, 1, :], all_atom_positions[:, 1, :], mask.astype(ms.float32))
        rmsd_loss_c = self.distance_rmsd_loss(final_atom_positions[:, 0, :], all_atom_positions[:, 0, :], mask.astype(ms.float32))
        rmsd_loss_n = self.distance_rmsd_loss(final_atom_positions[:, 2, :], all_atom_positions[:, 2, :], mask.astype(ms.float32))
        backbone_drmsd_loss = (rmsd_loss_ca + rmsd_loss_c + rmsd_loss_n + rmsd_loss_cb)
        return backbone_drmsd_loss

    def get_asym_centres(self, pos, asym_mask, eps):
        pos = P.ExpandDims()(pos, 0) * P.ExpandDims()(asym_mask, 2) # [NC, NR, 3]
        return mnp.sum(pos, -2) / (mnp.sum(asym_mask, -1)[..., None] + eps) # [NC, 3]

    def chain_centre_mass_loss(self, pseudo_beta, pseudo_beta_mask, aatype, final_atom_positions, asym_mask, eps=1e-8):
        pseudo_beta_pred = pseudo_beta_fn(aatype, final_atom_positions, None)
        asym_mask = asym_mask * P.ExpandDims()(pseudo_beta_mask, 0)  # [NC, NR]
        asym_exists = P.Cast()(asym_mask.sum(-1) > 0, ms.float16)
        # asym_exists = asym_mask.any(axis=-1) # [NC, ]

        pred_centres = self.get_asym_centres(pseudo_beta_pred, asym_mask, eps)  # [NC, 3]
        true_centres = self.get_asym_centres(pseudo_beta, asym_mask, eps)  # [NC, 3]

        pred_dists = P.Sqrt()((P.Square()(pred_centres[None] - pred_centres[:, None])).sum(-1) + 1e-8) # [NC, NC]
        true_dists = P.Sqrt()((P.Square()(true_centres[None] - true_centres[:, None])).sum(-1) + 1e-8) # [NC, NC]
        chain_centre_mass_loss = P.Square()(mnp.clip(P.Abs()(pred_dists - true_dists) - 4, xmin=0, xmax=None)) * 0.0025
        # chain_centre_mass_loss = P.Square()(mnp.clip(pred_dists - true_dists + 4, xmin=None, xmax=0)) * 0.0025

        chain_centre_mask = (asym_exists[None, :] * asym_exists[:, None]).astype(ms.float32)
        chain_centre_mass_loss = (chain_centre_mass_loss * chain_centre_mask).sum() / (chain_centre_mask.sum() + eps)

        return chain_centre_mass_loss
    

    def sbr_drmsd_loss(self, final_atom_positions, all_atom_positions, pseudo_beta_gt, aatype, sbr_intra_mask, sbr_inter_mask):

        pseudo_beta_pred = pseudo_beta_fn(aatype, final_atom_positions, None) # CA as CB for glycine
        # positional rmsd sbr loss
        sbr_intra_drmsd_loss = self.backbone_drmsd_loss(pseudo_beta_pred, pseudo_beta_gt, final_atom_positions, all_atom_positions, \
                                                            sbr_intra_mask)
        sbr_inter_drmsd_loss = self.backbone_drmsd_loss(pseudo_beta_pred, pseudo_beta_gt, final_atom_positions, all_atom_positions, \
                                                            sbr_inter_mask)
        return sbr_intra_drmsd_loss, sbr_inter_drmsd_loss, pseudo_beta_pred
    
    def compute_sbr_loss(self, pseudo_pred_dist, bin_edges_sbr, sbr, sbr_intra_mask, sbr_inter_mask, delta=2.0):
        not_high_bin = (sbr <= 1.0/(len(bin_edges_sbr) + 1)).astype(ms.float32)
        upper_1d = P.Concat()((bin_edges_sbr, Tensor([10000], ms.float32)))
        lower_1d = P.Concat()((Tensor([0], ms.float32), bin_edges_sbr))
        upper_2d = (upper_1d-1e6*not_high_bin).max(-1)
        lower_2d = (lower_1d+1e6*not_high_bin).min(-1)
        lower_error = mnp.clip(lower_2d- delta - pseudo_pred_dist, 0, 30)
        upper_error = mnp.clip(pseudo_pred_dist - upper_2d - delta, 0, 30)
        error = (lower_error + upper_error)*(upper_2d > lower_2d)
        error_inter = (error * sbr_inter_mask).sum() / (sbr_inter_mask.sum() + 1e-8)
        error_intra = (error * sbr_intra_mask).sum() / (sbr_intra_mask.sum() + 1e-8)
        recall = (error<=0).astype(ms.float32)
        recall_inter1 = (recall * sbr_inter_mask).sum() / (sbr_inter_mask.sum() + 1e-8)
        recall_intra1 = (recall * sbr_intra_mask).sum() / (sbr_intra_mask.sum() + 1e-8)
        return error_intra, error_inter, recall_inter1, recall_intra1

    def compute_recall(self, pseudo_pred_dist, bin_edges_sbr, sbr, sbr_intra_mask, sbr_inter_mask):
        # compute recall
        sbr_binary = (sbr > 1.0/(len(bin_edges_sbr) + 1)).astype(ms.float32)
        aa = (mnp.expand_dims(pseudo_pred_dist, -1) > bin_edges_sbr).astype(ms.float32)
        pred_bins = P.ReduceSum()(aa, -1)
        pred_bins = pred_bins.astype(ms.int32)
        sbr_pred = mnp.sum(self.distogram_one_hot_sbr(pred_bins) * sbr_binary, axis=-1)
        recall_intra = (sbr_pred * sbr_intra_mask).sum() / (sbr_intra_mask.sum() + 1e-8)
        recall_inter = (sbr_pred * sbr_inter_mask).sum() / (sbr_inter_mask.sum() + 1e-8)
        return  recall_intra, recall_inter

    def interface_loss(self, interface_mask, asym_id, pseudo_pred_dist, pseudo_beta_mask, true_dist, delta=1.0, eps=1e-8):
        inter_chain_mask = P.Cast()(asym_id[:, None] != asym_id[None, :], ms.float32)
        pseudo_pred_dist += (1.0 - pseudo_beta_mask * inter_chain_mask) * 1e9
        # dist += (1.0 - pseudo_beta_mask * inter_chain_mask) * 1e9
        perfect_dist = pseudo_pred_dist + (true_dist > 8) * 1e9
        interface_min_dist = pseudo_pred_dist.min(axis=-1)
        
        
        error = mnp.clip(interface_min_dist - (8.0 + delta), 0.0, 30.0)
        error = (error * interface_mask).sum() / (interface_mask.sum() + eps)

        is_interface = P.Cast()(interface_min_dist < 8.0, ms.float32)
        is_perfect_interface = P.Cast()(perfect_dist.min(axis=-1) < 8.0, ms.float32)
        recall_interface = (is_interface * interface_mask).sum() / interface_mask.sum()
        pefect_recall_interface = (is_perfect_interface * interface_mask).sum() / interface_mask.sum()
        return  error, recall_interface, pefect_recall_interface

    def construct(self, distogram_logits, bin_edges, pseudo_beta, pseudo_beta_mask, experimentally_logits,
                  atom37_atom_exists, all_atom_mask, true_msa, masked_logits, bert_mask,
                  final_atom14_positions, residue_index, aatype, residx_atom14_to_atom37, lower_bound, upper_bound,
                  seq_mask, atomtype_radius, final_affines, pae_breaks, pae_logits, angles_sin_cos,
                  um_angles_sin_cos, backbone_affine_tensor, backbone_affine_mask, atom14_gt_positions,
                  atom14_alt_gt_positions, atom14_atom_is_ambiguous, atom14_gt_exists, atom14_atom_exists,
                  atom14_alt_gt_exists, final_atom_positions, all_atom_positions, predicted_lddt_logits, traj,
                  rigidgroups_gt_frames, rigidgroups_gt_exists, rigidgroups_alt_gt_frames,
                  pred_frames, pred_positions, sin_cos_true_chi, torsion_angle_mask, use_clamped_fape,
                  filter_by_solution, asym_id, asym_mask, sbr, sbr_mask, interface_mask):
        """construct"""
        distogram_loss = 0.0
        sbr_intra_disto_loss = 0.0
        sbr_inter_disto_loss = 0.0
        masked_loss = 0.0
        sbr_intra_mask, sbr_inter_mask = self.get_mask(sbr_mask, asym_id)


        if self.train_fold:
            distogram_loss, dist = \
                self.distogram_loss(distogram_logits, bin_edges, pseudo_beta, 
                                    pseudo_beta_mask, sbr_intra_mask, sbr_inter_mask)
            distogram_loss = distogram_loss * self.distogram_weight # 0.3

            masked_loss = self.masked_head_loss(true_msa, masked_logits, bert_mask)
            masked_loss = self.masked_weight * masked_loss #2
            # masked_loss = Tensor(0.0)

            # self.aligned_error_loss(final_affines, backbone_affine_tensor, backbone_affine_mask, pae_breaks,
            #                         pae_logits, filter_by_solution)
            # self.experimentally_loss(experimentally_logits, atom37_atom_exists, all_atom_mask, filter_by_solution)

        fape_loss, loss_sidechain, angle_norm_loss, structure_violation_loss, no_clamp, bond_loss, clash_loss, \
            fape_nc_intra, fape_nc_inter, sbr_intra_fape_loss, sbr_inter_fape_loss = \
            self.structure_loss(atom14_gt_positions, atom14_alt_gt_positions, atom14_atom_is_ambiguous,
                                atom14_gt_exists, atom14_atom_exists, final_atom14_positions,
                                atom14_alt_gt_exists, residue_index, aatype, residx_atom14_to_atom37,
                                lower_bound, upper_bound, seq_mask, atomtype_radius, angles_sin_cos,
                                um_angles_sin_cos, traj, backbone_affine_tensor,
                                backbone_affine_mask, rigidgroups_gt_frames, rigidgroups_gt_exists,
                                rigidgroups_alt_gt_frames,
                                pred_frames, pred_positions, sin_cos_true_chi, torsion_angle_mask, use_clamped_fape,
                                asym_id, sbr_mask)
        structure_violation_loss = structure_violation_loss * 0.03

        predict_lddt_loss = self.predicted_lddt_loss(final_atom_positions, all_atom_positions, all_atom_mask,
                                                     predicted_lddt_logits, filter_by_solution)
        predict_lddt_loss = self.plddt_weight * predict_lddt_loss # 0.01

        chain_centre_mass_loss = self.chain_centre_mass_loss(pseudo_beta, pseudo_beta_mask, aatype,
                                                             final_atom_positions, asym_mask)
        # # todo check whether to use it
        aligned_error_loss = self.aligned_error_loss(final_affines, backbone_affine_tensor,
                                                     backbone_affine_mask, pae_breaks, pae_logits, filter_by_solution)
        aligned_error_loss = aligned_error_loss * 0.1

        l_fape_side = 0.5 * loss_sidechain
        l_fape_backbone = 0.5 * fape_loss
        l_anglenorm = angle_norm_loss

        # sbr loss
        sbr_intra_drmsd_loss, sbr_inter_drmsd_loss, pseudo_beta_pred \
            = self.sbr_drmsd_loss(final_atom_positions, all_atom_positions, pseudo_beta, aatype, \
                                sbr_intra_mask, sbr_inter_mask)
        #sbr recall
        # bin_edges_sbr = mnp.linspace(8.25, 20.75, 11)
        bin_edges_sbr = mnp.arange(4, 33, 1).astype(ms.float32)
        pseudo_pred_dist = P.Sqrt()(P.ReduceSum()(P.Square()(pseudo_beta_pred[:, None] - pseudo_beta_pred[None]), -1) + 1e-8)
        true_dist = P.Sqrt()(P.ReduceSum()(P.Square()(pseudo_beta[:, None] - pseudo_beta[None]), -1) + 1e-8)
        
        recall_intra, recall_inter = self.compute_recall(pseudo_pred_dist, bin_edges_sbr, sbr, sbr_intra_mask, sbr_inter_mask)
        sbr_intra_disto_loss, sbr_inter_disto_loss, recall_inter1, recall_intra1 = self.compute_sbr_loss(pseudo_pred_dist, bin_edges_sbr, sbr, sbr_intra_mask, sbr_inter_mask)
        

        # interface loss
        
        
        sbr_inter_fape_loss = sbr_inter_fape_loss * 0.5
        sbr_intra_fape_loss = sbr_intra_fape_loss * 0.5

        sbr_inter_drmsd_loss = sbr_inter_drmsd_loss * 0.05
        sbr_intra_drmsd_loss = sbr_intra_drmsd_loss * 0.05

        sbr_inter_disto_loss *= 0.01
        sbr_intra_disto_loss *= 0.01

        all_sbr_loss = sbr_intra_disto_loss + sbr_inter_disto_loss + \
            mnp.clip(sbr_inter_fape_loss + sbr_inter_drmsd_loss, 0.0, 1.5) + \
            mnp.clip(sbr_intra_fape_loss + sbr_intra_drmsd_loss, 0.0, 1.5)

        interface_loss, recall_interface, perfect_recall_interface = self.interface_loss(interface_mask, asym_id, pseudo_pred_dist, pseudo_beta_mask, true_dist)
        interface_loss *= 0.5
        
        loss = l_fape_side + \
               l_fape_backbone + \
               l_anglenorm + \
               distogram_loss + \
               masked_loss + \
               predict_lddt_loss + \
               mnp.clip(structure_violation_loss, 0.0, 1) + \
               aligned_error_loss + \
               mnp.clip(chain_centre_mass_loss, 0.0, 1) + \
               all_sbr_loss + \
               mnp.clip(interface_loss, 0.0, 1)

        loss = loss * P.Sqrt()(P.ReduceSum()(all_atom_mask[:, 0]))

        return loss, l_fape_side, l_fape_backbone, l_anglenorm, distogram_loss, masked_loss, predict_lddt_loss,\
            structure_violation_loss, no_clamp,  fape_nc_intra, fape_nc_inter, chain_centre_mass_loss, aligned_error_loss,\
            sbr_inter_fape_loss, sbr_inter_drmsd_loss, sbr_inter_disto_loss,\
            sbr_intra_fape_loss, sbr_intra_drmsd_loss, sbr_intra_disto_loss, interface_loss, \
            recall_intra, recall_inter, recall_interface, perfect_recall_interface, recall_inter1, recall_intra1

        # structure_violation_loss, no_clamp, bond_loss, clash_loss, chain_centre_mass_loss, aligned_error_loss
        # predict_lddt_loss, predict_lddt_loss, predict_lddt_loss, predict_lddt_loss, predict_lddt_loss, predict_lddt_loss
        
