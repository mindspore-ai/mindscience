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
from mindsponge.common import residue_constants
from mindsponge.common.geometry import invert_point, quaternion_from_tensor, vecs_expand_dims
from mindsponge.metrics.structure_violations import get_structural_violations, compute_renamed_ground_truth, backbone, \
    sidechain, supervised_chi, local_distance_difference_test
from mindsponge.metrics import BalancedMSE, BinaryFocal, MultiClassFocal


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

        self.masked_one_hot = nn.OneHot(depth=23, axis=-1)
        self.masked_weight = config.heads.masked_msa.weight
        self.sidechain_weight_frac = config.structure_module.sidechain.weight_frac
        self.angle_norm_weight = config.structure_module.angle_norm_weight
        self.chi_weight = config.structure_module.chi_weight
        self.chi_pi_periodic = mnp.asarray(residue_constants.chi_pi_periodic, ms.float32)

        self.violation_tolerance_factor = config.structure_module.violation_tolerance_factor
        self.clash_overlap_tolerance = config.structure_module.clash_overlap_tolerance
        self.sidechain_atom_clamp_distance = config.structure_module.sidechain.atom_clamp_distance
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
        loss = -mnp.sum(labels * P.Log()(nn.Softmax()(logits)), axis=-1)
        return mnp.asarray(loss)

    def distogram_loss(self, logits, bin_edges, pseudo_beta, pseudo_beta_mask):
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
        avg_error = (P.ReduceSum()(errors * square_mask, (-2, -1)) /
                     (1e-6 + P.ReduceSum()(square_mask.astype(ms.float32), (-2, -1))))

        dist2 = dist2[..., 0]
        loss = avg_error
        true_dist = mnp.sqrt(1e-6 + dist2)

        return loss, true_dist

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
                       pred_frames, pred_positions, sin_cos_true_chi, torsion_angle_mask, use_clamped_fape):
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
         _, _, clashes_per_atom_loss_sum, _, per_atom_loss_sum, _, _, _) = \
            get_structural_violations(atom14_atom_exists, residue_index, aatype, residx_atom14_to_atom37,
                                      atom14_pred_positions, self.violation_tolerance_factor,
                                      self.clash_overlap_tolerance, lower_bound, upper_bound, atomtype_radius,
                                      self.c_one_hot(self.twos), self.n_one_hot(self.zeros), self.dists_mask_i,
                                      self.cys_sg_idx)
        num_atoms = P.ReduceSum()(atom14_atom_exists.astype(ms.float32))
        structure_violation_loss = bonds_c_n_loss_mean + angles_ca_c_n_loss_mean + angles_c_n_ca_loss_mean + \
                                   P.ReduceSum()(clashes_per_atom_loss_sum + per_atom_loss_sum) / (1e-6 + num_atoms)

        # from structure module result
        _, fape_loss, no_clamp = backbone(traj, backbone_affine_tensor, backbone_affine_mask,
                                          self.fape_clamp_distance, self.fape_loss_unit_distance, use_clamped_fape)

        loss_sidechain = sidechain(alt_naming_is_better, rigidgroups_gt_frames, rigidgroups_alt_gt_frames,
                                   rigidgroups_gt_exists, renamed_atom14_gt_positions, renamed_atom14_gt_exists,
                                   self.sidechain_atom_clamp_distance, self.sidechain_length_scale, pred_frames,
                                   pred_positions)
        angle_norm_loss = supervised_chi(seq_mask, aatype, sin_cos_true_chi, torsion_angle_mask,
                                         angles_sin_cos, um_angles_sin_cos, self.chi_weight,
                                         self.angle_norm_weight, self.chi_pi_periodic)
        return fape_loss, loss_sidechain, angle_norm_loss, structure_violation_loss, no_clamp

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

        bin_index = mnp.floor(lddt_ca * self.predicted_lddt_num_bins).astype(mnp.int32)

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
        # # Compute the squared error for each alignment.
        sq_breaks = mnp.square(breaks)
        true_bins = P.ReduceSum()((error_dist2[..., None] > sq_breaks).astype(mnp.float32), -1)

        errors = self.softmax_cross_entropy(labels=self.aligned_one_hot(true_bins.astype(ms.int32)), logits=logits)

        loss = (P.ReduceSum()(errors * square_mask, (-2, -1)) /
                (1e-8 + P.ReduceSum()(square_mask, (-2, -1))))
        loss = loss * filter_by_solution
        loss *= self.ael_res_mask

        return loss

    def rmsd_loss(self, predicted_atom_positions, label_atom_positions, pseudo_beta_mask_2d):
        """rmsd_loss"""
        dist1 = P.Sqrt()((P.Square()(predicted_atom_positions[None] -
                                     predicted_atom_positions[:, None])).sum(-1) + 1e-8)
        dist2 = P.Sqrt()((P.Square()(label_atom_positions[None] - label_atom_positions[:, None])).sum(-1) + 1e-8)
        return P.Sqrt()((P.Square()(dist1 - dist2) * pseudo_beta_mask_2d).mean() + 1e-8)

    def construct(self, distogram_logits, bin_edges, pseudo_beta, pseudo_beta_mask, experimentally_logits,
                  atom37_atom_exists, all_atom_mask, true_msa, masked_logits, bert_mask,
                  final_atom14_positions, residue_index, aatype, residx_atom14_to_atom37, lower_bound, upper_bound,
                  seq_mask, atomtype_radius, final_affines, pae_breaks, pae_logits, angles_sin_cos,
                  um_angles_sin_cos, backbone_affine_tensor, backbone_affine_mask, atom14_gt_positions,
                  atom14_alt_gt_positions, atom14_atom_is_ambiguous, atom14_gt_exists, atom14_atom_exists,
                  atom14_alt_gt_exists, final_atom_positions, all_atom_positions, predicted_lddt_logits, traj,
                  rigidgroups_gt_frames, rigidgroups_gt_exists, rigidgroups_alt_gt_frames,
                  pred_frames, pred_positions, sin_cos_true_chi, torsion_angle_mask, use_clamped_fape,
                  filter_by_solution):
        """construct"""
        distogram_loss = 0.0
        masked_loss = 0.0
        if self.train_fold:
            distogram_loss, _ = self.distogram_loss(distogram_logits, bin_edges, pseudo_beta, pseudo_beta_mask)
            distogram_loss = distogram_loss * self.distogram_weight

            masked_loss = self.masked_head_loss(true_msa, masked_logits, bert_mask)
            masked_loss = self.masked_weight * masked_loss

            self.aligned_error_loss(final_affines, backbone_affine_tensor, backbone_affine_mask, pae_breaks,
                                    pae_logits, filter_by_solution)
            self.experimentally_loss(experimentally_logits, atom37_atom_exists, all_atom_mask, filter_by_solution)

        fape_loss, loss_sidechain, angle_norm_loss, _, _ = \
            self.structure_loss(atom14_gt_positions, atom14_alt_gt_positions, atom14_atom_is_ambiguous,
                                atom14_gt_exists, atom14_atom_exists, final_atom14_positions,
                                atom14_alt_gt_exists, residue_index, aatype, residx_atom14_to_atom37,
                                lower_bound, upper_bound, seq_mask, atomtype_radius, angles_sin_cos,
                                um_angles_sin_cos, traj, backbone_affine_tensor,
                                backbone_affine_mask, rigidgroups_gt_frames, rigidgroups_gt_exists,
                                rigidgroups_alt_gt_frames,
                                pred_frames, pred_positions, sin_cos_true_chi, torsion_angle_mask, use_clamped_fape)

        predict_lddt_loss = self.predicted_lddt_loss(final_atom_positions, all_atom_positions, all_atom_mask,
                                                     predicted_lddt_logits, filter_by_solution)
        predict_lddt_loss = self.plddt_weight * predict_lddt_loss

        # # todo check whether to use it
        # aligned_error_loss = self.aligned_error_loss(final_affines, backbone_affine_tensor,
        #                                              backbone_affine_mask, pae_breaks, pae_logits, filter_by_solution)

        l_fape_side = 0.5 * loss_sidechain
        l_fape_backbone = 0.5 * fape_loss
        l_anglenorm = angle_norm_loss

        loss = l_fape_side + \
               l_fape_backbone + \
               l_anglenorm + \
               distogram_loss + \
               masked_loss + \
               predict_lddt_loss

        loss = loss * P.Sqrt()(P.ReduceSum()(all_atom_mask[:, 0]))
        return loss, l_fape_side, l_fape_backbone, l_anglenorm, distogram_loss, masked_loss, predict_lddt_loss


class LossNetAssessment(nn.Cell):
    """loss net"""

    def __init__(self, config):
        super(LossNetAssessment, self).__init__()
        self.orign_loss = LossNet(config, train_fold=False)

        self.num_bins = config.heads.distogram.num_bins
        self.cutoff = 15.0
        self.within_cutoff_clip = 0.3
        self.beyond_cutoff_clip = 3.0
        self.beyond_cutoff_weight = 0.2
        self.regressor_idx = 1
        self.regressor_weight = 2.
        self.parallel_mode = _get_parallel_mode()
        self.reducer_flag = self.parallel_mode in (ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL)
        if self.reducer_flag:
            self.allreduce = P.AllReduce()
            self.device_num = D.get_group_size()

        self.reg_loss_distogram = RegressionLosses(first_break=2., last_break=22.,
                                                   num_bins=config.heads.distogram.num_bins, bin_shift=True,
                                                   charbonnier_eps=0.1, reducer_flag=self.reducer_flag)
        self.reg_loss_lddt = RegressionLosses(first_break=0., last_break=1.,
                                              num_bins=config.heads.predicted_lddt.num_bins, bin_shift=False,
                                              charbonnier_eps=1e-5, reducer_flag=self.reducer_flag)

        self.binary_focal_loss = BinaryFocal(alpha=0.25, gamma=1., feed_in=False, not_focal=False)
        self.softmax_focal_loss_lddt = MultiClassFocal(num_class=config.heads.predicted_lddt.num_bins,
                                                       gamma=1., e=0.1, neighbors=2, not_focal=False,
                                                       reducer_flag=self.reducer_flag)
        self.softmax_focal_loss_distogram = MultiClassFocal(num_class=config.heads.distogram.num_bins,
                                                            gamma=1., e=0.1, neighbors=2, not_focal=False,
                                                            reducer_flag=self.reducer_flag)
        self.cameo_focal_loss = BinaryFocal(alpha=0.2, gamma=0.5, feed_in=True, not_focal=False)
        self.distogram_one_hot = nn.OneHot(depth=self.num_bins, axis=-1)
        self.breaks = mnp.linspace(2.0, 22.0, self.num_bins)
        self.width = self.breaks[1] - self.breaks[0]
        self.centers = self.breaks + 0.5 * self.width

    def distogram_loss(self, logits, bin_edges, pseudo_beta, pseudo_beta_mask):
        """Log loss of a distogram."""
        positions = pseudo_beta
        mask = pseudo_beta_mask

        sq_breaks = mnp.square(bin_edges)
        dist_t = mnp.square(mnp.expand_dims(positions, axis=-2) - mnp.expand_dims(positions, axis=-3))
        dist2 = P.ReduceSum(True)(dist_t.astype(ms.float32), -1)
        aa = (dist2 > sq_breaks).astype(ms.float32)

        square_mask = mnp.expand_dims(mask, axis=-2) * mnp.expand_dims(mask, axis=-1)
        probs = nn.Softmax(-1)(logits)
        dmat_pred = mnp.sum(probs * mnp.reshape(self.centers, (1, 1, -1)), -1)
        dist2 = dist2[..., 0]
        dmat_true = mnp.sqrt(1e-6 + dist2)

        within_cutoff_mask = F.cast(dmat_true < self.cutoff, mnp.float32)
        within_cutoff_mask *= (1. - mnp.eye(within_cutoff_mask.shape[1]))
        beyond_cutoff_mask = F.cast(dmat_true > self.cutoff, mnp.float32)
        beyond_cutoff_mask *= self.beyond_cutoff_weight

        true_bins = P.ReduceSum()(aa, -1)
        true_bins = true_bins.astype(ms.int32)

        nres, nres, nbins = logits.shape
        logits = mnp.reshape(logits, (-1, nbins))
        labels = self.distogram_one_hot(true_bins)
        labels = mnp.reshape(labels, (-1, nbins))

        error = self.softmax_focal_loss_distogram(logits, labels)
        error = mnp.reshape(error, (nres, nres))
        focal_error = within_cutoff_mask * error + beyond_cutoff_mask * error

        focal_loss = (P.ReduceSum()(focal_error * square_mask, (-2, -1)) /
                      (1e-6 + P.ReduceSum()(square_mask.astype(ms.float32), (-2, -1))))

        error_tuple = self.reg_loss_distogram(dmat_pred, dmat_true)
        regression_error = error_tuple[1]

        regression_error_clip_within = mnp.clip(regression_error, self.within_cutoff_clip,
                                                20.) - self.within_cutoff_clip
        regression_error_clip_beyond = mnp.clip(regression_error, self.beyond_cutoff_clip,
                                                20.) - self.beyond_cutoff_clip

        regression_error = regression_error_clip_within * within_cutoff_mask + regression_error_clip_beyond \
                           * beyond_cutoff_mask

        square_mask_off_diagonal = square_mask * (1 - mnp.eye(square_mask.shape[1]))

        regression_loss = (P.ReduceSum()(regression_error * square_mask_off_diagonal, (-2, -1)) /
                           (1e-6 + P.ReduceSum()(square_mask_off_diagonal.astype(ms.float32), (-2, -1))))

        loss = focal_loss + self.regressor_weight * regression_loss

        dist_loss = loss, focal_loss, regression_loss, dmat_true

        return dist_loss

    def construct(self, distogram_logits, bin_edges, pseudo_beta, pseudo_beta_mask,
                  atom37_atom_exists, all_atom_mask, true_msa, bert_mask,
                  final_atom14_positions, residue_index, aatype, residx_atom14_to_atom37, lower_bound, upper_bound,
                  seq_mask, atomtype_radius, final_affines, angles_sin_cos,
                  um_angles_sin_cos, backbone_affine_tensor, backbone_affine_mask, atom14_gt_positions,
                  atom14_alt_gt_positions, atom14_atom_is_ambiguous, atom14_gt_exists, atom14_atom_exists,
                  atom14_alt_gt_exists, final_atom_positions, all_atom_positions, predicted_lddt_logits, traj,
                  rigidgroups_gt_frames, rigidgroups_gt_exists, rigidgroups_alt_gt_frames,
                  pred_frames, pred_positions, sin_cos_true_chi, torsion_angle_mask,
                  decoy_pseudo_beta, decoy_pseudo_beta_mask, decoy_predicted_lddt_logits, plddt_dist, pred_mask2d):
        """construct"""
        _, l_fape_side, l_fape_backbone, l_anglenorm, _, _, predict_lddt_loss = self.orign_loss(
            distogram_logits, bin_edges, pseudo_beta, pseudo_beta_mask, None,
            atom37_atom_exists, all_atom_mask, true_msa, None, bert_mask,
            final_atom14_positions, residue_index, aatype, residx_atom14_to_atom37, lower_bound, upper_bound,
            seq_mask, atomtype_radius, final_affines, None, None, angles_sin_cos,
            um_angles_sin_cos, backbone_affine_tensor, backbone_affine_mask, atom14_gt_positions,
            atom14_alt_gt_positions, atom14_atom_is_ambiguous, atom14_gt_exists, atom14_atom_exists,
            atom14_alt_gt_exists, final_atom_positions, all_atom_positions, predicted_lddt_logits, traj,
            rigidgroups_gt_frames, rigidgroups_gt_exists, rigidgroups_alt_gt_frames,
            pred_frames, pred_positions, sin_cos_true_chi, torsion_angle_mask, 1.0, 1.0)

        fold_loss = l_fape_side + l_fape_backbone + l_anglenorm + predict_lddt_loss

        lddt_cb = local_distance_difference_test(
            predicted_points=decoy_pseudo_beta[None, ...],
            true_points=pseudo_beta[None, ...],
            true_points_mask=decoy_pseudo_beta_mask[None, ..., None].astype(mnp.float32),
            cutoff=15.,
            per_residue=True)[0]
        lddt_cb = F.stop_gradient(lddt_cb)

        distogram_loss, distogram_focal_loss, distogram_regression_loss, dmat_true = self.distogram_loss(
            distogram_logits, bin_edges, pseudo_beta, pseudo_beta_mask)

        mask1d = decoy_pseudo_beta_mask
        mask2d = mnp.expand_dims(mask1d, 1) * mnp.expand_dims(mask1d, 0)
        error_tuple = self.reg_loss_lddt(plddt_dist, lddt_cb)
        plddt2_error = error_tuple[self.regressor_idx]

        plddt2_regression_loss = mnp.sum(plddt2_error * mask1d) / (mnp.sum(mask1d) + 1e-8)
        plddt2_loss = self.regressor_weight * plddt2_regression_loss

        true_mask2d = P.Cast()(dmat_true < self.cutoff, ms.float32)
        mask_error = self.binary_focal_loss(mnp.reshape(pred_mask2d, (-1,)), mnp.reshape(true_mask2d, (-1,)))
        mask_error = mnp.reshape(mask_error, true_mask2d.shape)
        mask_loss = mnp.sum(mask_error * mask2d) / (mnp.sum(mask2d) + 1e-6)
        confidence_pred = mnp.sum(plddt_dist * mask1d) / (mnp.sum(mask1d) + 1e-6)
        confidence_gt = mnp.sum(lddt_cb * mask1d) / (mnp.sum(mask1d) + 1e-6)
        confidence_loss = nn.MSELoss()(confidence_pred, confidence_gt)
        confidence_loss = mnp.sqrt(confidence_loss + 1e-5)

        cameo_label = F.cast(lddt_cb < 0.6, mnp.float32)
        cameo_scale = decoy_predicted_lddt_logits[:, 0]
        cameo_shift = decoy_predicted_lddt_logits[:, 1]
        cameo_scale = 5. * P.Tanh()(cameo_scale / 5.)
        decoy_cameo_logit = -F.exp(cameo_scale) * (plddt_dist + cameo_shift - 0.6)
        cameo_error = self.cameo_focal_loss(decoy_cameo_logit, cameo_label)
        cameo_loss = mnp.sum(cameo_error * mask1d) / (mnp.sum(mask1d) + 1e-6)

        score_loss = distogram_loss + plddt2_loss + mask_loss + 2.0 * confidence_loss + 2.0 * cameo_loss

        loss = 0.5 * fold_loss + score_loss

        seq_len = F.cast(P.ReduceSum()(pseudo_beta_mask), mnp.float32)
        loss_weight = mnp.power(seq_len, 0.5)
        if self.reducer_flag:
            loss_weight_sum = self.allreduce(loss_weight) / self.device_num
            loss_weight = loss_weight / loss_weight_sum
        loss_weight *= 64.

        loss = loss * loss_weight

        loss_all = loss, l_fape_side, l_fape_backbone, l_anglenorm, predict_lddt_loss, \
                   distogram_focal_loss, distogram_regression_loss, plddt2_regression_loss, mask_loss, \
                   confidence_loss, cameo_loss

        return loss_all


class RegressionLosses(nn.Cell):
    """Return various regressor losses"""

    def __init__(self, first_break, last_break, num_bins, bin_shift=True, beta=0.99, charbonnier_eps=1e-5,
                 reducer_flag=False):
        super(RegressionLosses, self).__init__()

        self.beta = beta
        self.charbonnier_eps = charbonnier_eps

        self.first_break = first_break
        self.last_break = last_break
        self.num_bins = num_bins
        self.breaks = mnp.linspace(self.first_break, self.last_break, self.num_bins)
        self.width = self.breaks[1] - self.breaks[0]

        bin_width = 2
        start_n = 1
        stop = self.num_bins * 2
        centers = mnp.divide(mnp.arange(start=start_n, stop=stop, step=bin_width), self.num_bins * 2.0)
        self.centers = centers / (self.last_break - self.first_break) + self.first_break

        if bin_shift:
            centers = mnp.linspace(self.first_break, self.last_break, self.num_bins)
            self.centers = centers + 0.5 * self.width
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()
        self.bmse = BalancedMSE(first_break, last_break, num_bins, beta, reducer_flag)

    def construct(self, prediction, target):
        """construct"""
        target = mnp.clip(target, self.centers[0], self.centers[-1])

        mse = self.mse(prediction, target)
        mae = self.mae(prediction, target)
        bmse = self.bmse(prediction, target)
        return [mse, mae, bmse]
