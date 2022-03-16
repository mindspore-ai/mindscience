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
import mindspore.nn as nn
import mindspore.numpy as mnp
from mindspore import Tensor
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from module import all_atom
from common import residue_constants
from common.utils import invert_point, mask_mean, from_tensor

def softmax_cross_entropy(logits, labels):
    """Computes softmax cross entropy given logits and one-hot class labels."""
    loss = -mnp.sum(labels * nn.LogSoftmax()(logits), axis=-1)
    return mnp.asarray(loss)


def sigmoid_cross_entropy(logits, labels):
    """Computes sigmoid cross entropy given logits and multiple class labels."""
    log_p = nn.LogSigmoid()(logits)
    log_not_p = nn.LogSigmoid()(-logits)
    loss = -labels * log_p - (1. - labels) * log_not_p
    return mnp.asarray(loss)


def compute_renamed_ground_truth(atom14_gt_positions,
                                 atom14_alt_gt_positions,
                                 atom14_atom_is_ambiguous,
                                 atom14_gt_exists,
                                 atom14_pred_positions,
                                 atom14_alt_gt_exists):
    """Find optimal renaming of ground truth based on the predicted positions.

    Jumper et al. (2021) Suppl. Alg. 26 "renameSymmetricGroundTruthAtoms"

    This renamed ground truth is then used for all losses,
    such that each loss moves the atoms in the same direction.
    Shape (N).

    Args:
        atom14_gt_positions: Ground truth positions.
        atom14_alt_gt_positions: Ground truth positions with renaming swaps.
        atom14_atom_is_ambiguous: 1.0 for atoms that are affected by
            renaming swaps.
        atom14_gt_exists: Mask for which atoms exist in ground truth.
        atom14_alt_gt_exists: Mask for which atoms exist in ground truth
            after renaming.
        atom14_atom_exists: Mask for whether each atom is part of the given
            amino acid type.
        atom14_pred_positions: Array of atom positions in global frame with shape
            (N, 14, 3).
    Returns:
        alt_naming_is_better: Array with 1.0 where alternative swap is better.
        renamed_atom14_gt_positions: Array of optimal ground truth positions
            after renaming swaps are performed.
        renamed_atom14_gt_exists: Mask after renaming swap is performed.
    """

    alt_naming_is_better = all_atom.find_optimal_renaming(atom14_gt_positions,
                                                          atom14_alt_gt_positions,
                                                          atom14_atom_is_ambiguous,
                                                          atom14_gt_exists,
                                                          atom14_pred_positions)

    renamed_atom14_gt_positions = ((1. - alt_naming_is_better[:, None, None]) * atom14_gt_positions +
                                   alt_naming_is_better[:, None, None] * atom14_alt_gt_positions)

    renamed_atom14_gt_mask = ((1. - alt_naming_is_better[:, None]) * atom14_gt_exists +
                              alt_naming_is_better[:, None] * atom14_alt_gt_exists)

    return alt_naming_is_better, renamed_atom14_gt_positions, renamed_atom14_gt_mask


def find_structural_violations(atom14_atom_exists, residue_index, aatype, residx_atom14_to_atom37,
                               atom14_pred_positions, violation_tolerance_factor, clash_overlap_tolerance,
                               lower_bound, upper_bound, atomtype_radius, c_one_hot, n_one_hot, dists_mask_i,
                               cys_sg_idx):
    """Computes several checks for structural violations."""

    # Compute between residue backbone violations of bonds and angles.
    c_n_loss_mean, ca_c_n_loss_mean, c_n_ca_loss_mean, per_residue_loss_sum, per_residue_violation_mask = \
        all_atom.between_residue_bond_loss(
            pred_atom_positions=atom14_pred_positions,
            pred_atom_mask=atom14_atom_exists.astype(mnp.float32),
            residue_index=residue_index.astype(mnp.float32),
            aatype=aatype,
            tolerance_factor_soft=violation_tolerance_factor,
            tolerance_factor_hard=violation_tolerance_factor)

    # Compute the Van der Waals radius for every atom
    # (the first letter of the atom name is the element type).
    # Shape: (N, 14).
    atom14_atom_radius = atom14_atom_exists * P.Gather()(atomtype_radius, residx_atom14_to_atom37, 0)

    # Compute the between residue clash loss.
    mean_loss, clashes_per_atom_loss_sum, per_atom_clash_mask = all_atom.between_residue_clash_loss(
        atom14_pred_positions=atom14_pred_positions,
        atom14_atom_exists=atom14_atom_exists,
        atom14_atom_radius=atom14_atom_radius,
        residue_index=residue_index,
        c_one_hot=c_one_hot,
        n_one_hot=n_one_hot,
        overlap_tolerance_soft=clash_overlap_tolerance,
        overlap_tolerance_hard=clash_overlap_tolerance,
        cys_sg_idx=cys_sg_idx
    )

    # Compute all within-residue violations (clashes,
    # bond length and angle violations).
    atom14_dists_lower_bound = P.Gather()(lower_bound, aatype, 0)
    atom14_dists_upper_bound = P.Gather()(upper_bound, aatype, 0)
    per_atom_loss_sum, per_atom_violations = all_atom.within_residue_violations(
        atom14_pred_positions=atom14_pred_positions,
        atom14_atom_exists=atom14_atom_exists,
        atom14_dists_lower_bound=atom14_dists_lower_bound,
        atom14_dists_upper_bound=atom14_dists_upper_bound,
        tighten_bounds_for_loss=0.0,
        dists_mask_i=dists_mask_i)

    # Combine them to a single per-residue violation mask (used later for LDDT).
    per_residue_violations_mask = mnp.max(mnp.stack([per_residue_violation_mask, mnp.max(per_atom_clash_mask, axis=-1),
                                                     mnp.max(per_atom_violations, axis=-1)]), axis=0)
    bonds_c_n_loss_mean = c_n_loss_mean
    angles_ca_c_n_loss_mean = ca_c_n_loss_mean
    angles_c_n_ca_loss_mean = c_n_ca_loss_mean
    connections_per_residue_loss_sum = per_residue_loss_sum
    connections_per_residue_violation_mask = per_residue_violation_mask
    clashes_mean_loss = mean_loss
    clashes_per_atom_loss_sum = clashes_per_atom_loss_sum
    clashes_per_atom_clash_mask = per_atom_clash_mask
    per_atom_loss_sum = per_atom_loss_sum
    per_atom_violations = per_atom_violations
    total_per_residue_violations_mask = per_residue_violations_mask
    return (bonds_c_n_loss_mean, angles_ca_c_n_loss_mean, angles_c_n_ca_loss_mean, connections_per_residue_loss_sum,
            connections_per_residue_violation_mask, clashes_mean_loss, clashes_per_atom_loss_sum,
            clashes_per_atom_clash_mask, per_atom_loss_sum, per_atom_violations, total_per_residue_violations_mask)


def compute_violation_metrics(atom14_atom_exists, residue_index, seq_mask, connections_per_residue_violation_mask,
                              per_atom_violations, total_per_residue_violations_mask, atom14_pred_positions,
                              clashes_per_atom_clash_mask):
    """Compute several metrics to assess the structural violations."""

    extreme_ca_ca_violations = all_atom.extreme_ca_ca_distance_violations(
        pred_atom_positions=atom14_pred_positions,
        pred_atom_mask=atom14_atom_exists.astype(mnp.float32),
        residue_index=residue_index.astype(mnp.float32))
    violations_extreme_ca_ca_distance = extreme_ca_ca_violations
    violations_between_residue_bond = mask_mean(mask=seq_mask, value=connections_per_residue_violation_mask)
    violations_between_residue_clash = mask_mean(mask=seq_mask, value=mnp.max(clashes_per_atom_clash_mask, axis=-1))
    violations_within_residue = mask_mean(mask=seq_mask, value=mnp.max(per_atom_violations, axis=-1))
    violations_per_residue = mask_mean(mask=seq_mask, value=total_per_residue_violations_mask)
    return (violations_extreme_ca_ca_distance, violations_between_residue_bond, violations_between_residue_clash,
            violations_within_residue, violations_per_residue)


def backbone_loss(traj, backbone_affine_tensor, backbone_affine_mask, fape_clamp_distance, fape_loss_unit_distance,
                  use_clamped_fape):
    """Backbone FAPE Loss.

    Jumper et al. (2021) Suppl. Alg. 20 "StructureModule" line 17

    Args:
      ret: Dictionary to write outputs into, needs to contain 'loss'.
      batch: Batch, needs to contain 'backbone_affine_tensor',
        'backbone_affine_mask'.
      value: Dictionary containing structure module output, needs to contain
        'traj', a trajectory of rigids.
      config: Configuration of loss, should contain 'fape.clamp_distance' and
        'fape.loss_unit_distance'.
    """
    _, rotation, translation = from_tensor(traj)
    pred_frames = [rotation[0][0], rotation[0][1], rotation[0][2],
                   rotation[1][0], rotation[1][1], rotation[1][2],
                   rotation[2][0], rotation[2][1], rotation[2][2],
                   translation[0], translation[1], translation[2]]
    pred_positions = [translation[0], translation[1], translation[2]]

    _, rotation_gt, translation_gt = from_tensor(backbone_affine_tensor)
    target_frames = [rotation_gt[0][0], rotation_gt[0][1], rotation_gt[0][2],
                     rotation_gt[1][0], rotation_gt[1][1], rotation_gt[1][2],
                     rotation_gt[2][0], rotation_gt[2][1], rotation_gt[2][2],
                     translation_gt[0], translation_gt[1], translation_gt[2]]
    target_positions = [translation_gt[0], translation_gt[1], translation_gt[2]]

    frames_mask = backbone_affine_mask
    positions_mask = backbone_affine_mask

    fape_loss_clamp, fape_loss_no_clamp = all_atom.frame_aligned_point_error_map(pred_frames,
                                                                                 target_frames,
                                                                                 frames_mask,
                                                                                 pred_positions,
                                                                                 target_positions,
                                                                                 positions_mask,
                                                                                 fape_clamp_distance,
                                                                                 fape_loss_unit_distance)
    fape_loss = (fape_loss_clamp * use_clamped_fape + fape_loss_no_clamp * (1 - use_clamped_fape))
    no_clamp = fape_loss_no_clamp[-1]
    fape = fape_loss[-1]
    loss = mnp.mean(fape_loss)
    return fape, loss, no_clamp


def sidechain_loss(alt_naming_is_better, rigidgroups_gt_frames, rigidgroups_alt_gt_frames, rigidgroups_gt_exists,
                   renamed_atom14_gt_positions, renamed_atom14_gt_exists, sidechain_atom_clamp_distance,
                   sidechain_length_scale, pred_frames, pred_positions):
    """All Atom FAPE Loss using renamed rigids."""
    # Rename Frames
    # Jumper et al. (2021) Suppl. Alg. 26 "renameSymmetricGroundTruthAtoms" line 7
    renamed_gt_frames = ((1. - alt_naming_is_better[:, None, None]) * rigidgroups_gt_frames
                         + alt_naming_is_better[:, None, None] * rigidgroups_alt_gt_frames)
    flat_gt_frames = mnp.moveaxis(mnp.reshape(renamed_gt_frames, [-1, 12]), -1, 0)
    # flat_gt_frames_rots = flat_gt_frames[:9]
    # flat_gt_frames_vecs = flat_gt_frames[9:]
    flat_frames_mask = mnp.reshape(rigidgroups_gt_exists, [-1])

    flat_gt_positions_t = mnp.reshape(renamed_atom14_gt_positions, [-1, 3])
    flat_gt_positions = [flat_gt_positions_t[..., 0], flat_gt_positions_t[..., 1], flat_gt_positions_t[..., 2]]
    flat_positions_mask = mnp.reshape(renamed_atom14_gt_exists, [-1])

    # Compute frame_aligned_point_error score for the final layer.
    flat_pred_frames = mnp.reshape(pred_frames[:, -1, ...], [12, -1])
    flat_pred_positions = mnp.reshape(pred_positions[:, -1, ...], [3, -1])

    # FAPE Loss on sidechains
    fape = all_atom.frame_aligned_point_error(
        pred_frames=flat_pred_frames,
        target_frames=flat_gt_frames,
        frames_mask=flat_frames_mask,
        pred_positions=flat_pred_positions,
        target_positions=flat_gt_positions,
        positions_mask=flat_positions_mask,
        l1_clamp_distance=sidechain_atom_clamp_distance,
        length_scale=sidechain_length_scale)
    loss = fape
    return fape, loss


def supervised_chi_loss(sequence_mask, aatype, sin_cos_true_chi, torsion_angle_mask, sin_cos_pred_chi,
                        sin_cos_unnormalized_pred, chi_weight, angle_norm_weight, chi_pi_periodic):
    """Computes loss for direct chi angle supervision.

    Jumper et al. (2021) Suppl. Alg. 27 "torsionAngleLoss"

    Args:
      ret: Dictionary to write outputs into, needs to contain 'loss'.
      batch: Batch, needs to contain 'seq_mask', 'chi_mask', 'chi_angles'.
      value: Dictionary containing structure module output, needs to contain
        value['sidechains']['angles_sin_cos'] for angles and
        value['sidechains']['unnormalized_angles_sin_cos'] for unnormalized
        angles.
      config: Configuration of loss, should contain 'chi_weight' and
        'angle_norm_weight', 'angle_norm_weight' scales angle norm term,
        'chi_weight' scales torsion term.
    """
    eps = 1e-6

    num_res = sequence_mask.shape[0]
    chi_mask = torsion_angle_mask
    pred_angles = mnp.reshape(sin_cos_pred_chi, [-1, num_res, 7, 2])
    pred_angles = pred_angles[:, :, 3:]

    residue_type_one_hot = nn.OneHot(depth=21)(aatype)[None]
    chi_pi_periodic = mnp.matmul(residue_type_one_hot, chi_pi_periodic)

    # This is -1 if chi is pi-periodic and +1 if it's 2pi-periodic
    shifted_mask = (1 - 2 * chi_pi_periodic)[..., None]
    sin_cos_true_chi_shifted = shifted_mask * sin_cos_true_chi

    sq_chi_error = mnp.sum(mnp.square(sin_cos_true_chi - pred_angles), -1)
    sq_chi_error_shifted = mnp.sum(mnp.square(sin_cos_true_chi_shifted - pred_angles), -1)
    sq_chi_error = mnp.minimum(sq_chi_error, sq_chi_error_shifted)

    sq_chi_loss = P.ReduceSum()(chi_mask[None] * sq_chi_error, (0, 1, 2)) / \
                  (P.ReduceSum()(chi_mask[None], (0, 1, 2)) * 8 + 1e-10)

    loss = chi_weight * sq_chi_loss
    unnormed_angles = mnp.reshape(sin_cos_unnormalized_pred[-1], [-1, num_res, 7, 2])
    angle_norm = mnp.sqrt(mnp.sum(mnp.square(unnormed_angles), axis=-1) + eps)
    norm_error = mnp.abs(angle_norm - 1.)
    angle_norm_loss = P.ReduceSum()(sequence_mask[None, :, None] * norm_error, (0, 1, 2)) / \
                      (P.ReduceSum()(sequence_mask[None, :, None].astype(ms.float32), (0, 1, 2)) * 7 + 1e-10)

    loss += angle_norm_weight * angle_norm_loss
    return loss


def structural_violation_loss(ret, batch, value, config):
    """Computes loss for structural violations."""

    # Put all violation losses together to one large loss.
    violations = value['violations']
    num_atoms = mnp.sum(batch['atom14_atom_exists']).astype(mnp.float32)
    ret['loss'] += (config.structural_violation_loss_weight *
                    (violations['between_residues']['bonds_c_n_loss_mean'] +
                     violations['between_residues']['angles_ca_c_n_loss_mean'] +
                     violations['between_residues']['angles_c_n_ca_loss_mean'] +
                     mnp.sum(violations['between_residues']['clashes_per_atom_loss_sum'] +
                             violations['within_residues']['per_atom_loss_sum']) / (1e-6 + num_atoms)))


def lddt_loss(predicted_points, true_points, true_points_mask, cutoff=15, per_residue=False):
    """Compute true and predicted distance matrices."""
    dmat_true = mnp.sqrt(1e-10 + mnp.sum((true_points[:, :, None] - true_points[:, None, :]) ** 2, axis=-1))

    dmat_predicted = mnp.sqrt(1e-10 + mnp.sum((predicted_points[:, :, None] - predicted_points[:, None, :]) ** 2,
                                              axis=-1))

    dists_to_score = ((dmat_true < cutoff).astype(mnp.float32) * true_points_mask *
                      mnp.transpose(true_points_mask, [0, 2, 1]) *
                      (1. - mnp.eye(dmat_true.shape[1]))  # Exclude self-interaction.
                      )

    # Shift unscored distances to be far away.
    dist_l1 = mnp.abs(dmat_true - dmat_predicted)

    # True lDDT uses a number of fixed bins.
    # We ignore the physical plausibility correction to lDDT, though.
    score = 0.25 * ((dist_l1 < 0.5).astype(mnp.float32) +
                    (dist_l1 < 1.0).astype(mnp.float32) +
                    (dist_l1 < 2.0).astype(mnp.float32) +
                    (dist_l1 < 4.0).astype(mnp.float32))

    # Normalize over the appropriate axes.
    reduce_axes = (-1,) if per_residue else (-2, -1)
    norm = 1. / (1e-10 + mnp.sum(dists_to_score, axis=reduce_axes))
    score = norm * (1e-10 + mnp.sum(dists_to_score * score, axis=reduce_axes))

    return score


class LossNet(nn.Cell):
    """loss net"""
    def __init__(self, config):
        super(LossNet, self).__init__()
        self.config = config
        self.num_res = config.data.eval.crop_size
        self.num_bins = config.model.heads.distogram.num_bins
        self.resolution = config.resolution
        self.distogram_weight = config.model.heads.distogram.weight
        self.distogram_one_hot = nn.OneHot(depth=self.num_bins, axis=-1)
        self.exp_min_resolution = config.model.heads.experimentally_resolved.min_resolution
        self.exp_max_resolution = config.model.heads.experimentally_resolved.max_resolution
        self.exp_res_filter_by_resolution = config.model.heads.experimentally_resolved.filter_by_resolution
        self.experimentally_weight = config.model.heads.experimentally_resolved.weight
        self.exp_res_mask = Tensor(1, ms.float32)\
            if not self.exp_res_filter_by_resolution or\
               (self.exp_min_resolution <= self.resolution <= self.exp_max_resolution) else Tensor(0, ms.float32)

        self.ael_min_resolution = config.model.heads.predicted_aligned_error.min_resolution
        self.ael_max_resolution = config.model.heads.predicted_aligned_error.max_resolution
        self.ael_res_filter_by_resolution = config.model.heads.predicted_aligned_error.filter_by_resolution
        self.ael_res_mask = Tensor(1, ms.float32)\
            if not self.ael_res_filter_by_resolution or\
               (self.ael_min_resolution <= self.resolution <= self.ael_max_resolution) else Tensor(0, ms.float32)
        self.aligned_one_hot = nn.OneHot(depth=config.model.heads.predicted_aligned_error.num_bins)

        self.plddt_min_resolution = config.model.heads.predicted_lddt.min_resolution
        self.plddt_max_resolution = config.model.heads.predicted_lddt.max_resolution
        self.plddt_res_filter_by_resolution = config.model.heads.predicted_lddt.filter_by_resolution
        self.plddt_res_mask = Tensor(1, ms.float32)\
            if not self.plddt_res_filter_by_resolution or\
               (self.plddt_min_resolution <= self.resolution <= self.plddt_max_resolution) else Tensor(0, ms.float32)
        self.plddt_weight = config.model.heads.predicted_lddt.weight

        self.masked_one_hot = nn.OneHot(depth=23, axis=-1)
        self.masked_weight = config.model.heads.masked_msa.weight
        self.sidechain_weight_frac = config.model.heads.structure_module.sidechain.weight_frac
        self.angle_norm_weight = config.model.heads.structure_module.angle_norm_weight
        self.chi_weight = config.model.heads.structure_module.chi_weight
        self.chi_pi_periodic = mnp.asarray(residue_constants.chi_pi_periodic, ms.float32)

        self.violation_tolerance_factor = config.model.heads.structure_module.violation_tolerance_factor
        self.clash_overlap_tolerance = config.model.heads.structure_module.clash_overlap_tolerance
        self.sidechain_atom_clamp_distance = config.model.heads.structure_module.sidechain.atom_clamp_distance
        self.sidechain_length_scale = config.model.heads.structure_module.sidechain.length_scale
        self.fape_clamp_distance = config.model.heads.structure_module.fape.clamp_distance
        self.fape_loss_unit_distance = config.model.heads.structure_module.fape.loss_unit_distance
        self.predicted_lddt_num_bins = config.model.heads.predicted_lddt.num_bins
        self.c_one_hot = nn.OneHot(depth=14)
        self.n_one_hot = nn.OneHot(depth=14)
        self.zeros = Tensor(0, ms.int32)
        self.twos = Tensor(2, ms.int32)
        self.dists_mask_i = mnp.eye(14, 14)
        self.cys_sg_idx = Tensor(5, ms.int32)

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
        errors = softmax_cross_entropy(labels=self.distogram_one_hot(true_bins), logits=logits)
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

        xent = sigmoid_cross_entropy(logits, all_atom_mask)
        loss = P.ReduceSum()(xent * atom_exists) / (1e-8 + P.ReduceSum()(atom_exists.astype(ms.float32)))
        loss = loss * filter_by_solution
        loss *= self.exp_res_mask
        return loss

    def masked_head_loss(self, true_msa, logits, bert_mask):
        """masked_head_loss"""
        errors = softmax_cross_entropy(logits=logits, labels=self.masked_one_hot(true_msa))
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
         connections_per_residue_violation_mask, _, clashes_per_atom_loss_sum,
         clashes_per_atom_clash_mask, per_atom_loss_sum, per_atom_violations, total_per_residue_violations_mask) = \
            find_structural_violations(atom14_atom_exists, residue_index, aatype, residx_atom14_to_atom37,
                                       atom14_pred_positions, self.violation_tolerance_factor,
                                       self.clash_overlap_tolerance, lower_bound, upper_bound, atomtype_radius,
                                       self.c_one_hot(self.twos), self.n_one_hot(self.zeros), self.dists_mask_i,
                                       self.cys_sg_idx)
        num_atoms = P.ReduceSum()(atom14_atom_exists.astype(ms.float32))
        structure_violation_loss = bonds_c_n_loss_mean + angles_ca_c_n_loss_mean + angles_c_n_ca_loss_mean + \
                                   P.ReduceSum()(clashes_per_atom_loss_sum + per_atom_loss_sum) / (1e-6 + num_atoms)
        # Several violation metrics:
        compute_violation_metrics(atom14_atom_exists, residue_index, seq_mask,
                                  connections_per_residue_violation_mask, per_atom_violations,
                                  total_per_residue_violations_mask, atom14_pred_positions,
                                  clashes_per_atom_clash_mask)

        # from structure module result
        _, fape_loss, no_clamp = backbone_loss(traj, backbone_affine_tensor, backbone_affine_mask,
                                               self.fape_clamp_distance, self.fape_loss_unit_distance, use_clamped_fape)

        _, loss_sidechain = sidechain_loss(alt_naming_is_better, rigidgroups_gt_frames, rigidgroups_alt_gt_frames,
                                           rigidgroups_gt_exists, renamed_atom14_gt_positions, renamed_atom14_gt_exists,
                                           self.sidechain_atom_clamp_distance, self.sidechain_length_scale, pred_frames,
                                           pred_positions)
        angle_norm_loss = supervised_chi_loss(seq_mask, aatype, sin_cos_true_chi, torsion_angle_mask,
                                              angles_sin_cos, um_angles_sin_cos, self.chi_weight,
                                              self.angle_norm_weight, self.chi_pi_periodic)
        return fape_loss, loss_sidechain, angle_norm_loss, structure_violation_loss, no_clamp

    def predicted_lddt_loss(self, final_atom_positions, all_atom_positions, all_atom_mask, predicted_lddt_logits,
                            filter_by_solution):
        """predicted_lddt_loss"""
        # Shape (num_res, 37, 3)
        pred_all_atom_pos = final_atom_positions
        # Shape (num_res, 37, 3)
        true_all_atom_pos = all_atom_positions
        # Shape (num_res, 37)

        # Shape (num_res,)
        lddt_ca = lddt_loss(
            # Shape (batch_size, num_res, 3)
            predicted_points=pred_all_atom_pos[None, :, 1, :],
            # Shape (batch_size, num_res, 3)
            true_points=true_all_atom_pos[None, :, 1, :],
            # Shape (batch_size, num_res, 1)
            true_points_mask=all_atom_mask[None, :, 1:2].astype(mnp.float32),
            cutoff=15.,
            per_residue=True)[0]

        lddt_ca = F.stop_gradient(lddt_ca)

        bin_index = mnp.floor(lddt_ca * self.predicted_lddt_num_bins).astype(mnp.int32)

        # protect against out of range for lddt_ca == 1
        bin_index = mnp.minimum(bin_index, self.predicted_lddt_num_bins - 1)
        lddt_ca_one_hot = nn.OneHot(depth=self.predicted_lddt_num_bins)(bin_index)

        # Shape (num_res, num_channel)
        logits = predicted_lddt_logits
        errors = softmax_cross_entropy(labels=lddt_ca_one_hot, logits=logits)

        # Shape (num_res,)
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
        _, rotation_pd, translation_pd = from_tensor(final_affines)
        translation_point_pd = mnp.expand_dims(translation_pd, axis=-2)
        rotation_pd_tensor = mnp.stack(
            [mnp.stack(rotation_pd[0]), mnp.stack(rotation_pd[1]), mnp.stack(rotation_pd[2])])
        # Shape (num_res, 7) true affine
        _, rotation_gt, translation_gt = from_tensor(backbone_affine_tensor)
        translation_point_tr = mnp.expand_dims(translation_gt, axis=-2)
        rotation_gt_tensor = mnp.stack(
            [mnp.stack(rotation_gt[0]), mnp.stack(rotation_gt[1]), mnp.stack(rotation_gt[2])])
        # Shape (num_res)
        mask = backbone_affine_mask
        # Shape (num_res, num_res)
        square_mask = (mask[:, None] * mask[None, :]).astype(ms.float32)
        # (1, num_bins - 1)
        breaks = pae_breaks
        # (1, num_bins)
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

        errors = softmax_cross_entropy(labels=self.aligned_one_hot(true_bins.astype(ms.int32)), logits=logits)

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
        distogram_loss, _ = self.distogram_loss(distogram_logits, bin_edges, pseudo_beta, pseudo_beta_mask)
        distogram_loss = distogram_loss * self.distogram_weight

        masked_loss = self.masked_head_loss(true_msa, masked_logits, bert_mask)
        masked_loss = self.masked_weight * masked_loss

        fape_loss, loss_sidechain, angle_norm_loss, _, _ = \
            self.structure_loss(atom14_gt_positions, atom14_alt_gt_positions, atom14_atom_is_ambiguous,
                                atom14_gt_exists, atom14_atom_exists, final_atom14_positions,
                                atom14_alt_gt_exists, residue_index, aatype, residx_atom14_to_atom37,
                                lower_bound, upper_bound, seq_mask, atomtype_radius, angles_sin_cos,
                                um_angles_sin_cos, traj, backbone_affine_tensor,
                                backbone_affine_mask, rigidgroups_gt_frames, rigidgroups_gt_exists,
                                rigidgroups_alt_gt_frames,
                                pred_frames, pred_positions, sin_cos_true_chi, torsion_angle_mask, use_clamped_fape)

        # experimentally_loss = self.experimentally_loss(experimentally_logits, atom37_atom_exists, all_atom_mask,
        #                                                filter_by_solution)
        # experimentally_loss = experimentally_loss * self.experimentally_weight
        self.experimentally_loss(experimentally_logits, atom37_atom_exists, all_atom_mask, filter_by_solution)

        predict_lddt_loss = self.predicted_lddt_loss(final_atom_positions, all_atom_positions, all_atom_mask,
                                                     predicted_lddt_logits, filter_by_solution)
        predict_lddt_loss = self.plddt_weight * predict_lddt_loss

        self.aligned_error_loss(final_affines, backbone_affine_tensor, backbone_affine_mask, pae_breaks,
                                pae_logits, filter_by_solution)
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
