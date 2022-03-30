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
"""all atom"""

import mindspore as ms
import mindspore.numpy as mnp
from mindspore import nn
from mindspore.ops import operations as P
from commons import residue_constants
from commons.utils import mask_mean

def find_optimal_renaming(
        atom14_gt_positions,  # (N, 14, 3)
        atom14_alt_gt_positions,  # (N, 14, 3)
        atom14_atom_is_ambiguous,  # (N, 14)
        atom14_gt_exists,  # (N, 14)
        atom14_pred_positions,  # (N, 14, 3)
):  # (N):
    """Find optimal renaming for ground truth that maximizes LDDT.

    Jumper et al. (2021) Suppl. Alg. 26
    "renameSymmetricGroundTruthAtoms" lines 1-5

    Args:
      atom14_gt_positions: Ground truth positions in global frame of ground truth.
      atom14_alt_gt_positions: Alternate ground truth positions in global frame of
        ground truth with coordinates of ambiguous atoms swapped relative to
        'atom14_gt_positions'.
      atom14_atom_is_ambiguous: Mask denoting whether atom is among ambiguous
        atoms, see Jumper et al. (2021) Suppl. Table 3
      atom14_gt_exists: Mask denoting whether atom at positions exists in ground
        truth.
      atom14_pred_positions: Predicted positions of atoms in
        global prediction frame
      atom14_atom_exists: Mask denoting whether atom at positions exists for given
        amino acid type

    Returns:
      Float array of shape [N] with 1. where atom14_alt_gt_positions is closer to
      prediction and 0. otherwise
    """

    # Create the pred distance matrix.
    # shape (N, N, 14, 14)
    pred_dists = mnp.sqrt(1e-10 + mnp.sum(
        mnp.square(atom14_pred_positions[:, None, :, None, :] - atom14_pred_positions[None, :, None, :, :]), axis=-1))

    # Compute distances for ground truth with original and alternative names.
    # shape (N, N, 14, 14)
    gt_dists = mnp.sqrt(1e-10 + mnp.sum(
        mnp.square(atom14_gt_positions[:, None, :, None, :] - atom14_gt_positions[None, :, None, :, :]), axis=-1))
    alt_gt_dists = mnp.sqrt(1e-10 + mnp.sum(
        mnp.square(atom14_alt_gt_positions[:, None, :, None, :] - atom14_alt_gt_positions[None, :, None, :, :]),
        axis=-1))

    # Compute LDDT's.
    # shape (N, N, 14, 14)
    lddt = mnp.sqrt(1e-10 + mnp.square(pred_dists - gt_dists))
    alt_lddt = mnp.sqrt(1e-10 + mnp.square(pred_dists - alt_gt_dists))

    # Create a mask for ambiguous atoms in rows vs. non-ambiguous atoms
    # in cols.
    # shape (N ,N, 14, 14)
    mask = (atom14_gt_exists[:, None, :, None] *  # rows
            atom14_atom_is_ambiguous[:, None, :, None] *  # rows
            atom14_gt_exists[None, :, None, :] *  # cols
            (1. - atom14_atom_is_ambiguous[None, :, None, :]))  # cols

    # Aggregate distances for each residue to the non-amibuguous atoms.
    # shape (N)
    per_res_lddt = P.ReduceSum()(mask * lddt, (1, 2, 3))
    alt_per_res_lddt = P.ReduceSum()(mask * alt_lddt, (1, 2, 3))

    # Decide for each residue, whether alternative naming is better.
    # shape (N)
    alt_naming_is_better = (alt_per_res_lddt < per_res_lddt).astype(ms.float32)

    return alt_naming_is_better  # shape (N)


def between_residue_bond_loss(
        pred_atom_positions,  # (N, 37(14), 3)
        pred_atom_mask,  # (N, 37(14))
        residue_index,  # (N)
        aatype,  # (N)
        tolerance_factor_soft=12.0,
        tolerance_factor_hard=12.0
):
    """Flat-bottom loss to penalize structural violations between residues.

    This is a loss penalizing any violation of the geometry around the peptide
    bond between consecutive amino acids. This loss corresponds to
    Jumper et al. (2021) Suppl. Sec. 1.9.11, eq 44, 45.

    Args:
      pred_atom_positions: Atom positions in atom37/14 representation
      pred_atom_mask: Atom mask in atom37/14 representation
      residue_index: Residue index for given amino acid, this is assumed to be
        monotonically increasing.
      aatype: Amino acid type of given residue
      tolerance_factor_soft: soft tolerance factor measured in standard deviations
        of pdb distributions
      tolerance_factor_hard: hard tolerance factor measured in standard deviations
        of pdb distributions

    Returns:
      Dict containing:
        * 'c_n_loss_mean': Loss for peptide bond length violations
        * 'ca_c_n_loss_mean': Loss for violations of bond angle around C spanned
            by CA, C, N
        * 'c_n_ca_loss_mean': Loss for violations of bond angle around N spanned
            by C, N, CA
        * 'per_residue_loss_sum': sum of all losses for each residue
        * 'per_residue_violation_mask': mask denoting all residues with violation
            present.
    """

    # Get the positions of the relevant backbone atoms.
    this_ca_pos = pred_atom_positions[:-1, 1, :]  # (N - 1, 3)
    this_ca_mask = pred_atom_mask[:-1, 1]  # (N - 1)
    this_c_pos = pred_atom_positions[:-1, 2, :]  # (N - 1, 3)
    this_c_mask = pred_atom_mask[:-1, 2]  # (N - 1)
    next_n_pos = pred_atom_positions[1:, 0, :]  # (N - 1, 3)
    next_n_mask = pred_atom_mask[1:, 0]  # (N - 1)
    next_ca_pos = pred_atom_positions[1:, 1, :]  # (N - 1, 3)
    next_ca_mask = pred_atom_mask[1:, 1]  # (N - 1)
    has_no_gap_mask = ((residue_index[1:] - residue_index[:-1]) == 1.0).astype(ms.float32)

    # Compute loss for the C--N bond.
    c_n_bond_length = mnp.sqrt(1e-6 + mnp.sum(mnp.square(this_c_pos - next_n_pos), axis=-1))

    # The C-N bond to proline has slightly different length because of the ring.
    next_is_proline = (aatype[1:] == residue_constants.resname_to_idx['PRO']).astype(ms.float32)
    gt_length = ((1. - next_is_proline) * residue_constants.between_res_bond_length_c_n[0]
                 + next_is_proline * residue_constants.between_res_bond_length_c_n[1])
    gt_stddev = ((1. - next_is_proline) * residue_constants.between_res_bond_length_stddev_c_n[0] +
                 next_is_proline * residue_constants.between_res_bond_length_stddev_c_n[1])
    c_n_bond_length_error = mnp.sqrt(1e-6 + mnp.square(c_n_bond_length - gt_length))
    c_n_loss_per_residue = nn.ReLU()(c_n_bond_length_error - tolerance_factor_soft * gt_stddev)
    mask = this_c_mask * next_n_mask * has_no_gap_mask
    c_n_loss_mean = mnp.sum(mask * c_n_loss_per_residue) / (mnp.sum(mask) + 1e-6)
    c_n_violation_mask = mask * (c_n_bond_length_error > (tolerance_factor_hard * gt_stddev))

    # Compute loss for the angles.
    ca_c_bond_length = mnp.sqrt(1e-6 + mnp.sum(mnp.square(this_ca_pos - this_c_pos), axis=-1))
    n_ca_bond_length = mnp.sqrt(1e-6 + mnp.sum(mnp.square(next_n_pos - next_ca_pos), axis=-1))

    c_ca_unit_vec = (this_ca_pos - this_c_pos) / ca_c_bond_length[:, None]
    c_n_unit_vec = (next_n_pos - this_c_pos) / c_n_bond_length[:, None]
    n_ca_unit_vec = (next_ca_pos - next_n_pos) / n_ca_bond_length[:, None]

    ca_c_n_cos_angle = mnp.sum(c_ca_unit_vec * c_n_unit_vec, axis=-1)
    gt_angle = residue_constants.between_res_cos_angles_ca_c_n[0]
    gt_stddev = residue_constants.between_res_bond_length_stddev_c_n[0]
    ca_c_n_cos_angle_error = mnp.sqrt(1e-6 + mnp.square(ca_c_n_cos_angle - gt_angle))
    ca_c_n_loss_per_residue = nn.ReLU()(ca_c_n_cos_angle_error - tolerance_factor_soft * gt_stddev)
    mask = this_ca_mask * this_c_mask * next_n_mask * has_no_gap_mask
    ca_c_n_loss_mean = mnp.sum(mask * ca_c_n_loss_per_residue) / (mnp.sum(mask) + 1e-6)
    ca_c_n_violation_mask = mask * (ca_c_n_cos_angle_error > (tolerance_factor_hard * gt_stddev))

    c_n_ca_cos_angle = mnp.sum((-c_n_unit_vec) * n_ca_unit_vec, axis=-1)
    gt_angle = residue_constants.between_res_cos_angles_c_n_ca[0]
    gt_stddev = residue_constants.between_res_cos_angles_c_n_ca[1]
    c_n_ca_cos_angle_error = mnp.sqrt(1e-6 + mnp.square(c_n_ca_cos_angle - gt_angle))
    c_n_ca_loss_per_residue = nn.ReLU()(c_n_ca_cos_angle_error - tolerance_factor_soft * gt_stddev)
    mask = this_c_mask * next_n_mask * next_ca_mask * has_no_gap_mask
    c_n_ca_loss_mean = mnp.sum(mask * c_n_ca_loss_per_residue) / (mnp.sum(mask) + 1e-6)
    c_n_ca_violation_mask = mask * (c_n_ca_cos_angle_error > (tolerance_factor_hard * gt_stddev))

    # Compute a per residue loss (equally distribute the loss to both
    # neighbouring residues).
    per_residue_loss_sum = c_n_loss_per_residue + ca_c_n_loss_per_residue + c_n_ca_loss_per_residue
    per_residue_loss_sum = 0.5 * (mnp.pad(per_residue_loss_sum, [[0, 1]]) + mnp.pad(per_residue_loss_sum, [[1, 0]]))

    # Compute hard violations.
    per_residue_violation_mask = mnp.max(mnp.stack([c_n_violation_mask, ca_c_n_violation_mask, c_n_ca_violation_mask]),
                                         axis=0)
    per_residue_violation_mask = mnp.maximum(mnp.pad(per_residue_violation_mask, [[0, 1]]),
                                             mnp.pad(per_residue_violation_mask, [[1, 0]]))

    return c_n_loss_mean, ca_c_n_loss_mean, c_n_ca_loss_mean, per_residue_loss_sum, per_residue_violation_mask


def between_residue_clash_loss(
        atom14_pred_positions,  # (N, 14, 3)
        atom14_atom_exists,  # (N, 14)
        atom14_atom_radius,  # (N, 14)
        residue_index,  # (N)
        c_one_hot,
        n_one_hot,
        overlap_tolerance_soft,
        overlap_tolerance_hard,
        cys_sg_idx):
    """Loss to penalize steric clashes between residues.

    This is a loss penalizing any steric clashes due to non bonded atoms in
    different peptides coming too close. This loss corresponds to the part with
    different residues of
    Jumper et al. (2021) Suppl. Sec. 1.9.11, eq 46.

    Args:
      atom14_pred_positions: Predicted positions of atoms in
        global prediction frame
      atom14_atom_exists: Mask denoting whether atom at positions exists for given
        amino acid type
      atom14_atom_radius: Van der Waals radius for each atom.
      residue_index: Residue index for given amino acid.
      overlap_tolerance_soft: Soft tolerance factor.
      overlap_tolerance_hard: Hard tolerance factor.

    Returns:
      Dict containing:
        * 'mean_loss': average clash loss
        * 'per_atom_loss_sum': sum of all clash losses per atom, shape (N, 14)
        * 'per_atom_clash_mask': mask whether atom clashes with any other atom
            shape (N, 14)
    """

    # Create the distance matrix.
    # (N, N, 14, 14)
    dists = mnp.sqrt(1e-10 + mnp.sum(
        mnp.square(atom14_pred_positions[:, None, :, None, :] - atom14_pred_positions[None, :, None, :, :]), axis=-1))

    # Create the mask for valid distances.
    # shape (N, N, 14, 14)
    dists_mask = atom14_atom_exists[:, None, :, None] * atom14_atom_exists[None, :, None, :]

    # Mask out all the duplicate entries in the lower triangular matrix.
    # Also mask out the diagonal (atom-pairs from the same residue) -- these atoms
    # are handled separately.
    dists_mask *= (residue_index[:, None, None, None] < residue_index[None, :, None, None])

    # Backbone C--N bond between subsequent residues is no clash.

    neighbour_mask = ((residue_index[:, None, None, None] + 1) == residue_index[None, :, None, None])
    c_n_bonds = neighbour_mask * c_one_hot[None, None, :, None] * n_one_hot[None, None, None, :]
    dists_mask *= (1. - c_n_bonds)

    # Disulfide bridge between two cysteines is no clash.

    cys_sg_one_hot = nn.OneHot(depth=14)(cys_sg_idx)
    disulfide_bonds = (cys_sg_one_hot[None, None, :, None] * cys_sg_one_hot[None, None, None, :])
    dists_mask *= (1. - disulfide_bonds)

    # Compute the lower bound for the allowed distances.
    # shape (N, N, 14, 14)
    dists_lower_bound = dists_mask * (atom14_atom_radius[:, None, :, None] + atom14_atom_radius[None, :, None, :])

    # Compute the error.
    # shape (N, N, 14, 14)
    dists_to_low_error = dists_mask * nn.ReLU()(dists_lower_bound - overlap_tolerance_soft - dists)

    # Compute the mean loss.
    # shape ()
    mean_loss = mnp.sum(dists_to_low_error) / (1e-6 + mnp.sum(dists_mask))

    # Compute the per atom loss sum.
    # shape (N, 14)
    per_atom_loss_sum = P.ReduceSum()(dists_to_low_error, (0, 2)) + P.ReduceSum()(dists_to_low_error, (1, 3))

    # Compute the hard clash mask.
    # shape (N, N, 14, 14)
    clash_mask = dists_mask * (dists < (dists_lower_bound - overlap_tolerance_hard))

    # Compute the per atom clash.
    # shape (N, 14)
    per_atom_clash_mask = mnp.maximum(mnp.max(clash_mask, axis=[0, 2]), mnp.max(clash_mask, axis=[1, 3]))

    return mean_loss, per_atom_loss_sum, per_atom_clash_mask


def within_residue_violations(
        atom14_pred_positions,  # (N, 14, 3)
        atom14_atom_exists,  # (N, 14)
        atom14_dists_lower_bound,  # (N, 14, 14)
        atom14_dists_upper_bound,  # (N, 14, 14)
        tighten_bounds_for_loss,
        dists_mask_i
):
    """Loss to penalize steric clashes within residues.

    This is a loss penalizing any steric violations or clashes of non-bonded atoms
    in a given peptide. This loss corresponds to the part with
    the same residues of
    Jumper et al. (2021) Suppl. Sec. 1.9.11, eq 46.

    Args:
      atom14_pred_positions: Predicted positions of atoms in
        global prediction frame
      atom14_atom_exists: Mask denoting whether atom at positions exists for given
        amino acid type
      atom14_dists_lower_bound: Lower bound on allowed distances.
      atom14_dists_upper_bound: Upper bound on allowed distances
      tighten_bounds_for_loss: Extra factor to tighten loss

    Returns:
      Dict containing:
        * 'per_atom_loss_sum': sum of all clash losses per atom, shape (N, 14)
        * 'per_atom_clash_mask': mask whether atom clashes with any other atom
            shape (N, 14)
    """

    # Compute the mask for each residue.
    # shape (N, 14, 14)
    dists_masks = (1. - dists_mask_i[None])
    dists_masks *= (atom14_atom_exists[:, :, None] * atom14_atom_exists[:, None, :])

    # Distance matrix
    # shape (N, 14, 14)
    dists = mnp.sqrt(1e-10 + mnp.sum(
        mnp.square(atom14_pred_positions[:, :, None, :] - atom14_pred_positions[:, None, :, :]), axis=-1))

    # Compute the loss.
    # shape (N, 14, 14)
    dists_to_low_error = nn.ReLU()(atom14_dists_lower_bound + tighten_bounds_for_loss - dists)
    dists_to_high_error = nn.ReLU()(dists - (atom14_dists_upper_bound - tighten_bounds_for_loss))
    loss = dists_masks * (dists_to_low_error + dists_to_high_error)

    # Compute the per atom loss sum.
    # shape (N, 14)
    per_atom_loss_sum = mnp.sum(loss, axis=1) + mnp.sum(loss, axis=2)

    # Compute the violations mask.
    # shape (N, 14, 14)
    lower = (dists < atom14_dists_lower_bound).astype(ms.int32)
    high = (dists > atom14_dists_upper_bound).astype(ms.int32)
    violations = dists_masks * ((lower + high).astype(bool))

    # Compute the per atom violations.
    # shape (N, 14)
    per_atom_violations = mnp.maximum(mnp.max(violations, axis=1), mnp.max(violations, axis=2))

    return per_atom_loss_sum, per_atom_violations


def extreme_ca_ca_distance_violations(
        pred_atom_positions,  # (N, 37(14), 3)
        pred_atom_mask,  # (N, 37(14))
        residue_index,  # (N)
        max_angstrom_tolerance=1.5):
    """Counts residues whose Ca is a large distance from its neighbor.

    Measures the fraction of CA-CA pairs between consecutive amino acids that
    are more than 'max_angstrom_tolerance' apart.

    Args:
      pred_atom_positions: Atom positions in atom37/14 representation
      pred_atom_mask: Atom mask in atom37/14 representation
      residue_index: Residue index for given amino acid, this is assumed to be
        monotonically increasing.
      max_angstrom_tolerance: Maximum distance allowed to not count as violation.
    Returns:
      Fraction of consecutive CA-CA pairs with violation.
    """
    this_ca_pos = pred_atom_positions[:-1, 1, :]  # (N - 1, 3)
    this_ca_mask = pred_atom_mask[:-1, 1]  # (N - 1)
    next_ca_pos = pred_atom_positions[1:, 1, :]  # (N - 1, 3)
    next_ca_mask = pred_atom_mask[1:, 1]  # (N - 1)
    has_no_gap_mask = ((residue_index[1:] - residue_index[:-1]) == 1.0).astype(ms.float32)
    ca_ca_distance = mnp.sqrt(
        1e-6 + mnp.sum(mnp.square(this_ca_pos - next_ca_pos), axis=-1))
    violations = (ca_ca_distance -
                  residue_constants.ca_ca) > max_angstrom_tolerance
    mask = this_ca_mask * next_ca_mask * has_no_gap_mask
    return mask_mean(mask=mask, value=violations)


def frame_aligned_point_error_map(pred_frames,
                                  target_frames,
                                  frames_mask,
                                  pred_positions,
                                  target_positions,
                                  positions_mask,
                                  length_scale,
                                  l1_clamp_distance,
                                  epsilon=1e-4):
    """Measure point error under different alignments.

    Jumper et al. (2021) Suppl. Alg. 28 "computeFAPE"

    Computes error between two structures with B points under A alignments derived
    from the given pairs of frames.
    Args:
      pred_frames: num_frames reference frames for 'pred_positions'.
      target_frames: num_frames reference frames for 'target_positions'.
      frames_mask: Mask for frame pairs to use.
      pred_positions: num_positions predicted positions of the structure.
      target_positions: num_positions target positions of the structure.
      positions_mask: Mask on which positions to score.
      length_scale: length scale to divide loss by.
      l1_clamp_distance: Distance cutoff on error beyond which gradients will
        be zero.
      epsilon: small value used to regularize denominator for masked average.
    Returns:
      Masked Frame Aligned Point Error.
    """

    # Compute array of predicted positions in the predicted frames.
    # r3.Vecs (num_frames, num_positions)
    xx = pred_frames[0]
    xy = pred_frames[1]
    xz = pred_frames[2]
    yx = pred_frames[3]
    yy = pred_frames[4]
    yz = pred_frames[5]
    zx = pred_frames[6]
    zy = pred_frames[7]
    zz = pred_frames[8]
    t0_p = pred_frames[9]
    t1_p = pred_frames[10]
    t2_p = pred_frames[11]
    t0 = pred_positions[0]
    t1 = pred_positions[1]
    t2 = pred_positions[2]

    v1 = -(xx * t0_p + yx * t1_p + zx * t2_p)
    v2 = -(xy * t0_p + yy * t1_p + zy * t2_p)
    v3 = -(xz * t0_p + yz * t1_p + zz * t2_p)
    # r = invert_rots + [v1, v2, v3]
    # v = [t0, t1, t2]
    ## all is 0 ????
    local_pred_pos = [
        xx[..., None] * t0[:, None, ...] + yx[..., None] * t1[:, None, ...] + zx[..., None] * t2[:, None, ...] + v1[
            ..., None],
        xy[..., None] * t0[:, None, ...] + yy[..., None] * t1[:, None, ...] + zy[..., None] * t2[:, None, ...] + v2[
            ..., None],
        xz[..., None] * t0[:, None, ...] + yz[..., None] * t1[:, None, ...] + zz[..., None] * t2[:, None, ...] + v3[
            ..., None]
    ]
    xx_gt = target_frames[0]
    xy_gt = target_frames[1]
    xz_gt = target_frames[2]
    yx_gt = target_frames[3]
    yy_gt = target_frames[4]
    yz_gt = target_frames[5]
    zx_gt = target_frames[6]
    zy_gt = target_frames[7]
    zz_gt = target_frames[8]
    t0_t = target_frames[9]
    t1_t = target_frames[10]
    t2_t = target_frames[11]
    t0_gt = target_positions[0]
    t1_gt = target_positions[1]
    t2_gt = target_positions[2]

    v1_gt = -(xx_gt * t0_t + yx_gt * t1_t + zx_gt * t2_t)
    v2_gt = -(xy_gt * t0_t + yy_gt * t1_t + zy_gt * t2_t)
    v3_gt = -(xz_gt * t0_t + yz_gt * t1_t + zz_gt * t2_t)

    local_target_pos = [xx_gt[:, None] * t0_gt[None, :] + yx_gt[:, None] * t1_gt[None, :] +
                        zx_gt[:, None] * t2_gt[None, :] + v1_gt[:, None], xy_gt[:, None] * t0_gt[None, :] +
                        yy_gt[:, None] * t1_gt[None, :] + zy_gt[:, None] * t2_gt[None, :] +
                        v2_gt[:, None], xz_gt[:, None] * t0_gt[None, :] + yz_gt[:, None] * t1_gt[None, :] +
                        zz_gt[:, None] * t2_gt[None, :] + v3_gt[:, None]]
    error_dist = mnp.sqrt(mnp.square(local_pred_pos[0] - local_target_pos[0]) +
                          mnp.square(local_pred_pos[1] - local_target_pos[1]) +
                          mnp.square(local_pred_pos[2] - local_target_pos[2]) + epsilon)
    normalization_factor = (mnp.sum(frames_mask.astype(ms.float32), axis=-1) *
                            mnp.sum(positions_mask.astype(ms.float32), axis=-1))
    # fape with clamp
    error_dist_clamp = mnp.clip(error_dist, 0, l1_clamp_distance)
    normed_error_clamp = error_dist_clamp / length_scale
    normed_error_clamp *= mnp.expand_dims(frames_mask, axis=-1)
    normed_error_clamp *= mnp.expand_dims(positions_mask, axis=-2)
    erro_clamp = P.ReduceSum()(normed_error_clamp, (-2, -1)) / (epsilon + normalization_factor)

    # fape with no clamp
    normed_error_no_clamp = error_dist / length_scale
    normed_error_no_clamp *= mnp.expand_dims(frames_mask, axis=-1)
    normed_error_no_clamp *= mnp.expand_dims(positions_mask, axis=-2)
    erro_no_clamp = P.ReduceSum()(normed_error_no_clamp, (-2, -1)) / (epsilon + normalization_factor)

    return erro_clamp, erro_no_clamp


def frame_aligned_point_error(pred_frames,
                              target_frames,
                              frames_mask,
                              pred_positions,
                              target_positions,
                              positions_mask,
                              length_scale,
                              l1_clamp_distance,
                              epsilon=1e-4):
    """Measure point error under different alignments.

    Jumper et al. (2021) Suppl. Alg. 28 "computeFAPE"

    Computes error between two structures with B points under A alignments derived
    from the given pairs of frames.
    Args:
      pred_frames: num_frames reference frames for 'pred_positions'.
      target_frames: num_frames reference frames for 'target_positions'.
      frames_mask: Mask for frame pairs to use.
      pred_positions: num_positions predicted positions of the structure.
      target_positions: num_positions target positions of the structure.
      positions_mask: Mask on which positions to score.
      length_scale: length scale to divide loss by.
      l1_clamp_distance: Distance cutoff on error beyond which gradients will
        be zero.
      epsilon: small value used to regularize denominator for masked average.
    Returns:
      Masked Frame Aligned Point Error.
    """

    # Compute array of predicted positions in the predicted frames.
    # r3.Vecs (num_frames, num_positions)
    xx = pred_frames[0]
    xy = pred_frames[1]
    xz = pred_frames[2]
    yx = pred_frames[3]
    yy = pred_frames[4]
    yz = pred_frames[5]
    zx = pred_frames[6]
    zy = pred_frames[7]
    zz = pred_frames[8]
    t0_p = pred_frames[9]
    t1_p = pred_frames[10]
    t2_p = pred_frames[11]
    t0 = pred_positions[0]
    t1 = pred_positions[1]
    t2 = pred_positions[2]

    v1 = -(xx * t0_p + yx * t1_p + zx * t2_p)
    v2 = -(xy * t0_p + yy * t1_p + zy * t2_p)
    v3 = -(xz * t0_p + yz * t1_p + zz * t2_p)
    # r = invert_rots + [v1, v2, v3]
    # v = [t0, t1, t2]
    # all is 0 ????
    local_pred_pos = [
        xx[..., None] * t0[None, ...] + yx[..., None] * t1[None, ...] + zx[..., None] * t2[None, ...] + v1[..., None],
        xy[..., None] * t0[None, ...] + yy[..., None] * t1[None, ...] + zy[..., None] * t2[None, ...] + v2[..., None],
        xz[..., None] * t0[None, ...] + yz[..., None] * t1[None, ...] + zz[..., None] * t2[None, ...] + v3[..., None]
    ]
    xx_gt = target_frames[0]
    xy_gt = target_frames[1]
    xz_gt = target_frames[2]
    yx_gt = target_frames[3]
    yy_gt = target_frames[4]
    yz_gt = target_frames[5]
    zx_gt = target_frames[6]
    zy_gt = target_frames[7]
    zz_gt = target_frames[8]
    t0_t = target_frames[9]
    t1_t = target_frames[10]
    t2_t = target_frames[11]
    t0_gt = target_positions[0]
    t1_gt = target_positions[1]
    t2_gt = target_positions[2]

    v1_gt = -(xx_gt * t0_t + yx_gt * t1_t + zx_gt * t2_t)
    v2_gt = -(xy_gt * t0_t + yy_gt * t1_t + zy_gt * t2_t)
    v3_gt = -(xz_gt * t0_t + yz_gt * t1_t + zz_gt * t2_t)

    local_target_pos = [xx_gt[:, None] * t0_gt[None, :] + yx_gt[:, None] * t1_gt[None, :] +
                        zx_gt[:, None] * t2_gt[None, :] + v1_gt[:, None], xy_gt[:, None] * t0_gt[None, :] +
                        yy_gt[:, None] * t1_gt[None, :] + zy_gt[:, None] * t2_gt[None, :] +
                        v2_gt[:, None], xz_gt[:, None] * t0_gt[None, :] + yz_gt[:, None] * t1_gt[None, :] +
                        zz_gt[:, None] * t2_gt[None, :] + v3_gt[:, None]]
    error_dist = mnp.sqrt(mnp.square(local_pred_pos[0] - local_target_pos[0]) +
                          mnp.square(local_pred_pos[1] - local_target_pos[1]) +
                          mnp.square(local_pred_pos[2] - local_target_pos[2]) + epsilon)
    if l1_clamp_distance:
        error_dist = mnp.clip(error_dist, 0, l1_clamp_distance)

    normed_error = error_dist / length_scale
    normed_error *= mnp.expand_dims(frames_mask, axis=-1)
    normed_error *= mnp.expand_dims(positions_mask, axis=-2)

    normalization_factor = mnp.sum(frames_mask, axis=-1) * mnp.sum(positions_mask, axis=-1)
    return mnp.sum(normed_error, axis=(-2, -1)) / (epsilon + normalization_factor)
