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
"""Modules and utilities for the structure module."""
import numpy as np
from mindsponge.common import residue_constants
from . import utils

VIOLATION_TOLERANCE_ACTOR = 12.0
CLASH_OVERLAP_TOLERANCE = 1.5

# one hot encoding for C and N atoms (using atom14 representation)
C_ONE_HOT = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
N_ONE_HOT = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

# Van der Waals radii for each atom
ATOMTYPE_RADIUS = \
    np.array([residue_constants.van_der_waals_radius.get(name[0]) for name in residue_constants.atom_types])
DISTS_MASK_I = np.eye(14, 14)

# lower bound and upper bound between each atoms used for clashes calculation
LOWER_BOUND, UPPER_BOUND, _ = \
    residue_constants.make_atom14_dists_bounds(overlap_tolerance=CLASH_OVERLAP_TOLERANCE,
                                               bond_length_tolerance_factor=VIOLATION_TOLERANCE_ACTOR)
CYS_SG_IDX = 5


def within_residue_violations(
        atom14_pred_positions,
        atom14_atom_exists,
        atom14_dists_lower_bound,
        atom14_dists_upper_bound,
        tighten_bounds_for_loss,
        dists_mask_i
):
    """Loss to penalize steric clashes within residues.
    This is a loss penalizing any steric violations or clashes of non-bonded atoms in a given peptide.

    Args:
        atom14_pred_positions (Tensor):    predicted positions of atoms in global prediction frame.
                                           shape :math:`(N_{res}, 14, 3)` .
        atom14_atom_exists (Tensor):       mask denoting whether atom at positions exists for given amino acid type.
                                           shape :math:`(N_{res}, 14)` .
        atom14_dists_lower_bound (Tensor): lower bond on allowed distances. shape :math:`(N_{res}, 14, 14)` .
        atom14_dists_upper_bound (Tensor): upper bond on allowed distances. shape :math:`(N_{res}, 14, 14)` .
        tighten_bounds_for_loss (float):  Extra factor to tighten loss. Default: 0.0.
        dists_mask_i (Tensor):             initial distants mask, shape: :math:`(14, 14)` .

    Returns:
        - **per_atom_loss_sum** (Tensor) - sum of all clash losses per atom, shape :math:`(N_{res}, 14)` .
        - **per_atom_violations** (Tensor) - violation per atom, shape :math:`(N_{res}, 14)` .

    Symbol:
        :math:`N_{res}`, number of amino acids.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import mindspore as ms
        >>> from mindspore import Tensor
        >>> import numpy as np
        >>> from mindsponge.metrics import within_residue_violations
        >>> atom14_pred_positions = Tensor(np.random.random(size=(50, 14, 3)), ms.float32)
        >>> atom14_atom_exists = Tensor(np.random.random(size=(50, 14)), ms.float32)
        >>> atom14_dists_lower_bound = Tensor(np.random.random(size=(50, 14, 14)), ms.float32)
        >>> atom14_dists_upper_bound = Tensor(np.random.random(size=(50, 14, 14)), ms.float32)
        >>> tighten_bounds_for_loss = 0.0
        >>> dists_mask_i = Tensor(np.eye(14, 14), ms.int32)
        >>> per_atom_loss_sum, per_atom_violations = within_residue_violations(atom14_pred_positions,
        ...                                                                   atom14_atom_exists,
        ...                                                                   atom14_dists_lower_bound,
        ...                                                                   atom14_dists_upper_bound,
        ...                                                                   tighten_bounds_for_loss,
        ...                                                                   dists_mask_i)
        >>> print(per_atom_loss_sum.shape, per_atom_violations.shape)
        (50, 14) (50, 14)

    """

    dists_masks = (1. - dists_mask_i[None])
    dists_masks = dists_masks * (atom14_atom_exists[:, :, None] * atom14_atom_exists[:, None, :])

    dists = np.sqrt(1e-10 + np.sum(
        np.square(atom14_pred_positions[:, :, None, :] - atom14_pred_positions[:, None, :, :]), axis=-1))
    dists_to_low_error = np.maximum(0, atom14_dists_lower_bound + tighten_bounds_for_loss - dists)
    dists_to_high_error = np.maximum(0, dists - (atom14_dists_upper_bound - tighten_bounds_for_loss))
    loss = dists_masks * (dists_to_low_error + dists_to_high_error)
    per_atom_loss_sum = np.sum(loss, axis=1) + np.sum(loss, axis=2)
    lower = (dists < atom14_dists_lower_bound).astype(np.int32)
    high = (dists > atom14_dists_upper_bound).astype(np.int32)
    violations = dists_masks * ((lower + high).astype(bool))

    per_atom_violations = np.maximum(np.max(violations, axis=1), np.max(violations, axis=2))

    return per_atom_loss_sum, per_atom_violations


def between_residue_clash(
        atom14_pred_positions,
        atom14_atom_exists,
        atom14_atom_radius,
        residue_index,
        c_one_hot,
        n_one_hot,
        overlap_tolerance_soft,
        overlap_tolerance_hard,
        cys_sg_idx):
    """
    This is a loss penalizing any steric clashes due to non bonded atoms in different peptides coming too close.

    Args:
        atom14_pred_positions (Tensor): predicted positions of atoms in global prediction frame.
                                        shape is :math:`(N_{res}, 14, 3)` .
        atom14_atom_exists (Tensor):    mask denoting whether atom at positions exists for given amino acid type.
                                        shape is :math:`(N_{res}, 14)` .
        atom14_atom_radius (Tensor):    Van der Waals radius for each atom. shape is :math:`(N_{res}, 14)` .
        residue_index (Tensor):         Residue index for given amino acid. shape is :math:`(N_{res}, )` ,
                                        range from 1 to :math:`N_{res}` .
        c_one_hot (Tensor):             one hot encoding for C atoms (using atom14 representation). shape is (14, ) .
        n_one_hot (Tensor):             one hot encoding for N atoms (using atom14 representation). shape is (14, ) .
        overlap_tolerance_soft (float): soft tolerance factor. in default: 12.0.
        overlap_tolerance_hard (float): hard tolerance factor. in default: 1.5.
        cys_sg_idx (Tensor):            CYS amino acid index. Default: 5.
                                        see more at `mindsponge.common.residue_constants`. Shape: `()` .

    Returns:
        - Tensor, mean_loss, average clash loss. Shape is `()` .
        - Tensor, per_atom_loss_sum, sum of all clash losses per atom, shape is :math:`(N_{res}, 14)` .
        - Tensor, per_atom_clash_mask, mask whether atom clashes with any other atom,
          shape is :math:`(N_{res}, 14)` .

    Symbol:
        :math:`N_{res}`, number of amino acids.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import mindspore as ms
        >>> from mindspore import Tensor
        >>> import numpy as np
        >>> from mindsponge.metrics import between_residue_clash
        >>> atom14_pred_positions = Tensor(np.random.random(size=(50, 14, 3)), ms.float32)
        >>> atom14_atom_exists = Tensor(np.random.randint(2, size=(50, 14)))
        >>> atom14_atom_radius = Tensor(np.random.random(size=(50, 14)), ms.float32)
        >>> residue_index = Tensor(np.array(range(50)), ms.int32)
        >>> c_one_hot = Tensor(np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), ms.int32)
        >>> n_one_hot = Tensor(np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), ms.int32)
        >>> overlap_tolerance_soft = 12.0
        >>> overlap_tolerance_hard = 1.5
        >>> cys_sg_idx = Tensor(5, ms.int32)
        >>> mean_loss, per_atom_loss_sum, per_atom_clash_mask = between_residue_clash(atom14_pred_positions,
        ...                                                                           atom14_atom_exists,
        ...                                                                           atom14_atom_radius,
        ...                                                                           residue_index,
        ...                                                                           c_one_hot,
        ...                                                                           n_one_hot,
        ...                                                                           overlap_tolerance_soft,
        ...                                                                           overlap_tolerance_hard,
        ...                                                                           cys_sg_idx)
        >>> print(mean_loss.shape, per_atom_loss_sum.shape, per_atom_clash_mask.shape)
        () (50,14) (50,14)

    """

    dists = np.sqrt(1e-10 + np.sum(
        np.square(atom14_pred_positions[:, None, :, None, :] - atom14_pred_positions[None, :, None, :, :]), axis=-1))
    dists_mask = atom14_atom_exists[:, None, :, None] * atom14_atom_exists[None, :, None, :]
    dists_mask *= (residue_index[:, None, None, None] < residue_index[None, :, None, None])

    # Backbone C--N bond between subsequent residues is no clash.
    neighbour_mask = ((residue_index[:, None, None, None] + 1) == residue_index[None, :, None, None])
    c_n_bonds = neighbour_mask * c_one_hot[None, None, :, None] * n_one_hot[None, None, None, :]
    dists_mask *= (1. - c_n_bonds)

    # Disulfide bridge between two cysteines is no clash.
    cys_sg_one_hot = np.eye(14)[cys_sg_idx]
    disulfide_bonds = (cys_sg_one_hot[None, None, :, None] * cys_sg_one_hot[None, None, None, :])
    dists_mask *= (1. - disulfide_bonds)

    dists_lower_bound = dists_mask * (atom14_atom_radius[:, None, :, None] + atom14_atom_radius[None, :, None, :])
    dists_to_low_error = dists_mask * np.maximum(0, dists_lower_bound - overlap_tolerance_soft - dists)
    mean_loss = np.sum(dists_to_low_error) / (1e-6 + np.sum(dists_mask))
    per_atom_loss_sum = (dists_to_low_error.sum(axis=(0, 2)) + dists_to_low_error.sum(axis=(1, 3)))
    clash_mask = dists_mask * (dists < (dists_lower_bound - overlap_tolerance_hard))
    per_atom_clash_mask = np.maximum(clash_mask.max(axis=(0, 2)), clash_mask.max(axis=(1, 3)))

    return mean_loss, per_atom_loss_sum, per_atom_clash_mask


def between_residue_bond(
        pred_atom_positions,
        pred_atom_mask,
        residue_index,
        aatype,
        tolerance_factor_soft=12.0,
        tolerance_factor_hard=12.0
):
    """
    Flat-bottom loss to penalize structural violations between residues. This is a loss penalizing any violation
    of the geometry around the peptide bond between consecutive amino acids.

    Args:
        pred_atom_positions (Tensor):   Atom positions in atom37/14 representation, shape :math:`(N_{res}, 37, 3)`.
                                        or shape :math:`(N_{res}, 14, 3)` .
        pred_atom_mask (Tensor):        Atom mask in atom37/14 representation. shape :math:`(N_{res}, 37)` or
                                        shape :math:`(N_{res}, 14)` .
        residue_index (Tensor):         Residue index for given amino acid, this is assumed to be monotonically
                                        increasing. Range from 1 to :math:`N_{res}`. shape :math:`(N_{res}, )` .
        aatype (Tensor):                amino acid types. Range is :math:`[0,20]`. shape :math:`(N_{res}, )` .
        tolerance_factor_soft (float):  soft tolerance factor measured in standard deviations of pdb distributions.
                                        Default: 12.0 .
        tolerance_factor_hard (float):  hard tolerance factor measured in standard deviations of pdb distributions.
                                        Default: 12.0 .

    Returns:
        - Tensor, c_n_loss_mean, loss for peptide bond length violations. shape is :math:`( )` .
        - Tensor, ca_c_n_loss_mean, loss for violations of bond angle around C spanned by CA, C, N.
          shape is :math:`( )` .
        - Tensor, c_n_ca_loss_mean, loss for violations of bond angle around N spanned by C, N, CA.
          shape is :math:`( )` .
        - Tensor, per_residue_loss_sum, sum of all losses of each residue. shape is :math:`(N_{res}, )` .
        - Tensor, per_residue_violation_mask, mask denoting all residues with violation present.
          shape is :math:`(N_{res}, )` .

    Symbol:
        :math:`N_{res}`, number of amino acids.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import mindspore as ms
        >>> from mindspore import Tensor
        >>> import numpy as np
        >>> from mindsponge.metrics import between_residue_bond
        >>> np.random.seed(1)
        >>> pred_atom_positions = Tensor(np.random.random(size=(50,37,3)), ms.float32)
        >>> pred_atom_mask = Tensor(np.random.randint(2,size=(50,37)), ms.int32)
        >>> residue_index = Tensor(np.array(range(50)), ms.int32)
        >>> aatype = Tensor(np.random.randint(20, size=(50,)), ms.int32)
        >>> tolerance_factor_soft = 12.0
        >>> tolerance_factor_hard = 12.0
        >>> result = between_residue_bond(pred_atom_positions, pred_atom_mask, residue_index, aatype,
        >>>                              tolerance_factor_soft, tolerance_factor_hard)
        >>> for x in result:
        >>>    print(x)
        0.52967054
        0.6045412
        0.39251995
        [0.62809587 1.6770853  1.7221183  1.0325309  1.3417522  1.79882
         1.7718308  1.5092779  1.5653987  1.9564128  1.6804926  1.6051245
         1.5033073  1.5895741  2.1686926  2.126039   1.3837843  1.2554975
         1.8135165  2.1593785  1.9408598  1.7281027  1.8666006  1.9623451
         1.8177024  1.7543832  1.5969353  1.2150483  0.9833115  1.219868
         1.7008476  1.6968286  1.7648234  1.5584714  1.370602   1.8525059
         1.7938454  1.5313196  1.6940074  1.8512855  1.8222975  1.6600168
         1.9163743  1.7201058  1.6288358  1.6055745  1.521946   1.6553445
         1.6175683  0.894606 ]
         [1. 1. 0. 1. 1. 0. 0. 1. 1. 1. 1. 0. 0. 0. 0. 1. 1. 1. 1. 1. 0. 1. 1. 0.
          0. 1. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 1. 1. 1. 0.
          1. 1.]

    """

    # Get the positions of the relevant backbone atoms.
    this_ca_pos = pred_atom_positions[:-1, 1, :]
    this_ca_mask = pred_atom_mask[:-1, 1]
    this_c_pos = pred_atom_positions[:-1, 2, :]
    this_c_mask = pred_atom_mask[:-1, 2]
    next_n_pos = pred_atom_positions[1:, 0, :]
    next_n_mask = pred_atom_mask[1:, 0]
    next_ca_pos = pred_atom_positions[1:, 1, :]
    next_ca_mask = pred_atom_mask[1:, 1]
    has_no_gap_mask = ((residue_index[1:] - residue_index[:-1]) == 1.0).astype(np.float32)

    # Compute loss for the C--N bond.
    c_n_bond_length = np.sqrt(1e-6 + np.sum(np.square(this_c_pos - next_n_pos), axis=-1))

    # The C-N bond to proline has slightly different length because of the ring.
    next_is_proline = (aatype[1:] == residue_constants.resname_to_idx['PRO']).astype(np.float32)
    gt_length = ((1. - next_is_proline) * residue_constants.between_res_bond_length_c_n[0]
                 + next_is_proline * residue_constants.between_res_bond_length_c_n[1])
    gt_stddev = ((1. - next_is_proline) * residue_constants.between_res_bond_length_stddev_c_n[0] +
                 next_is_proline * residue_constants.between_res_bond_length_stddev_c_n[1])
    c_n_bond_length_error = np.sqrt(1e-6 + np.square(c_n_bond_length - gt_length))
    c_n_loss_per_residue = np.maximum(0, c_n_bond_length_error - tolerance_factor_soft * gt_stddev)
    mask = this_c_mask * next_n_mask * has_no_gap_mask
    c_n_loss_mean = np.sum(mask * c_n_loss_per_residue) / (np.sum(mask) + 1e-6)
    c_n_violation_mask = mask * (c_n_bond_length_error > (tolerance_factor_hard * gt_stddev))

    # Compute loss for the angles.
    ca_c_bond_length = np.sqrt(1e-6 + np.sum(np.square(this_ca_pos - this_c_pos), axis=-1))
    n_ca_bond_length = np.sqrt(1e-6 + np.sum(np.square(next_n_pos - next_ca_pos), axis=-1))

    c_ca_unit_vec = (this_ca_pos - this_c_pos) / ca_c_bond_length[:, None]
    c_n_unit_vec = (next_n_pos - this_c_pos) / c_n_bond_length[:, None]
    n_ca_unit_vec = (next_ca_pos - next_n_pos) / n_ca_bond_length[:, None]

    ca_c_n_cos_angle = np.sum(c_ca_unit_vec * c_n_unit_vec, axis=-1)
    gt_angle = residue_constants.between_res_cos_angles_ca_c_n[0]
    gt_stddev = residue_constants.between_res_cos_angles_ca_c_n[1]
    ca_c_n_cos_angle_error = np.sqrt(1e-6 + np.square(ca_c_n_cos_angle - gt_angle))
    ca_c_n_loss_per_residue = np.maximum(0, ca_c_n_cos_angle_error - tolerance_factor_soft * gt_stddev)
    mask = this_ca_mask * this_c_mask * next_n_mask * has_no_gap_mask
    ca_c_n_loss_mean = np.sum(mask * ca_c_n_loss_per_residue) / (np.sum(mask) + 1e-6)
    ca_c_n_violation_mask = mask * (ca_c_n_cos_angle_error > (tolerance_factor_hard * gt_stddev))

    c_n_ca_cos_angle = np.sum((-c_n_unit_vec) * n_ca_unit_vec, axis=-1)
    gt_angle = residue_constants.between_res_cos_angles_c_n_ca[0]
    gt_stddev = residue_constants.between_res_cos_angles_c_n_ca[1]
    c_n_ca_cos_angle_error = np.sqrt(1e-6 + np.square(c_n_ca_cos_angle - gt_angle))
    c_n_ca_loss_per_residue = np.maximum(0, c_n_ca_cos_angle_error - tolerance_factor_soft * gt_stddev)
    mask = this_c_mask * next_n_mask * next_ca_mask * has_no_gap_mask
    c_n_ca_loss_mean = np.sum(mask * c_n_ca_loss_per_residue) / (np.sum(mask) + 1e-6)
    c_n_ca_violation_mask = mask * (c_n_ca_cos_angle_error > (tolerance_factor_hard * gt_stddev))

    # Compute a per residue loss (equally distribute the loss to both neighbouring residues).
    per_residue_loss_sum = c_n_loss_per_residue + ca_c_n_loss_per_residue + c_n_ca_loss_per_residue
    per_residue_loss_sum = 0.5 * (np.pad(per_residue_loss_sum, [[0, 1]]) + np.pad(per_residue_loss_sum, [[1, 0]]))

    # Compute hard violations.
    per_residue_violation_mask = np.max(np.stack([c_n_violation_mask, ca_c_n_violation_mask, c_n_ca_violation_mask]),
                                        axis=0)
    per_residue_violation_mask = np.maximum(np.pad(per_residue_violation_mask, [[0, 1]]),
                                            np.pad(per_residue_violation_mask, [[1, 0]]))

    result = (c_n_loss_mean, ca_c_n_loss_mean, c_n_ca_loss_mean, per_residue_loss_sum, per_residue_violation_mask)
    return result


def get_structural_violations(atom14_atom_exists, residue_index, aatype, residx_atom14_to_atom37,
                              atom14_pred_positions, violation_tolerance_factor=VIOLATION_TOLERANCE_ACTOR,
                              clash_overlap_tolerance=CLASH_OVERLAP_TOLERANCE, lower_bound=LOWER_BOUND,
                              upper_bound=UPPER_BOUND, atomtype_radius=ATOMTYPE_RADIUS,
                              c_one_hot=C_ONE_HOT, n_one_hot=N_ONE_HOT, dists_mask_i=DISTS_MASK_I,
                              cys_sg_idx=CYS_SG_IDX):
    """Computes several checks for structural violations.

    Args:
        atom14_atom_exists (Tensor):        mask denoting whether atom at positions exists for given amino acid type.
                                            shape :math:`(N_{res}, 14)` .
        residue_index (Tensor):             Residue index for given amino acid range from 0 to :math:`N_{res} - 1`.
                                            Shape :math:`(N_{res}, )` .
        aatype (Tensor):                    amino acid types. shape :math:`(N_{res}, )` . Range is  :math:`[0,20]` .
        residx_atom14_to_atom37 (Tensor):   mapping for (residx, atom14) --> atom37. shape :math:`(N_{res}, 14)` .
        atom14_pred_positions (Tensor):     predicted positions of atoms in global prediction frame.
                                            shape :math:`(N_{res}, 14, 3)` .
        violation_tolerance_factor (float): violation between amino acid tolerance factor. Default: 12.0 .
        clash_overlap_tolerance (float):    clash overlap tolerance factor. Default: 1.5 .
        lower_bound (Tensor):               lower bond on allowed distances. shape :math:`(N_{res}, 14, 14)` .
        upper_bound (Tensor):               upper bond on allowed distances. shape :math:`(N_{res}, 14, 14)` .
        atomtype_radius (Tensor):           Van der Waals radius for each amino acid. shape: :math:`(37, )` .
        c_one_hot (Tensor):                 one hot encoding for C atoms (using atom14 representation).
                                            shape: :math:`(14, )` .
        n_one_hot (Tensor):                 one hot encoding for N atoms (using atom14 representation).
                                            shape: :math:`(14, )` .
        dists_mask_i (Tensor):              initial distants mask, shape: :math:`(14, 14)` .
        cys_sg_idx (Tensor):                CYS amino acid index. Default: 5 .
                                            see more at `mindsponge.common.residue_constants`.

    Returns:
        - bonds_c_n_loss_mean (Tensor), loss for peptide bond length violations. shape is :math:`()`.
        - angles_ca_c_n_loss_mean (Tensor), loss for violations of bond angle around C spanned by CA, C, N.
          Shape is :math:`()`.
        - angles_c_n_ca_loss_mean (Tensor), loss for violations of bond angle around N spanned by C, N, CA.
          Shape is :math:`()`.
        - connections_per_residue_loss_sum (Tensor), sum of all losses of each residue. shape is :math:`(N_{res}, )` .
        - connections_per_residue_violation_mask (Tensor), mask denoting all residues with violation present.
          shape is :math:`(N_{res}, )` .
        - clashes_mean_loss (Tensor),  average clash loss. shape: :math:`()` .
        - clashes_per_atom_loss_sum (Tensor), sum of all clash losses per atom, shape :math:`(N_{res}, 14)` .
        - clashes_per_atom_clash_mask (Tensor), mask whether atom clashes with any other atom.
          shape :math:`(N_{res}, 14)` .
        - per_atom_loss_sum (Tensor), sum of all clash losses per atom, shape :math:`(N_{res}, 14)` .
        - per_atom_violations (Tensor), violation per atom, shape :math:`(N_{res}, 14)` .
        - total_per_residue_violations_mask (Tensor), violation masks for all residues, shape :math:`(N_{res}, )` .
        - structure_violation_loss (Tensor), total violations for all amino acids. shape is :math:`()` .

    Symbol:
        :math:`N_{res}`, number of amino acids.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import mindspore as ms
        >>> from mindspore import Tensor
        >>> import numpy as np
        >>> from mindsponge.metrics import get_structural_violations
        >>> atom14_atom_exists = Tensor(np.random.random(size=(50, 14)), ms.float32)
        >>> residue_index = Tensor(np.array(range(50)), ms.int32)
        >>> aatype = Tensor(np.random.randint(20, size=(50,)), ms.int32)
        >>> residx_atom14_to_atom37 = Tensor(np.random.randint(2, size=(50, 14)), ms.int32)
        >>> atom14_pred_positions = Tensor(np.random.random(size=(50, 14, 3)), ms.float32)
        >>> violation_tolerance_factor = 12.0
        >>> clash_overlap_tolerance = 1.5
        >>> lower_bound = Tensor(np.random.random(size=(50, 14, 14)), ms.float32)
        >>> upper_bound = Tensor(np.random.random(size=(50, 14, 14)), ms.float32)
        >>> atomtype_radius =Tensor([1.55, 1.7, 1.7, 1.7, 1.52, 1.7, 1.7, 1.7, 1.52, 1.52, 1.8,
        ...                          1.7, 1.7, 1.7, 1.55, 1.55, 1.52, 1.52, 1.8, 1.7, 1.7, 1.7,
        ...                          1.7, 1.55, 1.55, 1.55, 1.52, 1.52, 1.7, 1.55, 1.55, 1.52, 1.7,
        ...                          1.7, 1.7, 1.55, 1.52], ms.float32)
        >>> c_one_hot = Tensor(np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), ms.int32)
        >>> n_one_hot = Tensor(np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), ms.int32)
        >>> dists_mask_i = Tensor(np.eye(14, 14), ms.int32)
        >>> cys_sg_idx = Tensor(5, ms.int32)
        >>> result = get_structural_violations(atom14_atom_exists, residue_index, aatype, residx_atom14_to_atom37,
        ...                                    atom14_pred_positions, violation_tolerance_factor,
        ...                                    clash_overlap_tolerance, lower_bound, upper_bound, atomtype_radius,
        ...                                    c_one_hot, n_one_hot, dists_mask_i,cys_sg_idx)
        >>> for r in result:
        >>>     print(r.shape)
        ()
        ()
        ()
        (50,)
        (50,)
        ()
        (50, 14)
        (50, 14)
        (50, 14)
        (50, 14)
        (50,)
        ()

    """

    # Compute between residue backbone violations of bonds and angles.
    result = \
        between_residue_bond(
            pred_atom_positions=atom14_pred_positions,
            pred_atom_mask=atom14_atom_exists.astype(np.float32),
            residue_index=residue_index.astype(np.float32),
            aatype=aatype,
            tolerance_factor_soft=violation_tolerance_factor,
            tolerance_factor_hard=violation_tolerance_factor)
    c_n_loss_mean, ca_c_n_loss_mean, c_n_ca_loss_mean, per_residue_loss_sum, per_residue_violation_mask = result
    # Compute the Van der Waals radius for every atom (the first letter of the atom name is the element type).
    # Shape: (N, 14).
    atom14_atom_radius = atom14_atom_exists * utils.batched_gather(
        atomtype_radius, residx_atom14_to_atom37)

    # Compute the between residue clash loss.
    mean_loss, clashes_per_atom_loss_sum, per_atom_clash_mask = between_residue_clash(
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
    atom14_dists_lower_bound = utils.batched_gather(lower_bound, aatype)
    atom14_dists_upper_bound = utils.batched_gather(upper_bound, aatype)
    per_atom_loss_sum, per_atom_violations = within_residue_violations(
        atom14_pred_positions=atom14_pred_positions,
        atom14_atom_exists=atom14_atom_exists,
        atom14_dists_lower_bound=atom14_dists_lower_bound,
        atom14_dists_upper_bound=atom14_dists_upper_bound,
        tighten_bounds_for_loss=0.0,
        dists_mask_i=dists_mask_i)

    # Combine them to a single per-residue violation mask (used later for LDDT).
    per_residue_violations_mask = np.max(np.stack([per_residue_violation_mask, np.max(per_atom_clash_mask, axis=-1),
                                                   np.max(per_atom_violations, axis=-1)]), axis=0)
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
    return {
        'between_residues': {
            'bonds_c_n_loss_mean':
                bonds_c_n_loss_mean,  # ()
            'angles_ca_c_n_loss_mean':
                angles_ca_c_n_loss_mean,  # ()
            'angles_c_n_ca_loss_mean':
                angles_c_n_ca_loss_mean,  # ()
            'connections_per_residue_loss_sum':
                connections_per_residue_loss_sum,  # (N)
            'connections_per_residue_violation_mask':
                connections_per_residue_violation_mask,  # (N)
            'clashes_mean_loss':
                clashes_mean_loss,  # ()
            'clashes_per_atom_loss_sum':
                clashes_per_atom_loss_sum,  # (N, 14)
            'clashes_per_atom_clash_mask':
                clashes_per_atom_clash_mask,  # (N, 14)
        },
        'within_residues': {
            'per_atom_loss_sum':
                per_atom_loss_sum,  # (N, 14)
            'per_atom_violations':
                per_atom_violations,  # (N, 14),
        },
        'total_per_residue_violations_mask':
            total_per_residue_violations_mask,  # (N)
    }


def extreme_ca_ca_distance_violations(
        pred_atom_positions,  # (N, 37(14), 3)
        pred_atom_mask,  # (N, 37(14))
        residue_index,  # (N)
        max_angstrom_tolerance=1.5
):
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
    has_no_gap_mask = ((residue_index[1:] - residue_index[:-1]) == 1.0).astype(
        np.float32)
    ca_ca_distance = np.sqrt(
        1e-6 + np.sum(np.square(this_ca_pos - next_ca_pos), axis=-1))
    violations = (ca_ca_distance -
                  residue_constants.ca_ca) > max_angstrom_tolerance
    mask = this_ca_mask * next_ca_mask * has_no_gap_mask
    return utils.mask_mean(mask=mask, value=violations)


def compute_violation_metrics(
        batch,
        atom14_pred_positions,  # (N, 14, 3)
        violations,
):
    """Compute several metrics to assess the structural violations."""

    ret = {}
    extreme_ca_ca_violations = extreme_ca_ca_distance_violations(
        pred_atom_positions=atom14_pred_positions,
        pred_atom_mask=batch['atom14_atom_exists'].astype(np.float32),
        residue_index=batch['residue_index'].astype(np.float32))
    ret['violations_extreme_ca_ca_distance'] = extreme_ca_ca_violations
    ret['violations_between_residue_bond'] = utils.mask_mean(
        mask=batch['seq_mask'],
        value=violations['between_residues'][
            'connections_per_residue_violation_mask'])
    ret['violations_between_residue_clash'] = utils.mask_mean(
        mask=batch['seq_mask'],
        value=np.max(violations['between_residues']['clashes_per_atom_clash_mask'], axis=-1))
    ret['violations_within_residue'] = utils.mask_mean(
        mask=batch['seq_mask'],
        value=np.max(violations['within_residues']['per_atom_violations'], axis=-1))
    ret['violations_per_residue'] = utils.mask_mean(
        mask=batch['seq_mask'],
        value=violations['total_per_residue_violations_mask'])
    return ret
