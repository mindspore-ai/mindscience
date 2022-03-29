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
"""train dataset"""
import datetime
import json
import os
import pickle
import time
import numpy as np
from mindspore import dataset as ds
from mindspore.communication import get_rank

import data.tools.quat_affine_np as quat_affine
import data.tools.r3_np as r3
from data.feature import data_transforms
from data.feature.feature_extraction import process_features_train
from commons import residue_constants, protein

# Internal import (7716).
MAX_TEMPLATE_HITS = 20
RELAX_MAX_ITERATIONS = 0
RELAX_ENERGY_TOLERANCE = 2.39
RELAX_STIFFNESS = 10.0
RELAX_EXCLUDE_RESIDUES = []
RELAX_MAX_OUTER_ITERATIONS = 1


def gather(params, indices, axis=0):
    """gather"""
    take_fn = lambda p, i: np.take(p, i, axis=axis)
    return take_fn(params, indices)


def np_gather(params, indices, axis=0, batch_dims=0):
    """numpy gather"""
    if batch_dims == 0:
        return gather(params, indices)
    result = []
    if batch_dims == 1:
        for p, i in zip(params, indices):
            axis = axis - batch_dims if axis - batch_dims > 0 else 0
            r = gather(p, i, axis=axis)
            result.append(r)
        return np.stack(result)
    for p, i in zip(params[0], indices[0]):
        r = gather(p, i, axis=axis)
        result.append(r)
    res = np.stack(result)
    return res.reshape((1,) + res.shape)


def rigids_to_quataffine(r):
    """Convert Rigids r into QuatAffine, inverse of 'rigids_from_quataffine'."""
    return quat_affine.QuatAffine(
        quaternion=None,
        rotation=[[r.rot.xx, r.rot.xy, r.rot.xz],
                  [r.rot.yx, r.rot.yy, r.rot.yz],
                  [r.rot.zx, r.rot.zy, r.rot.zz]],
        translation=[r.trans.x, r.trans.y, r.trans.z])


def make_atom14_positions(prot):
    # import pdb; pdb.set_trace()
    """Constructs denser atom positions (14 dimensions instead of 37)."""
    restype_atom14_to_atom37 = []  # mapping (restype, atom14) --> atom37
    restype_atom37_to_atom14 = []  # mapping (restype, atom37) --> atom14
    restype_atom14_mask = []

    for rt in residue_constants.restypes:
        atom_names = residue_constants.restype_name_to_atom14_names[
            residue_constants.restype_1to3[rt]]

        restype_atom14_to_atom37.append([
            (residue_constants.atom_order[name] if name else 0)
            for name in atom_names
        ])

        atom_name_to_idx14 = {name: i for i, name in enumerate(atom_names)}
        restype_atom37_to_atom14.append([
            (atom_name_to_idx14[name] if name in atom_name_to_idx14 else 0)
            for name in residue_constants.atom_types
        ])

        restype_atom14_mask.append([(1. if name else 0.) for name in atom_names])

    # Add dummy mapping for restype 'UNK'.
    restype_atom14_to_atom37.append([0] * 14)
    restype_atom37_to_atom14.append([0] * 37)
    restype_atom14_mask.append([0.] * 14)

    restype_atom14_to_atom37 = np.array(restype_atom14_to_atom37, dtype=np.int32)
    restype_atom37_to_atom14 = np.array(restype_atom37_to_atom14, dtype=np.int32)
    restype_atom14_mask = np.array(restype_atom14_mask, dtype=np.float32)

    # Create the mapping for (residx, atom14) --> atom37, i.e. an array
    # with shape (num_res, 14) containing the atom37 indices for this protein.
    residx_atom14_to_atom37 = restype_atom14_to_atom37[prot["aatype"]]
    residx_atom14_mask = restype_atom14_mask[prot["aatype"]]

    # Create a mask for known ground truth positions.
    residx_atom14_gt_mask = residx_atom14_mask * np.take_along_axis(
        prot["all_atom_mask"], residx_atom14_to_atom37, axis=1).astype(np.float32)

    # Gather the ground truth positions.
    residx_atom14_gt_positions = residx_atom14_gt_mask[:, :, None] * (
        np.take_along_axis(prot["all_atom_positions"],
                           residx_atom14_to_atom37[..., None],
                           axis=1))

    prot["atom14_atom_exists"] = residx_atom14_mask
    prot["atom14_gt_exists"] = residx_atom14_gt_mask
    prot["atom14_gt_positions"] = residx_atom14_gt_positions

    prot["residx_atom14_to_atom37"] = residx_atom14_to_atom37

    # Create the gather indices for mapping back.
    residx_atom37_to_atom14 = restype_atom37_to_atom14[prot["aatype"]]
    prot["residx_atom37_to_atom14"] = residx_atom37_to_atom14

    # Create the corresponding mask.
    restype_atom37_mask = np.zeros([21, 37], dtype=np.float32)
    for restype, restype_letter in enumerate(residue_constants.restypes):
        restype_name = residue_constants.restype_1to3[restype_letter]
        atom_names = residue_constants.residue_atoms[restype_name]
        for atom_name in atom_names:
            atom_type = residue_constants.atom_order[atom_name]
            restype_atom37_mask[restype, atom_type] = 1

    residx_atom37_mask = restype_atom37_mask[prot["aatype"]]
    prot["atom37_atom_exists"] = residx_atom37_mask

    # As the atom naming is ambiguous for 7 of the 20 amino acids, provide
    # alternative ground truth coordinates where the naming is swapped
    restype_3 = [
        residue_constants.restype_1to3[res] for res in residue_constants.restypes
    ]
    restype_3 += ["UNK"]

    # Matrices for renaming ambiguous atoms.
    all_matrices = {res: np.eye(14, dtype=np.float32) for res in restype_3}
    for resname, swap in residue_constants.residue_atom_renaming_swaps.items():
        correspondences = np.arange(14)
        for source_atom_swap, target_atom_swap in swap.items():
            source_index = residue_constants.restype_name_to_atom14_names[
                resname].index(source_atom_swap)
            target_index = residue_constants.restype_name_to_atom14_names[
                resname].index(target_atom_swap)
            correspondences[source_index] = target_index
            correspondences[target_index] = source_index
            renaming_matrix = np.zeros((14, 14), dtype=np.float32)
            for index, correspondence in enumerate(correspondences):
                renaming_matrix[index, correspondence] = 1.
        all_matrices[resname] = renaming_matrix.astype(np.float32)
    renaming_matrices = np.stack([all_matrices[restype] for restype in restype_3])

    # Pick the transformation matrices for the given residue sequence
    # shape (num_res, 14, 14).
    renaming_transform = renaming_matrices[prot["aatype"]]

    # Apply it to the ground truth positions. shape (num_res, 14, 3).
    alternative_gt_positions = np.einsum("rac,rab->rbc",
                                         residx_atom14_gt_positions,
                                         renaming_transform)
    prot["atom14_alt_gt_positions"] = alternative_gt_positions

    # Create the mask for the alternative ground truth (differs from the
    # ground truth mask, if only one of the atoms in an ambiguous pair has a
    # ground truth position).
    alternative_gt_mask = np.einsum("ra,rab->rb",
                                    residx_atom14_gt_mask,
                                    renaming_transform)

    prot["atom14_alt_gt_exists"] = alternative_gt_mask

    # Create an ambiguous atoms mask.  shape: (21, 14).
    restype_atom14_is_ambiguous = np.zeros((21, 14), dtype=np.float32)
    for resname, swap in residue_constants.residue_atom_renaming_swaps.items():
        for atom_name1, atom_name2 in swap.items():
            restype = residue_constants.restype_order[
                residue_constants.restype_3to1[resname]]
            atom_idx1 = residue_constants.restype_name_to_atom14_names[resname].index(
                atom_name1)
            atom_idx2 = residue_constants.restype_name_to_atom14_names[resname].index(
                atom_name2)
            restype_atom14_is_ambiguous[restype, atom_idx1] = 1
            restype_atom14_is_ambiguous[restype, atom_idx2] = 1

    # From this create an ambiguous_mask for the given sequence.
    prot["atom14_atom_is_ambiguous"] = (
        restype_atom14_is_ambiguous[prot["aatype"]])
    return prot


def atom37_to_frames(
        aatype,  # (...)
        all_atom_positions,  # (..., 37, 3)
        all_atom_mask,  # (..., 37)
        is_affine=False
):
    """Computes the frames for the up to 8 rigid groups for each residue.

  The rigid groups are defined by the possible torsions in a given amino acid.
  We group the atoms according to their dependence on the torsion angles into
  "rigid groups".  E.g., the position of atoms in the chi2-group depend on
  chi1 and chi2, but do not depend on chi3 or chi4.
  Jumper et al. (2021) Suppl. Table 2 and corresponding text.

  Args:
    aatype: Amino acid type, given as array with integers.
    all_atom_positions: atom37 representation of all atom coordinates.
    all_atom_mask: atom37 representation of mask on all atom coordinates.
    is_affine: whether get backbone_affine_tensor
  Returns:
    Dictionary containing:
      * 'rigidgroups_gt_frames': 8 Frames corresponding to 'all_atom_positions'
           represented as flat 12 dimensional array.
      * 'rigidgroups_gt_exists': Mask denoting whether the atom positions for
          the given frame are available in the ground truth, e.g. if they were
          resolved in the experiment.
      * 'rigidgroups_group_exists': Mask denoting whether given group is in
          principle present for given amino acid type.
      * 'rigidgroups_group_is_ambiguous': Mask denoting whether frame is
          affected by naming ambiguity.
      * 'rigidgroups_alt_gt_frames': 8 Frames with alternative atom renaming
          corresponding to 'all_atom_positions' represented as flat
          12 dimensional array.
  """
    # 0: 'backbone group',
    # 1: 'pre-omega-group', (empty)
    # 2: 'phi-group', (currently empty, because it defines only hydrogens)
    # 3: 'psi-group',
    # 4,5,6,7: 'chi1,2,3,4-group'
    aatype_in_shape = aatype.shape

    # If there is a batch axis, just flatten it away, and reshape everything
    # back at the end of the function.
    aatype = np.reshape(aatype, [-1])
    all_atom_positions = np.reshape(all_atom_positions, [-1, 37, 3])
    all_atom_mask = np.reshape(all_atom_mask, [-1, 37])

    # Create an array with the atom names.
    # shape (num_restypes, num_rigidgroups, 3_atoms): (21, 8, 3)
    restype_rigidgroup_base_atom_names = np.full([21, 8, 3], '', dtype=object)

    # 0: backbone frame
    restype_rigidgroup_base_atom_names[:, 0, :] = ['C', 'CA', 'N']

    # 3: 'psi-group'
    restype_rigidgroup_base_atom_names[:, 3, :] = ['CA', 'C', 'O']

    # 4,5,6,7: 'chi1,2,3,4-group'
    for restype, restype_letter in enumerate(residue_constants.restypes):
        resname = residue_constants.restype_1to3[restype_letter]
        for chi_idx in range(4):
            if residue_constants.chi_angles_mask[restype][chi_idx]:
                atom_names = residue_constants.chi_angles_atoms[resname][chi_idx]
                restype_rigidgroup_base_atom_names[restype, chi_idx + 4, :] = atom_names[1:]

    # Create mask for existing rigid groups.
    restype_rigidgroup_mask = np.zeros([21, 8], dtype=np.float32)
    restype_rigidgroup_mask[:, 0] = 1
    restype_rigidgroup_mask[:, 3] = 1
    restype_rigidgroup_mask[:20, 4:] = residue_constants.chi_angles_mask

    # Translate atom names into atom37 indices.
    lookuptable = residue_constants.atom_order.copy()
    lookuptable[''] = 0
    restype_rigidgroup_base_atom37_idx = np.vectorize(lambda x: lookuptable[x])(
        restype_rigidgroup_base_atom_names)

    # Compute the gather indices for all residues in the chain.
    # shape (N, 8, 3)
    residx_rigidgroup_base_atom37_idx = np_gather(
        restype_rigidgroup_base_atom37_idx, aatype)

    # Gather the base atom positions for each rigid group.
    base_atom_pos = np_gather(
        all_atom_positions,
        residx_rigidgroup_base_atom37_idx,
        batch_dims=1)

    # Compute the Rigids.
    gt_frames = r3.rigids_from_3_points(
        point_on_neg_x_axis=r3.vecs_from_tensor(base_atom_pos[:, :, 0, :]),
        origin=r3.vecs_from_tensor(base_atom_pos[:, :, 1, :]),
        point_on_xy_plane=r3.vecs_from_tensor(base_atom_pos[:, :, 2, :])
    )

    # Compute a mask whether the group exists.
    # (N, 8)
    group_exists = np_gather(restype_rigidgroup_mask, aatype)

    # Compute a mask whether ground truth exists for the group
    gt_atoms_exist = np_gather(  # shape (N, 8, 3)
        all_atom_mask.astype(np.float32),
        residx_rigidgroup_base_atom37_idx,
        batch_dims=1)
    gt_exists = np.min(gt_atoms_exist, axis=-1) * group_exists  # (N, 8)

    # Adapt backbone frame to old convention (mirror x-axis and z-axis).
    rots = np.tile(np.eye(3, dtype=np.float32), [8, 1, 1])
    rots[0, 0, 0] = -1
    rots[0, 2, 2] = -1
    gt_frames = r3.rigids_mul_rots(gt_frames, r3.rots_from_tensor3x3(rots))

    # The frames for ambiguous rigid groups are just rotated by 180 degree around
    # the x-axis. The ambiguous group is always the last chi-group.
    restype_rigidgroup_is_ambiguous = np.zeros([21, 8], dtype=np.float32)
    restype_rigidgroup_rots = np.tile(np.eye(3, dtype=np.float32), [21, 8, 1, 1])

    for resname, _ in residue_constants.residue_atom_renaming_swaps.items():
        restype = residue_constants.restype_order[
            residue_constants.restype_3to1[resname]]
        chi_idx = int(sum(residue_constants.chi_angles_mask[restype]) - 1)
        restype_rigidgroup_is_ambiguous[restype, chi_idx + 4] = 1
        restype_rigidgroup_rots[restype, chi_idx + 4, 1, 1] = -1
        restype_rigidgroup_rots[restype, chi_idx + 4, 2, 2] = -1

    # Gather the ambiguity information for each residue.
    residx_rigidgroup_is_ambiguous = np_gather(
        restype_rigidgroup_is_ambiguous, aatype)
    residx_rigidgroup_ambiguity_rot = np_gather(
        restype_rigidgroup_rots, aatype)

    # Create the alternative ground truth frames.
    alt_gt_frames = r3.rigids_mul_rots(
        gt_frames, r3.rots_from_tensor3x3(residx_rigidgroup_ambiguity_rot))

    gt_frames_flat12 = r3.rigids_to_tensor_flat12(gt_frames)
    alt_gt_frames_flat12 = r3.rigids_to_tensor_flat12(alt_gt_frames)

    # reshape back to original residue layout
    gt_frames_flat12 = np.reshape(gt_frames_flat12, aatype_in_shape + (8, 12))
    gt_exists = np.reshape(gt_exists, aatype_in_shape + (8,))
    group_exists = np.reshape(group_exists, aatype_in_shape + (8,))
    gt_frames_flat12 = np.reshape(gt_frames_flat12, aatype_in_shape + (8, 12))
    residx_rigidgroup_is_ambiguous = np.reshape(residx_rigidgroup_is_ambiguous,
                                                aatype_in_shape + (8,))
    alt_gt_frames_flat12 = np.reshape(alt_gt_frames_flat12,
                                      aatype_in_shape + (8, 12,))
    if not is_affine:
        return {
            'rigidgroups_gt_frames': gt_frames_flat12,  # (..., 8, 12)
            'rigidgroups_gt_exists': gt_exists,  # (..., 8)
            'rigidgroups_group_exists': group_exists,  # (..., 8)
            'rigidgroups_group_is_ambiguous':
                residx_rigidgroup_is_ambiguous,  # (..., 8)
            'rigidgroups_alt_gt_frames': alt_gt_frames_flat12,  # (..., 8, 12)
        }
    quataffine = rigids_to_quataffine(gt_frames)
    backbone_affine_tensor = quataffine.to_tensor()[:, 0, :]
    return {
        'rigidgroups_gt_frames': gt_frames_flat12,  # (..., 8, 12)
        'rigidgroups_gt_exists': gt_exists,  # (..., 8)
        'rigidgroups_group_exists': group_exists,  # (..., 8)
        'rigidgroups_group_is_ambiguous':
            residx_rigidgroup_is_ambiguous,  # (..., 8)
        'rigidgroups_alt_gt_frames': alt_gt_frames_flat12,  # (..., 8, 12)
        'backbone_affine_tensor': backbone_affine_tensor
    }


def get_chi_atom_indices():
    """Returns atom indices needed to compute chi angles for all residue types.

  Returns:
    A tensor of shape [residue_types=21, chis=4, atoms=4]. The residue types are
    in the order specified in residue_constants.restypes + unknown residue type
    at the end. For chi angles which are not defined on the residue, the
    positions indices are by default set to 0.
  """
    chi_atom_indices = []
    for residue_name in residue_constants.restypes:
        residue_name = residue_constants.restype_1to3[residue_name]
        residue_chi_angles = residue_constants.chi_angles_atoms[residue_name]
        atom_indices = []
        for chi_angle in residue_chi_angles:
            atom_indices.append(
                [residue_constants.atom_order[atom] for atom in chi_angle])
        for _ in range(4 - len(atom_indices)):
            atom_indices.append([0, 0, 0, 0])  # For chi angles not defined on the AA.
        chi_atom_indices.append(atom_indices)

    chi_atom_indices.append([[0, 0, 0, 0]] * 4)  # For UNKNOWN residue.

    return np.array(chi_atom_indices)


def atom37_to_torsion_angles(
        aatype: np.ndarray,  # (B, N)
        all_atom_pos: np.ndarray,  # (B, N, 37, 3)
        all_atom_mask: np.ndarray,  # (B, N, 37)
        placeholder_for_undefined=False,
):
    """Computes the 7 torsion angles (in sin, cos encoding) for each residue.

  The 7 torsion angles are in the order
  '[pre_omega, phi, psi, chi_1, chi_2, chi_3, chi_4]',
  here pre_omega denotes the omega torsion angle between the given amino acid
  and the previous amino acid.

  Args:
    aatype: Amino acid type, given as array with integers.
    all_atom_pos: atom37 representation of all atom coordinates.
    all_atom_mask: atom37 representation of mask on all atom coordinates.
    placeholder_for_undefined: flag denoting whether to set masked torsion
      angles to zero.
  Returns:
    Dict containing:
      * 'torsion_angles_sin_cos': Array with shape (B, N, 7, 2) where the final
        2 dimensions denote sin and cos respectively
      * 'alt_torsion_angles_sin_cos': same as 'torsion_angles_sin_cos', but
        with the angle shifted by pi for all chi angles affected by the naming
        ambiguities.
      * 'torsion_angles_mask': Mask for which chi angles are present.
  """

    # Map aatype > 20 to 'Unknown' (20).
    aatype = np.minimum(aatype, 20)

    # Compute the backbone angles.
    num_batch, num_res = aatype.shape

    pad = np.zeros([num_batch, 1, 37, 3], np.float32)
    prev_all_atom_pos = np.concatenate([pad, all_atom_pos[:, :-1, :, :]], axis=1)

    pad = np.zeros([num_batch, 1, 37], np.float32)
    prev_all_atom_mask = np.concatenate([pad, all_atom_mask[:, :-1, :]], axis=1)

    # For each torsion angle collect the 4 atom positions that define this angle.
    # shape (B, N, atoms=4, xyz=3)
    pre_omega_atom_pos = np.concatenate(
        [prev_all_atom_pos[:, :, 1:3, :],  # prev CA, C
         all_atom_pos[:, :, 0:2, :]  # this N, CA
         ], axis=-2)
    phi_atom_pos = np.concatenate(
        [prev_all_atom_pos[:, :, 2:3, :],  # prev C
         all_atom_pos[:, :, 0:3, :]  # this N, CA, C
         ], axis=-2)
    psi_atom_pos = np.concatenate(
        [all_atom_pos[:, :, 0:3, :],  # this N, CA, C
         all_atom_pos[:, :, 4:5, :]  # this O
         ], axis=-2)

    # Collect the masks from these atoms.
    # Shape [batch, num_res]
    pre_omega_mask = (np.prod(prev_all_atom_mask[:, :, 1:3], axis=-1) *
                      np.prod(all_atom_mask[:, :, 0:2], axis=-1))  # this N, CA
    phi_mask = (prev_all_atom_mask[:, :, 2] * np.prod(all_atom_mask[:, :, 0:3], axis=-1))  # this N, CA, C
    psi_mask = (np.prod(all_atom_mask[:, :, 0:3], axis=-1) * all_atom_mask[:, :, 4])  # this O

    # Collect the atoms for the chi-angles.
    # Compute the table of chi angle indices. Shape: [restypes, chis=4, atoms=4].
    chi_atom_indices = get_chi_atom_indices()
    # Select atoms to compute chis. Shape: [batch, num_res, chis=4, atoms=4].
    atom_indices = np_gather(params=chi_atom_indices, indices=aatype, axis=0, batch_dims=0)
    # Gather atom positions. Shape: [batch, num_res, chis=4, atoms=4, xyz=3].
    chis_atom_pos = np_gather(params=all_atom_pos, indices=atom_indices, axis=-2, batch_dims=2)

    # Copy the chi angle mask, add the UNKNOWN residue. Shape: [restypes, 4].
    chi_angles_mask = list(residue_constants.chi_angles_mask)
    chi_angles_mask.append([0.0, 0.0, 0.0, 0.0])
    chi_angles_mask = np.array(chi_angles_mask)

    # Compute the chi angle mask. I.e. which chis angles exist according to the
    # aatype. Shape [batch, num_res, chis=4].
    chis_mask = np_gather(params=chi_angles_mask, indices=aatype,
                          axis=0, batch_dims=0)

    # Constrain the chis_mask to those chis, where the ground truth coordinates of
    # all defining four atoms are available.
    # Gather the chi angle atoms mask. Shape: [batch, num_res, chis=4, atoms=4].
    chi_angle_atoms_mask = np_gather(
        params=all_atom_mask, indices=atom_indices, axis=-1,
        batch_dims=2)
    # Check if all 4 chi angle atoms were set. Shape: [batch, num_res, chis=4].
    chi_angle_atoms_mask = np.prod(chi_angle_atoms_mask, axis=-1)
    chis_mask = chis_mask * (chi_angle_atoms_mask).astype(np.float32)

    # Stack all torsion angle atom positions.
    # Shape (B, N, torsions=7, atoms=4, xyz=3)
    torsions_atom_pos = np.concatenate(
        [pre_omega_atom_pos[:, :, None, :, :],
         phi_atom_pos[:, :, None, :, :],
         psi_atom_pos[:, :, None, :, :],
         chis_atom_pos
         ], axis=2)

    # Stack up masks for all torsion angles.
    # shape (B, N, torsions=7)
    torsion_angles_mask = np.concatenate(
        [pre_omega_mask[:, :, None],
         phi_mask[:, :, None],
         psi_mask[:, :, None],
         chis_mask
         ], axis=2)

    # Create a frame from the first three atoms:
    # First atom: point on x-y-plane
    # Second atom: point on negative x-axis
    # Third atom: origin
    # r3.Rigids (B, N, torsions=7)
    torsion_frames = r3.rigids_from_3_points(
        point_on_neg_x_axis=r3.vecs_from_tensor(torsions_atom_pos[:, :, :, 1, :]),
        origin=r3.vecs_from_tensor(torsions_atom_pos[:, :, :, 2, :]),
        point_on_xy_plane=r3.vecs_from_tensor(torsions_atom_pos[:, :, :, 0, :]))

    # Compute the position of the forth atom in this frame (y and z coordinate
    # define the chi angle)
    # r3.Vecs (B, N, torsions=7)
    forth_atom_rel_pos = r3.rigids_mul_vecs(
        r3.invert_rigids(torsion_frames),
        r3.vecs_from_tensor(torsions_atom_pos[:, :, :, 3, :]))

    # Normalize to have the sin and cos of the torsion angle.
    # np.ndarray (B, N, torsions=7, sincos=2)
    torsion_angles_sin_cos = np.stack(
        [forth_atom_rel_pos.z, forth_atom_rel_pos.y], axis=-1)
    torsion_angles_sin_cos /= np.sqrt(
        np.sum(np.square(torsion_angles_sin_cos), axis=-1, keepdims=True)
        + 1e-8)

    # Mirror psi, because we computed it from the Oxygen-atom.
    torsion_angles_sin_cos *= np.array(
        [1., 1., -1., 1., 1., 1., 1.])[None, None, :, None]

    # Create alternative angles for ambiguous atom names.
    chi_is_ambiguous = np_gather(
        np.array(residue_constants.chi_pi_periodic), aatype)
    mirror_torsion_angles = np.concatenate(
        [np.ones([num_batch, num_res, 3]),
         1.0 - 2.0 * chi_is_ambiguous], axis=-1)
    alt_torsion_angles_sin_cos = (torsion_angles_sin_cos * mirror_torsion_angles[:, :, :, None])

    if placeholder_for_undefined:
        # Add placeholder torsions in place of undefined torsion angles
        # (e.g. N-terminus pre-omega)
        placeholder_torsions = np.stack([np.ones(torsion_angles_sin_cos.shape[:-1]),
                                         np.zeros(torsion_angles_sin_cos.shape[:-1])], axis=-1)
        torsion_angles_sin_cos = torsion_angles_sin_cos * torsion_angles_mask[
            ..., None] + placeholder_torsions * (1 - torsion_angles_mask[..., None])
        alt_torsion_angles_sin_cos = alt_torsion_angles_sin_cos * torsion_angles_mask[
            ..., None] + placeholder_torsions * (1 - torsion_angles_mask[..., None])

    return {
        'torsion_angles_sin_cos': torsion_angles_sin_cos,  # (B, N, 7, 2)
        'alt_torsion_angles_sin_cos': alt_torsion_angles_sin_cos,  # (B, N, 7, 2)
        'torsion_angles_mask': torsion_angles_mask  # (B, N, 7)
    }


class SeedMaker:
    """Return unique seeds."""

    def __init__(self, initial_seed=0):
        self.next_seed = initial_seed

    def __call__(self):
        i = self.next_seed
        self.next_seed += 1
        return i


global_seed = SeedMaker()


def crop_to_fix_size(gt_features, config, seed):
    """crop and pad to fix size"""
    common_cfg = config.data.common
    eval_cfg = config.data.eval
    num_res = 'num residues placeholder'
    crop_feats = {'aatype': [num_res],
                  'all_atom_mask': [num_res, None],
                  'all_atom_positions': [num_res, None, None],
                  'atom14_alt_gt_exists': [num_res, None],
                  'atom14_alt_gt_positions': [num_res, None, None],
                  'atom14_atom_exists': [num_res, None],
                  'atom14_atom_is_ambiguous': [num_res, None],
                  'atom14_gt_exists': [num_res, None],
                  'atom14_gt_positions': [num_res, None, None],
                  'rigidgroups_alt_gt_frames': [num_res, None, None],
                  'rigidgroups_group_exists': [num_res, None],
                  'rigidgroups_group_is_ambiguous': [num_res, None],
                  'rigidgroups_gt_exists': [num_res, None],
                  'rigidgroups_gt_frames': [num_res, None, None],
                  'pseudo_beta': [num_res, None],
                  'pseudo_beta_mask': [num_res],
                  'torsion_angles_sin_cos': [num_res, None, None],
                  'alt_torsion_angles_sin_cos': [num_res, None, None],
                  'torsion_angles_mask': [num_res, None],
                  'backbone_affine_tensor': [num_res, None],
                  'seq_length': []}
    map_fns = [
        data_transforms.select_feat(list(crop_feats)),
        data_transforms.random_crop_to_size(eval_cfg.crop_size, eval_cfg.max_templates, crop_feats,
                                            eval_cfg.subsample_templates, seed),
        data_transforms.make_fixed_size(crop_feats, 5, common_cfg.max_extra_msa, eval_cfg.crop_size,
                                        eval_cfg.max_templates)]
    gt_features = data_transforms.compose(map_fns)(gt_features)
    return gt_features


def get_train_data(train_data_dir, prot_name, config, raw_feature_path, names_all, is_crop=True):
    """get train data"""
    flag = True
    feature_dict = None
    prot_pdb = None
    try_count = 0
    while flag:
        try:
            pdb_path = os.path.join(train_data_dir, prot_name + '.pdb')
            with open(pdb_path, 'r') as f:
                prot_pdb = protein.from_pdb_string(f.read())
            with open(os.path.join(raw_feature_path, prot_name + '.pkl'), "rb") as f:
                feature_dict = pickle.load(f)
            flag = False
        except ValueError:
            print("cannot get pickle data: ", prot_name, "will try to get: ", names_all[try_count])
            prot_name = names_all[try_count]
            try_count += 1
    aatype = prot_pdb.aatype
    seq_len = len(aatype)
    atom37_positions = prot_pdb.atom_positions.astype(np.float32)
    atom37_mask = prot_pdb.atom_mask.astype(np.float32)

    # get ground truth of atom14
    features = {'aatype': aatype,
                'all_atom_positions': atom37_positions,
                'all_atom_mask': atom37_mask}
    features = make_atom14_positions(features)

    # get ground truth of rigid groups
    rigidgroups_gt_feature = atom37_to_frames(aatype, atom37_positions, atom37_mask, is_affine=True)

    # get ground truth of angle
    angle_gt_feature = atom37_to_torsion_angles(aatype.reshape((1, -1)), atom37_positions.reshape((1, seq_len, 37, 3)),
                                                atom37_mask.reshape((1, seq_len, 37)), True)

    # get pseudo_beta, pseudo_beta_mask
    pseudo_beta, pseudo_beta_mask = data_transforms.pseudo_beta_fn(aatype, atom37_positions, atom37_mask)

    # combine all gt features
    gt_features = {'seq_length': seq_len, 'all_atom_positions': atom37_positions, 'all_atom_mask': atom37_mask,
                   'pseudo_beta': pseudo_beta, 'pseudo_beta_mask': pseudo_beta_mask}
    atom14_gt_features = ["atom14_gt_positions", "atom14_alt_gt_positions", "atom14_atom_is_ambiguous",
                          "atom14_gt_exists", "atom14_atom_exists", "atom14_alt_gt_exists"]
    for gt in atom14_gt_features:
        gt_features[gt] = features[gt]
    for key in rigidgroups_gt_feature:
        gt_features[key] = rigidgroups_gt_feature[key]
    for key in angle_gt_feature:
        gt_features[key] = angle_gt_feature[key][0]

    seed = global_seed()
    # crop to fix size
    if is_crop:
        gt_features = crop_to_fix_size(gt_features, config, seed)

    processed_feature_dict = process_features_train(feature_dict, config, seed)
    gt_features.pop('seq_length')
    gt_features["chi_mask"] = gt_features["torsion_angles_mask"][:, 3:]
    processed_feature_dict.update(gt_features)

    int_key = ['aatype', 'seq_length']
    float_key = ['extra_msa_mask', 'extra_msa_row_mask', 'extra_has_deletion', 'extra_deletion_value', 'msa_feat',
                 'target_feat', 'alt_torsion_angles_sin_cos']
    for key in int_key:
        processed_feature_dict[key] = processed_feature_dict[key].astype(np.int32)
    for key in float_key:
        processed_feature_dict[key] = processed_feature_dict[key].astype(np.float32)
    processed_feature_dict['backbone_affine_mask'] = processed_feature_dict['pseudo_beta_mask']
    return processed_feature_dict


def create_dataset(train_data_dir, raw_feature_dir, names, model_config, center_name_path, shuffle=False,
                   num_parallel_worker=4,
                   is_parallel=False):
    """create train dataset"""
    column_name = ["target_feat", "msa_feat", "msa_mask", "seq_mask_batch", "aatype_batch",
                   "template_aatype", "template_all_atom_masks",
                   "template_all_atom_positions", "template_mask",
                   "template_pseudo_beta_mask", "template_pseudo_beta",
                   "template_sum_probs", "extra_msa", "extra_has_deletion",
                   "extra_deletion_value", "extra_msa_mask", "residx_atom37_to_atom14",
                   "atom37_atom_exists_batch", "residue_index_batch", "prev_pos",
                   "prev_msa_first_row", "prev_pair", "pseudo_beta_gt",
                   "pseudo_beta_mask_gt", "all_atom_mask_gt", "atomtype_radius",
                   "true_msa", "bert_mask", "residue_index", "seq_mask",
                   "atom37_atom_exists", "aatype", "restype_atom14_bond_lower_bound",
                   "restype_atom14_bond_upper_bound", "residx_atom14_to_atom37",
                   "atom14_atom_exists", "backbone_affine_tensor", "backbone_affine_mask",
                   "atom14_gt_positions", "atom14_alt_gt_positions",
                   "atom14_atom_is_ambiguous", "atom14_gt_exists", "atom14_alt_gt_exists",
                   "all_atom_positions", "rigidgroups_gt_frames", "rigidgroups_gt_exists",
                   "rigidgroups_alt_gt_frames", "torsion_angles_sin_cos_gt", "use_clamped_fape", "filter_by_solution",
                   'prot_name_index', 'chi_mask']

    dataset_generator = DatasetGenerator(train_data_dir, raw_feature_dir, names, model_config, center_name_path)
    ds.config.set_prefetch_size(1)

    if is_parallel:
        rank_id = get_rank() % 8
        rank_size = 8
        train_dataset = ds.GeneratorDataset(source=dataset_generator, column_names=column_name,
                                            num_parallel_workers=num_parallel_worker, shuffle=shuffle,
                                            num_shards=rank_size,
                                            shard_id=rank_id, max_rowsize=16)
    else:
        train_dataset = ds.GeneratorDataset(source=dataset_generator, column_names=column_name,
                                            num_parallel_workers=num_parallel_worker, shuffle=shuffle, max_rowsize=16)
    return train_dataset


class DatasetGenerator:
    """dataset generator"""
    def __init__(self, train_data_dir, raw_feature_dir, names, model_config, resolution_data):
        self.t1 = time.time()
        print("start dataset init: ", str(datetime.datetime.now()))
        self.model_config = model_config
        self.model_config.data.eval.num_ensemble = 1
        self.num_residues = model_config.data.eval.crop_size
        self.msa_channel = model_config.model.embeddings_and_evoformer.msa_channel
        self.pair_channel = model_config.model.embeddings_and_evoformer.pair_channel
        self.train_data_dir = train_data_dir
        self.raw_feature_dir = raw_feature_dir
        self.names = [name.replace("\n", "") for name in names]

        self.resolution_info = resolution_data
        print("end dataset init: ", time.time() - self.t1)

    def get_resolution_info(self, resolution_path):
        with open(resolution_path, 'r') as f:
            data = json.load(f)
        return data

    def get_solution_flag(self, prot_name):
        prot_new_name = prot_name.rsplit('_', 1)[0]
        if prot_new_name not in self.resolution_info:
            return np.array(1.0).astype(np.float32)
        resolution = float(self.resolution_info[prot_new_name]['resolution'])
        nmr = self.resolution_info[prot_new_name]['method']
        if resolution < 3 and nmr != 'NMR':
            return np.array(1.0).astype(np.float32)
        return np.array(0.0).astype(np.float32)

    def __getitem__(self, index):
        prot_name = self.names[index]
        prot_name_index = np.asarray([index]).astype(np.int32)
        features = get_train_data(self.train_data_dir, prot_name, self.model_config, self.raw_feature_dir, self.names,
                                  is_crop=True)
        target_feat = features["target_feat"].astype(np.float32)
        msa_feat = features["msa_feat"].astype(np.float32)
        msa_mask = features["msa_mask"]
        bert_mask = features['bert_mask']
        true_msa = features['true_msa']
        residx_atom14_to_atom37 = features['residx_atom14_to_atom37'][0]
        seq_mask_batch = features["seq_mask"]
        seq_mask = features["seq_mask"][0]
        aatype_batch = features["aatype"]
        aatype = features["aatype"][0]
        residue_index_batch = features["residue_index"]
        residue_index = features["residue_index"][0]
        atom37_atom_exists_batch = features["atom37_atom_exists"]
        atom37_atom_exists = features["atom37_atom_exists"][0]

        template_aatype = features["template_aatype"]
        template_all_atom_masks = features["template_all_atom_masks"]
        template_all_atom_positions = features["template_all_atom_positions"]
        template_mask = features["template_mask"]
        template_pseudo_beta_mask = features["template_pseudo_beta_mask"]
        template_pseudo_beta = features["template_pseudo_beta"]
        template_sum_probs = features["template_sum_probs"]

        extra_msa = features["extra_msa"]
        extra_has_deletion = features["extra_has_deletion"].astype(np.float32)
        extra_deletion_value = features["extra_deletion_value"].astype(np.float32)
        extra_msa_mask = features["extra_msa_mask"].astype(np.float32)
        atom14_atom_exists = features["atom14_atom_exists"].astype(np.int32)
        residx_atom37_to_atom14 = features["residx_atom37_to_atom14"].astype(np.int32)
        atomtype_radius = np.array(
            [1.55, 1.7, 1.7, 1.7, 1.52, 1.7, 1.7, 1.7, 1.52, 1.52, 1.8, 1.7, 1.7, 1.7, 1.55, 1.55,
             1.52, 1.52, 1.8, 1.7, 1.7, 1.7, 1.7, 1.55, 1.55, 1.55, 1.52, 1.52, 1.7, 1.55, 1.55,
             1.52, 1.7, 1.7, 1.7, 1.55, 1.52], np.float32)
        prev_pos = np.zeros([self.num_residues, 37, 3]).astype(np.float32)
        prev_msa_first_row = np.zeros([self.num_residues, self.msa_channel]).astype(np.float32)
        prev_pair = np.zeros([self.num_residues, self.num_residues, self.pair_channel]).astype(np.float32)
        restype_atom14_bond_lower_bound, restype_atom14_bond_upper_bound, _ = \
            residue_constants.make_atom14_dists_bounds(
                overlap_tolerance=self.model_config.model.heads.structure_module.clash_overlap_tolerance,
                bond_length_tolerance_factor=self.model_config.model.heads.structure_module.violation_tolerance_factor)
        pseudo_beta_gt = features["pseudo_beta"]
        pseudo_beta_mask_gt = features["pseudo_beta_mask"]
        all_atom_mask_gt = features["all_atom_mask"]
        backbone_affine_tensor = features["backbone_affine_tensor"]
        backbone_affine_mask = features["backbone_affine_mask"]
        atom14_gt_positions = features["atom14_gt_positions"]
        atom14_alt_gt_positions = features["atom14_alt_gt_positions"]
        atom14_atom_is_ambiguous = features["atom14_atom_is_ambiguous"]
        atom14_gt_exists = features["atom14_gt_exists"]
        atom14_alt_gt_exists = features["atom14_alt_gt_exists"]
        all_atom_positions = features["all_atom_positions"]
        rigidgroups_gt_frames = features["rigidgroups_gt_frames"]
        rigidgroups_gt_exists = features["rigidgroups_gt_exists"]
        rigidgroups_alt_gt_frames = features["rigidgroups_alt_gt_frames"]
        torsion_angles_sin_cos_gt = features["torsion_angles_sin_cos"][None, :, 3:, :].astype(np.float32)
        chi_mask = features["chi_mask"].astype(np.float32)
        # use_clamped_fape = np.random.binomial(1, 0.9, size=1).astype(np.int32)//TODO
        # filter_by_solution = self.get_solution_flag(prot_name) //TODO
        use_clamped_fape = np.array(0.0).astype(np.float32)
        filter_by_solution = np.array(1.0).astype(np.float32)

        return (target_feat, msa_feat, msa_mask, seq_mask_batch, aatype_batch, template_aatype,
                template_all_atom_masks, template_all_atom_positions, template_mask,
                template_pseudo_beta_mask, template_pseudo_beta, template_sum_probs, extra_msa,
                extra_has_deletion, extra_deletion_value, extra_msa_mask, residx_atom37_to_atom14,
                atom37_atom_exists_batch, residue_index_batch, prev_pos, prev_msa_first_row,
                prev_pair, pseudo_beta_gt, pseudo_beta_mask_gt, all_atom_mask_gt, atomtype_radius, true_msa, bert_mask,
                residue_index, seq_mask, atom37_atom_exists, aatype, restype_atom14_bond_lower_bound,
                restype_atom14_bond_upper_bound, residx_atom14_to_atom37, atom14_atom_exists, backbone_affine_tensor,
                backbone_affine_mask, atom14_gt_positions, atom14_alt_gt_positions, atom14_atom_is_ambiguous,
                atom14_gt_exists, atom14_alt_gt_exists, all_atom_positions, rigidgroups_gt_frames,
                rigidgroups_gt_exists, rigidgroups_alt_gt_frames, torsion_angles_sin_cos_gt, use_clamped_fape,
                filter_by_solution, prot_name_index, chi_mask)

    def __len__(self):
        return len(self.names)
