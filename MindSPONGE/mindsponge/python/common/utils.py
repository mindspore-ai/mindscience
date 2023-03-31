# Copyright 2021 The AIMM Group at Shenzhen Bay Laboratory & Peking University & Huawei Technologies Co., Ltd
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
"""utils module"""

import numpy as np
from Bio import Align
from Bio.Align import substitution_matrices
from mindspore import nn
import mindspore.numpy as mnp
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from . import geometry
from . import residue_constants, protein


def _memory_reduce(body, batched_inputs, nonbatched_inputs, slice_num, dim=0):
    """memory reduce function"""
    if slice_num <= 1:
        inputs = batched_inputs + nonbatched_inputs
        return body(*inputs)
    inner_batched_inputs = []
    for val in batched_inputs:
        inner_val = P.Split(dim, slice_num)(val)
        inner_batched_inputs.append(inner_val)
    # for depend
    inner_split_batched_inputs = ()
    for j in range(len(inner_batched_inputs)):
        inner_split_batched_inputs = inner_split_batched_inputs + (inner_batched_inputs[j][0],)
    inner_split_inputs = inner_split_batched_inputs + nonbatched_inputs
    inner_split_res = body(*inner_split_inputs)
    res = (inner_split_res,)
    for i in range(1, slice_num):
        inner_split_batched_inputs = ()
        for j in range(len(inner_batched_inputs)):
            inner_split_batched_inputs = inner_split_batched_inputs + (inner_batched_inputs[j][i],)
        inner_split_inputs = inner_split_batched_inputs + nonbatched_inputs
        inner_split_inputs = F.depend(inner_split_inputs, res[-1])
        inner_split_res = body(*inner_split_inputs)
        res = res + (inner_split_res,)
    res = P.Concat()(res)
    return res


def pseudo_beta_fn(aatype, all_atom_positions, all_atom_masks):
    """Create pseudo beta features."""

    is_gly = mnp.equal(aatype, residue_constants.restype_order['G'])
    ca_idx = residue_constants.atom_order['CA']
    cb_idx = residue_constants.atom_order['CB']
    pseudo_beta = mnp.where(
        mnp.tile(is_gly[..., None], [1,] * len(is_gly.shape) + [3,]),
        all_atom_positions[..., ca_idx, :],
        all_atom_positions[..., cb_idx, :])
    if all_atom_masks is not None:
        pseudo_beta_mask = mnp.where(is_gly, all_atom_masks[..., ca_idx], all_atom_masks[..., cb_idx])
        pseudo_beta_mask = pseudo_beta_mask.astype(mnp.float32)
        return pseudo_beta, pseudo_beta_mask
    return pseudo_beta


def dgram_from_positions(positions, num_bins, min_bin, max_bin, ret_type):
    """Compute distogram from amino acid positions.

    Arguments:
    positions: [N_res, 3] Position coordinates.
    num_bins: The number of bins in the distogram.
    min_bin: The left edge of the first bin.
    max_bin: The left edge of the final bin. The final bin catches
    everything larger than `max_bin`.

    Returns:
    Distogram with the specified number of bins.
    """

    def squared_difference(x, y):
        return mnp.square(x - y)

    lower_breaks = mnp.linspace(min_bin, max_bin, num_bins)
    lower_breaks = mnp.square(lower_breaks)
    upper_breaks = mnp.concatenate([lower_breaks[1:], mnp.array([1e8], dtype=mnp.float32)], axis=-1)
    dist2 = mnp.sum(squared_difference(mnp.expand_dims(positions, axis=-2),
                                       mnp.expand_dims(positions, axis=-3)), axis=-1, keepdims=True)
    dgram = ((dist2 > lower_breaks).astype(ret_type) * (dist2 < upper_breaks).astype(ret_type))
    return dgram


def atom37_to_torsion_angles(
        aatype,  # (B, N)
        all_atom_pos,  # (B, N, 37, 3)
        all_atom_mask,  # (B, N, 37)
        chi_atom_indices,
        chi_angles_mask,
        mirror_psi_mask,
        chi_pi_periodic,
        indices0,
        indices1
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
    aatype = mnp.minimum(aatype, 20)

    # Compute the backbone angles.
    num_batch, num_res = aatype.shape

    pad = mnp.zeros([num_batch, 1, 37, 3], mnp.float32)
    prev_all_atom_pos = mnp.concatenate([pad, all_atom_pos[:, :-1, :, :]], axis=1)

    pad = mnp.zeros([num_batch, 1, 37], mnp.float32)
    prev_all_atom_mask = mnp.concatenate([pad, all_atom_mask[:, :-1, :]], axis=1)

    # For each torsion angle collect the 4 atom positions that define this angle.
    # shape (B, N, atoms=4, xyz=3)
    pre_omega_atom_pos = mnp.concatenate([prev_all_atom_pos[:, :, 1:3, :], all_atom_pos[:, :, 0:2, :]], axis=-2)
    phi_atom_pos = mnp.concatenate([prev_all_atom_pos[:, :, 2:3, :], all_atom_pos[:, :, 0:3, :]], axis=-2)
    psi_atom_pos = mnp.concatenate([all_atom_pos[:, :, 0:3, :], all_atom_pos[:, :, 4:5, :]], axis=-2)
    # # Collect the masks from these atoms.
    # # Shape [batch, num_res]
    # ERROR NO PROD
    pre_omega_mask = (P.ReduceProd()(prev_all_atom_mask[:, :, 1:3], -1)  # prev CA, C
                      * P.ReduceProd()(all_atom_mask[:, :, 0:2], -1))  # this N, CA
    phi_mask = (prev_all_atom_mask[:, :, 2]  # prev C
                * P.ReduceProd()(all_atom_mask[:, :, 0:3], -1))  # this N, CA, C
    psi_mask = (P.ReduceProd()(all_atom_mask[:, :, 0:3], -1) *  # this N, CA, C
                all_atom_mask[:, :, 4])  # this O
    # Collect the atoms for the chi-angles.
    # Compute the table of chi angle indices. Shape: [restypes, chis=4, atoms=4].
    # Select atoms to compute chis. Shape: [batch, num_res, chis=4, atoms=4].
    atom_indices = mnp.take(chi_atom_indices, aatype, axis=0)

    # # Gather atom positions Batch Gather. Shape: [batch, num_res, chis=4, atoms=4, xyz=3].

    # 4 seq_length 4 4  batch, sequence length, chis, atoms
    seq_length = all_atom_pos.shape[1]
    atom_indices = atom_indices.reshape((4, seq_length, 4, 4, 1)).astype("int32")
    new_indices = P.Concat(4)((indices0, indices1, atom_indices))  # 4, seq_length, 4, 4, 3
    chis_atom_pos = P.GatherNd()(all_atom_pos, new_indices)
    chis_mask = mnp.take(chi_angles_mask, aatype, axis=0)
    chi_angle_atoms_mask = P.GatherNd()(all_atom_mask, new_indices)

    # Check if all 4 chi angle atoms were set. Shape: [batch, num_res, chis=4].
    chi_angle_atoms_mask = P.ReduceProd()(chi_angle_atoms_mask, -1)
    chis_mask = chis_mask * (chi_angle_atoms_mask).astype(mnp.float32)

    # Stack all torsion angle atom positions.
    # Shape (B, N, torsions=7, atoms=4, xyz=3)ls
    torsions_atom_pos = mnp.concatenate([pre_omega_atom_pos[:, :, None, :, :],
                                         phi_atom_pos[:, :, None, :, :],
                                         psi_atom_pos[:, :, None, :, :],
                                         chis_atom_pos], axis=2)
    # Stack up masks for all torsion angles.
    # shape (B, N, torsions=7)
    torsion_angles_mask = mnp.concatenate([pre_omega_mask[:, :, None],
                                           phi_mask[:, :, None],
                                           psi_mask[:, :, None],
                                           chis_mask], axis=2)

    torsion_rigid = geometry.rigids_from_3_points(
        geometry.vecs_from_tensor(torsions_atom_pos[:, :, :, 1, :]),
        geometry.vecs_from_tensor(torsions_atom_pos[:, :, :, 2, :]),
        geometry.vecs_from_tensor(torsions_atom_pos[:, :, :, 0, :]))
    inv_torsion_rigid = geometry.invert_rigids(torsion_rigid)
    forth_atom_rel_pos = geometry.rigids_mul_vecs(inv_torsion_rigid,
                                                  geometry.vecs_from_tensor(torsions_atom_pos[:, :, :, 3, :]))
    # Compute the position of the forth atom in this frame (y and z coordinate
    torsion_angles_sin_cos = mnp.stack([forth_atom_rel_pos[2], forth_atom_rel_pos[1]], axis=-1)
    torsion_angles_sin_cos /= mnp.sqrt(mnp.sum(mnp.square(torsion_angles_sin_cos), axis=-1, keepdims=True) + 1e-8)
    # Mirror psi, because we computed it from the Oxygen-atom.
    torsion_angles_sin_cos *= mirror_psi_mask
    chi_is_ambiguous = mnp.take(chi_pi_periodic, aatype, axis=0)
    mirror_torsion_angles = mnp.concatenate([mnp.ones([num_batch, num_res, 3]), 1.0 - 2.0 * chi_is_ambiguous], axis=-1)
    alt_torsion_angles_sin_cos = (torsion_angles_sin_cos * mirror_torsion_angles[:, :, :, None])
    return torsion_angles_sin_cos, alt_torsion_angles_sin_cos, torsion_angles_mask


def rigids_from_tensor4x4(m):
    """Construct Rigids object from an 4x4 array.

    Here the 4x4 is representing the transformation in homogeneous coordinates.

    Args:
    m: Array representing transformations in homogeneous coordinates.
    Returns:
    Rigids object corresponding to transformations m
    """
    rotation = (m[..., 0, 0], m[..., 0, 1], m[..., 0, 2],
                m[..., 1, 0], m[..., 1, 1], m[..., 1, 2],
                m[..., 2, 0], m[..., 2, 1], m[..., 2, 2])
    trans = (m[..., 0, 3], m[..., 1, 3], m[..., 2, 3])
    rigid = (rotation, trans)
    return rigid


def frames_and_literature_positions_to_atom14_pos(aatype, all_frames_to_global, restype_atom14_to_rigid_group,
                                                  restype_atom14_rigid_group_positions, restype_atom14_mask):  # (N, 14)
    """Put atom literature positions (atom14 encoding) in each rigid group.

    Jumper et al. (2021) Suppl. Alg. 24 "computeAllAtomCoordinates" line 11

    Args:
    aatype: aatype for each residue.
    all_frames_to_global: All per residue coordinate frames.
    Returns:
    Positions of all atom coordinates in global frame.
    """

    # Pick the appropriate transform for every atom.
    residx_to_group_idx = P.Gather()(restype_atom14_to_rigid_group, aatype, 0)
    group_mask = nn.OneHot(depth=8, axis=-1)(residx_to_group_idx)

    # Rigids with shape (N, 14)
    map_atoms_to_global = map_atoms_to_global_func(all_frames_to_global, group_mask)

    # Gather the literature atom positions for each residue.
    # Vecs with shape (N, 14)
    lit_positions = geometry.vecs_from_tensor(P.Gather()(restype_atom14_rigid_group_positions, aatype, 0))

    # Transform each atom from its local frame to the global frame.
    # Vecs with shape (N, 14)
    pred_positions = geometry.rigids_mul_vecs(map_atoms_to_global, lit_positions)

    # Mask out non-existing atoms.
    mask = P.Gather()(restype_atom14_mask, aatype, 0)

    pred_positions = geometry.vecs_scale(pred_positions, mask)

    return pred_positions


def rigids_concate_all(xall, x5, x6, x7):
    """rigids concate all."""
    x5 = (geometry.rots_expand_dims(x5[0], -1), geometry.vecs_expand_dims(x5[1], -1))
    x6 = (geometry.rots_expand_dims(x6[0], -1), geometry.vecs_expand_dims(x6[1], -1))
    x7 = (geometry.rots_expand_dims(x7[0], -1), geometry.vecs_expand_dims(x7[1], -1))
    xall_rot = xall[0]
    xall_rot_slice = []
    for val in xall_rot:
        xall_rot_slice.append(val[:, 0:5])
    xall_trans = xall[1]
    xall_trans_slice = []
    for val in xall_trans:
        xall_trans_slice.append(val[:, 0:5])
    xall = (xall_rot_slice, xall_trans_slice)
    res_rot = []
    for i in range(9):
        res_rot.append(mnp.concatenate((xall[0][i], x5[0][i], x6[0][i], x7[0][i]), axis=-1))
    res_trans = []
    for i in range(3):
        res_trans.append(mnp.concatenate((xall[1][i], x5[1][i], x6[1][i], x7[1][i]), axis=-1))
    return (res_rot, res_trans)


def torsion_angles_to_frames(aatype, backb_to_global, torsion_angles_sin_cos, restype_rigid_group_default_frame):
    """Compute rigid group frames from torsion angles."""

    # Gather the default frames for all rigid groups.
    m = P.Gather()(restype_rigid_group_default_frame, aatype, 0)

    default_frames = rigids_from_tensor4x4(m)

    # Create the rotation matrices according to the given angles (each frame is
    # defined such that its rotation is around the x-axis).
    sin_angles = torsion_angles_sin_cos[..., 0]
    cos_angles = torsion_angles_sin_cos[..., 1]

    # insert zero rotation for backbone group.
    num_residues, = aatype.shape
    sin_angles = mnp.concatenate([mnp.zeros([num_residues, 1]), sin_angles], axis=-1)
    cos_angles = mnp.concatenate([mnp.ones([num_residues, 1]), cos_angles], axis=-1)
    zeros = mnp.zeros_like(sin_angles)
    ones = mnp.ones_like(sin_angles)

    all_rots = (ones, zeros, zeros,
                zeros, cos_angles, -sin_angles,
                zeros, sin_angles, cos_angles)

    # Apply rotations to the frames.
    all_frames = geometry.rigids_mul_rots(default_frames, all_rots)
    # chi2, chi3, and chi4 frames do not transform to the backbone frame but to
    # the previous frame. So chain them up accordingly.
    chi2_frame_to_frame = ((all_frames[0][0][:, 5], all_frames[0][1][:, 5], all_frames[0][2][:, 5],
                            all_frames[0][3][:, 5], all_frames[0][4][:, 5], all_frames[0][5][:, 5],
                            all_frames[0][6][:, 5], all_frames[0][7][:, 5], all_frames[0][8][:, 5]),
                           (all_frames[1][0][:, 5], all_frames[1][1][:, 5], all_frames[1][2][:, 5]))
    chi3_frame_to_frame = ((all_frames[0][0][:, 6], all_frames[0][1][:, 6], all_frames[0][2][:, 6],
                            all_frames[0][3][:, 6], all_frames[0][4][:, 6], all_frames[0][5][:, 6],
                            all_frames[0][6][:, 6], all_frames[0][7][:, 6], all_frames[0][8][:, 6]),
                           (all_frames[1][0][:, 6], all_frames[1][1][:, 6], all_frames[1][2][:, 6]))

    chi4_frame_to_frame = ((all_frames[0][0][:, 7], all_frames[0][1][:, 7], all_frames[0][2][:, 7],
                            all_frames[0][3][:, 7], all_frames[0][4][:, 7], all_frames[0][5][:, 7],
                            all_frames[0][6][:, 7], all_frames[0][7][:, 7], all_frames[0][8][:, 7]),
                           (all_frames[1][0][:, 7], all_frames[1][1][:, 7], all_frames[1][2][:, 7]))

    chi1_frame_to_backb = ((all_frames[0][0][:, 4], all_frames[0][1][:, 4], all_frames[0][2][:, 4],
                            all_frames[0][3][:, 4], all_frames[0][4][:, 4], all_frames[0][5][:, 4],
                            all_frames[0][6][:, 4], all_frames[0][7][:, 4], all_frames[0][8][:, 4]),
                           (all_frames[1][0][:, 4], all_frames[1][1][:, 4], all_frames[1][2][:, 4]))

    chi2_frame_to_backb = geometry.rigids_mul_rigids(chi1_frame_to_backb, chi2_frame_to_frame)
    chi3_frame_to_backb = geometry.rigids_mul_rigids(chi2_frame_to_backb, chi3_frame_to_frame)
    chi4_frame_to_backb = geometry.rigids_mul_rigids(chi3_frame_to_backb, chi4_frame_to_frame)

    # Recombine them to a Rigids with shape (N, 8).
    all_frames_to_backb = rigids_concate_all(all_frames, chi2_frame_to_backb,
                                             chi3_frame_to_backb, chi4_frame_to_backb)

    backb_to_global = (geometry.rots_expand_dims(backb_to_global[0], -1),
                       geometry.vecs_expand_dims(backb_to_global[1], -1))
    # Create the global frames.
    all_frames_to_global = geometry.rigids_mul_rigids(backb_to_global, all_frames_to_backb)
    return all_frames_to_global


def map_atoms_to_global_func(all_frames, group_mask):
    """map atoms to global."""
    all_frames_rot = all_frames[0]
    all_frames_trans = all_frames[1]
    rot = geometry.rots_scale(geometry.rots_expand_dims(all_frames_rot, 1), group_mask)
    res_rot = []
    for val in rot:
        res_rot.append(mnp.sum(val, axis=-1))
    trans = geometry.vecs_scale(geometry.vecs_expand_dims(all_frames_trans, 1), group_mask)
    res_trans = []
    for val in trans:
        res_trans.append(mnp.sum(val, axis=-1))
    return (res_rot, res_trans)


def atom14_to_atom37(atom14_data, residx_atom37_to_atom14, atom37_atom_exists, indices0):
    """Convert atom14 to atom37 representation."""

    seq_length = atom14_data.shape[0]
    residx_atom37_to_atom14 = residx_atom37_to_atom14.reshape((seq_length, 37, 1))
    new_indices = P.Concat(2)((indices0, residx_atom37_to_atom14))

    atom37_data = P.GatherNd()(atom14_data, new_indices)

    if len(atom14_data.shape) == 2:
        atom37_data *= atom37_atom_exists
    elif len(atom14_data.shape) == 3:
        atom37_data *= atom37_atom_exists[:, :, None].astype(atom37_data.dtype)

    return atom37_data


def make_atom14_positions(aatype, all_atom_mask, all_atom_positions):
    """
    The function of transforming sparse encoding method to densely encoding method.

    Total coordinate encoding for atoms in proteins comes in two forms.

    - Sparse encoding, 20 amino acids contain a total of 37 atom types as shown in
      `common.residue_constants.atom_types`. So coordinates of atoms in protein can be encoded
      as a Tensor with shape :math:`(N_{res}, 37, 3)`.
    - Densely encoding. 20 amino acids contain a total of 14 atom types as shown in
      `common.residue_constants.restype_name_to_atom14_names`. So coordinates of atoms in protein can be encoded
      as a Tensor with shape :math:`(N_{res}, 14, 3)`.

    Args:
        aatype(numpy.ndarray):              Protein sequence encoding. the encoding method refers to
                                            `common.residue_constants.restype_order`. Value range is :math:`[0,20]`.
                                            20 means the amino acid is unknown (`UNK`).
        all_atom_mask(numpy.ndarray):       Mask of coordinates of all atoms in proteins. Shape is
                                            :math:`(N_{res}, 37)`. If the corresponding position is 0, the amino acid
                                            does not contain the atom.
        all_atom_positions(numpy.ndarray):  Coordinates of all atoms in protein. Shape is :math:`(N_{res}, 37, 3)` .

    Returns:
        - numpy.array. Densely encoding, mask of all atoms in protein, including unknown amino acid atoms.
          Shape is :math:`(N_{res}, 14)`.
        - numpy.array. Densely encoding, mask of all atoms in protein, excluding unknown amino acid atoms.
          Shape is :math:`(N_{res}, 14)`.
        - numpy.array. Densely encoding, coordinates of all atoms in protein. Shape is :math:`(N_{res}, 14, 3)`.
        - numpy.array. Index of mapping sparse encoding atoms with densely encoding method.
          Shape is :math:`(N_{res}, 14)` .
        - numpy.array. Index of mapping densely encoding atoms with sparse encoding method.
          Shape is :math:`(N_{res}, 37)` .
        - numpy.array. Sparse encoding, mask of all atoms in protein, including unknown amino acid atoms.
          Shape is :math:`(N_{res}, 14)`
        - numpy.array. The atomic coordinates after chiral transformation for the atomic coordinates of
          densely encoding method. Shape is :math:`(N_{res}, 14, 3)` .
        - numpy.array. Atom mask after chiral transformation. Shape is :math:`(N_{res}, 14)` .
        - numpy.array. Atom identifier of the chiral transformation. 1 is transformed and 0 is not transformed.
          Shape is :math:`(N_{res}, 14)` .

    Symbol:
        - ** :math:`N_{res}` ** - The number of amino acids in a protein, according to the sequence of the protein.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> from mindsponge.common import make_atom14_positions
        >>> from mindsponge.common import protein
        >>> import numpy as np
        >>> pdb_path = "YOUR_PDB_FILE"
        >>> with open(pdb_path, 'r', encoding = 'UTF-8') as f:
        >>>     prot_pdb = protein.from_pdb_string(f.read())
        >>> result = make_atom14_positions(prot_pdb.aatype, prot_pdb.atom_mask.astype(np.float32),
        >>>                                prot_pdb.atom_positions.astype(np.float32))
        >>> for val in result:
        >>>     print(val.shape)
        (Nres, 14)
        (Nres, 14)
        (Nres, 14, 3)
        (Nres, 14)
        (Nres, 37)
        (Nres, 37)
        (Nres, 14, 3)
        (Nres, 14)
        (Nres, 14)
    """
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
    residx_atom14_to_atom37 = restype_atom14_to_atom37[aatype]
    residx_atom14_mask = restype_atom14_mask[aatype]

    # Create a mask for known ground truth positions.
    residx_atom14_gt_mask = residx_atom14_mask * np.take_along_axis(
        all_atom_mask, residx_atom14_to_atom37, axis=1).astype(np.float32)

    # Gather the ground truth positions.
    residx_atom14_gt_positions = residx_atom14_gt_mask[:, :, None] * (
        np.take_along_axis(all_atom_positions, residx_atom14_to_atom37[..., None], axis=1))

    atom14_atom_exists = residx_atom14_mask
    atom14_gt_exists = residx_atom14_gt_mask
    atom14_gt_positions = residx_atom14_gt_positions

    residx_atom14_to_atom37 = residx_atom14_to_atom37

    # Create the gather indices for mapping back.
    residx_atom37_to_atom14 = restype_atom37_to_atom14[aatype]

    # Create the corresponding mask.
    restype_atom37_mask = np.zeros([21, 37], dtype=np.float32)
    for restype, restype_letter in enumerate(residue_constants.restypes):
        restype_name = residue_constants.restype_1to3[restype_letter]
        atom_names = residue_constants.residue_atoms[restype_name]
        for atom_name in atom_names:
            atom_type = residue_constants.atom_order[atom_name]
            restype_atom37_mask[restype, atom_type] = 1

    atom37_atom_exists = restype_atom37_mask[aatype]

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
            source_index = residue_constants.restype_name_to_atom14_names.get(resname).index(source_atom_swap)
            target_index = residue_constants.restype_name_to_atom14_names.get(resname).index(target_atom_swap)
            correspondences[source_index] = target_index
            correspondences[target_index] = source_index
            renaming_matrix = np.zeros((14, 14), dtype=np.float32)
            for index, correspondence in enumerate(correspondences):
                renaming_matrix[index, correspondence] = 1.
        all_matrices[resname] = renaming_matrix.astype(np.float32)
    renaming_matrices = np.stack([all_matrices[restype] for restype in restype_3])

    # Pick the transformation matrices for the given residue sequence
    # shape (num_res, 14, 14).
    renaming_transform = renaming_matrices[aatype]

    # Apply it to the ground truth positions. shape (num_res, 14, 3).
    alternative_gt_positions = np.einsum("rac,rab->rbc", residx_atom14_gt_positions, renaming_transform)
    atom14_alt_gt_positions = alternative_gt_positions

    # Create the mask for the alternative ground truth (differs from the
    # ground truth mask, if only one of the atoms in an ambiguous pair has a
    # ground truth position).
    alternative_gt_mask = np.einsum("ra,rab->rb", residx_atom14_gt_mask, renaming_transform)

    atom14_alt_gt_exists = alternative_gt_mask

    # Create an ambiguous atoms mask.  shape: (21, 14).
    restype_atom14_is_ambiguous = np.zeros((21, 14), dtype=np.float32)
    for resname, swap in residue_constants.residue_atom_renaming_swaps.items():
        for atom_name1, atom_name2 in swap.items():
            restype = residue_constants.restype_order[
                residue_constants.restype_3to1[resname]]
            atom_idx1 = residue_constants.restype_name_to_atom14_names.get(resname).index(atom_name1)
            atom_idx2 = residue_constants.restype_name_to_atom14_names.get(resname).index(atom_name2)
            restype_atom14_is_ambiguous[restype, atom_idx1] = 1
            restype_atom14_is_ambiguous[restype, atom_idx2] = 1

    # From this create an ambiguous_mask for the given sequence.
    atom14_atom_is_ambiguous = restype_atom14_is_ambiguous[aatype]
    return_pack = (atom14_atom_exists, atom14_gt_exists, atom14_gt_positions, residx_atom14_to_atom37,
                   residx_atom37_to_atom14, atom37_atom_exists, atom14_alt_gt_positions, atom14_alt_gt_exists,
                   atom14_atom_is_ambiguous)
    return return_pack


def get_pdb_info(pdb_path):
    """
    get atom positions, residue index etc. info from pdb file.

    Args:
        pdb_path(str): the path of the input pdb.

    Returns:
        features(dict), the information of pdb, including these keys

        - aatype, numpy.array. Protein sequence encoding. Encoding method refers to
          `common.residue_constants_restype_order`, :math:`[0,20]` . 20 means the amino acid is `UNK`.
          Shape :math:`(N_{res}, )` .
        - all_atom_positions, numpy.array. Coordinates of all residues in pdb. Shape :math:`(N_{res}, 37)` .
        - all_atom_mask, numpy.array. Mask of atoms in pdb. Shape :math:`(N_{res}, 37)` .
          0 means the atom inexistence.
        - atom14_atom_exists, numpy.array. Densely encoding, mask of all atoms in protein.
          The position with atoms is 1 and the position without atoms is 0. Shape is :math:`(N_{res}, 14)`.
        - atom14_gt_exists, numpy.array. Densely encoding, mask of all atoms in protein.
          Keep the same as `atom14_atom_exist`. Shape is :math:`(N_{res}, 14)`.
        - atom14_gt_positions, numpy.array. Densely encoding, coordinates of all atoms in the protein.
          Shape is :math:`(N_{res}, 14, 3)`.
        - residx_atom14_to_atom37, numpy.array. Index of mapping sparse encoding atoms with densely encoding method.
          Shape is :math:`(N_{res}, 14)` .
        - residx_atom37_to_atom14, numpy.array. Index of mapping densely encoding atoms with sparse encoding method.
          Shape is :math:`(N_{res}, 37)` .
        - atom37_atom_exists, numpy.array. Sparse encoding, mask of all atoms in protein.
          The position with atoms is 1 and the position without atoms is 0. Shape is :math:`(N_{res}, 37)`.
        - atom14_alt_gt_positions, numpy.array. Densely encoding, coordinates of all atoms in chiral proteins.
          Shape is :math:`(N_{res}, 14, 3)` .
        - atom14_alt_gt_exists, numpy.array. Densely encoding, mask of all atoms in chiral proteins.
          Shape is :math:`(N_{res}, 14)` .
        - atom14_atom_is_ambiguous, numpy.array. Because of the local symmetry of some amino acid structures,
          the symmetric atomic codes can be transposed. Specific atoms can be found in
          `common.residue_atom_renaming_swaps`. This feature records the uncertain atom encoding positions.
          Shape is :math:`(N_{res}, 14)` .
        - residue_index, numpy.array. Residue index information of protein sequence, ranging from 1 to :math:`N_{res}` .
          Shape is :math:`(N_{res}, )` .

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> from mindsponge.common import get_pdb_info
        >>> pdb_path = "YOUR PDB PATH"
        >>> pdb_feature = get_pdb_info(pdb_path)
        >>> for feature in pdb_feature:
        >>>     print(feature, pdb_feature[feature])
        # Nres represents the Amino acid num of the input pdb.
        aatype (Nres,)
        all_atom_positions (Nres, 37, 3)
        all_atom_mask (Nres, 37)
        atom14_atom_exists (Nres, 14)
        atom14_gt_exists (Nres, 14)
        atom14_gt_positions (Nres, 14, 3)
        residx_atom14_to_atom37 (Nres, 14)
        residx_atom37_to_atom14 (Nres, 37)
        atom37_atom_exists (Nres, 37)
        atom14_alt_gt_positions (Nres, 14, 3)
        atom14_alt_gt_exists (Nres, 14)
        atom14_atom_is_ambiguous (Nres, 14)
        residue_index (Nres, )

    """
    with open(pdb_path, 'r', encoding="UTF-8") as f:
        prot_pdb = protein.from_pdb_string(f.read())
    aatype = prot_pdb.aatype
    atom37_positions = prot_pdb.atom_positions.astype(np.float32)
    atom37_mask = prot_pdb.atom_mask.astype(np.float32)

    # get ground truth of atom14
    features = {'aatype': aatype,
                'all_atom_positions': atom37_positions,
                'all_atom_mask': atom37_mask}
    atom14_atom_exists, atom14_gt_exists, atom14_gt_positions, residx_atom14_to_atom37, residx_atom37_to_atom14, \
    atom37_atom_exists, atom14_alt_gt_positions, atom14_alt_gt_exists, atom14_atom_is_ambiguous = \
        make_atom14_positions(aatype, atom37_mask, atom37_positions)
    features.update({"atom14_atom_exists": atom14_atom_exists,
                     "atom14_gt_exists": atom14_gt_exists,
                     "atom14_gt_positions": atom14_gt_positions,
                     "residx_atom14_to_atom37": residx_atom14_to_atom37,
                     "residx_atom37_to_atom14": residx_atom37_to_atom14,
                     "atom37_atom_exists": atom37_atom_exists,
                     "atom14_alt_gt_positions": atom14_alt_gt_positions,
                     "atom14_alt_gt_exists": atom14_alt_gt_exists,
                     "atom14_atom_is_ambiguous": atom14_atom_is_ambiguous})

    features["residue_index"] = prot_pdb.residue_index

    return features


def get_fasta_info(pdb_path):
    """
    Put in a pdb file and get fasta information from it. Return the sequence of the pdb.

    Args:
        pdb_path(str): path of the input pdb.

    Returns:
        fasta(str), fasta of input pdb. The sequence is the order of residues in the protein and has no
        relationship with residue index, such as "GSHMGVQ".

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> from mindsponge.common import get_fasta_info
        >>> pdb_path = "YOUR PDB PATH"
        >>> fasta = get_fasta_info(pdb_path)
        >>> print(fasta)
        "GSHMGVQ"

    """
    with open(pdb_path, 'r', encoding='UTF-8') as f:
        prot_pdb = protein.from_pdb_string(f.read())
    aatype = prot_pdb.aatype
    fasta = [residue_constants.order_restype_with_x.get(x, "X") for x in aatype]

    return ''.join(fasta)


def get_aligned_seq(gt_seq, pr_seq):
    """
    Align two protein fasta sequence. Return two aligned sequences and the position of same residues.

    Args:
        gt_seq(str): one protein fasta sequence, such as "ABAAABAA".
        pr_seq(str): another protein fasta sequence, such as "A-AABBBA".

    Returns:
        - target(str), one protein fasta sequence.
        - align_relationship(str), the differences of the two sequences.
        - query(str), another protein fasta sequence.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> from mindsponge.common import get_aligned_seq
        >>> gt_seq = "ABAAABAA"
        >>> pr_seq = "AAABBBA"
        >>> aligned_gt_seq, aligned_info, aligned_pr_seq = get_aligned_seq(gt_seq, pr_seq)
        >>> print(aligned_gt_seq)
        ABAAABAA
        >>> print(aligned_info)
        |-||.|.|
        >>> print(aligned_pr_seq)
        A-AABBBA

    """
    aligner = Align.PairwiseAligner()
    substitution_matrices.load()
    matrix = substitution_matrices.load("BLOSUM62")
    for i in range(len(str(matrix.alphabet))):
        res = matrix.alphabet[i]
        matrix['X'][res] = 0
        matrix[res]['X'] = 0
    aligner.substitution_matrix = matrix
    aligner.open_gap_score = -10
    aligner.extend_gap_score = -1
    # many align results, get only the one w/ highest score. gt_seq as reference
    alignments = aligner.align(gt_seq, pr_seq)
    align = alignments[0]
    align_str = str(align)
    align_str_len = len(align_str)
    point = []
    target = ''
    align_relationship = ''
    query = ''
    for i in range(align_str_len):
        if align_str[i] == '\n':
            point.append(i)
    for i in range(int(point[0])):
        target = target + align_str[i]
    for i in range(int(point[1])-int(point[0])-1):
        align_relationship = align_relationship + align_str[i + int(point[0])+1]
    for i in range(int(point[2])-int(point[1])-1):
        query = query + align_str[i + int(point[1])+1]
    return target, align_relationship, query


def find_optimal_renaming(
        atom14_gt_positions,
        atom14_alt_gt_positions,
        atom14_atom_is_ambiguous,
        atom14_gt_exists,
        atom14_pred_positions,
):  # (N):
    """
    Find optimal renaming for ground truth that maximizes LDDT.

    Reference:
        `Jumper et al. (2021) Suppl. Alg. 26 "renameSymmetricGroundTruthAtoms"
        <https://www.nature.com/articles/s41586-021-03819-2>`_

    Args:
      atom14_gt_positions (Tensor):      Ground truth positions in global frame with shape :math:`(N_{res}, 14, 3)`.
      atom14_alt_gt_positions (Tensor):  Alternate ground truth positions in global frame with coordinates of
                                         ambiguous atoms swapped relative to 'atom14_gt_positions'.
                                         The shape is :math:`(N_{res}, 14, 3)`.
      atom14_atom_is_ambiguous (Tensor): Mask denoting whether atom is among ambiguous atoms,
                                         see Jumper et al. (2021) Suppl. Table 3. The shape is :math:`(N_{res}, 14)`.
      atom14_gt_exists (Tensor):         Mask denoting whether atom at positions exists in ground truth with
                                         shape :math:`(N_{res}, 14)`.
      atom14_pred_positions(Tensor):     Predicted positions of atoms in global prediction frame with
                                         shape :math:`(N_{res}, 14, 3)`.

    Returns:
        Tensor, :math:`(N_{res},)` with 1.0 where atom14_alt_gt_positions is closer to prediction and otherwise 0.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> from mindsponge.common.utils import find_optimal_renaming
        >>> from mindspore import Tensor
        >>> n_res = 16
        >>> atom14_gt_positions = Tensor(np.random.randn(n_res, 14, 3).astype(np.float32))
        >>> atom14_alt_gt_positions = Tensor(np.random.randn(n_res, 14, 3).astype(np.float32))
        >>> atom14_atom_is_ambiguous = Tensor(np.random.randn(n_res, 14).astype(np.float32))
        >>> atom14_gt_exists = Tensor(np.random.randn(n_res, 14).astype(np.float32))
        >>> atom14_pred_positions = Tensor(np.random.randn(n_res, 14, 3).astype(np.float32))
        >>> out = find_optimal_renaming(atom14_gt_positions, atom14_alt_gt_positions,
        ...                             atom14_atom_is_ambiguous, atom14_gt_exists, atom14_pred_positions)
        >>> print(out.shape)
        (16,)
    """

    # Create the pred distance matrix.
    atom14_pred_positions = P.Pad(((0, 0), (0, 0), (0, 5)))(atom14_pred_positions)
    pred_dists = mnp.sqrt(1e-10 + mnp.sum(
        mnp.square(atom14_pred_positions[:, None, :, None, :] - atom14_pred_positions[None, :, None, :, :]), axis=-1))

    # Compute distances for ground truth with original and alternative names.
    gt_dists = mnp.sqrt(1e-10 + mnp.sum(
        mnp.square(atom14_gt_positions[:, None, :, None, :] - atom14_gt_positions[None, :, None, :, :]), axis=-1))
    alt_gt_dists = mnp.sqrt(1e-10 + mnp.sum(
        mnp.square(atom14_alt_gt_positions[:, None, :, None, :] - atom14_alt_gt_positions[None, :, None, :, :]),
        axis=-1))

    # Compute LDDT's.
    lddt = mnp.sqrt(1e-10 + mnp.square(pred_dists - gt_dists))
    alt_lddt = mnp.sqrt(1e-10 + mnp.square(pred_dists - alt_gt_dists))

    # Create a mask for ambiguous atoms in rows vs. non-ambiguous atoms
    # in cols.
    mask = (atom14_gt_exists[:, None, :, None] *  # rows
            atom14_atom_is_ambiguous[:, None, :, None] *  # rows
            atom14_gt_exists[None, :, None, :] *  # cols
            (1. - atom14_atom_is_ambiguous[None, :, None, :]))  # cols

    # Aggregate distances for each residue to the non-amibuguous atoms.
    per_res_lddt = P.ReduceSum()(mask * lddt, (1, 2, 3))
    alt_per_res_lddt = P.ReduceSum()(mask * alt_lddt, (1, 2, 3))

    # Decide for each residue, whether alternative naming is better.
    alt_naming_is_better = (alt_per_res_lddt < per_res_lddt)

    return alt_naming_is_better
