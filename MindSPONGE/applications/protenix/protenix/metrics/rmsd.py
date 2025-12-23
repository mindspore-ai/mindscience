# Copyright 2025 Huawei Technologies Co., Ltd
# Copyright 2024 ByteDance and/or its affiliates.
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

"""rmsd"""

from typing import Optional

import numpy as np
import mindspore as ms
from mindspore import ops, Tensor


def rmsd(
    pred_pose: Tensor,
    true_pose: Tensor,
    mask: Tensor = None,
    eps: float = 0.0,
    reduce: bool = True,
):
    """
    compute rmsd between two poses, with the same shape
    Arguments:
        pred_pose/true_pose: [...,N,3], two poses with the same shape
        mask: [..., N], mask to indicate which atoms/pseudo_betas/etc to compute
        eps: add a tolerance to avoid floating number issue
        reduce: decide the return shape of rmsd;
    Return:
        rmsd_value: if reduce = true, return the mean of rmsd over batches;
            else return a tensor containing each rmsd separately
    """

    # mask [..., N]
    if pred_pose.shape != true_pose.shape:  # [..., N, 3]
        raise ValueError(f"Pred pose shape {pred_pose.shape} is not equal to true pose shape {true_pose.shape}")

    if mask is None:
        mask = ops.ones(true_pose.shape[:-1])

    # [...]
    err2 = (ops.square(pred_pose - true_pose).sum(dim=-1) * mask).sum(
        dim=-1
    ) / mask.sum(dim=-1)
    rmsd_value = err2.add(eps).sqrt()
    if reduce:
        rmsd_value = rmsd_value.mean()
    return rmsd_value


def align_pred_to_true(
    pred_pose: Tensor,
    true_pose: Tensor,
    atom_mask: Optional[Tensor] = None,
    weight: Optional[Tensor] = None,
    allowing_reflection: bool = False,
):
    """Find optimal transformation, rotation (and reflection) of two poses.
    Arguments:
        pred_pose: [...,N,3] the pose to perform transformation on
        true_pose: [...,N,3] the target pose to align pred_pose to
        atom_mask: [..., N] a mask for atoms
        weight: [..., N] a weight vector to be applied.
        allow_reflection: whether to allow reflection when finding optimal alignment
    return:
        aligned_pose: [...,N,3] the transformed pose
        rot: optimal rotation
        translate: optimal translation
    """
    if atom_mask is not None:
        pred_pose = pred_pose * atom_mask.unsqueeze(-1)
        true_pose = true_pose * atom_mask.unsqueeze(-1)
    else:
        atom_mask = ops.ones((pred_pose.shape[:-1]))

    if weight is None:
        weight = atom_mask
    else:
        weight = weight * atom_mask

    weighted_n_atoms = ops.sum(weight, dim=-1, keepdim=True).unsqueeze(-1)
    pred_pose_centroid = (
        ops.sum(pred_pose * weight.unsqueeze(-1), dim=-2, keepdim=True)
        / weighted_n_atoms
    )
    pred_pose_centered = pred_pose - pred_pose_centroid
    true_pose_centroid = (
        ops.sum(true_pose * weight.unsqueeze(-1), dim=-2, keepdim=True)
        / weighted_n_atoms
    )
    true_pose_centered = true_pose - true_pose_centroid
    h_mat = ops.matmul(
        (pred_pose_centered * weight.unsqueeze(-1)).transpose(-2, -1),
        true_pose_centered * atom_mask.unsqueeze(-1),
    )

    _, u, v = ops.svd(h_mat, full_matrices=True)
    u = u.transpose(-1, -2)

    if not allowing_reflection:
        det = ms.Tensor(np.linalg.det(ops.matmul(v, u).asnumpy()))

        diagonal = ops.stack(
            [ops.ones_like(det), ops.ones_like(det), det], axis=-1
        )
        rot = ops.matmul(
            ops.diag_embed(diagonal),
            u,
        )
        rot = ops.matmul(v, rot)
    else:
        rot = ops.matmul(v, u)
    translate = true_pose_centroid - ops.matmul(
        pred_pose_centroid, rot.transpose(-1, -2)
    )

    pred_pose_translated = (
        ops.matmul(pred_pose_centered, rot.transpose(-1, -2)) + true_pose_centroid
    )

    return pred_pose_translated, rot, translate


def partially_aligned_rmsd(
    pred_pose: Tensor,
    true_pose: Tensor,
    align_mask: Tensor,
    atom_mask: Tensor,
    weight: Optional[Tensor] = None,
    eps: float = 0.0,
    reduce: bool = True,
    allowing_reflection: bool = False,
):
    """RMSD when aligning parts of the complex coordinate, does NOT take permutation symmetricity into consideration
    Arguments:
        pred_pose: native predicted pose, [..., N,3]
        true_pose: ground truth pose, [..., N, 3]
        align_mask: a mask representing which coordinates to align [..., N]
        atom_mask: a mask representing which coordinates to compute loss [..., N]
        weight: a weight tensor assining weights in alignment for each atom [..., N]
        eps: add a tolerance to avoid floating number issue in sqrt
        reduce: decide the return shape of rmsd;
        allowing_reflection: whether to allow reflection when finding optimal alignment
    return:
        aligned_part_rmsd: the rmsd of part being align_masked
        unaligned_part_rmsd: the rmsd of unaligned part
        transformed_pred_pose:
        rot: optimal rotation
        trans: optimal translation
    """
    _, rot, translate = align_pred_to_true(
        pred_pose,
        true_pose,
        atom_mask=atom_mask * align_mask,
        weight=weight,
        allowing_reflection=allowing_reflection,
    )
    transformed_pose = ops.matmul(pred_pose, rot.transpose(-1, -2)) + translate
    err_atom = ops.square(transformed_pose - true_pose).sum(dim=-1) * atom_mask
    aligned_mask, unaligned_mask = atom_mask * align_mask.float(), atom_mask * (
        1 - align_mask.float()
    )
    aligned_part_err_square = (err_atom * aligned_mask).sum(dim=-1) / aligned_mask.sum(
        dim=-1
    )
    unaligned_part_err_square = (err_atom * unaligned_mask).sum(
        dim=-1
    ) / unaligned_mask.sum(dim=-1)
    aligned_part_rmsd = aligned_part_err_square.add(eps).sqrt()
    unaligned_part_rmsd = unaligned_part_err_square.add(eps).sqrt()
    if reduce:
        aligned_part_rmsd = aligned_part_rmsd.mean()
        unaligned_part_rmsd = unaligned_part_rmsd.mean()
    return aligned_part_rmsd, unaligned_part_rmsd, transformed_pose, rot, translate


def self_aligned_rmsd(
    pred_pose: Tensor,
    true_pose: Tensor,
    atom_mask: Tensor,
    eps: float = 0.0,
    reduce: bool = True,
    allowing_reflection: bool = False,
):
    """RMSD when aligning one molecule with ground truth and compute rmsd.
    Arguments:
        pred_pose: native predicted pose, [..., N,3]
        true_pose: ground truth pose, [..., N, 3]
        atom_mask: a mask representing which coordinates to compute loss [..., N]
        eps: add a tolerance to avoid floating number issue in sqrt
        reduce: decide the return shape of rmsd;
        allowing_reflection: whether to allow reflection when finding optimal alignment
    return:
        aligned_rmsd: the rmsd of part being align_masked
        transformed_pred_pose: the aligned pose
        rot: optimal rotation matrix
        trans: optimal translation
    """
    aligned_rmsd, _, transformed_pred_pose, rot, trans = partially_aligned_rmsd(
        pred_pose=pred_pose,
        true_pose=true_pose,
        align_mask=atom_mask,
        atom_mask=atom_mask,
        eps=eps,
        reduce=reduce,
        allowing_reflection=allowing_reflection,
    )
    return aligned_rmsd, transformed_pred_pose, rot, trans


def weighted_rigid_align(
    x: Tensor,
    x_target: Tensor,
    atom_weight: Tensor,
) -> tuple[Tensor]:
    """Implements Algorithm 28 in AF3. Wrap `align_pred_to_true`.

    Args:
        x (Tensor): input coordinates, it will be moved to match x_target.
            [..., N_atom, 3]
        x_target (Tensor): target coordinates for the input to match.
            [..., N_atom, 3]
        atom_weight (Tensor): weights for each atom.
            [..., N_atom] or [N_atom]

    Returns:
        x_aligned (Tensor): rotated, translated x which should be closer to x_target.
            [..., N_atom, 3]
    """

    if len(atom_weight.shape) == len(x.shape) - 1:
        if atom_weight.shape[:-1] != x.shape[:-2]:
            raise ValueError(f"Atom weight shape {atom_weight.shape[:-1]} is not equal to x shape {x.shape[:-2]}")
    else:
        if len(atom_weight.shape) != 1 or atom_weight.shape[-1] != x.shape[-2]:
            raise ValueError(f"Atom weight shape {atom_weight.shape} is not equal to x shape {x.shape}")

    x_aligned, _, _ = align_pred_to_true(
        pred_pose=x,
        true_pose=x_target,
        atom_mask=None,
        weight=atom_weight,
        allowing_reflection=False,
    )
    return x_aligned
