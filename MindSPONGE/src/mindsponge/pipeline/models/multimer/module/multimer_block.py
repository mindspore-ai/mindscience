# Copyright 2023 @ Shenzhen Bay Laboratory &
#                  Peking University &
#                  Huawei Technologies Co., Ltd
#
# This code is a part of MindSPONGE:
# MindSpore Simulation Package tOwards Next Generation molecular modelling.
#
# MindSPONGE is open-source software based on the AI-framework:
# MindSpore (https://www.mindspore.cn/)
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
"""Evoformer"""

import numpy as np
import mindspore.nn as nn
import mindspore.common.dtype as mstype
import mindspore.numpy as mnp
import mindspore.ops as ops
from mindspore import Parameter
from mindspore.ops import operations as P
from mindspore.common.tensor import Tensor
from mindsponge.common.geometry import apply_to_point, invert_point, vecs_from_tensor, \
    vecs_dot_vecs, vecs_sub, vecs_cross_vecs, vecs_scale, \
    rots_expand_dims, vecs_expand_dims, invert_rigids, rigids_mul_vecs
from mindsponge.pipeline.cell.initializer import lecun_init


def compute_chi_angles(aatype,  # (B, N)
                       all_atom_pos,  # (B, N, 37, 3)
                       all_atom_mask,  # (B, N, 37)
                       chi_atom_indices,
                       chi_angles_mask,
                       indices0,
                       indices1,
                       batch_size=1):
    """compute chi angles"""

    aatype = mnp.minimum(aatype, 20)
    # Collect the atoms for the chi-angles.
    # Compute the table of chi angle indices. Shape: [restypes, chis=4, atoms=4].
    # Select atoms to compute chis. Shape: [batch, num_res, chis=4, atoms=4].
    atom_indices = mnp.take(chi_atom_indices, aatype, axis=0)

    # # Gather atom positions Batch Gather. Shape: [batch, num_res, chis=4, atoms=4, xyz=3].

    # 4 seq_length 4 4  batch, sequence length, chis, atoms
    seq_length = all_atom_pos.shape[1]
    atom_indices = atom_indices.reshape((4, seq_length, 4, 4, 1)).astype("int32")
    new_indices = P.Concat(4)((indices0, indices1, atom_indices))
    chis_atom_pos = P.GatherNd()(all_atom_pos, new_indices)
    chis_mask = mnp.take(chi_angles_mask, aatype, axis=0)
    chi_angle_atoms_mask = P.GatherNd()(all_atom_mask, new_indices)

    # Check if all 4 chi angle atoms were set. Shape: [batch, num_res, chis=4].
    chi_angle_atoms_mask = P.ReduceProd()(chi_angle_atoms_mask, -1)
    chis_mask = chis_mask * (chi_angle_atoms_mask).astype(mnp.float32)
    all_chi_angles = []
    for i in range(batch_size):
        template_chi_angles = multimer_rigids_compute_dihedral_angle(vecs_from_tensor(chis_atom_pos[i, :, :, 0, :]),
                                                                     vecs_from_tensor(chis_atom_pos[i, :, :, 1, :]),
                                                                     vecs_from_tensor(chis_atom_pos[i, :, :, 2, :]),
                                                                     vecs_from_tensor(chis_atom_pos[i, :, :, 3, :]))
        all_chi_angles.append(template_chi_angles)
    chi_angles = mnp.stack(all_chi_angles, axis=0)
    return chi_angles, chis_mask


def multimer_square_euclidean_distance(v1, v2, epsilon):
    """multimer_square_euclidean_distance."""
    difference = vecs_sub(v1, v2)
    distance = vecs_dot_vecs(difference, difference)
    if epsilon:
        distance = mnp.maximum(distance, epsilon)
    return distance


def multimer_vecs_robust_norm(v, epsilon=1e-6):
    """multime computes norm of vectors 'v'."""
    v_l2_norm = v[0] * v[0] + v[1] * v[1] + v[2] * v[2]
    if epsilon:
        v_l2_norm = mnp.maximum(v_l2_norm, epsilon**2)
    return mnp.sqrt(v_l2_norm)


def multimer_vecs_robust_normalize(v, epsilon=1e-6):
    """multimer normalizes vectors 'v'."""
    norms = multimer_vecs_robust_norm(v, epsilon)
    return (v[0] / norms, v[1] / norms, v[2] / norms)


def multimer_rots_from_two_vecs(e0_unnormalized, e1_unnormalized):
    """multimer_rots_from_two_vecs."""
    e0 = multimer_vecs_robust_normalize(e0_unnormalized)
    c = vecs_dot_vecs(e1_unnormalized, e0)
    e1 = vecs_sub(e1_unnormalized, vecs_scale(e0, c))
    e1 = multimer_vecs_robust_normalize(e1)
    e2 = vecs_cross_vecs(e0, e1)

    rots = (e0[0], e1[0], e2[0],
            e0[1], e1[1], e2[1],
            e0[2], e1[2], e2[2])
    return rots


def multimer_rigids_from_3_points(vec_a, vec_b, vec_c):
    """Create multimer Rigids from 3 points. """
    m = multimer_rots_from_two_vecs(
        e0_unnormalized=vecs_sub(vec_c, vec_b),
        e1_unnormalized=vecs_sub(vec_a, vec_b))
    rigid = (m, vec_b)
    return rigid


def multimer_rigids_get_unit_vector(point_a, point_b, point_c):
    """multimer_rigids_get_unit_vector."""
    rigid = multimer_rigids_from_3_points(vecs_from_tensor(point_a),
                                          vecs_from_tensor(point_b),
                                          vecs_from_tensor(point_c))
    rot, trans = rigid
    rotation = rots_expand_dims(rot, -1)
    translation = vecs_expand_dims(trans, -1)
    inv_rigid = invert_rigids((rotation, translation))
    rigid_vec = rigids_mul_vecs(inv_rigid, vecs_expand_dims(trans, -2))
    unit_vector = multimer_vecs_robust_normalize(rigid_vec)
    return unit_vector


def multimer_rigids_compute_dihedral_angle(a, b, c, d):
    """multimer_rigids_compute_dihedral_angle."""
    v1 = vecs_sub(a, b)
    v2 = vecs_sub(b, c)
    v3 = vecs_sub(d, c)

    c1 = vecs_cross_vecs(v1, v2)
    c2 = vecs_cross_vecs(v3, v2)
    c3 = vecs_cross_vecs(c2, c1)

    v2_mag = multimer_vecs_robust_norm(v2)
    return mnp.arctan2(vecs_dot_vecs(c3, v2), v2_mag * vecs_dot_vecs(c1, c2))


class MultimerInvariantPointAttention(nn.Cell):
    """Invariant Point attention module."""

    def __init__(self, num_head, num_scalar_qk, num_scalar_v, num_point_v, num_point_qk, num_channel, pair_dim):
        """

        Args:
          pair_dim: pair representation dimension.
        """

        super(MultimerInvariantPointAttention, self).__init__()

        self._dist_epsilon = 1e-8
        self.num_head = num_head
        self.num_scalar_qk = num_scalar_qk
        self.num_scalar_v = num_scalar_v
        self.num_point_v = num_point_v
        self.num_point_qk = num_point_qk
        self.num_channel = num_channel
        self.projection_num = self.num_head * self.num_scalar_v + self.num_head * self.num_point_v * 4 + \
                              self.num_head * pair_dim
        self.q_scalar = nn.Dense(self.num_channel, self.num_head * self.num_scalar_qk,
                                 weight_init=lecun_init(self.num_channel), has_bias=False)
        self.k_scalar = nn.Dense(self.num_channel, self.num_head * self.num_scalar_qk,
                                 weight_init=lecun_init(self.num_channel), has_bias=False)
        self.v_scalar = nn.Dense(self.num_channel, self.num_head * self.num_scalar_v,
                                 weight_init=lecun_init(self.num_channel), has_bias=False)
        self.q_point_local = nn.Dense(self.num_channel, self.num_head * 3 * self.num_point_qk,
                                      weight_init=lecun_init(self.num_channel))
        self.k_point_local = nn.Dense(self.num_channel, self.num_head * 3 * self.num_point_qk,
                                      weight_init=lecun_init(self.num_channel))
        self.v_point_local = nn.Dense(self.num_channel, self.num_head * 3 * self.num_point_v,
                                      weight_init=lecun_init(self.num_channel))
        self.soft_max = nn.Softmax(axis=-2)
        self.trainable_point_weights = Parameter(Tensor(np.ones((12,)), mstype.float32), name="trainable_point_weights")
        self.attention_2d = nn.Dense(pair_dim, self.num_head, weight_init=lecun_init(pair_dim))
        self.output_projection = nn.Dense(self.projection_num, self.num_channel, weight_init='zeros')

        self.point_weights = np.sqrt(1.0 / (max(num_point_qk, 1) * 9. / 2))
        self.scalar_weights = np.sqrt(1.0 / (max(num_scalar_qk, 1) * 1.))

    def construct(self, inputs_1d, inputs_2d, mask, rotation, translation):
        """Compute geometry-aware attention.

        Args:
          inputs_1d: (N, C) 1D input embedding that is the basis for the
            scalar queries.
          inputs_2d: (N, M, C') 2D input embedding, used for biases and values.
          mask: (N, 1) mask to indicate which elements of inputs_1d participate
            in the attention.
          rotation: describe the orientation of every element in inputs_1d
          translation: describe the position of every element in inputs_1d

        Returns:
          Transformation of the input embedding.
        """
        num_residues, _ = inputs_1d.shape

        num_head = self.num_head
        attn_logits = 0.
        num_point_qk = self.num_point_qk
        point_weights = self.point_weights

        trainable_point_weights = mnp.logaddexp(self.trainable_point_weights,
                                                mnp.zeros_like(self.trainable_point_weights))
        point_weights = point_weights * trainable_point_weights

        q_point_local = self.q_point_local(inputs_1d)
        q_point_local = mnp.reshape(q_point_local, (num_residues, num_head, num_point_qk * 3))
        q_point_local = mnp.split(q_point_local, 3, axis=-1)
        q_point_local = (ops.Squeeze()(q_point_local[0]), ops.Squeeze()(q_point_local[1]),
                         ops.Squeeze()(q_point_local[2]))
        # Project query points into global frame.
        q_point_global = apply_to_point(rotation, translation, q_point_local, 2)
        q_point = [q_point_global[0][:, None, :, :], q_point_global[1][:, None, :, :], q_point_global[2][:, None, :, :]]

        k_point_local = self.k_point_local(inputs_1d)
        k_point_local = mnp.reshape(k_point_local, (num_residues, num_head, num_point_qk * 3))
        k_point_local = mnp.split(k_point_local, 3, axis=-1)
        k_point_local = (ops.Squeeze()(k_point_local[0]), ops.Squeeze()(k_point_local[1]),
                         ops.Squeeze()(k_point_local[2]))

        # Project query points into global frame.
        k_point_global = apply_to_point(rotation, translation, k_point_local, 2)
        k_point = [k_point_global[0][None, :, :, :], k_point_global[1][None, :, :, :], k_point_global[2][None, :, :, :]]

        dist2 = multimer_square_euclidean_distance(q_point, k_point, epsilon=0.)

        attn_qk_point = -0.5 * mnp.sum(point_weights[:, None] * dist2, axis=-1)
        attn_logits += attn_qk_point

        num_scalar_qk = self.num_scalar_qk

        scalar_weights = self.scalar_weights
        q_scalar = self.q_scalar(inputs_1d)
        q_scalar = mnp.reshape(q_scalar, [num_residues, num_head, num_scalar_qk])

        k_scalar = self.k_scalar(inputs_1d)
        k_scalar = mnp.reshape(k_scalar, [num_residues, num_head, num_scalar_qk])

        q_scalar *= scalar_weights
        q = mnp.swapaxes(q_scalar, -2, -3)
        k = mnp.swapaxes(k_scalar, -2, -3)
        attn_qk_scalar = ops.matmul(q, mnp.swapaxes(k, -2, -1))
        attn_qk_scalar = mnp.swapaxes(attn_qk_scalar, -2, -3)
        attn_qk_scalar = mnp.swapaxes(attn_qk_scalar, -2, -1)
        attn_logits += attn_qk_scalar

        attention_2d = self.attention_2d(inputs_2d)
        attn_logits += attention_2d

        mask_2d = mask * mnp.swapaxes(mask, -1, -2)
        attn_logits -= 1e5 * (1. - mask_2d[..., None])
        attn_logits *= mnp.sqrt(1. / 3)
        attn = self.soft_max(attn_logits)

        num_scalar_v = self.num_scalar_v
        v_scalar = self.v_scalar(inputs_1d)
        v_scalar = mnp.reshape(v_scalar, [num_residues, num_head, num_scalar_v])

        attn_tmp = mnp.swapaxes(attn, -1, -2)
        attn_tmp = mnp.swapaxes(attn_tmp, -2, -3)
        result_scalar = ops.matmul(attn_tmp, mnp.swapaxes(v_scalar, -2, -3))
        result_scalar = mnp.swapaxes(result_scalar, -2, -3)

        num_point_v = self.num_point_v

        v_point_local = self.v_point_local(inputs_1d)
        v_point_local = mnp.reshape(v_point_local, (num_residues, num_head, num_point_v * 3))
        v_point_local = mnp.split(v_point_local, 3, axis=-1)
        v_point_local = (ops.Squeeze()(v_point_local[0]), ops.Squeeze()(v_point_local[1]),
                         ops.Squeeze()(v_point_local[2]))
        # Project query points into global frame.
        v_point_global = apply_to_point(rotation, translation, v_point_local, 2)
        v_point = [v_point_global[0][None], v_point_global[1][None], v_point_global[2][None]]

        result_point_global = [mnp.sum(attn[..., None] * v_point[0], axis=-3),
                               mnp.sum(attn[..., None] * v_point[1], axis=-3),
                               mnp.sum(attn[..., None] * v_point[2], axis=-3)
                               ]

        num_query_residues, _ = inputs_1d.shape

        result_scalar = mnp.reshape(result_scalar, [num_query_residues, -1])

        output_feature1 = result_scalar

        result_point_global = [mnp.reshape(result_point_global[0], [num_query_residues, -1]),
                               mnp.reshape(result_point_global[1], [num_query_residues, -1]),
                               mnp.reshape(result_point_global[2], [num_query_residues, -1])]
        result_point_local = invert_point(result_point_global, rotation, translation, 1)
        output_feature20 = result_point_local[0]
        output_feature21 = result_point_local[1]
        output_feature22 = result_point_local[2]
        point_norms = multimer_vecs_robust_norm(result_point_local, self._dist_epsilon)
        output_feature3 = point_norms

        result_attention_over_2d = ops.matmul(mnp.swapaxes(attn, 1, 2), inputs_2d)
        output_feature4 = mnp.reshape(result_attention_over_2d, [num_query_residues, -1])
        final_act = mnp.concatenate([output_feature1, output_feature20, output_feature21,
                                     output_feature22, output_feature3, output_feature4], axis=-1)
        final_result = self.output_projection(final_act)
        return final_result
