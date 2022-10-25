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
"""Equivariant"""
import numpy as np
import mindspore.nn as nn
import mindspore.common.dtype as mstype
import mindspore.numpy as mnp
import mindspore.ops as ops
from mindspore import Parameter
from mindspore.common.tensor import Tensor
from mindsponge.common.geometry import apply_to_point, invert_point
from mindsponge.cell.initializer import lecun_init
from common.geometry import multimer_vecs_robust_norm, multimer_square_euclidean_distance


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
