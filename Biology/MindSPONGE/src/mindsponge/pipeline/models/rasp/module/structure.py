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
"""structure module"""
import numpy as np
import mindspore.common.dtype as mstype
import mindspore.nn as nn
import mindspore.numpy as mnp
import mindspore.ops as ops
from mindspore import Tensor, Parameter
from mindspore.ops import functional as F
import mindsponge.common.residue_constants as residue_constants
from mindsponge.cell.initializer import lecun_init
from mindsponge.common.utils import torsion_angles_to_frames, frames_and_literature_positions_to_atom14_pos, \
    atom14_to_atom37, pseudo_beta_fn
from mindsponge.common.geometry import initial_affine, quaternion_to_tensor, pre_compose, vecs_scale, \
    vecs_to_tensor, vecs_expand_dims, rots_expand_dims, apply_to_point, invert_point


class InvariantPointContactAttention(nn.Cell):
    r"""
    Invariant Point attention module.
    This module is used to update the sequence representation ,which is the first input--inputs_1d,
    adding location information to the sequence representation.

    The attention consists of three parts, namely, q, k, v obtained by the sequence representation,
    q'k'v' obtained by the interaction between the sequence representation and the rigid body group,
    and b , which is th bias, obtained from the pair representation (the second inputs -- inputs_2d).

    .. math::
        a_{ij} = Softmax(w_l(c_1{q_i}^Tk_j+b{ij}-c_2\sum {\left \| T_i\circ q'_i-T_j\circ k'_j \right \| ^{2 } })

    where i and j represent the ith and jth amino acids in the sequence, respectively,
    and T is the rotation and translation in the input.

    `Jumper et al. (2021) Suppl. Alg. 22 "InvariantPointContactAttention"
    <https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-021-03819-2/MediaObjects/41586_2021_3819_MOESM1_ESM.pdf>`_.

    Args:
        num_head (int):         The number of the heads.
        num_scalar_qk (int):    The number of the scalar query/key.
        num_scalar_v (int):     The number of the scalar value.
        num_point_v (int):      The number of the point value.
        num_point_qk (int):     The number of the point query/key.
        num_channel (int):      The number of the channel.
        pair_dim (int):         The last dimension length of pair.

    Inputs:
        - **inputs_1d** (Tensor) - The first row of msa representation which is the output of evoformer module,
          also called the sequence representation, shape :math:`[N_{res}, num\_channel]`.
        - **inputs_2d** (Tensor) - The pair representation which is the output of evoformer module,
          shape :math:`[N_{res}, N_{res}, pair\_dim]`.
        - **mask** (Tensor) - A mask that determines which elements of inputs_1d are involved in the
          attention calculation, shape :math:`[N_{res}, 1]`
        - **rotation** (tuple) - A rotation term in a rigid body group T(r,t),
          A tuple of length 9, The shape of each elements in the tuple is :math:`[N_{res}]`.
        - **translation** (tuple) - A translation term in a rigid body group T(r,t),
          A tuple of length 3, The shape of each elements in the tuple is :math:`[N_{res}]`.

    Outputs:
        Tensor, the update of inputs_1d, shape :math:`[N_{res}, num\_channel]`.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> from mindsponge.cell import InvariantPointContactAttention
        >>> from mindspore import dtype as mstype
        >>> from mindspore import Tensor
        >>> import mindspore.context as context
        >>> context.set_context(mode=context.GRAPH_MODE)
        >>> model = InvariantPointContactAttention(num_head=12, num_scalar_qk=16, num_scalar_v=16,
        ...                                 num_point_v=8, num_point_qk=4,
        ...                                 num_channel=384, pair_dim=128)
        >>> inputs_1d = Tensor(np.ones((256, 384)), mstype.float32)
        >>> inputs_2d = Tensor(np.ones((256, 256, 128)), mstype.float32)
        >>> mask = Tensor(np.ones((256, 1)), mstype.float32)
        >>> rotation = tuple([Tensor(np.ones(256), mstype.float16) for _ in range(9)])
        >>> translation = tuple([Tensor(np.ones(256), mstype.float16) for _ in range(3)])
        >>> attn_out = model(inputs_1d, inputs_2d, mask, rotation, translation)
        >>> print(attn_out.shape)
        (256, 384)
    """

    def __init__(self, num_head, num_scalar_qk, num_scalar_v, num_point_v, num_point_qk, num_channel, pair_dim):
        super(InvariantPointContactAttention, self).__init__()

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
                                 weight_init=lecun_init(self.num_channel))
        self.kv_scalar = nn.Dense(self.num_channel, self.num_head * (self.num_scalar_qk + self.num_scalar_v),
                                  weight_init=lecun_init(self.num_channel))
        self.q_point_local = nn.Dense(self.num_channel, self.num_head * 3 * self.num_point_qk,
                                      weight_init=lecun_init(self.num_channel)
                                      )
        self.kv_point_local = nn.Dense(self.num_channel, self.num_head * 3 * (self.num_point_qk + self.num_point_v),
                                       weight_init=lecun_init(self.num_channel))
        self.contact_layer = nn.Dense(32, self.num_head)
        self.soft_max = nn.Softmax()
        self.soft_plus = ops.Softplus()
        self.trainable_point_weights = Parameter(Tensor(np.ones((12,)), mstype.float32), name="trainable_point_weights")
        self.attention_2d = nn.Dense(pair_dim, self.num_head, weight_init=lecun_init(pair_dim))
        self.output_projection = nn.Dense(self.projection_num, self.num_channel, weight_init='zeros'
                                          )
        self.scalar_weights = Tensor(np.sqrt(1.0 / (3 * 16)).astype(np.float32))
        self.point_weights = Tensor(np.sqrt(1.0 / (3 * 18)).astype(np.float32))
        self.attention_2d_weights = Tensor(np.sqrt(1.0 / 3).astype(np.float32))

    def construct(self, inputs_1d, inputs_2d, mask, rotation, translation, contact_act=None, contact_info_mask=None):
        '''construct'''
        num_residues, _ = inputs_1d.shape

        # Improve readability by removing a large number of 'self's.
        num_head = self.num_head
        num_scalar_qk = self.num_scalar_qk
        num_point_qk = self.num_point_qk
        num_scalar_v = self.num_scalar_v
        num_point_v = self.num_point_v

        # Construct scalar queries of shape:
        q_scalar = self.q_scalar(inputs_1d)
        q_scalar = mnp.reshape(q_scalar, [num_residues, num_head, num_scalar_qk])

        # Construct scalar keys/values of shape:
        kv_scalar = self.kv_scalar(inputs_1d)
        kv_scalar = mnp.reshape(kv_scalar, [num_residues, num_head, num_scalar_v + num_scalar_qk])
        k_scalar, v_scalar = mnp.split(kv_scalar, [num_scalar_qk], axis=-1)

        # Construct query points of shape:
        # First construct query points in local frame.
        q_point_local = self.q_point_local(inputs_1d)

        q_point_local = mnp.split(q_point_local, 3, axis=-1)
        q_point_local = (ops.Squeeze()(q_point_local[0]), ops.Squeeze()(q_point_local[1]),
                         ops.Squeeze()(q_point_local[2]))
        # Project query points into global frame.
        q_point_global = apply_to_point(rotation, translation, q_point_local, 1)

        # Reshape query point for later use.
        q_point0 = mnp.reshape(q_point_global[0], (num_residues, num_head, num_point_qk))
        q_point1 = mnp.reshape(q_point_global[1], (num_residues, num_head, num_point_qk))
        q_point2 = mnp.reshape(q_point_global[2], (num_residues, num_head, num_point_qk))

        # Construct key and value points.
        # Key points have shape [num_residues, num_head, num_point_qk]
        # Value points have shape [num_residues, num_head, num_point_v]

        # Construct key and value points in local frame.
        kv_point_local = self.kv_point_local(inputs_1d)

        kv_point_local = mnp.split(kv_point_local, 3, axis=-1)
        kv_point_local = (ops.Squeeze()(kv_point_local[0]), ops.Squeeze()(kv_point_local[1]),
                          ops.Squeeze()(kv_point_local[2]))
        # Project key and value points into global frame.
        kv_point_global = apply_to_point(rotation, translation, kv_point_local, 1)

        kv_point_global0 = mnp.reshape(kv_point_global[0], (num_residues, num_head, (num_point_qk + num_point_v)))
        kv_point_global1 = mnp.reshape(kv_point_global[1], (num_residues, num_head, (num_point_qk + num_point_v)))
        kv_point_global2 = mnp.reshape(kv_point_global[2], (num_residues, num_head, (num_point_qk + num_point_v)))

        # Split key and value points.
        k_point0, v_point0 = mnp.split(kv_point_global0, [num_point_qk], axis=-1)
        k_point1, v_point1 = mnp.split(kv_point_global1, [num_point_qk], axis=-1)
        k_point2, v_point2 = mnp.split(kv_point_global2, [num_point_qk], axis=-1)

        trainable_point_weights = self.soft_plus(self.trainable_point_weights)
        point_weights = self.point_weights * ops.expand_dims(trainable_point_weights, axis=1)

        v_point = [mnp.swapaxes(v_point0, -2, -3), mnp.swapaxes(v_point1, -2, -3), mnp.swapaxes(v_point2, -2, -3)]
        q_point = [mnp.swapaxes(q_point0, -2, -3), mnp.swapaxes(q_point1, -2, -3), mnp.swapaxes(q_point2, -2, -3)]
        k_point = [mnp.swapaxes(k_point0, -2, -3), mnp.swapaxes(k_point1, -2, -3), mnp.swapaxes(k_point2, -2, -3)]

        dist2 = ops.Square()(ops.expand_dims(q_point[0], 2) - ops.expand_dims(k_point[0], 1)) + \
                ops.Square()(ops.expand_dims(q_point[1], 2) - ops.expand_dims(k_point[1], 1)) + \
                ops.Square()(ops.expand_dims(q_point[2], 2) - ops.expand_dims(k_point[2], 1))

        attn_qk_point = -0.5 * mnp.sum(ops.expand_dims(ops.expand_dims(point_weights, 1), 1) * dist2, axis=-1)

        v = mnp.swapaxes(v_scalar, -2, -3)
        q = mnp.swapaxes(self.scalar_weights * q_scalar, -2, -3)
        k = mnp.swapaxes(k_scalar, -2, -3)
        attn_qk_scalar = ops.matmul(q, mnp.swapaxes(k, -2, -1))
        attn_logits = attn_qk_scalar + attn_qk_point

        attention_2d = self.attention_2d(inputs_2d)
        attention_2d = mnp.transpose(attention_2d, [2, 0, 1])
        attention_2d = self.attention_2d_weights * attention_2d

        attn_logits += attention_2d

        # modify wch
        contact_act = self.contact_layer(contact_act)
        contact_act = ops.Transpose()(contact_act, (2, 0, 1))
        contact_act = contact_act * contact_info_mask[None, :, :]

        attn_logits += contact_act

        mask_2d = mask * mnp.swapaxes(mask, -1, -2)
        attn_logits -= 50 * (1. - mask_2d)

        attn = self.soft_max(attn_logits)

        result_scalar = ops.matmul(attn, v)

        result_point_global = [mnp.swapaxes(mnp.sum(attn[:, :, :, None] * v_point[0][:, None, :, :], axis=-2), -2, -3),
                               mnp.swapaxes(mnp.sum(attn[:, :, :, None] * v_point[1][:, None, :, :], axis=-2), -2, -3),
                               mnp.swapaxes(mnp.sum(attn[:, :, :, None] * v_point[2][:, None, :, :], axis=-2), -2, -3)
                               ]

        result_point_global = [mnp.reshape(result_point_global[0], [num_residues, num_head * num_point_v]),
                               mnp.reshape(result_point_global[1], [num_residues, num_head * num_point_v]),
                               mnp.reshape(result_point_global[2], [num_residues, num_head * num_point_v])]
        result_scalar = mnp.swapaxes(result_scalar, -2, -3)

        result_scalar = mnp.reshape(result_scalar, [num_residues, num_head * num_scalar_v])

        result_point_local = invert_point(result_point_global, rotation, translation, 1)

        output_feature1 = result_scalar
        output_feature20 = result_point_local[0]
        output_feature21 = result_point_local[1]
        output_feature22 = result_point_local[2]

        output_feature3 = mnp.sqrt(self._dist_epsilon +
                                   ops.Square()(result_point_local[0]) +
                                   ops.Square()(result_point_local[1]) +
                                   ops.Square()(result_point_local[2]))

        result_attention_over_2d = ops.matmul(mnp.swapaxes(attn, 0, 1), inputs_2d)
        num_out = num_head * result_attention_over_2d.shape[-1]
        output_feature4 = mnp.reshape(result_attention_over_2d, [num_residues, num_out])

        final_act = mnp.concatenate([output_feature1, output_feature20, output_feature21,
                                     output_feature22, output_feature3, output_feature4], axis=-1)
        final_result = self.output_projection(final_act)
        return final_result


class MultiRigidSidechain(nn.Cell):
    """Class to make side chain atoms."""

    def __init__(self, config, single_repr_dim):
        super().__init__()
        self.config = config
        self.input_projection = nn.Dense(single_repr_dim, self.config.num_channel,
                                         weight_init=lecun_init(single_repr_dim))
        self.input_projection_1 = nn.Dense(single_repr_dim, self.config.num_channel,
                                           weight_init=lecun_init(single_repr_dim))
        self.relu = nn.ReLU()
        self.resblock1 = nn.Dense(self.config.num_channel, self.config.num_channel,
                                  weight_init=lecun_init(self.config.num_channel,
                                                         initializer_name='relu'))
        self.resblock2 = nn.Dense(self.config.num_channel, self.config.num_channel, weight_init='zeros')
        self.resblock1_1 = nn.Dense(self.config.num_channel, self.config.num_channel,
                                    weight_init=lecun_init(self.config.num_channel, initializer_name='relu'))
        self.resblock2_1 = nn.Dense(self.config.num_channel, self.config.num_channel, weight_init='zeros')
        self.unnormalized_angles = nn.Dense(self.config.num_channel, 14,
                                            weight_init=lecun_init(self.config.num_channel))
        self.restype_atom14_to_rigid_group = Tensor(residue_constants.restype_atom14_to_rigid_group)
        self.restype_atom14_rigid_group_positions = Tensor(residue_constants.restype_atom14_rigid_group_positions)
        self.restype_atom14_mask = Tensor(residue_constants.restype_atom14_mask)
        self.restype_rigid_group_default_frame = Tensor(residue_constants.restype_rigid_group_default_frame)
        self.l2_normalize = ops.L2Normalize(axis=-1, epsilon=1e-12)

    def construct(self, rotation, translation, act, initial_act, aatype):
        """Predict side chains using rotation and translation representations.

        Args:
          rotation: The rotation matrices.
          translation: A translation matrices.
          act: updated pair activations from structure module
          initial_act: initial act representations (input of structure module)
          aatype: Amino acid type representations

        Returns:
          angles, positions and new frames
        """

        act1 = self.input_projection(self.relu(act))
        init_act1 = self.input_projection_1(self.relu(initial_act))
        # Sum the activation list (equivalent to concat then Linear).
        act = act1 + init_act1

        # Mapping with some residual blocks.
        # resblock1
        old_act = act
        act = self.resblock1(self.relu(act))
        act = self.resblock2(self.relu(act))
        act += old_act
        # resblock2
        old_act = act
        act = self.resblock1_1(self.relu(act))
        act = self.resblock2_1(self.relu(act))
        act += old_act

        # Map activations to torsion angles. Shape: (num_res, 14).
        num_res = act.shape[0]
        unnormalized_angles = self.unnormalized_angles(self.relu(act))

        unnormalized_angles = mnp.reshape(unnormalized_angles, [num_res, 7, 2])
        angles = self.l2_normalize(unnormalized_angles)

        backb_to_global = ((rotation[0], rotation[1], rotation[2],
                            rotation[3], rotation[4], rotation[5],
                            rotation[6], rotation[7], rotation[8]),
                           (translation[0], translation[1], translation[2]))

        all_frames_to_global = torsion_angles_to_frames(aatype, backb_to_global, angles,
                                                        self.restype_rigid_group_default_frame)

        pred_positions = frames_and_literature_positions_to_atom14_pos(aatype, all_frames_to_global,
                                                                       self.restype_atom14_to_rigid_group,
                                                                       self.restype_atom14_rigid_group_positions,
                                                                       self.restype_atom14_mask)

        atom_pos = pred_positions
        frames = all_frames_to_global
        res = (angles, unnormalized_angles, atom_pos, frames)
        return res


class FoldIteration(nn.Cell):
    """A single iteration of the main structure module loop."""

    def __init__(self, config, pair_dim, single_repr_dim):
        super().__init__()
        self.config = config
        self.drop_out = nn.Dropout(p=0.1)
        self.attention_layer_norm = nn.LayerNorm([self.config.num_channel,], epsilon=1e-5)
        self.transition_layer_norm = nn.LayerNorm([self.config.num_channel,], epsilon=1e-5)
        self.transition = nn.Dense(self.config.num_channel, config.num_channel,
                                   weight_init=lecun_init(self.config.num_channel, initializer_name='relu'))
        self.transition_1 = nn.Dense(self.config.num_channel, self.config.num_channel,
                                     weight_init=lecun_init(self.config.num_channel, initializer_name='relu'))
        self.transition_2 = nn.Dense(self.config.num_channel, self.config.num_channel, weight_init='zeros')
        self.relu = nn.ReLU()
        self.affine_update = nn.Dense(self.config.num_channel, 6, weight_init='zeros')
        self.attention_module = InvariantPointContactAttention(self.config.num_head,
                                                               self.config.num_scalar_qk,
                                                               self.config.num_scalar_v,
                                                               self.config.num_point_v,
                                                               self.config.num_point_qk,
                                                               self.config.num_channel,
                                                               pair_dim)
        self.mu_side_chain = MultiRigidSidechain(self.config.sidechain, single_repr_dim)
        self.print = ops.Print()

    def construct(self, act, static_feat_2d, sequence_mask, quaternion, rotation, \
                  translation, initial_act, aatype, contact_act2, contact_info_mask2):
        """construct"""
        attn = self.attention_module(act, static_feat_2d, sequence_mask, \
                                     rotation, translation, contact_act2, contact_info_mask2)
        act += attn
        act = self.drop_out(act)
        act = self.attention_layer_norm(act)
        # Transition
        input_act = act
        act = self.transition(act)
        act = self.relu(act)
        act = self.transition_1(act)
        act = self.relu(act)
        act = self.transition_2(act)

        act += input_act
        act = self.drop_out(act)
        act = self.transition_layer_norm(act)

        # This block corresponds to
        # Jumper et al. (2021) Alg. 23 "Backbone update"
        # Affine update
        affine_update = self.affine_update(act)
        quaternion, rotation, translation = pre_compose(quaternion, rotation, translation, affine_update)
        translation1 = vecs_scale(translation, 10.0)
        rotation1 = rotation
        angles_sin_cos, unnormalized_angles_sin_cos, atom_pos, frames = \
            self.mu_side_chain(rotation1, translation1, act, initial_act, aatype)

        affine_output = quaternion_to_tensor(quaternion, translation)
        quaternion = F.stop_gradient(quaternion)
        rotation = F.stop_gradient(rotation)
        res = (act, quaternion, translation, rotation, affine_output, angles_sin_cos, unnormalized_angles_sin_cos, \
               atom_pos, frames)
        return res


class ContactStructureModule(nn.Cell):
    """StructureModule as a network head."""

    def __init__(self, config, single_repr_dim, pair_dim):
        super(ContactStructureModule, self).__init__()
        self.config = config.model.structure_module
        self.seq_length = config.seq_length
        self.fold_iteration = FoldIteration(self.config, pair_dim, single_repr_dim)
        self.single_layer_norm = nn.LayerNorm([single_repr_dim,], epsilon=1e-5)
        self.initial_projection = nn.Dense(single_repr_dim, self.config.num_channel,
                                           weight_init=lecun_init(single_repr_dim))
        self.pair_layer_norm = nn.LayerNorm([pair_dim,], epsilon=1e-5)
        self.num_layer = self.config.num_layer
        self.indice0 = Tensor(
            np.arange(self.seq_length).reshape((-1, 1, 1)).repeat(37, axis=1).astype("int32"))
        self.traj_w = Tensor(np.array([1.] * 4 + [self.config.position_scale] * 3), mstype.float32)
        self.use_sumcons = True

    def construct(self, single, pair, seq_mask, aatype, contact_act2, contact_info_mask2, residx_atom37_to_atom14=None,
                  atom37_atom_exists=None):
        """construct"""
        sequence_mask = seq_mask[:, None]
        act = self.single_layer_norm(single)
        initial_act = act
        act = self.initial_projection(act)
        quaternion, rotation, translation = initial_affine(self.seq_length)
        act_2d = self.pair_layer_norm(pair)
        # folder iteration
        atom_pos, affine_output_new, angles_sin_cos_new, um_angles_sin_cos_new, sidechain_frames, act_iter = \
            self.iteration_operation(act, act_2d, sequence_mask, quaternion, rotation, translation, initial_act, aatype,
                                     contact_act2, contact_info_mask2)
        atom14_pred_positions = vecs_to_tensor(atom_pos)[-1]
        sidechain_atom_pos = atom_pos

        atom37_pred_positions = atom14_to_atom37(atom14_pred_positions,
                                                 residx_atom37_to_atom14,
                                                 atom37_atom_exists,
                                                 self.indice0)

        structure_traj = affine_output_new * self.traj_w
        final_affines = affine_output_new[-1]
        final_atom_positions = atom37_pred_positions
        final_atom_mask = atom37_atom_exists
        rp_structure_module = act_iter
        if self.use_sumcons:
            pseudo_beta_pred = pseudo_beta_fn(aatype, atom37_pred_positions, None)
            coord_diffs = pseudo_beta_pred[None] - pseudo_beta_pred[:, None]
            distance = ops.Sqrt()(ops.ReduceSum()(ops.Square()(coord_diffs), -1) + 1e-8)
            scale = (8.10 / distance - 1) * contact_info_mask2 * (distance > 8.10)
            contact_translation_2 = scale[:, :, None] * coord_diffs / 2
            contact_translation = ops.ReduceSum(keep_dims=True)(contact_translation_2, 1)
            atom14_pred_positions = atom14_pred_positions - contact_translation
            final_atom_positions = final_atom_positions - contact_translation
        res = (final_atom_positions, final_atom_mask, rp_structure_module, atom14_pred_positions, final_affines, \
               angles_sin_cos_new, um_angles_sin_cos_new, sidechain_frames, sidechain_atom_pos, structure_traj)
        return res

    def iteration_operation(self, act, act_2d, sequence_mask, quaternion, rotation, translation, initial_act,
                            aatype, contact_act2, contact_info_mask2):
        """iteration_operation"""
        affine_init = ()
        angles_sin_cos_init = ()
        um_angles_sin_cos_init = ()
        atom_pos_batch = ()
        frames_batch = ()

        for _ in range(self.num_layer):
            act, quaternion, translation, rotation, affine_output, angles_sin_cos, unnormalized_angles_sin_cos, \
            atom_pos, frames = \
                self.fold_iteration(act, act_2d, sequence_mask, quaternion, rotation, translation, initial_act, aatype,
                                    contact_act2, contact_info_mask2)

            affine_init = affine_init + (affine_output[None, ...],)
            angles_sin_cos_init = angles_sin_cos_init + (angles_sin_cos[None, ...],)
            um_angles_sin_cos_init = um_angles_sin_cos_init + (unnormalized_angles_sin_cos[None, ...],)
            atom_pos_batch += (mnp.concatenate(vecs_expand_dims(atom_pos, 0), axis=0)[:, None, ...],)
            frames_batch += (mnp.concatenate(rots_expand_dims(frames[0], 0) +
                                             vecs_expand_dims(frames[1], 0), axis=0)[:, None, ...],)
        affine_output_new = mnp.concatenate(affine_init, axis=0)
        angles_sin_cos_new = mnp.concatenate(angles_sin_cos_init, axis=0)
        um_angles_sin_cos_new = mnp.concatenate(um_angles_sin_cos_init, axis=0)
        frames_new = mnp.concatenate(frames_batch, axis=1)
        atom_pos_new = mnp.concatenate(atom_pos_batch, axis=1)
        res = (atom_pos_new, affine_output_new, angles_sin_cos_new, um_angles_sin_cos_new, frames_new, act)
        return res
