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
"""evoformer model"""

import mindspore.common.dtype as mstype
import mindspore.nn as nn
from mindspore.common.tensor import Tensor
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from module.basic_module import Attention, MSARowAttentionWithPairBias, MSAColumnAttention, MSAColumnGlobalAttention, \
    Transition, OuterProductMean, TriangleMultiplication, TriangleAttention, MSARowAttentionWithPairBiasMSA, \
    CustomDropout
from common import residue_constants
from common.utils import dgram_from_positions, batch_make_transform_from_reference, batch_quat_affine, \
    batch_invert_point, batch_rot_to_quat, lecun_init

class EvoformerIterationTail(nn.Cell):
    """EvoformerIterationTail"""
    def __init__(self, config, msa_act_dim, pair_act_dim, is_extra_msa, global_config):
        super(EvoformerIterationTail, self).__init__()
        self.config = config
        self.is_extra_msa = is_extra_msa
        self.attn_mod = MSAColumnGlobalAttention(self.config.msa_column_attention, msa_act_dim,
                                                 global_config.extra_msa_stack.msa_column_global_attention.slice_num)
        self.msa_transition = Transition(self.config.msa_transition, msa_act_dim,
                                         global_config.extra_msa_stack.msa_transition.slice_num)
        self.outer_product_mean = OuterProductMean(self.config.outer_product_mean, msa_act_dim, pair_act_dim,
                                                   global_config.extra_msa_stack.outer_product_mean.slice_num)
        self.triangle_attention_starting_node = TriangleAttention(self.config.triangle_attention_starting_node,
                                                                  pair_act_dim,
                                                                  global_config.extra_msa_stack.
                                                                  triangle_attention_starting_node.slice_num)
        self.triangle_attention_ending_node = TriangleAttention(self.config.triangle_attention_ending_node,
                                                                pair_act_dim,
                                                                global_config.extra_msa_stack.
                                                                triangle_attention_ending_node.slice_num)
        self.pair_transition = Transition(self.config.pair_transition, pair_act_dim,
                                          global_config.extra_msa_stack.pair_transition.slice_num)

        self.triangle_multiplication_outgoing = TriangleMultiplication(self.config.triangle_multiplication_outgoing,
                                                                       pair_act_dim)
        self.triangle_multiplication_incoming = TriangleMultiplication(self.config.triangle_multiplication_incoming,
                                                                       pair_act_dim)

        self.dropout_attn_mod = CustomDropout(keep_prob=1 - self.attn_mod.config.dropout_rate)
        self.dropout_msa_transition = CustomDropout(keep_prob=1 - self.msa_transition.config.dropout_rate)
        self.dropout_outer_product_mean = CustomDropout(keep_prob=1 - self.outer_product_mean.config.dropout_rate)
        self.dropout_triangle_multiplication_outgoing = CustomDropout(
            keep_prob=1 - self.triangle_multiplication_outgoing.config.dropout_rate, broadcast_dim=0)
        self.dropout_triangle_multiplication_incoming = CustomDropout(
            keep_prob=1 - self.triangle_multiplication_incoming.config.dropout_rate, broadcast_dim=0)
        self.dropout_triangle_attention_starting_node = CustomDropout(
            keep_prob=1 - self.triangle_attention_starting_node.config.dropout_rate, broadcast_dim=0)
        self.dropout_triangle_attention_ending_node = CustomDropout(
            keep_prob=1 - self.triangle_attention_ending_node.config.dropout_rate, broadcast_dim=1)
        self.dropout_pair_transition = CustomDropout(keep_prob=1 - self.pair_transition.config.dropout_rate)

    def construct(self, msa_act, pair_act, msa_mask, msa_mask_norm, pair_mask):
        """construct"""
        msa_act = P.Add()(msa_act, self.dropout_attn_mod(self.attn_mod(msa_act, msa_mask)))
        msa_act = P.Add()(msa_act, self.dropout_msa_transition(self.msa_transition(msa_act)))
        pair_act = P.Add()(pair_act,
                           self.dropout_outer_product_mean(self.outer_product_mean(msa_act, msa_mask, msa_mask_norm)))
        pair_act = P.Add()(pair_act, self.dropout_triangle_multiplication_outgoing(
            self.triangle_multiplication_outgoing(pair_act, pair_mask)))
        pair_act = P.Add()(pair_act, self.dropout_triangle_multiplication_incoming(
            self.triangle_multiplication_incoming(pair_act, pair_mask)))
        pair_act = P.Add()(pair_act, self.dropout_triangle_attention_starting_node(
            self.triangle_attention_starting_node(pair_act, pair_mask)))
        pair_act = P.Add()(pair_act, self.dropout_triangle_attention_ending_node(
            self.triangle_attention_ending_node(pair_act, pair_mask)))
        pair_act = P.Add()(pair_act, self.dropout_pair_transition(self.pair_transition(pair_act)))
        return msa_act, pair_act


class EvoformerIterationMSA(nn.Cell):
    """EvoformerIterationMSA"""
    def __init__(self, config, msa_act_dim, pair_act_dim, is_extra_msa, global_config):
        super(EvoformerIterationMSA, self).__init__()
        self.config = config
        self.is_extra_msa = is_extra_msa

        self.msa_row_attention_with_pair_bias = MSARowAttentionWithPairBiasMSA(
            self.config.msa_row_attention_with_pair_bias, msa_act_dim, pair_act_dim,
            global_config.extra_msa_stack.msa_row_attention_with_pair_bias.slice_num)
        self.msa_row_attention_with_pair_bias.recompute()

        self.dropout_msa_row_attention_with_pair_bias = CustomDropout(
            keep_prob=1 - self.msa_row_attention_with_pair_bias.config.dropout_rate, broadcast_dim=0)
        self.tail = EvoformerIterationTail(self.config,
                                           msa_act_dim=64,
                                           pair_act_dim=128,
                                           is_extra_msa=True,
                                           global_config=global_config)
        self.tail.recompute()

    def construct(self, msa_act, pair_act, msa_mask, msa_mask_norm, pair_mask):
        """construct"""
        pair_act = P.Cast()(pair_act, mstype.float16)
        msa_act = P.Cast()(msa_act, mstype.float16)
        msa_act = P.Add()(msa_act, self.dropout_msa_row_attention_with_pair_bias(
            self.msa_row_attention_with_pair_bias(msa_act, msa_mask, pair_act)))
        msa_act, pair_act = self.tail(msa_act, pair_act, msa_mask, msa_mask_norm, pair_mask)

        return msa_act, pair_act


class EvoformerIteration(nn.Cell):
    """EvoformerIteration"""
    def __init__(self, config, msa_act_dim, pair_act_dim, is_extra_msa, global_config):
        super(EvoformerIteration, self).__init__()
        self.config = config
        self.is_extra_msa = is_extra_msa
        if not self.is_extra_msa:
            self.msa_row_attention_with_pair_bias = MSARowAttentionWithPairBias(
                self.config.msa_row_attention_with_pair_bias, msa_act_dim, pair_act_dim,
                global_config.evoformer_iteration.msa_row_attention_with_pair_bias.slice_num)
            self.attn_mod = MSAColumnAttention(self.config.msa_column_attention, msa_act_dim,
                                               global_config.evoformer_iteration.msa_column_attention.slice_num)
            self.msa_transition = Transition(self.config.msa_transition, msa_act_dim,
                                             global_config.evoformer_iteration.msa_transition.slice_num)
            self.outer_product_mean = OuterProductMean(self.config.outer_product_mean, msa_act_dim, pair_act_dim,
                                                       global_config.evoformer_iteration.outer_product_mean.slice_num)
            self.triangle_attention_starting_node = TriangleAttention(self.config.triangle_attention_starting_node,
                                                                      pair_act_dim,
                                                                      global_config.evoformer_iteration.
                                                                      triangle_attention_starting_node.slice_num)
            self.triangle_attention_ending_node = TriangleAttention(self.config.triangle_attention_ending_node,
                                                                    pair_act_dim,
                                                                    global_config.evoformer_iteration.
                                                                    triangle_attention_ending_node.slice_num)
            self.pair_transition = Transition(self.config.pair_transition, pair_act_dim,
                                              global_config.evoformer_iteration.pair_transition.slice_num)
        else:
            self.msa_row_attention_with_pair_bias = MSARowAttentionWithPairBias(
                self.config.msa_row_attention_with_pair_bias, msa_act_dim, pair_act_dim,
                global_config.extra_msa_stack.msa_row_attention_with_pair_bias.slice_num)
            self.attn_mod = MSAColumnGlobalAttention(self.config.msa_column_attention, msa_act_dim,
                                                     global_config.extra_msa_stack.
                                                     msa_column_global_attention.slice_num)
            self.msa_transition = Transition(self.config.msa_transition, msa_act_dim,
                                             global_config.extra_msa_stack.msa_transition.slice_num)
            self.outer_product_mean = OuterProductMean(self.config.outer_product_mean, msa_act_dim, pair_act_dim,
                                                       global_config.extra_msa_stack.outer_product_mean.slice_num)
            self.triangle_attention_starting_node = TriangleAttention(self.config.triangle_attention_starting_node,
                                                                      pair_act_dim,
                                                                      global_config.extra_msa_stack.
                                                                      triangle_attention_starting_node.slice_num)
            self.triangle_attention_ending_node = TriangleAttention(self.config.triangle_attention_ending_node,
                                                                    pair_act_dim,
                                                                    global_config.extra_msa_stack.
                                                                    triangle_attention_ending_node.slice_num)
            self.pair_transition = Transition(self.config.pair_transition, pair_act_dim,
                                              global_config.extra_msa_stack.pair_transition.slice_num)

        self.triangle_multiplication_outgoing = TriangleMultiplication(self.config.triangle_multiplication_outgoing,
                                                                       pair_act_dim)
        self.triangle_multiplication_incoming = TriangleMultiplication(self.config.triangle_multiplication_incoming,
                                                                       pair_act_dim)

        self.dropout_msa_row_attention_with_pair_bias = CustomDropout(
            keep_prob=1 - self.msa_row_attention_with_pair_bias.config.dropout_rate, broadcast_dim=0)
        self.dropout_attn_mod = CustomDropout(keep_prob=1 - self.attn_mod.config.dropout_rate)
        self.dropout_msa_transition = CustomDropout(keep_prob=1 - self.msa_transition.config.dropout_rate)
        self.dropout_outer_product_mean = CustomDropout(keep_prob=1 - self.outer_product_mean.config.dropout_rate)
        self.dropout_triangle_multiplication_outgoing = CustomDropout(
            keep_prob=1 - self.triangle_multiplication_outgoing.config.dropout_rate, broadcast_dim=0)
        self.dropout_triangle_multiplication_incoming = CustomDropout(
            keep_prob=1 - self.triangle_multiplication_incoming.config.dropout_rate, broadcast_dim=0)
        self.dropout_triangle_attention_starting_node = CustomDropout(
            keep_prob=1 - self.triangle_attention_starting_node.config.dropout_rate, broadcast_dim=0)
        self.dropout_triangle_attention_ending_node = CustomDropout(
            keep_prob=1 - self.triangle_attention_ending_node.config.dropout_rate, broadcast_dim=1)
        self.dropout_pair_transition = CustomDropout(keep_prob=1 - self.pair_transition.config.dropout_rate)

    def construct(self, msa_act, pair_act, msa_mask, msa_mask_norm, pair_mask):
        """construct"""
        pair_act = P.Cast()(pair_act, mstype.float16)
        msa_act = P.Cast()(msa_act, mstype.float16)
        msa_act = P.Add()(msa_act, self.dropout_msa_row_attention_with_pair_bias(
            self.msa_row_attention_with_pair_bias(msa_act, msa_mask, pair_act)))
        msa_act = P.Add()(msa_act, self.dropout_attn_mod(self.attn_mod(msa_act, msa_mask)))
        msa_act = P.Add()(msa_act, self.dropout_msa_transition(self.msa_transition(msa_act)))
        pair_act = P.Add()(pair_act,
                           self.dropout_outer_product_mean(self.outer_product_mean(msa_act, msa_mask, msa_mask_norm)))
        pair_act = P.Add()(pair_act, self.dropout_triangle_multiplication_outgoing(
            self.triangle_multiplication_outgoing(pair_act, pair_mask)))
        pair_act = P.Add()(pair_act, self.dropout_triangle_multiplication_incoming(
            self.triangle_multiplication_incoming(pair_act, pair_mask)))
        pair_act = P.Add()(pair_act, self.dropout_triangle_attention_starting_node(
            self.triangle_attention_starting_node(pair_act, pair_mask)))
        pair_act = P.Add()(pair_act, self.dropout_triangle_attention_ending_node(
            self.triangle_attention_ending_node(pair_act, pair_mask)))
        pair_act = P.Add()(pair_act, self.dropout_pair_transition(self.pair_transition(pair_act)))
        return msa_act, pair_act


class TemplatePairStack(nn.Cell):
    """TemplatePairStack"""
    def __init__(self, config, global_config=None):
        super(TemplatePairStack, self).__init__()
        self.config = config
        self.global_config = global_config
        self.num_block = self.config.num_block

        self.triangle_attention_starting_node = TriangleAttention(self.config.triangle_attention_starting_node,
                                                                  64,
                                                                  global_config.template_pair_stack.
                                                                  triangle_attention_starting_node.slice_num)
        self.triangle_attention_ending_node = TriangleAttention(self.config.triangle_attention_ending_node,
                                                                64,
                                                                global_config.template_pair_stack.
                                                                triangle_attention_ending_node.slice_num)
        # Hard Code
        self.pair_transition = Transition(self.config.pair_transition, 64,
                                          global_config.template_pair_stack.pair_transition.slice_num)
        self.triangle_multiplication_outgoing = TriangleMultiplication(self.config.triangle_multiplication_outgoing,
                                                                       layer_norm_dim=64)
        self.triangle_multiplication_incoming = TriangleMultiplication(self.config.triangle_multiplication_incoming,
                                                                       layer_norm_dim=64)

        self.dropout_triangle_attention_starting_node = CustomDropout(
            keep_prob=1 - self.triangle_attention_starting_node.config.dropout_rate, broadcast_dim=0)
        self.dropout_triangle_attention_ending_node = CustomDropout(
            keep_prob=1 - self.triangle_attention_ending_node.config.dropout_rate, broadcast_dim=1)
        self.dropout_pair_transition = CustomDropout(keep_prob=1 - self.pair_transition.config.dropout_rate)
        self.dropout_triangle_multiplication_outgoing = CustomDropout(
            keep_prob=1 - self.triangle_multiplication_outgoing.config.dropout_rate, broadcast_dim=0)
        self.dropout_triangle_multiplication_incoming = CustomDropout(
            keep_prob=1 - self.triangle_multiplication_incoming.config.dropout_rate, broadcast_dim=0)

    def construct(self, pair_act, pair_mask):
        """construct"""
        if not self.num_block:
            return pair_act

        pair_act = pair_act + self.dropout_triangle_attention_starting_node(
            self.triangle_attention_starting_node(pair_act, pair_mask))
        pair_act = pair_act + self.dropout_triangle_attention_ending_node(
            self.triangle_attention_ending_node(pair_act, pair_mask))
        pair_act = pair_act + self.dropout_triangle_multiplication_outgoing(
            self.triangle_multiplication_outgoing(pair_act, pair_mask))
        pair_act = pair_act + self.dropout_triangle_multiplication_incoming(
            self.triangle_multiplication_incoming(pair_act, pair_mask))
        pair_act = pair_act + self.dropout_pair_transition(self.pair_transition(pair_act))
        return pair_act


class SingleTemplateEmbedding(nn.Cell):
    """SingleTemplateEmbedding"""
    def __init__(self, config, global_config=None):
        super(SingleTemplateEmbedding, self).__init__()
        self.config = config
        self.num_channels = (self.config.template_pair_stack.triangle_attention_ending_node.value_dim)
        self.embedding2d = nn.Dense(88, self.num_channels,
                                    weight_init=lecun_init(88, initializer_name='relu')).to_float(mstype.float16)

        template_layers = nn.CellList()
        for _ in range(self.config.template_pair_stack.num_block):
            template_pair_stack_block = TemplatePairStack(self.config.template_pair_stack, global_config)
            template_layers.append(template_pair_stack_block)
        self.template_pair_stack = template_layers
        self.num_bins = self.config.dgram_features.num_bins
        self.min_bin = self.config.dgram_features.min_bin
        self.max_bin = self.config.dgram_features.max_bin

        self.one_hot = nn.OneHot(depth=22, axis=-1)
        self.n, self.ca, self.c = [residue_constants.atom_order[a] for a in ('N', 'CA', 'C')]

        self.use_template_unit_vector = self.config.use_template_unit_vector
        layer_norm_dim = 64
        self.output_layer_norm = nn.LayerNorm([layer_norm_dim,], epsilon=1e-5)
        self.zeros = Tensor(0, mstype.int32)
        self.num_block = self.config.template_pair_stack.num_block
        self.batch_block = 4

    def construct(self, query_embedding, mask_2d, template_aatype, template_all_atom_masks, template_all_atom_positions,
                  template_pseudo_beta_mask, template_pseudo_beta):
        """construct"""

        num_res = template_aatype[0, ...].shape[0]

        template_mask_2d_temp = P.Cast()(P.ExpandDims()(template_pseudo_beta_mask, -1) *
                                         P.ExpandDims()(template_pseudo_beta_mask, 1), query_embedding.dtype)
        template_dgram_temp = dgram_from_positions(template_pseudo_beta, self.num_bins, self.min_bin, self.max_bin)
        template_dgram_temp = P.Cast()(template_dgram_temp, query_embedding.dtype)

        to_concat_temp = (template_dgram_temp, P.ExpandDims()(template_mask_2d_temp, -1))
        aatype_temp = self.one_hot(template_aatype)  ## slice 0
        to_concat_temp = to_concat_temp + (P.Tile()(P.ExpandDims()(aatype_temp, 1), (1, num_res, 1, 1)),
                                           P.Tile()(P.ExpandDims()(aatype_temp, 2), (1, 1, num_res, 1)))
        rot_temp, trans_temp = batch_make_transform_from_reference(template_all_atom_positions[:, :, self.n],
                                                                   template_all_atom_positions[:, :, self.ca],
                                                                   template_all_atom_positions[:, :, self.c])

        _, rotation_tmp, translation_tmp = batch_quat_affine(batch_rot_to_quat(rot_temp, unstack_inputs=True),
                                                             translation=trans_temp, rotation=rot_temp,
                                                             unstack_inputs=True)
        points_tmp = P.ExpandDims()(translation_tmp, -2)
        affine_vec_tmp = batch_invert_point(points_tmp, rotation_tmp, translation_tmp, extra_dims=1)
        inv_distance_scalar_tmp = P.Rsqrt()(1e-6 + P.ReduceSum()(P.Square()(affine_vec_tmp), 1))
        template_mask_tmp = (template_all_atom_masks[:, :, self.n] * template_all_atom_masks[:, :, self.ca] *
                             template_all_atom_masks[:, :, self.c])
        template_mask_2d_tmp = P.ExpandDims()(template_mask_tmp, -1) * P.ExpandDims()(template_mask_tmp, 1)

        inv_distance_scalar_tmp = inv_distance_scalar_tmp * P.Cast()(template_mask_2d_tmp,
                                                                     inv_distance_scalar_tmp.dtype)
        unit_vector_tmp = P.Transpose()((affine_vec_tmp * P.ExpandDims()(inv_distance_scalar_tmp, 1)), (0, 2, 3, 1))
        template_mask_2d_tmp = P.Cast()(template_mask_2d_tmp, query_embedding.dtype)
        if not self.use_template_unit_vector:
            unit_vector_tmp = P.ZerosLike()(unit_vector_tmp)
        to_concat_temp = to_concat_temp + (unit_vector_tmp, P.ExpandDims()(template_mask_2d_tmp, -1),)
        act_tmp = P.Concat(-1)(to_concat_temp)
        act_tmp = act_tmp * P.ExpandDims()(template_mask_2d_tmp, -1)
        act_tmp = self.embedding2d(act_tmp)

        output = []
        idx_batch_loop = self.zeros
        for _ in range(self.batch_block):
            act_bacth = P.Gather()(act_tmp, idx_batch_loop, 0)
            for j in range(self.num_block):
                act_bacth = self.template_pair_stack[j](act_bacth, mask_2d)
            a_act = P.Reshape()(act_bacth, ((1,) + P.Shape()(act_bacth)))
            output.append(a_act)
            idx_batch_loop = F.depend(idx_batch_loop + 1, output[-1])

        act_tmp_loop = P.Concat()(output)
        act_tmp_loop = P.Cast()(act_tmp_loop, mstype.float32)
        act_tmp = self.output_layer_norm(act_tmp_loop)
        act_tmp = P.Cast()(act_tmp, mstype.float16)
        return act_tmp


class TemplateEmbedding(nn.Cell):
    """TemplateEmbedding"""
    def __init__(self, config, slice_num, global_config=None):
        super(TemplateEmbedding, self).__init__()
        self.config = config
        self.global_config = global_config
        self.num_channels = (self.config.template_pair_stack.triangle_attention_ending_node.value_dim)
        self.template_embedder = SingleTemplateEmbedding(self.config, self.global_config)
        self.template_pointwise_attention = Attention(self.config.attention, q_data_dim=128, m_data_dim=64,
                                                      output_dim=128)
        self.slice_num = slice_num
        if slice_num == 0:
            slice_num = 1

    def construct(self, query_embedding, template_aatype, template_all_atom_masks, template_all_atom_positions,
                  template_mask, template_pseudo_beta_mask, template_pseudo_beta, mask_2d):
        """construct"""
        num_templates = template_mask.shape[0]
        num_channels = self.num_channels
        num_res = query_embedding.shape[0]
        query_num_channels = query_embedding.shape[-1]
        template_mask = P.Cast()(template_mask, query_embedding.dtype)

        mask_2d = F.depend(mask_2d, query_embedding)
        template_pair_representation = self.template_embedder(query_embedding, mask_2d, template_aatype,
                                                              template_all_atom_masks, template_all_atom_positions,
                                                              template_pseudo_beta_mask,
                                                              template_pseudo_beta)

        flat_query = P.Reshape()(query_embedding, (num_res * num_res, 1, query_num_channels))
        flat_templates = P.Reshape()(
            P.Transpose()(template_pair_representation, (1, 2, 0, 3)),
            (num_res * num_res, num_templates, num_channels))

        template_mask_bias = P.ExpandDims()(P.ExpandDims()(P.ExpandDims()(template_mask, 0), 1), 2) - 1.0
        template_mask_bias = P.Cast()(template_mask_bias, mstype.float32)
        bias = 1e9 * template_mask_bias

        embedding = self.template_pointwise_attention(flat_query, flat_templates, bias, nonbatched_bias=None)
        embedding = P.Reshape()(embedding, (num_res, num_res, query_num_channels))
        # No gradients if no templates.
        template_mask = P.Cast()(template_mask, embedding.dtype)
        embedding = embedding * P.Cast()((P.ReduceSum()(template_mask) > 0.), embedding.dtype)
        return embedding
