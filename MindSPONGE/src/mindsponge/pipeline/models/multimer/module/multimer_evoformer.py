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

import mindspore.nn as nn
from mindspore.ops import operations as P
from mindsponge.pipeline.cell import MSARowAttentionWithPairBias, Transition, OuterProductMean, \
    TriangleAttention, TriangleMultiplication, \
    MSAColumnGlobalAttention, MSAColumnAttention


class MultimerEvoformer(nn.Cell):
    '''multimerevoformer'''

    def __init__(self, config, msa_act_dim, pair_act_dim, is_extra_msa, batch_size):
        super(MultimerEvoformer, self).__init__()
        if is_extra_msa:
            self.slice_cfg = config.slice.extra_msa_stack
        else:
            self.slice_cfg = config.slice.msa_stack
        self.config = config.evoformer
        self.msa_row_attention_with_pair_bias = MSARowAttentionWithPairBias(
            self.config.msa_row_attention_with_pair_bias.num_head,
            msa_act_dim,
            self.config.msa_row_attention_with_pair_bias.gating,
            msa_act_dim,
            pair_act_dim,
            batch_size,
            self.slice_cfg.msa_row_attention_with_pair_bias)

        self.msa_transition = Transition(self.config.msa_transition.num_intermediate_factor,
                                         msa_act_dim,
                                         batch_size,
                                         self.slice_cfg.msa_transition)

        self.outer_product_mean = OuterProductMean(self.config.outer_product_mean.num_outer_channel,
                                                   msa_act_dim,
                                                   pair_act_dim,
                                                   batch_size,
                                                   self.slice_cfg.outer_product_mean)

        self.triangle_attention_starting_node = TriangleAttention(
            self.config.triangle_attention_starting_node.orientation,
            self.config.triangle_attention_starting_node.num_head,
            pair_act_dim,
            self.config.triangle_attention_starting_node.gating,
            pair_act_dim,
            batch_size,
            self.slice_cfg.triangle_attention_starting_node)

        self.triangle_attention_ending_node = TriangleAttention(self.config.triangle_attention_ending_node.orientation,
                                                                self.config.triangle_attention_ending_node.num_head,
                                                                pair_act_dim,
                                                                self.config.triangle_attention_ending_node.gating,
                                                                pair_act_dim,
                                                                batch_size,
                                                                self.slice_cfg.triangle_attention_ending_node)

        self.pair_transition = Transition(self.config.pair_transition.num_intermediate_factor,
                                          pair_act_dim,
                                          batch_size,
                                          self.slice_cfg.pair_transition)

        self.triangle_multiplication_outgoing = TriangleMultiplication(
            self.config.triangle_multiplication_outgoing.num_intermediate_channel,
            self.config.triangle_multiplication_outgoing.equation,
            layer_norm_dim=pair_act_dim,
            batch_size=batch_size)

        self.triangle_multiplication_incoming = TriangleMultiplication(
            self.config.triangle_multiplication_incoming.num_intermediate_channel,
            self.config.triangle_multiplication_incoming.equation,
            layer_norm_dim=pair_act_dim,
            batch_size=batch_size)
        if is_extra_msa:
            self.attn_mod = MSAColumnGlobalAttention(self.config.msa_column_attention.num_head,
                                                     self.config.msa_column_attention.gating,
                                                     msa_act_dim,
                                                     batch_size,
                                                     self.slice_cfg.msa_column_global_attention)
        else:
            self.attn_mod = MSAColumnAttention(self.config.msa_column_attention.num_head,
                                               msa_act_dim,
                                               self.config.msa_column_attention.gating,
                                               msa_act_dim,
                                               batch_size,
                                               self.slice_cfg.msa_column_attention)

    def construct(self, msa_act, pair_act, msa_mask, extra_msa_norm, pair_mask, index=None):
        '''construct'''
        pair_act = P.Add()(pair_act, self.outer_product_mean(msa_act, msa_mask, extra_msa_norm, index))
        msa_act = P.Add()(msa_act, self.msa_row_attention_with_pair_bias(msa_act, msa_mask, pair_act, index))
        msa_act = P.Add()(msa_act, self.attn_mod(msa_act, msa_mask, index))
        msa_act = P.Add()(msa_act, self.msa_transition(msa_act, index))
        pair_act = P.Add()(pair_act, self.triangle_multiplication_outgoing(pair_act, pair_mask, index))
        pair_act = P.Add()(pair_act, self.triangle_multiplication_incoming(pair_act, pair_mask, index))
        pair_act = P.Add()(pair_act, self.triangle_attention_starting_node(pair_act, pair_mask, index))
        pair_act = P.Add()(pair_act, self.triangle_attention_ending_node(pair_act, pair_mask, index))
        pair_act = P.Add()(pair_act, self.pair_transition(pair_act, index))
        return msa_act, pair_act
