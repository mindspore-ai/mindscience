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
'''TEMPLATE'''
import mindspore.common.dtype as mstype
import mindspore.nn as nn
import mindspore.numpy as mnp
from mindspore.ops import operations as P
from mindsponge.pipeline.cell.initializer import lecun_init
from mindsponge.common.utils import dgram_from_positions, pseudo_beta_fn
from mindsponge.common.residue_constants import atom_order
from mindsponge.pipeline.cell import TriangleAttention, Transition, TriangleMultiplication
from .multimer_block import multimer_rigids_get_unit_vector


class MultimerTemplatePairStack(nn.Cell):
    '''multimer template pair stack'''

    def __init__(self, config):
        super(MultimerTemplatePairStack, self).__init__()
        self.config = config.template.template_pair_stack
        self.num_block = self.config.num_block
        batch_size = 0
        self.slice = config.slice.template_pair_stack
        start_node_cfg = self.config.triangle_attention_starting_node
        self.triangle_attention_starting_node = TriangleAttention(start_node_cfg.orientation,
                                                                  start_node_cfg.num_head,
                                                                  start_node_cfg.key_dim,
                                                                  start_node_cfg.gating,
                                                                  64,
                                                                  batch_size,
                                                                  self.slice.triangle_attention_starting_node)
        end_node_cfg = self.config.triangle_attention_ending_node
        self.triangle_attention_ending_node = TriangleAttention(end_node_cfg.orientation,
                                                                end_node_cfg.num_head,
                                                                end_node_cfg.key_dim,
                                                                end_node_cfg.gating,
                                                                64,
                                                                batch_size,
                                                                self.slice.triangle_attention_ending_node)
        # Hard Code
        self.pair_transition = Transition(self.config.pair_transition.num_intermediate_factor,
                                          64,
                                          batch_size,
                                          self.slice.pair_transition)

        mul_outgoing_cfg = self.config.triangle_multiplication_outgoing
        self.triangle_multiplication_outgoing = TriangleMultiplication(mul_outgoing_cfg.num_intermediate_channel,
                                                                       mul_outgoing_cfg.equation,
                                                                       layer_norm_dim=64,
                                                                       batch_size=batch_size)
        mul_incoming_cfg = self.config.triangle_multiplication_incoming
        self.triangle_multiplication_incoming = TriangleMultiplication(mul_incoming_cfg.num_intermediate_channel,
                                                                       mul_incoming_cfg.equation,
                                                                       layer_norm_dim=64,
                                                                       batch_size=batch_size)

    def construct(self, pair_act, pair_mask, index=None):
        if not self.num_block:
            return pair_act

        pair_act = pair_act + self.triangle_multiplication_outgoing(pair_act, pair_mask, index)
        pair_act = pair_act + self.triangle_multiplication_incoming(pair_act, pair_mask, index)
        pair_act = pair_act + self.triangle_attention_starting_node(pair_act, pair_mask, index)
        pair_act = pair_act + self.triangle_attention_ending_node(pair_act, pair_mask, index)
        pair_act = pair_act + self.pair_transition(pair_act, index)
        return pair_act


class MultimerSingleTemplateEmbedding(nn.Cell):
    '''multimer single template embedding'''

    def __init__(self, config, mixed_precision):
        super(MultimerSingleTemplateEmbedding, self).__init__()
        self.config = config.template
        if mixed_precision:
            self._type = mstype.float16
        else:
            self._type = mstype.float32
        self.num_bins = self.config.dgram_features.num_bins
        self.min_bin = self.config.dgram_features.min_bin
        self.max_bin = self.config.dgram_features.max_bin

        self.num_channels = (self.config.template_pair_stack.triangle_attention_ending_node.value_dim)
        self.template_dgram_temp_dense = nn.Dense(39, self.num_channels,
                                                  weight_init=lecun_init(39, initializer_name='relu'))
        self.template_mask_2d_temp_dense = nn.Dense(1, self.num_channels,
                                                    weight_init=lecun_init(1, initializer_name='relu'))
        self.aatype_temp_0 = nn.Dense(22, self.num_channels,
                                      weight_init=lecun_init(22, initializer_name='relu'))
        self.aatype_temp_1 = nn.Dense(22, self.num_channels,
                                      weight_init=lecun_init(22, initializer_name='relu'))
        self.unit_vector_0 = nn.Dense(1, self.num_channels,
                                      weight_init=lecun_init(1, initializer_name='relu'))
        self.unit_vector_1 = nn.Dense(1, self.num_channels,
                                      weight_init=lecun_init(1, initializer_name='relu'))
        self.unit_vector_2 = nn.Dense(1, self.num_channels,
                                      weight_init=lecun_init(1, initializer_name='relu'))
        self.backbone_mask_2d_dense = nn.Dense(1, self.num_channels,
                                               weight_init=lecun_init(1, initializer_name='relu'))
        self.embedding2d = nn.Dense(128, self.num_channels,
                                    weight_init=lecun_init(128, initializer_name='relu'))
        template_layers = nn.CellList()
        for _ in range(self.config.template_pair_stack.num_block):
            template_pair_stack_block = MultimerTemplatePairStack(config)
            template_layers.append(template_pair_stack_block)
        self.template_pair_stack = template_layers

        self.one_hot = nn.OneHot(depth=22, axis=-1)
        self.n, self.ca, self.c = [atom_order[a] for a in ('N', 'CA', 'C')]

        layer_norm_dim = 64
        self.query_embedding_norm = nn.LayerNorm([128,], epsilon=1e-5)
        self.output_layer_norm = nn.LayerNorm([layer_norm_dim,], epsilon=1e-5)
        self.num_block = self.config.template_pair_stack.num_block
        self.batch_block = 4

    def construct(self, pair_activations, template_aatype,
                  template_all_atom_positions, template_all_atom_mask,
                  padding_mask_2d, multichain_mask_2d):
        '''construct'''
        num_templates = template_aatype.shape[0]
        template_positions, template_pseudo_beta_mask = pseudo_beta_fn(template_aatype,
                                                                       template_all_atom_positions,
                                                                       template_all_atom_mask)
        template_mask_2d_temp = P.ExpandDims()(template_pseudo_beta_mask, -1) * \
                                P.ExpandDims()(template_pseudo_beta_mask, 1)

        template_mask_2d_temp *= multichain_mask_2d
        template_dgram_temp = dgram_from_positions(template_positions, self.num_bins, self.min_bin,
                                                   self.max_bin, self._type)
        template_dgram_temp *= template_mask_2d_temp[..., None]
        act_tmp = self.template_dgram_temp_dense(template_dgram_temp)
        act_tmp += self.template_mask_2d_temp_dense((P.ExpandDims()(template_mask_2d_temp, -1)))
        aatype_temp = self.one_hot(template_aatype)
        aatype_temp = P.Cast()(aatype_temp, self._type)
        act_tmp += self.aatype_temp_0((P.ExpandDims()(aatype_temp, 1)))
        act_tmp += self.aatype_temp_1((P.ExpandDims()(aatype_temp, 2)))
        backbone_mask = (template_all_atom_mask[:, :, self.n] *
                         template_all_atom_mask[:, :, self.ca] *
                         template_all_atom_mask[:, :, self.c])
        unit_vector = multimer_rigids_get_unit_vector(template_all_atom_positions[:, :, self.n],
                                                      template_all_atom_positions[:, :, self.ca],
                                                      template_all_atom_positions[:, :, self.c])

        backbone_mask_2d = (P.ExpandDims()(backbone_mask, -1)) * (P.ExpandDims()(backbone_mask, 1))
        backbone_mask_2d *= multichain_mask_2d
        unit_vector = (P.ExpandDims()(backbone_mask_2d * unit_vector[0], -1),
                       P.ExpandDims()(backbone_mask_2d * unit_vector[1], -1),
                       P.ExpandDims()(backbone_mask_2d * unit_vector[2], -1))
        pair_activations = self.query_embedding_norm(pair_activations)
        num_res, _, query_num_channels = pair_activations.shape
        pair_init = mnp.zeros((num_templates, num_res, num_res, query_num_channels), dtype=self._type)
        pair_activations = pair_init + pair_activations
        act_tmp += self.unit_vector_0(unit_vector[0])
        act_tmp += self.unit_vector_1(unit_vector[1])
        act_tmp += self.unit_vector_2(unit_vector[2])
        act_tmp += self.backbone_mask_2d_dense(P.ExpandDims()(backbone_mask_2d, -1))
        act_tmp += self.embedding2d(pair_activations)

        act_tmp = P.Split(0, self.batch_block)(act_tmp)
        scan_init = mnp.zeros((num_res, num_res, self.num_channels), dtype=self._type)
        act = ()
        for i in range(self.batch_block):
            act = act + (P.Squeeze()(act_tmp[i]),)

        for i in range(self.batch_block):
            act_batch = act[i]
            for j in range(self.num_block):
                act_batch = self.template_pair_stack[j](act_batch, padding_mask_2d)
            scan_init += self.output_layer_norm(act_batch)
        return scan_init


class MultimerTemplateEmbedding(nn.Cell):
    '''multimer template embedding'''

    def __init__(self, config, mixed_precision=True):
        super(MultimerTemplateEmbedding, self).__init__()
        self.config = config.template
        if mixed_precision:
            self._type = mstype.float16
        else:
            self._type = mstype.float32
        self.num_channels = (self.config.template_pair_stack.triangle_attention_ending_node.value_dim)
        self.template_embedder = MultimerSingleTemplateEmbedding(config, mixed_precision)
        self.relu = nn.ReLU()
        self.output_linear = nn.Dense(self.num_channels, config.pair_channel,
                                      weight_init=lecun_init(self.num_channels, initializer_name='relu'))

    def construct(self, pair_activations, template_aatype, template_all_atom_mask, template_all_atom_positions,
                  padding_mask_2d, multichain_mask_2d):
        '''construct'''
        num_templates = template_aatype.shape[0]
        embedding = self.template_embedder(pair_activations, template_aatype,
                                           template_all_atom_positions,
                                           template_all_atom_mask,
                                           padding_mask_2d,
                                           multichain_mask_2d)
        embedding = embedding / num_templates
        embedding = self.relu(embedding)
        return self.output_linear(embedding)
