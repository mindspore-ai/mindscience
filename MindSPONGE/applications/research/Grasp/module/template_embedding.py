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
'''TEMPLATE'''
import mindspore.common.dtype as mstype
import mindspore.nn as nn
import mindspore.numpy as mnp
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindspore import Tensor
from mindsponge1.cell.initializer import lecun_init
from mindsponge1.common.utils import dgram_from_positions, _memory_reduce, pseudo_beta_fn#, DgramFromPositionsCell
from mindsponge1.common.geometry import make_transform_from_reference, quat_affine, invert_point
from mindsponge1.common.residue_constants import atom_order
from mindsponge1.cell import Attention, TriangleAttention, Transition, TriangleMultiplication
from common.geometry import multimer_rigids_get_unit_vector
# from mindspore import lazy_inline
# from mindspore import Layout


class TemplatePairStack(nn.Cell):
    '''template pair stack'''

    def __init__(self, config):
        super(TemplatePairStack, self).__init__()
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

        pair_act = pair_act + self.triangle_attention_starting_node(pair_act, pair_mask, index)
        pair_act = pair_act + self.triangle_attention_ending_node(pair_act, pair_mask, index)
        pair_act = pair_act + self.triangle_multiplication_outgoing(pair_act, pair_mask, index)
        pair_act = pair_act + self.triangle_multiplication_incoming(pair_act, pair_mask, index)
        pair_act = pair_act + self.pair_transition(pair_act, index)
        return pair_act


class SingleTemplateEmbedding(nn.Cell):
    '''single template embedding'''

    def __init__(self, config, mixed_precision):
        super(SingleTemplateEmbedding, self).__init__()
        self.config = config.template
        if mixed_precision:
            self._type = mstype.float16
        else:
            self._type = mstype.float32
        self.num_bins = self.config.dgram_features.num_bins
        self.min_bin = self.config.dgram_features.min_bin
        self.max_bin = self.config.dgram_features.max_bin

        self.num_channels = (self.config.template_pair_stack.triangle_attention_ending_node.value_dim)
        self.embedding2d = nn.Dense(88, self.num_channels,
                                    weight_init=lecun_init(88, initializer_name='relu'))
        # if is_training:
        template_layers = nn.CellList()
        for _ in range(self.config.template_pair_stack.num_block):
            template_pair_stack_block = TemplatePairStack(config)
            template_layers.append(template_pair_stack_block)
        self.template_pair_stack = template_layers

        self.one_hot = nn.OneHot(depth=22, axis=-1)
        self.n, self.ca, self.c = [atom_order[a] for a in ('N', 'CA', 'C')]

        self.use_template_unit_vector = self.config.use_template_unit_vector
        layer_norm_dim = 64
        self.output_layer_norm = nn.LayerNorm([layer_norm_dim,], epsilon=1e-5)
        self.num_block = self.config.template_pair_stack.num_block
        self.batch_block = 4

    def construct(self, mask_2d, template_aatype, template_all_atom_masks, template_all_atom_positions,
                  template_pseudo_beta_mask, template_pseudo_beta):
        '''construct'''
        num_res = template_aatype[0, ...].shape[0]
        template_mask_2d_temp = P.ExpandDims()(template_pseudo_beta_mask, -1) * \
                                P.ExpandDims()(template_pseudo_beta_mask, 1)
        template_dgram_temp = dgram_from_positions(template_pseudo_beta, self.num_bins, self.min_bin,
                                                   self.max_bin, self._type)

        to_concat_temp = (template_dgram_temp, P.ExpandDims()(template_mask_2d_temp, -1))
        aatype_temp = self.one_hot(template_aatype)
        aatype_temp = P.Cast()(aatype_temp, self._type)
        to_concat_temp = to_concat_temp + (P.Tile()(P.ExpandDims()(aatype_temp, 1), (1, num_res, 1, 1)),
                                           P.Tile()(P.ExpandDims()(aatype_temp, 2), (1, 1, num_res, 1)))

        rot_temp, trans_temp = make_transform_from_reference(template_all_atom_positions[:, :, self.n],
                                                             template_all_atom_positions[:, :, self.ca],
                                                             template_all_atom_positions[:, :, self.c])

        _, rotation_tmp, translation_tmp = quat_affine(None, trans_temp, rot_temp)
        points_tmp = [P.ExpandDims()(translation_tmp[0], -2),
                      P.ExpandDims()(translation_tmp[1], -2),
                      P.ExpandDims()(translation_tmp[2], -2)]
        affine_vec_tmp = invert_point(points_tmp, rotation_tmp, translation_tmp, extra_dims=1)
        inv_distance_scalar_tmp = P.Rsqrt()(1e-6 + P.Square()(affine_vec_tmp[0]) + P.Square()(affine_vec_tmp[1]) + \
                                            P.Square()(affine_vec_tmp[2]))
        template_mask_tmp = (template_all_atom_masks[:, :, self.n] *
                             template_all_atom_masks[:, :, self.ca] *
                             template_all_atom_masks[:, :, self.c])
        template_mask_2d_tmp = P.ExpandDims()(template_mask_tmp, -1) * P.ExpandDims()(template_mask_tmp, 1)

        inv_distance_scalar_tmp = inv_distance_scalar_tmp * template_mask_2d_tmp
        unit_vector_tmp = (P.ExpandDims()(inv_distance_scalar_tmp * affine_vec_tmp[0], -1),
                           P.ExpandDims()(inv_distance_scalar_tmp * affine_vec_tmp[1], -1),
                           P.ExpandDims()(inv_distance_scalar_tmp * affine_vec_tmp[2], -1))

        if not self.use_template_unit_vector:
            unit_vector_tmp = (P.ZerosLike()(unit_vector_tmp[0]), P.ZerosLike()(unit_vector_tmp[1]),
                               P.ZerosLike()(unit_vector_tmp[2]))
        to_concat_temp = to_concat_temp + unit_vector_tmp + (P.ExpandDims()(template_mask_2d_tmp, -1),)
        act_tmp = P.Concat(-1)(to_concat_temp)

        act_tmp = act_tmp * P.ExpandDims()(template_mask_2d_tmp, -1)
        act_tmp = self.embedding2d(act_tmp)

        act_tmp = P.Split(0, self.batch_block)(act_tmp)
        act = ()
        for i in range(self.batch_block):
            act = act + (P.Squeeze()(act_tmp[i]),)

        output = []
        for i in range(self.batch_block):
            act_batch = act[i]
            for j in range(self.num_block):
                act_batch = self.template_pair_stack[j](act_batch, mask_2d)
            slice_act = P.Reshape()(act_batch, ((1,) + P.Shape()(act_batch)))
            output.append(slice_act)

        act_tmp_loop = P.Concat()(output)
        act_tmp = self.output_layer_norm(act_tmp_loop)
        return act_tmp


class TemplateEmbedding(nn.Cell):
    '''template embedding'''

    def __init__(self, config, mixed_precision=True):
        super(TemplateEmbedding, self).__init__()
        self.config = config.template
        if mixed_precision:
            self._type = mstype.float16
        else:
            self._type = mstype.float32
        self.num_channels = (self.config.template_pair_stack.triangle_attention_ending_node.value_dim)
        self.template_embedder = SingleTemplateEmbedding(config, mixed_precision)
        self.template_pointwise_attention = Attention(self.config.attention.num_head,
                                                      self.config.attention.key_dim,
                                                      self.config.attention.gating,
                                                      q_data_dim=128, m_data_dim=64,
                                                      output_dim=128, batch_size=None)
        self.slice_num = config.slice.template_embedding


    def compute(self, flat_query, flat_templates, input_mask):
        embedding = self.template_pointwise_attention(flat_query, flat_templates, input_mask, index=None,
                                                      nonbatched_bias=None)
        return embedding


    def construct(self, query_embedding, template_aatype, template_all_atom_masks, template_all_atom_positions,
                  template_mask, template_pseudo_beta_mask, template_pseudo_beta, mask_2d):
        '''construct'''
        num_templates = template_mask.shape[0]
        num_channels = self.num_channels
        num_res = query_embedding.shape[0]
        query_num_channels = query_embedding.shape[-1]
        mask_2d = F.depend(mask_2d, query_embedding)
        template_pair_representation = self.template_embedder(mask_2d, template_aatype,
                                                              template_all_atom_masks, template_all_atom_positions,
                                                              template_pseudo_beta_mask,
                                                              template_pseudo_beta)
        flat_query = P.Reshape()(query_embedding, (num_res * num_res, 1, query_num_channels))
        flat_templates = P.Reshape()(
            P.Transpose()(template_pair_representation, (1, 2, 0, 3)),
            (num_res * num_res, num_templates, num_channels))
        template_mask_bias = P.ExpandDims()(P.ExpandDims()(P.ExpandDims()(template_mask, 0), 1), 2) - 1.0
        input_mask = 1e4 * template_mask_bias
        batched_inputs = (flat_query, flat_templates)
        nonbatched_inputs = (input_mask,)
        embedding = _memory_reduce(self.compute, batched_inputs, nonbatched_inputs, self.slice_num)
        embedding = P.Reshape()(embedding, (num_res, num_res, query_num_channels))
        # No gradients if no templates.
        embedding = embedding * (P.ReduceSum()(template_mask) > 0.)
        return embedding


class MultimerTemplatePairStack(nn.Cell):
    '''multimer template pair stack'''

    def __init__(self, config, device_num):
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
                                                                  device_num,
                                                                  batch_size,
                                                                  self.slice.triangle_attention_starting_node)
        end_node_cfg = self.config.triangle_attention_ending_node
        self.triangle_attention_ending_node = TriangleAttention(end_node_cfg.orientation,
                                                                end_node_cfg.num_head,
                                                                end_node_cfg.key_dim,
                                                                end_node_cfg.gating,
                                                                64,
                                                                device_num,
                                                                batch_size,
                                                                self.slice.triangle_attention_ending_node)
        # Hard Code
        self.pair_transition = Transition(self.config.pair_transition.num_intermediate_factor,
                                          64,
                                          device_num,
                                          batch_size,
                                          self.slice.pair_transition)

        mul_outgoing_cfg = self.config.triangle_multiplication_outgoing
        self.triangle_multiplication_outgoing = TriangleMultiplication(mul_outgoing_cfg.num_intermediate_channel,
                                                                       mul_outgoing_cfg.equation,
                                                                       64,
                                                                       device_num,
                                                                       batch_size=batch_size)
        mul_incoming_cfg = self.config.triangle_multiplication_incoming
        self.triangle_multiplication_incoming = TriangleMultiplication(mul_incoming_cfg.num_intermediate_channel,
                                                                       mul_incoming_cfg.equation,
                                                                       64,
                                                                       device_num,
                                                                       batch_size=batch_size)
        self.add = P.Add().shard(((1, device_num, 1),(1, device_num, 1)))

    def construct(self, pair_act, pair_mask, index=None):
        if not self.num_block:
            return pair_act
        # print("debug pair_act 277", pair_act)
        pair_act = pair_act + self.triangle_multiplication_outgoing(pair_act, pair_mask, index)
        # pair_act = self.add(pair_act, self.triangle_multiplication_outgoing(pair_act, pair_mask, index))
        # print("debug pair_act 279", pair_act)
        pair_act = pair_act + self.triangle_multiplication_incoming(pair_act, pair_mask, index)
        # pair_act = self.add(pair_act, self.triangle_multiplication_incoming(pair_act, pair_mask, index))
        # print("debug pair_act 281", pair_act)
        pair_act = pair_act + self.triangle_attention_starting_node(pair_act, pair_mask, index)
        # pair_act = self.add(pair_act, self.triangle_attention_starting_node(pair_act, pair_mask, index))
        # print("debug pair_act 283", pair_act)
        pair_act = pair_act + self.triangle_attention_ending_node(pair_act, pair_mask, index)
        # pair_act = self.add(pair_act, self.triangle_attention_ending_node(pair_act, pair_mask, index))
        # print("debug pair_act 285", pair_act)
        pair_act = pair_act + self.pair_transition(pair_act, index)
        # pair_act = self.add(pair_act, self.pair_transition(pair_act, index))
        # print("debug pair_act 287", pair_act)
        return pair_act


class MultimerSingleTemplateEmbedding(nn.Cell):
    '''multimer single template embedding'''

    def __init__(self, config, mixed_precision, device_num):
        super(MultimerSingleTemplateEmbedding, self).__init__()
        self.is_training = config.is_training
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
            # print("debug MultimerSingleTemplateEmbedding round", _)
            template_pair_stack_block = MultimerTemplatePairStack(config, device_num)
            if self.is_training:
                template_pair_stack_block.recompute()
            template_layers.append(template_pair_stack_block)
        self.template_pair_stack = template_layers

        self.one_hot = nn.OneHot(depth=22, axis=-1)
        self.n, self.ca, self.c = [atom_order[a] for a in ('N', 'CA', 'C')]

        layer_norm_dim = 64
        self.query_embedding_norm = nn.LayerNorm([128,], epsilon=1e-5)
        self.output_layer_norm = nn.LayerNorm([layer_norm_dim,], epsilon=1e-5)
        self.num_block = self.config.template_pair_stack.num_block
        self.batch_block = 4

        self.squeeze = P.Squeeze()
        self.gather2 = P.Gather().shard(((1,1),()))
        self.gather3 = P.Gather().shard(((1,1,1),()))
        self.gather4 = P.Gather().shard(((1,1,1,1),()))
        self.expand = P.ExpandDims().shard(((device_num,),))
        # self.dgram_from_positions_cell = DgramFromPositionsCell(self.num_bins, self.min_bin, self.max_bin, self._type, device_num)


    def construct(self, pair_activations, template_aatype,
                  template_all_atom_positions, template_all_atom_mask,
                  padding_mask_2d, multichain_mask_2d):
        '''construct'''
        # print("debug MultimerSingleTemplateEmbedding", pair_activations, template_aatype, template_all_atom_positions, template_all_atom_mask, padding_mask_2d, multichain_mask_2d)
        pair_activations = self.query_embedding_norm(pair_activations)

        # num_res, _, query_num_channels = pair_activations.shape

        # scan_init = mnp.zeros((num_res, num_res, self.num_channels), dtype=self._type)
        scan_init = None
        slice_act = None
        # print("scan_init] ",scan_init)
        # slice_act = None
        for i in range(self.batch_block):
            single_template_aatype = self.squeeze(self.gather2(template_aatype, Tensor(i), 0))
            single_template_all_atom_masks = self.squeeze(self.gather3(template_all_atom_mask, Tensor(i), 0))
            single_template_all_positions = self.squeeze(self.gather4(template_all_atom_positions, Tensor(i), 0))

            template_pseudo_beta, template_pseudo_beta_mask = pseudo_beta_fn(single_template_aatype,
                                                                       single_template_all_positions,
                                                                       single_template_all_atom_masks)
            # single_template_pseudo_beta = self.squeeze(self.gather3(template_pseudo_beta, Tensor(i), 0))#P.Squeeze()(template_pseudo_beta[i,...])
            # single_template_pseudo_beta_mask = self.squeeze(self.gather2(template_pseudo_beta_mask, Tensor(i), 0))

            template_mask_2d_temp = self.expand(template_pseudo_beta_mask, -1) * self.expand(template_pseudo_beta_mask, 0)

            template_mask_2d_temp = template_mask_2d_temp * multichain_mask_2d

            template_dgram_temp = dgram_from_positions(template_pseudo_beta, self.num_bins, self.min_bin,
                                            self.max_bin, self._type)
            # template_dgram_temp = self.dgram_from_positions_cell(template_pseudo_beta)
            # template_dgram_temp = self.squeeze(self.gather4(template_dgram_temp_raw, Tensor(i), 0))#P.Squeeze()(template_dgram_temp_raw[i,...])
            template_dgram_temp *= template_mask_2d_temp[..., None]

            act_tmp = self.template_dgram_temp_dense(template_dgram_temp)

            act_tmp += self.template_mask_2d_temp_dense((P.ExpandDims()(template_mask_2d_temp, -1)))

            aatype_temp = self.one_hot(single_template_aatype)

            aatype_temp = P.Cast()(aatype_temp, self._type)

            act_tmp += self.aatype_temp_0((P.ExpandDims()(aatype_temp, 0)))

            act_tmp += self.aatype_temp_1((P.ExpandDims()(aatype_temp, 1)))

            backbone_mask = (single_template_all_atom_masks[:, self.n] *
                            single_template_all_atom_masks[:, self.ca] *
                            single_template_all_atom_masks[:, self.c])

            unit_vector = multimer_rigids_get_unit_vector(single_template_all_positions[:, self.n],
                                                        single_template_all_positions[:, self.ca],
                                                        single_template_all_positions[:, self.c])

            backbone_mask_2d = (P.ExpandDims()(backbone_mask, -1)) * (P.ExpandDims()(backbone_mask, 0))

            backbone_mask_2d *= multichain_mask_2d

            digonal_mask = 1 - mnp.eye(multichain_mask_2d.shape[0])
            # unit_vector = (P.Squeeze()(unit_vector_raw[0][i]), P.Squeeze()(unit_vector_raw[1][i]), P.Squeeze()(unit_vector_raw[2][i]))

            unit_vector = (P.ExpandDims()(backbone_mask_2d * digonal_mask * unit_vector[0], -1),
                        P.ExpandDims()(backbone_mask_2d * digonal_mask * unit_vector[1], -1),
                        P.ExpandDims()(backbone_mask_2d * digonal_mask * unit_vector[2], -1))

            # unit_vector = (P.ExpandDims()(backbone_mask_2d * digonal_mask * unit_vector[0], -1),
            #             P.ExpandDims()(backbone_mask_2d * digonal_mask * unit_vector[1], -1),
            #             P.ExpandDims()(backbone_mask_2d * digonal_mask * unit_vector[2], -1))

            act_tmp += self.unit_vector_0(unit_vector[0])
            # print("debug act_tmp 379", act_tmp)
            act_tmp += self.unit_vector_1(unit_vector[1])
            # print("debug act_tmp 381", act_tmp)
            act_tmp += self.unit_vector_2(unit_vector[2])
            # print("debug act_tmp 383", act_tmp)
            act_tmp += self.backbone_mask_2d_dense(P.ExpandDims()(backbone_mask_2d, -1))
            # print("debug act_tmp 385", act_tmp)
            act_tmp += self.embedding2d(pair_activations)
            if i > 0:
                act_tmp = F.depend(act_tmp, slice_act)
            for j in range(self.num_block):
                act_tmp = self.template_pair_stack[j](act_tmp, padding_mask_2d)
            slice_act = self.output_layer_norm(act_tmp)
            if scan_init is None:
                scan_init = slice_act
            else:
                scan_init += slice_act
            # scan_init += self.output_layer_norm(act_tmp)

        return scan_init
        # num_templates = template_aatype.shape[0]

        # # template_pseudo_beta_mask (1, 248)
        # template_pseudo_beta, template_pseudo_beta_mask = pseudo_beta_fn(template_aatype,
        #                                                                template_all_atom_positions,
        #                                                                template_all_atom_mask)
        # # (1, 248, 1) (1, 1, 248)
        # template_mask_2d_temp = P.ExpandDims()(template_pseudo_beta_mask, -1) * \
        #                         P.ExpandDims()(template_pseudo_beta_mask, 1)

        # # (1, 248, 248) * (248, 248) multichain_mask_2d (62, 248) -> (248, 248)
        # template_mask_2d_temp *= multichain_mask_2d

        # template_dgram_temp = dgram_from_positions(template_pseudo_beta, self.num_bins, self.min_bin,
        #                                            self.max_bin, self._type)

        # # (1, 248, 248, 39) * (1, 248, 248, 1)
        # template_dgram_temp *= template_mask_2d_temp[..., None]

        # # weight: (64, 39)
        # # input: (61504, 39) -- (1, 248, 248, 39)
        # act_tmp = self.template_dgram_temp_dense(template_dgram_temp)


        # act_tmp += self.template_mask_2d_temp_dense((P.ExpandDims()(template_mask_2d_temp, -1)))
        # # print("debug act_tmp 356", act_tmp)
        # aatype_temp = self.one_hot(template_aatype)
        # aatype_temp = P.Cast()(aatype_temp, self._type)
        # act_tmp += self.aatype_temp_0((P.ExpandDims()(aatype_temp, 1)))
        # # print("debug act_tmp 359", act_tmp)
        # act_tmp += self.aatype_temp_1((P.ExpandDims()(aatype_temp, 2)))
        # # print("debug act_tmp 362", act_tmp)
        # backbone_mask = (template_all_atom_mask[:, :, self.n] *
        #                  template_all_atom_mask[:, :, self.ca] *
        #                  template_all_atom_mask[:, :, self.c])
        # # print("debug backbone_mask", backbone_mask)
        # # print("debug template_all_atom_positions", template_all_atom_positions)
        # unit_vector = multimer_rigids_get_unit_vector(template_all_atom_positions[:, :, self.n],
        #                                               template_all_atom_positions[:, :, self.ca],
        #                                               template_all_atom_positions[:, :, self.c])
        # # print("debug unit_vector 370", unit_vector)
        # backbone_mask_2d = (P.ExpandDims()(backbone_mask, -1)) * (P.ExpandDims()(backbone_mask, 1))
        # # print("debug backbone_mask_2d 372", backbone_mask_2d)
        # backbone_mask_2d *= multichain_mask_2d
        # # digonal_mask = 1 - self.eye
        # digonal_mask = 1 - mnp.eye(multichain_mask_2d.shape[0])

        # # digonal_mask = 1 - self.eye(multichain_mask_2d.shape[0], multichain_mask_2d.shape[0], mstype.float32)
        # # print("debug digonal_mask 375", digonal_mask)
        # unit_vector = (P.ExpandDims()(backbone_mask_2d * digonal_mask * unit_vector[0], -1),
        #                P.ExpandDims()(backbone_mask_2d * digonal_mask * unit_vector[1], -1),
        #                P.ExpandDims()(backbone_mask_2d * digonal_mask * unit_vector[2], -1))
        # # print("debug unit_vector 377", unit_vector)
        # pair_activations = self.query_embedding_norm(pair_activations)
        # num_res, _, query_num_channels = pair_activations.shape
        # act_tmp += self.unit_vector_0(unit_vector[0])
        # # print("debug act_tmp 379", act_tmp)
        # act_tmp += self.unit_vector_1(unit_vector[1])
        # # print("debug act_tmp 381", act_tmp)
        # act_tmp += self.unit_vector_2(unit_vector[2])
        # # print("debug act_tmp 383", act_tmp)
        # act_tmp += self.backbone_mask_2d_dense(P.ExpandDims()(backbone_mask_2d, -1))
        # # print("debug act_tmp 385", act_tmp)
        # act_tmp += self.embedding2d(pair_activations)
        # # print("debug act_tmp 387", act_tmp)
        # # print("act_tmp's shape:", act_tmp.shape) Tensor(shape=[4], dtype=Int64, value=[  4 256 256  64])
        
        # act_tmp = P.Split(0, self.batch_block)(act_tmp)
        # # print("debug act_tmp 390", act_tmp)
        # scan_init = mnp.zeros((num_res, num_res, self.num_channels), dtype=self._type)
        # act = ()
        # for i in range(self.batch_block):
        #     # print("debug act 394", "act", act, "act_tmp", act_tmp)
        #     act = act + (P.Squeeze()(act_tmp[i]),)

        # for i in range(self.batch_block):
        #     act_batch = act[i]
        #     for j in range(self.num_block):
        #         # print("debug MultimerSingleTemplateEmbedding act_batch round", j, "act_batch", act_batch, "padding_mask_2d", padding_mask_2d)
        #         act_batch = self.template_pair_stack[j](act_batch, padding_mask_2d)
        #     # print("debug MultimerSingleTemplateEmbedding round", i, "scan_init", scan_init, "act_batch", act_batch)
        #     scan_init += self.output_layer_norm(act_batch)
        # return scan_init


class MultimerTemplateEmbedding(nn.Cell):
    '''multimer template embedding'''
    # @lazy_inline
    def __init__(self, config, device_num, mixed_precision=True):
        super(MultimerTemplateEmbedding, self).__init__()
        self.config = config.template
        if mixed_precision:
            self._type = mstype.float16
        else:
            self._type = mstype.float32
        self.num_channels = (self.config.template_pair_stack.triangle_attention_ending_node.value_dim)
        self.template_embedder = MultimerSingleTemplateEmbedding(config, mixed_precision, device_num)
        self.relu = nn.ReLU()
        self.output_linear = nn.Dense(self.num_channels, config.pair_channel,
                                      weight_init=lecun_init(self.num_channels, initializer_name='relu'))

    def construct(self, pair_activations, template_aatype, template_all_atom_mask, template_all_atom_positions,
                  padding_mask_2d, multichain_mask_2d):
        '''construct'''
        num_templates = template_aatype.shape[0]
        # print("num_templates: ", num_templates.shape)
        embedding = self.template_embedder(pair_activations, template_aatype,
                                           template_all_atom_positions,
                                           template_all_atom_mask,
                                           padding_mask_2d,
                                           multichain_mask_2d)
        embedding = embedding / num_templates
        embedding = self.relu(embedding)
        output = self.output_linear(embedding)
        return output