# Copyright 2022 Huawei Technologies Co., Ltd
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
"""evogen block"""
import numpy as np
from mindspore import nn
from mindspore import Tensor
from mindspore import Parameter
import mindspore.ops as ops
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.common.initializer import initializer
import mindspore.common.dtype as mstype
import mindspore.nn.probability.distribution as msd
import mindspore.numpy as msnp

from mindsponge.cell import Attention, MSARowAttentionWithPairBias, Transition, \
    OuterProductMean, TriangleMultiplication, TriangleAttention
from mindsponge.cell.initializer import lecun_init
from mindsponge.cell.mask import MaskedLayerNorm


def absolute_position_embedding(length, depth, min_timescale=1, max_timescale=1e4):
    '''absolute_position_embedding'''
    depth = depth // 2
    positions = np.arange(length, dtype=np.float32)
    log_timescale_increment = (np.log(max_timescale / min_timescale) / (depth - 1))
    inv_timescales = min_timescale * np.exp(np.arange(depth, dtype=np.float32) * -log_timescale_increment)
    scaled_time = np.expand_dims(positions, 1) * np.expand_dims(inv_timescales, 0)
    x = np.concatenate([np.sin(scaled_time), np.cos(scaled_time)], axis=1)
    return x


class EvoGenFeatProcess(nn.Cell):
    '''EvoGenFeatProcess'''

    def __init__(
            self,
            config,
    ):
        super().__init__()
        self.num_msa = config.global_config.num_msa
        self.max_num_res = config.global_config.max_num_res
        self.num_aa_types = config.global_config.num_aa_types
        self.num_msa_types = self.num_aa_types + 1
        self.seq_weight_power = Tensor(config.train.seq_weight_power, mstype.float32)
        self.rel_pos_generator = RelativePositionEmbedding(config.model.embeddings_and_evoformer)
        self.label_smooting = config.train.label_smoothing
        self.pi = np.pi
        self.msa_onehot = nn.OneHot(depth=self.num_msa_types)
        self.q_onehot = nn.OneHot(depth=self.num_aa_types)

    def construct(self, query_input, msa_input, additional_input, random_input, random_mask):
        '''Transform input data into a set of labels and feats.'''
        msa_ = msa_input[:, :, 0]
        x_new_number = 7. * P.OnesLike()(msa_)
        x_where = P.Equal()(msa_, 20)
        msa_ = P.Select()(x_where, x_new_number, msa_)
        msa_input = P.Concat(-1)((P.ExpandDims()(msa_, -1), msa_input[:, :, 1:]))
        aa_labels = msa_input[:, :, 0]

        del_num = msa_input[:, :, 1]
        del_num_feat = 2. / self.pi * msnp.arctan(del_num / 3.)
        has_del_label = ops.clip_by_value(del_num, 0, 1)

        msa_mask = P.ExpandDims()(additional_input[:, 2], 0)
        pair_mask = P.MatMul(transpose_a=True)(msa_mask, msa_mask)
        context_mask = random_mask[:, 0]
        context_mask[0] = 1.
        context_mask = P.ExpandDims()(context_mask, 1)
        target_mask = random_mask[:, 1]
        target_mask[0] = 1.
        target_mask = P.ExpandDims()(target_mask, 1)
        norm_const_predict = P.ReduceSum()(msa_mask)

        if self.seq_weight_power > 1e-5:
            norm_const_predict = norm_const_predict / 100.
            norm_const_predict = P.Pow()(norm_const_predict, self.seq_weight_power)
        else:
            norm_const_predict = Tensor(1.0, mstype.float32)

        msa_raw_feat = msa_input[:, :, 0]
        msa_raw_feat = P.Concat(0)((P.ExpandDims()(query_input[:, 0], 0), msa_raw_feat[1:, :]))
        msa_raw_feat = self.msa_onehot(msa_raw_feat.astype(mstype.int32))
        msa_raw_feat = P.Concat(-1)((msa_raw_feat.astype(has_del_label.dtype), P.ExpandDims()(has_del_label, -1),
                                     P.ExpandDims()(del_num_feat, -1)))

        q_raw_feat = self.q_onehot(query_input[:, 0].astype(mstype.int32)).astype(query_input.dtype)

        res_idx = additional_input[:, 1]
        pair_raw_feat = self.rel_pos_generator(res_idx, res_idx)

        msa_labels_onehot = self.msa_onehot(msa_input[:, :, 0].astype(mstype.int32))

        msa_mask_full = msa_input[:, :, 2]

        msa_mask_full = P.ExpandDims()(msa_mask_full, -1)

        msa_profile = P.ReduceSum()(msa_labels_onehot * msa_mask_full, 0)
        num_seq = P.ReduceSum()(msa_mask_full[:, 0, 0])

        msa_profile = msa_profile / (num_seq + 1e-5)

        q_labels = msa_labels_onehot[:1]
        aa_labels_onehot = self.msa_onehot(aa_labels.astype(mstype.int32))
        aa_labels = P.Concat(0)((q_labels, aa_labels_onehot[1:]))
        aa_labels = (1. - self.label_smooting) * aa_labels + self.label_smooting * (P.ExpandDims()(msa_profile, 0))

        label_tuple = (aa_labels, del_num, has_del_label, del_num_feat, msa_profile, norm_const_predict)
        feat_tuple = (q_raw_feat, msa_raw_feat, pair_raw_feat, msa_mask, pair_mask, context_mask,
                      target_mask, res_idx, random_input)

        return label_tuple, feat_tuple


class LatentNormal(nn.Cell):
    '''LatentNormal'''

    def __init__(self):
        super().__init__()
        self.exp = F.exp
        self.log = P.Log()
        self.pi2 = Tensor(2 * np.pi, mstype.float32)
        self.standard_normal = msd.Normal(mean=Tensor(0, dtype=mstype.float32),
                                          sd=Tensor(1, dtype=mstype.float32), dtype=mstype.float32)
        self.tanh = ops.Tanh()

    def sample(self, mu, log_sigma, temp=1.):
        '''sample'''
        mu, sigma = self._process_data(mu, log_sigma, temp)
        eps = self.standard_normal.sample(mu.shape)
        return eps * sigma + mu

    def sample_given_eps(self, eps, mu, log_sigma, temp=1.):
        '''sample_given_eps'''
        mu, sigma = self._process_data(mu, log_sigma, temp)
        return eps * sigma + mu

    def construct(self, mu, log_sigma, normal_dist_mu, normal_dist_log_sigma):
        '''kl'''
        mu, sigma = self._process_data(mu, log_sigma)
        normal_dist_mu, normal_dist_sigma = self._process_data(normal_dist_mu, normal_dist_log_sigma)
        term1 = (mu - normal_dist_mu) / normal_dist_sigma
        term2 = sigma / normal_dist_sigma
        return 0.5 * (term1 * term1 + term2 * term2) - 0.5 - F.log(term2)

    def _process_data(self, mu, log_sigma, temp=1.):
        '''process_data'''
        mu = 5. * self.tanh(mu / 5.)
        log_sigma = 5. * self.tanh(log_sigma / 5.)
        sigma = self.exp(log_sigma)
        sigma *= temp
        return mu, sigma


class RelativePositionEmbedding(nn.Cell):
    '''RelativePositionEmbedding'''

    def __init__(self,
                 config,
                 ):
        super(RelativePositionEmbedding, self).__init__()

        self.exact_distance = config.exact_distance
        self.num_buckets = config.num_buckets
        self.max_distance = config.max_distance
        self.onehot = nn.OneHot(depth=2 * self.num_buckets + 1)

    @staticmethod
    def _relative_position_bucket(x, alpha=16.0, beta=32.0, gamma=64.0):
        '''_relative_position_bucket'''
        alpha = Tensor(alpha, mstype.float32)
        beta = Tensor(beta, mstype.float32)
        gamma = Tensor(gamma, mstype.float32)

        scale = (beta - alpha) / F.log(gamma / alpha)
        x_abs = P.Abs()(x)
        gx = F.log((x_abs + 1e-3) / alpha) * scale + alpha
        gx = P.Minimum()(beta, gx)
        gx = P.Sign()(x) * gx

        cond = P.Greater()(x_abs, alpha)
        ret = P.Select()(cond, gx, x)
        ret = ops.clip_by_value(ret, -beta, beta)

        ret += beta
        return ret

    def construct(self, q_idx, k_idx):
        """ Compute binned relative position encoding """

        context_position = P.ExpandDims()(q_idx, 1)
        memory_position = P.ExpandDims()(k_idx, 0)
        relative_position = memory_position - context_position
        rp_bucket = self._relative_position_bucket(relative_position)
        rp_onehot = self.onehot(rp_bucket.astype(mstype.int32))
        return rp_onehot


class EvogenAttention(Attention):
    '''EvogenAttention'''

    def __init__(self, config, q_data_dim, m_data_dim, output_dim):
        super(EvogenAttention, self).__init__(config.num_head, q_data_dim,
                                              config.gating, q_data_dim, m_data_dim, output_dim, batch_size=None)
        self.ape_table = config.ape_table
        if self.ape_table is not None:
            self.ape_table = Tensor(self.ape_table, mstype.float32)
        self.onehot = nn.OneHot(depth=1024)

    def rope(self, hidden_states, res_idx):
        '''rope'''
        c_m = hidden_states.shape[-1]
        n_res = res_idx.shape[0]

        idx_one_hot = self.onehot(res_idx.astype(mstype.int32))
        ape_sin, ape_cos = ops.Split(axis=-1, output_num=2)(self.ape_table)
        ape_table = P.Concat(-1)([ape_cos, ape_sin])

        rope = P.MatMul()(idx_one_hot, ape_table)
        rope_double = P.Reshape()(P.Tile()(P.ExpandDims()(rope, -1), (1, 1, 2)), (n_res, -1))
        rope_cos, rope_sin = P.Split(axis=-1, output_num=2)(rope_double)

        vec_ = P.Reshape()(hidden_states, (-1, c_m // 2, 2))
        vec_even, vec_odd = P.Split(axis=-1, output_num=2)(vec_)
        vec2 = P.Concat(axis=-1)([-vec_odd, vec_even])
        vec2 = P.Reshape()(vec2, hidden_states.shape)

        vec1 = P.Reshape()(hidden_states, (-1, n_res, c_m))
        vec2 = P.Reshape()(vec2, (-1, n_res, c_m))
        vec_rope = vec1 * P.ExpandDims()(rope_cos, 0) + \
                   vec2 * P.ExpandDims()(rope_sin, 0)

        return P.Reshape()(vec_rope, hidden_states.shape)

    def construct(self, q_data, m_data, bias, pair_bias=None, res_idx=None):
        '''construct'''
        linear_gating_weight = 0
        if self.gating:
            linear_gating_weight = self.linear_gating_weights

        b_dim, q_dim, a_dim = q_data.shape
        _, k_dim, c_dim = m_data.shape
        q_data = P.Reshape()(q_data, (-1, a_dim))
        m_data = P.Reshape()(m_data, (-1, c_dim))

        q = self.matmul(q_data, self.linear_q_weights) * self.dim_per_head ** (-0.5)
        k = self.matmul(m_data, self.linear_k_weights)
        v = self.matmul(m_data, self.linear_v_weights)

        if (res_idx is not None) and (self.ape_table is not None):
            q = self.rope(q, res_idx)
            k = self.rope(k, res_idx)

        q = P.Reshape()(q, (b_dim, q_dim, self.num_head, -1))
        k = P.Reshape()(k, (b_dim, k_dim, self.num_head, -1))
        v = P.Reshape()(v, (b_dim, k_dim, self.num_head, -1))

        tmp_q = P.Reshape()(P.Transpose()(q, (0, 2, 1, 3)), (b_dim * self.num_head, q_dim, -1))
        tmp_k = P.Reshape()(P.Transpose()(k, (0, 2, 1, 3)), (b_dim * self.num_head, k_dim, -1))
        logits = P.Add()(P.Reshape()(self.batch_matmul_trans_b(tmp_q, tmp_k), (b_dim, self.num_head, q_dim, k_dim)),
                         bias)

        if pair_bias is not None:
            bias_ = P.ExpandDims()(pair_bias, 0)
            logits = P.Add()(logits, bias_)

        probs = self.softmax(logits)
        tmp_v = P.Reshape()(P.Transpose()(v, (0, 2, 3, 1)), (b_dim * self.num_head, -1, k_dim))
        tmp_probs = P.Reshape()(probs, (b_dim * self.num_head, q_dim, k_dim))

        weighted_avg = P.Transpose()(
            P.Reshape()(self.batch_matmul_trans_b(tmp_probs, tmp_v), (b_dim, self.num_head, q_dim, -1)),
            (0, 2, 1, 3))

        if self.gating:
            gating_bias = P.ExpandDims()(P.ExpandDims()(self.gating_biases, 0), 0)
            gate_values = P.Add()(
                P.Reshape()(self.matmul(q_data, linear_gating_weight), (b_dim, q_dim, self.num_head, -1)),
                gating_bias)
            gate_values = gate_values
            gate_values = self.sigmoid(gate_values)
            gate_values = gate_values
            weighted_avg = weighted_avg * gate_values

        weighted_avg = P.Reshape()(weighted_avg, (b_dim * q_dim, -1))
        output = P.Add()(P.Reshape()(self.matmul(weighted_avg, self.linear_output_weights), (b_dim, q_dim, -1)),
                         P.ExpandDims()(self.o_biases, 0))
        return output


class EvogenMSARowAttentionWithPairBias(MSARowAttentionWithPairBias):
    '''EvogenMSARowAttentionWithPairBias'''

    def __init__(self, config, msa_act_dim, pair_act_dim):
        super(EvogenMSARowAttentionWithPairBias, self).__init__(config.num_head, msa_act_dim, config.gating,
                                                                msa_act_dim, pair_act_dim)
        self.config = config
        self.attn_mod = EvogenAttention(self.config, msa_act_dim, msa_act_dim, msa_act_dim)

    def _compute(self, msa_act, bias, pair_bias=None, res_idx=None):
        '''compute'''
        msa_act = self.attn_mod(msa_act, msa_act, bias, pair_bias=pair_bias, res_idx=res_idx)
        return msa_act


class MSAConditioner(nn.Cell):
    '''MSAConditioner'''

    def __init__(self, config, layer_norm_dim):
        super(MSAConditioner, self).__init__()
        self.config = config
        self.layer_norm_dim = layer_norm_dim
        self.num_intermediate = int(layer_norm_dim * self.config.num_intermediate_factor)
        self.act_fn = nn.ReLU()
        self.matmul = P.MatMul(transpose_b=True)
        self.sigmoid = nn.Sigmoid()
        self.masked_layer_norm = MaskedLayerNorm()
        self._init_parameter()

    def construct(self, act, mask):
        '''construct'''
        act_ = self.masked_layer_norm(act, self.input_layer_norm_gammas, self.input_layer_norm_betas, mask=mask)
        q_act = P.ExpandDims()(act_[0], 0)
        mix_act = P.Concat(-1)((P.Tile()(q_act, (act_.shape[0], 1, 1)), act_))
        act_shape = P.Shape()(mix_act)
        if len(act_shape) != 2:
            mix_act = P.Reshape()(mix_act, (-1, act_shape[-1]))
        mix_act = self.act_fn(P.BiasAdd()(self.matmul(mix_act, self.transition_weights), self.transition_biases))
        gate_values = P.BiasAdd()(self.matmul(mix_act, self.linear_gating_weights), self.gating_biases)
        gate_values = self.sigmoid(gate_values)
        gate_values = P.Reshape()(gate_values, act.shape)
        return act, gate_values

    def _init_parameter(self):
        '''init_parameter'''
        self.input_layer_norm_gammas = Parameter(Tensor(np.ones((self.layer_norm_dim)), mstype.float32))
        self.input_layer_norm_betas = Parameter(Tensor(np.zeros((self.layer_norm_dim)), mstype.float32))
        self.transition_weights = Parameter(initializer(lecun_init(2 * self.layer_norm_dim, initializer_name='relu'),
                                                        [self.num_intermediate, 2 * self.layer_norm_dim]))
        self.transition_biases = Parameter(Tensor(np.zeros((self.num_intermediate)), mstype.float32))
        self.linear_gating_weights = Parameter(
            Tensor(np.zeros([self.layer_norm_dim, self.num_intermediate]), mstype.float32))
        self.gating_biases = Parameter(Tensor(np.ones((self.layer_norm_dim)), mstype.float32))


class EvoformerSeqBlock(nn.Cell):
    '''EvoformerSeqBlock'''

    def __init__(self, config, msa_act_dim, pair_act_dim, encoding=True):
        super(EvoformerSeqBlock, self).__init__()
        self.config = config
        self.msa_row_attention_with_pair_bias = EvogenMSARowAttentionWithPairBias(
            self.config.msa_row_attention_with_pair_bias, msa_act_dim, pair_act_dim)
        self.msa_transition = Transition(self.config.msa_transition.num_intermediate_factor, msa_act_dim)
        self.encoding = encoding
        if self.encoding:
            self.msa_conditioner = MSAConditioner(self.config.msa_condition, msa_act_dim)

    def construct(self, msa_act, pair_act, msa_mask, pair_mask, res_idx=None):
        '''construct'''
        msa_act = P.Add()(msa_act,
                          self.msa_row_attention_with_pair_bias(msa_act, msa_mask, pair_act, 0,
                                                                msa_mask, pair_mask, res_idx=res_idx))
        msa_act = P.Add()(msa_act, self.msa_transition(msa_act, 0, msa_mask))

        if self.encoding:
            act, gate_values = self.msa_conditioner(msa_act, msa_mask)
        else:
            act, gate_values = msa_act, 1.
        msa_act = P.Add()(gate_values * act, (1. - gate_values) * P.ExpandDims()(act[0], 0))
        return msa_act


class EvoformerPairBlock(nn.Cell):
    '''EvoformerPairBlock'''

    def __init__(self, config, msa_act_dim, pair_act_dim):
        super(EvoformerPairBlock, self).__init__()
        self.config = config
        self.outer_product = OuterProductMean(self.config.outer_product.num_outer_channel, msa_act_dim, pair_act_dim)
        self.triangle_multiplication_outgoing = TriangleMultiplication(
            self.config.triangle_multiplication_outgoing.num_intermediate_channel,
            self.config.triangle_multiplication_outgoing.equation,
            pair_act_dim)
        self.triangle_multiplication_incoming = TriangleMultiplication(
            self.config.triangle_multiplication_incoming.num_intermediate_channel,
            self.config.triangle_multiplication_incoming.equation,
            pair_act_dim)
        self.triangle_attention_starting_node = TriangleAttention(
            self.config.triangle_attention_starting_node.orientation,
            self.config.triangle_attention_starting_node.num_head,
            pair_act_dim,
            self.config.triangle_attention_starting_node.gating,
            pair_act_dim)
        self.triangle_attention_ending_node = TriangleAttention(self.config.triangle_attention_ending_node.orientation,
                                                                self.config.triangle_attention_ending_node.num_head,
                                                                pair_act_dim,
                                                                self.config.triangle_attention_ending_node.gating,
                                                                pair_act_dim)
        self.pair_transition = Transition(self.config.pair_transition.num_intermediate_factor, pair_act_dim)

    def construct(self, msa_act, pair_act, msa_mask, pair_mask, context_mask, mask_norm=None):
        '''construct'''
        msa_mask_ = msa_mask * context_mask
        pair_act = P.Add()(pair_act, self.outer_product(msa_act, msa_mask_, mask_norm))
        pair_act = P.Add()(pair_act, self.triangle_multiplication_outgoing(pair_act, pair_mask))
        pair_act = P.Add()(pair_act, self.triangle_multiplication_incoming(pair_act, pair_mask))

        pair_act = P.Add()(pair_act, self.triangle_attention_starting_node(pair_act, pair_mask, mask=pair_mask))
        pair_mask_ = P.Transpose()(pair_mask, (1, 0))
        pair_act = P.Add()(pair_act, self.triangle_attention_ending_node(pair_act, pair_mask, mask=pair_mask_))
        pair_act = P.Add()(pair_act, self.pair_transition(pair_act, 0, pair_mask))
        return pair_act


class EvoformerIteration(nn.Cell):
    '''EvoformerIteration'''

    def __init__(self, config, msa_act_dim, pair_act_dim, encoding=True):
        super(EvoformerIteration, self).__init__()
        self.config = config.model.embeddings_and_evoformer.evoformer
        self.evoformer_seq_block = EvoformerSeqBlock(self.config, msa_act_dim, pair_act_dim, encoding=encoding)
        if config.global_config.recompute:
            self.evoformer_seq_block.recompute()
        self.encoding = encoding
        if self.encoding:
            self.evoformer_pair_block = EvoformerPairBlock(self.config, msa_act_dim, pair_act_dim)
            if config.global_config.recompute:
                self.evoformer_pair_block.recompute()

    def construct(self, msa_act, pair_act, msa_mask, pair_mask, context_mask, mask_norm=None, res_idx=None):
        '''construct'''
        msa_act_ = msa_act
        msa_act = self.evoformer_seq_block(msa_act_, pair_act, msa_mask, pair_mask, res_idx=res_idx)
        if self.encoding:
            pair_act = self.evoformer_pair_block(msa_act_, pair_act, msa_mask, pair_mask, context_mask,
                                                 mask_norm=mask_norm)
        return msa_act, pair_act


class LatentTransition(nn.Cell):
    '''LatentTransition'''

    def __init__(self, config, input_dim, output_dim):
        super(LatentTransition, self).__init__()
        self.config = config
        self.layer_norm_dim = input_dim
        self.num_intermediate = int(input_dim * self.config.num_intermediate_factor)
        self.output_dim = output_dim
        self.act_fn = nn.ReLU()
        self.matmul = P.MatMul(transpose_b=True)
        self.masked_layer_norm = MaskedLayerNorm()
        self._init_parameter()

    def construct(self, act, mask):
        '''construct'''
        act = self.masked_layer_norm(act, self.input_layer_norm_gammas, self.input_layer_norm_betas, mask=mask)
        act_shape = P.Shape()(act)
        if len(act_shape) != 2:
            act = P.Reshape()(act, (-1, act_shape[-1]))
        act1 = P.BiasAdd()(self.matmul(act, self.linear0_weights), self.linear0_biases)

        act = self.act_fn(P.BiasAdd()(self.matmul(act, self.linear1_weights), self.linear1_biases))
        act = self.act_fn(P.BiasAdd()(self.matmul(act, self.linear2_weights), self.linear2_biases))
        act = P.BiasAdd()(self.matmul(act, self.linear3_weights), self.linear3_biases)

        act = P.Add()(act, act1)
        act = P.Reshape()(act, act_shape[:-1] + (-1,))
        return act

    def _init_parameter(self):
        '''init parameter'''
        self.input_layer_norm_gammas = Parameter(Tensor(np.ones((self.layer_norm_dim)), mstype.float32))
        self.input_layer_norm_betas = Parameter(Tensor(np.zeros((self.layer_norm_dim)), mstype.float32))

        self.linear0_weights = Parameter(
            initializer(lecun_init(self.layer_norm_dim), [self.output_dim, self.layer_norm_dim]))
        self.linear0_biases = Parameter(Tensor(np.zeros((self.output_dim)), mstype.float32))

        self.linear1_weights = Parameter(initializer(lecun_init(self.layer_norm_dim, initializer_name='relu'),
                                                     [self.num_intermediate, self.layer_norm_dim]))
        self.linear1_biases = Parameter(Tensor(np.zeros((self.num_intermediate)), mstype.float32))
        self.linear2_weights = Parameter(initializer(lecun_init(self.layer_norm_dim, initializer_name='relu'),
                                                     [self.num_intermediate, self.layer_norm_dim]))
        self.linear2_biases = Parameter(Tensor(np.zeros((self.num_intermediate)), mstype.float32))

        self.linear3_weights = Parameter(Tensor(np.zeros((self.output_dim, self.layer_norm_dim)), mstype.float32))
        self.linear3_biases = Parameter(Tensor(np.zeros((self.output_dim)), mstype.float32))


class ColumnAttentionWithPairBias(nn.Cell):
    ''''ColumnAttentionWithPairBias'''

    def __init__(self, config, input_dim, output_dim):
        super(ColumnAttentionWithPairBias, self).__init__()
        self.attn_mod = EvogenAttention(config, input_dim, input_dim, output_dim)
        self.input_norm_gammas = Parameter(Tensor(np.ones([input_dim]), mstype.float32))
        self.input_norm_betas = Parameter(Tensor(np.zeros([input_dim]), mstype.float32))
        self.masked_layer_norm = MaskedLayerNorm()

    def construct(self, q, k, q_mask, k_mask):
        '''construct'''
        q_act = P.Transpose()(q, (1, 0, 2))
        k_act = P.Transpose()(k, (1, 0, 2))
        q_act = self.masked_layer_norm(q_act, self.input_norm_gammas, self.input_norm_betas, mask=q_mask)
        k_act = self.masked_layer_norm(k_act, self.input_norm_gammas, self.input_norm_betas, mask=k_mask)

        bias = 1e9 * (k_mask - 1.0)
        bias = P.ExpandDims()(P.ExpandDims()(bias, 1), 2)
        act = self.attn_mod(q_act, k_act, bias)
        act = P.Transpose()(act, (1, 0, 2))
        return act


class LatentTransformerBlock(nn.Cell):
    '''LatentTransformerBlock'''

    def __init__(self, config, input_dim, output_dim):
        super(LatentTransformerBlock, self).__init__()
        self.column_attention_with_pair_bias = ColumnAttentionWithPairBias(
            config.column_attention_with_pair_bias, input_dim, output_dim)
        self.transition = Transition(config.msa_transition.num_intermediate_factor, output_dim)

    def construct(self, q_act, k_act, q_mask, k_mask):
        '''construct'''
        act = P.Add()(q_act, self.column_attention_with_pair_bias(q_act, k_act, q_mask, k_mask))
        q_mask_t = P.Transpose()(q_mask, (1, 0))
        act = P.Add()(act, self.transition(act, 0, q_mask_t))
        return act


class LatentStatistics(nn.Cell):
    '''LatentStatistics'''

    def __init__(self, config, latent_dim):
        super(LatentStatistics, self).__init__()
        self.num_intermediate = int(latent_dim * config.num_intermediate_factor)
        self.act_fn = nn.ReLU()
        self.matmul = P.MatMul(transpose_b=True)
        self.split = ops.Split(axis=-1, output_num=2)
        self.prior_net1_weights = Parameter(
            initializer(lecun_init(latent_dim, initializer_name='relu'), [self.num_intermediate, latent_dim]))
        self.prior_net1_biases = Parameter(Tensor(np.zeros((self.num_intermediate)), mstype.float32))
        self.prior_net2_weights = Parameter(
            Tensor(np.zeros((2 * latent_dim, self.num_intermediate)), mstype.float32))
        self.prior_net2_biases = Parameter(Tensor(np.zeros((2 * latent_dim)), mstype.float32))

    def construct(self, w_act, v_act):
        '''construct'''
        act_shape = P.Shape()(w_act)
        if len(act_shape) != 2:
            w_act = P.Reshape()(w_act, (-1, act_shape[-1]))
        prior_state = self.act_fn(P.BiasAdd()(self.matmul(w_act, self.prior_net1_weights), self.prior_net1_biases))
        prior_state = P.BiasAdd()(self.matmul(prior_state, self.prior_net2_weights), self.prior_net2_biases)
        prior_state = P.Reshape()(prior_state, act_shape[:-1] + (-1,))
        mu_prior, log_sigma_prior = self.split(prior_state)

        act_shape = P.Shape()(v_act)
        if len(act_shape) != 2:
            v_act = P.Reshape()(v_act, (-1, act_shape[-1]))
        posterior_state = self.act_fn(P.BiasAdd()(self.matmul(v_act, self.prior_net1_weights), self.prior_net1_biases))
        posterior_state = P.BiasAdd()(self.matmul(posterior_state, self.prior_net2_weights), self.prior_net2_biases)
        posterior_state = P.Reshape()(posterior_state, act_shape[:-1] + (-1,))
        mu_posterior, log_sigma_posterior = self.split(posterior_state)
        latent_statistics_result = mu_prior, log_sigma_prior, mu_posterior, log_sigma_posterior
        return latent_statistics_result


class LatentRemap(nn.Cell):
    '''LatentRemap'''

    def __init__(self, config, input_dim, output_dim):
        super(LatentRemap, self).__init__()
        self.transition = Transition(config.msa_transition.num_intermediate_factor, output_dim)
        self.matmul = P.MatMul(transpose_b=True)
        self.linear_weights = Parameter(initializer(lecun_init(input_dim), [output_dim, input_dim]))
        self.linear_biases = Parameter(Tensor(np.zeros((output_dim)), mstype.float32))

    def construct(self, act, h_act, mask):
        '''construct'''
        act_shape = P.Shape()(act)
        if len(act_shape) != 2:
            act = P.Reshape()(act, (-1, act_shape[-1]))
        act = P.BiasAdd()(self.matmul(act, self.linear_weights), self.linear_biases)
        h_act_star = P.Reshape()(act, act_shape[:-1] + (-1,))
        delta_h = h_act_star - h_act
        delta_h = P.Add()(delta_h, self.transition(delta_h, 0, mask))
        return delta_h


class LatentBlock(nn.Cell):
    '''LatentBlock'''

    def __init__(self, config, msa_dim, latent_dim):
        super(LatentBlock, self).__init__()
        self.config = config.model.latent
        self.temperature = self.config.temperature
        self.encoder_latent_projection = LatentTransition(self.config.latent_transition, msa_dim, latent_dim)
        self.decoder_latent_projection = LatentTransition(self.config.latent_transition, msa_dim, latent_dim)
        self.context_transformer_layers = self.config.context_layers
        blocks = nn.CellList()
        for _ in range(self.context_transformer_layers):
            block = LatentTransformerBlock(self.config, latent_dim, latent_dim)
            if config.global_config.recompute:
                block.recompute()
            blocks.append(block)
        self.context_transformer = blocks
        self.match_transformer = LatentTransformerBlock(self.config, latent_dim, latent_dim)
        if config.global_config.recompute:
            self.match_transformer.recompute()

        self.noise_transformer = LatentTransformerBlock(self.config, latent_dim, latent_dim)
        if config.global_config.recompute:
            self.noise_transformer.recompute()

        self.latent_statistics = LatentStatistics(self.config.latent_statistics, latent_dim)
        self.latent_normal = LatentNormal()
        self.latent_mapper = LatentRemap(self.config, latent_dim, msa_dim)

    def construct(self, dec_act, enc_act, msa_mask, context_mask, target_mask, eps=None):
        '''construct'''
        q_mask_u = P.Reshape()(context_mask, (1, -1))
        q_mask_w = P.Reshape()(target_mask, (1, -1))

        u_act = self.encoder_latent_projection(enc_act, msa_mask)
        w_act = self.decoder_latent_projection(dec_act, msa_mask)
        u_act_star = u_act
        for i in range(self.context_transformer_layers):
            u_act_star = self.context_transformer[i](u_act, u_act, q_mask_u, q_mask_u)

        w_act_star = self.match_transformer(w_act, u_act_star, q_mask_w, q_mask_u)
        v_act_star = self.match_transformer(u_act, u_act_star, q_mask_w, q_mask_u)
        mu_prior, log_sigma_prior, mu_posterior, log_sigma_posterior = self.latent_statistics(w_act_star, v_act_star)
        target_mask = P.Reshape()(target_mask, (-1, 1, 1))

        mu_posterior = target_mask * mu_posterior + (1. - target_mask) * mu_prior
        log_sigma_posterior = target_mask * log_sigma_posterior + (1. - target_mask) * log_sigma_prior
        if eps is not None:
            eps[0] *= 0.
            z_act = self.latent_normal.sample_given_eps(eps, mu_posterior, log_sigma_posterior, temp=self.temperature)
        else:
            z_act = self.latent_normal.sample(mu_posterior, log_sigma_posterior, temp=self.temperature)

        z_act_star = self.noise_transformer(z_act, u_act_star, q_mask_w, q_mask_u)
        delta_h = self.latent_mapper(z_act_star, dec_act, msa_mask)
        dec_act = P.Add()(dec_act, delta_h)
        latent_block_result = dec_act, mu_prior, log_sigma_prior, mu_posterior, log_sigma_posterior
        return latent_block_result
