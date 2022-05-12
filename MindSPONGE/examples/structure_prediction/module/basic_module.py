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
"""basic model"""
import numpy as np
import mindspore.common.dtype as mstype
import mindspore.nn as nn
import mindspore.numpy as mnp
from mindspore import Parameter
from mindspore.common.initializer import initializer
from mindspore.common.tensor import Tensor
from mindspore.ops import functional as F
from mindspore.ops import operations as P

from commons.utils import lecun_init


def glorot_uniform(fan_in, fan_out, weight_shape):
    limit = np.sqrt(6 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, size=weight_shape)


class Attention(nn.Cell):
    '''attention module'''

    def __init__(self, config, q_data_dim, m_data_dim, output_dim, batch_size=None):
        super(Attention, self).__init__()
        self.config = config
        self.q_data_dim = q_data_dim
        self.m_data_dim = m_data_dim
        self.output_dim = output_dim
        self.num_head = self.config.num_head
        self.gating = self.config.gating
        self.key_dim = self.config.get('key_dim', int(q_data_dim))
        self.value_dim = self.config.get('value_dim', int(m_data_dim))
        self.key_dim = self.key_dim // self.num_head
        self.value_dim = self.value_dim // self.num_head
        self.batch_size = batch_size
        self.matmul = P.MatMul(transpose_b=True)
        self.batch_matmul_trans_b = P.BatchMatMul(transpose_b=True)
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
        self.batch_size = batch_size
        self._init_parameter()

    def _init_parameter(self):
        '''init parameter'''
        if self.batch_size:
            self.linear_q_weights = Parameter(Tensor(np.zeros([self.batch_size, self.num_head * self.key_dim,
                                                               self.q_data_dim]), mstype.float32))
            self.linear_k_weights = Parameter(Tensor(np.zeros([self.batch_size, self.num_head * self.key_dim,
                                                               self.m_data_dim]), mstype.float32))
            self.linear_v_weights = Parameter(Tensor(np.zeros([self.batch_size, self.num_head * self.value_dim,
                                                               self.m_data_dim]), mstype.float32))
            self.linear_output_weights = Parameter(Tensor(np.zeros([self.batch_size, self.output_dim,
                                                                    self.num_head * self.value_dim]), mstype.float32))
            self.o_biases = Parameter(Tensor(np.zeros([self.batch_size, self.output_dim]), mstype.float32))
            if self.gating:
                self.linear_gating_weights = Parameter(Tensor(np.zeros([self.batch_size, self.num_head * self.value_dim,
                                                                        self.q_data_dim]), mstype.float32))
                self.gating_biases = Parameter(Tensor(np.zeros((self.batch_size, self.num_head, self.value_dim)),
                                                      mstype.float32), name="gating_b")
        else:
            self.linear_q_weights = Parameter(Tensor(
                glorot_uniform(self.num_head * self.q_data_dim, self.key_dim * self.q_data_dim,
                               [self.num_head * self.key_dim, self.q_data_dim]), mstype.float32))
            self.linear_k_weights = Parameter(Tensor(
                glorot_uniform(self.num_head * self.m_data_dim, self.key_dim * self.m_data_dim,
                               [self.num_head * self.key_dim, self.m_data_dim]), mstype.float32))
            self.linear_v_weights = Parameter(Tensor(
                glorot_uniform(self.num_head * self.m_data_dim, self.value_dim * self.m_data_dim,
                               [self.num_head * self.value_dim, self.m_data_dim]), mstype.float32))
            self.linear_output_weights = Parameter(
                Tensor(np.zeros([self.output_dim, self.num_head * self.value_dim]), mstype.float32))
            self.o_biases = Parameter(Tensor(np.zeros([self.output_dim]), mstype.float32))
            if self.gating:
                self.linear_gating_weights = Parameter(
                    Tensor(np.zeros([self.num_head * self.value_dim, self.q_data_dim]), mstype.float32))
                self.gating_biases = Parameter(Tensor(np.ones((self.num_head, self.value_dim)), mstype.float32),
                                               name="gating_b")

    def construct(self, q_data, m_data, bias, index=None, nonbatched_bias=None):
        '''construct'''

        q_data = P.Cast()(q_data, mstype.float16)
        m_data = P.Cast()(m_data, mstype.float16)

        if self.batch_size:
            linear_q_weight = P.Cast()(P.Gather()(self.linear_q_weights, index, 0), mstype.float16)
            linear_k_weight = P.Cast()(P.Gather()(self.linear_k_weights, index, 0), mstype.float16)
            linear_v_weight = P.Cast()(P.Gather()(self.linear_v_weights, index, 0), mstype.float16)
            linear_output_weight = P.Cast()(P.Gather()(self.linear_output_weights, index, 0), mstype.float16)
            o_bias = P.Cast()(P.Gather()(self.o_biases, index, 0), mstype.float16)
            linear_gating_weight = 0
            gating_bias = 0
            if self.gating:
                linear_gating_weight = P.Cast()(P.Gather()(self.linear_gating_weights, index, 0), mstype.float16)
                gating_bias = P.Cast()(P.Gather()(self.gating_biases, index, 0), mstype.float16)
        else:
            linear_q_weight = P.Cast()(self.linear_q_weights, mstype.float16)
            linear_k_weight = P.Cast()(self.linear_k_weights, mstype.float16)
            linear_v_weight = P.Cast()(self.linear_v_weights, mstype.float16)
            linear_output_weight = P.Cast()(self.linear_output_weights, mstype.float16)
            o_bias = P.Cast()(self.o_biases, mstype.float16)
            linear_gating_weight = 0
            gating_bias = 0
            if self.gating:
                linear_gating_weight = P.Cast()(self.linear_gating_weights, mstype.float16)
                gating_bias = P.Cast()(self.gating_biases, mstype.float16)

        dim_b, dim_q, dim_a = q_data.shape
        _, dim_k, dim_c = m_data.shape
        dim_h = self.num_head

        q_data = P.Reshape()(q_data, (-1, dim_a))
        m_data = P.Reshape()(m_data, (-1, dim_c))

        q = self.matmul(q_data, linear_q_weight) * self.key_dim ** (-0.5)
        k = self.matmul(m_data, linear_k_weight)
        v = self.matmul(m_data, linear_v_weight)

        q = P.Reshape()(q, (dim_b, dim_q, dim_h, -1))
        k = P.Reshape()(k, (dim_b, dim_k, dim_h, -1))
        v = P.Reshape()(v, (dim_b, dim_k, dim_h, -1))

        tmp_q = P.Reshape()(P.Transpose()(q.astype(mstype.float16), (0, 2, 1, 3)), (dim_b * dim_h, dim_q, -1))
        tmp_k = P.Reshape()(P.Transpose()(k.astype(mstype.float16), (0, 2, 1, 3)), (dim_b * dim_h, dim_k, -1))
        bias = P.Cast()(bias, mstype.float32)
        logits = P.Add()(P.Cast()(P.Reshape()(self.batch_matmul_trans_b(tmp_q, tmp_k), (dim_b, dim_h, dim_q, dim_k)),
                                  mstype.float32), bias)

        if nonbatched_bias is not None:
            bias = P.Cast()(P.ExpandDims()(nonbatched_bias, 0), mstype.float32)
            logits = P.Add()(logits, bias)
        weights = self.softmax(logits)
        weights = P.Cast()(weights, mstype.float16)
        tmp_v = P.Reshape()(P.Transpose()(v, (0, 2, 3, 1)), (dim_b * dim_h, -1, dim_k))
        tmp_weights = P.Reshape()(weights, (dim_b * dim_h, dim_q, -1))
        weighted_avg = P.Transpose()(
            P.Reshape()(self.batch_matmul_trans_b(tmp_weights, tmp_v), (dim_b, dim_h, dim_q, -1)), (0, 2, 1, 3))

        if self.gating:
            gating_bias = P.ExpandDims()(P.ExpandDims()(gating_bias, 0), 0)
            gate_values = P.Add()(P.Reshape()(self.matmul(q_data, linear_gating_weight), (dim_b, dim_q, dim_h, -1)),
                                  gating_bias)
            gate_values = P.Cast()(gate_values, mstype.float32)
            gate_values = self.sigmoid(gate_values)
            gate_values = P.Cast()(gate_values, mstype.float16)
            weighted_avg = P.Reshape()(weighted_avg * gate_values, (dim_b * dim_q, -1))

        weighted_avg = P.Reshape()(weighted_avg, (dim_b * dim_q, -1))
        output = P.Add()(P.Reshape()(self.matmul(weighted_avg, linear_output_weight), (dim_b, dim_q, -1)),
                         P.ExpandDims()(o_bias, 0))
        return output


class MSARowAttentionWithPairBias(nn.Cell):
    '''MSA row attention'''

    def __init__(self, config, msa_act_dim, pair_act_dim, batch_size=None, slice_num=0):
        super(MSARowAttentionWithPairBias, self).__init__()
        self.config = config
        self.num_head = self.config.num_head
        self.batch_size = batch_size
        self.norm = P.LayerNorm(begin_norm_axis=-1, begin_params_axis=-1, epsilon=1e-5)
        self.matmul = P.MatMul(transpose_b=True)
        self.attn_mod = Attention(self.config, msa_act_dim, msa_act_dim, msa_act_dim, batch_size)
        self.msa_act_dim = msa_act_dim
        self.pair_act_dim = pair_act_dim
        self.batch_size = batch_size
        self.slice_num = slice_num
        self.idx = Tensor(0, mstype.int32)
        self._init_parameter()

    def _init_parameter(self):
        '''init parameter'''
        if self.batch_size:
            self.query_norm_gammas = Parameter(Tensor(np.zeros([self.batch_size, self.msa_act_dim, ]), mstype.float32))
            self.query_norm_betas = Parameter(Tensor(np.zeros([self.batch_size, self.msa_act_dim, ]), mstype.float32))
            self.feat_2d_norm_gammas = Parameter(
                Tensor(np.zeros([self.batch_size, self.pair_act_dim, ]), mstype.float32))
            self.feat_2d_norm_betas = Parameter(
                Tensor(np.zeros([self.batch_size, self.pair_act_dim, ]), mstype.float32))
            self.feat_2d_weights = Parameter(
                Tensor(np.zeros([self.batch_size, self.num_head, self.pair_act_dim]), mstype.float32))
        else:
            self.query_norm_gammas = Parameter(Tensor(np.ones([self.msa_act_dim, ]), mstype.float32))
            self.query_norm_betas = Parameter(Tensor(np.zeros([self.msa_act_dim, ]), mstype.float32))
            self.feat_2d_norm_gammas = Parameter(Tensor(np.ones([self.pair_act_dim, ]), mstype.float32))
            self.feat_2d_norm_betas = Parameter(Tensor(np.zeros([self.pair_act_dim, ]), mstype.float32))
            self.feat_2d_weights = Parameter(
                Tensor(np.random.normal(scale=1 / np.sqrt(self.pair_act_dim), size=[self.num_head, self.pair_act_dim]),
                       mstype.float32))

    def construct(self, msa_act, msa_mask, pair_act, index):
        '''construct'''
        if self.batch_size:
            query_norm_gamma = P.Gather()(self.query_norm_gammas, index, 0)
            query_norm_beta = P.Gather()(self.query_norm_betas, index, 0)
            feat_2d_norm_gamma = P.Gather()(self.feat_2d_norm_gammas, index, 0)
            feat_2d_norm_beta = P.Gather()(self.feat_2d_norm_betas, index, 0)
            feat_2d_weight = P.Cast()(P.Gather()(self.feat_2d_weights, index, 0), mstype.float16)
        else:
            query_norm_gamma = self.query_norm_gammas
            query_norm_beta = self.query_norm_betas
            feat_2d_norm_gamma = self.feat_2d_norm_gammas
            feat_2d_norm_beta = self.feat_2d_norm_betas
            feat_2d_weight = P.Cast()(self.feat_2d_weights, mstype.float16)

        q, k, _ = pair_act.shape
        # bias = (1e9 * (msa_mask - 1.))[:, None, None, :]
        msa_mask = P.Cast()(msa_mask, mstype.float32)
        bias = 1e9 * (msa_mask - 1.0)
        bias = P.ExpandDims()(P.ExpandDims()(bias, 1), 2)

        msa_act = P.Cast()(msa_act, mstype.float32)
        pair_act = P.Cast()(pair_act, mstype.float32)
        msa_act, _, _ = self.norm(msa_act, query_norm_gamma, query_norm_beta)
        pair_act, _, _ = self.norm(pair_act, feat_2d_norm_gamma, feat_2d_norm_beta)
        msa_act = P.Cast()(msa_act, mstype.float16)
        pair_act = P.Cast()(pair_act, mstype.float16)
        pair_act = P.Reshape()(pair_act, (-1, pair_act.shape[-1]))
        nonbatched_bias = P.Transpose()(P.Reshape()(self.matmul(pair_act, feat_2d_weight), (q, k, self.num_head)),
                                        (2, 0, 1))

        if self.slice_num:
            msa_act_ori_shape = P.Shape()(msa_act)
            slice_shape = (self.slice_num, -1) + msa_act_ori_shape[1:]
            msa_act = P.Reshape()(msa_act, slice_shape)
            bias_shape = P.Shape()(bias)
            bias = P.Reshape()(bias, slice_shape[:2] + bias_shape[1:])
            slice_idx = 0
            slice_idx_tensor = self.idx
            msa_act_tuple = ()

            msa_act_slice = P.Gather()(msa_act, slice_idx_tensor, 0)
            bias_slice = P.Gather()(bias, slice_idx_tensor, 0)
            msa_act_slice = self.attn_mod(msa_act_slice, msa_act_slice, bias_slice, index, nonbatched_bias)
            msa_act_slice = P.Reshape()(msa_act_slice, ((1,) + P.Shape()(msa_act_slice)))
            msa_act_tuple = msa_act_tuple + (msa_act_slice,)
            slice_idx += 1
            slice_idx_tensor += 1

            while slice_idx < self.slice_num:
                msa_act_slice = P.Gather()(msa_act, slice_idx_tensor, 0)
                msa_act_slice = F.depend(msa_act_slice, msa_act_tuple[-1])
                bias_slice = P.Gather()(bias, slice_idx_tensor, 0)
                msa_act_slice = self.attn_mod(msa_act_slice, msa_act_slice, bias_slice, index, nonbatched_bias)
                msa_act_slice = P.Reshape()(msa_act_slice, ((1,) + P.Shape()(msa_act_slice)))
                msa_act_tuple = msa_act_tuple + (msa_act_slice,)
                slice_idx += 1
                slice_idx_tensor += 1

            msa_act = P.Concat()(msa_act_tuple)
            msa_act = P.Reshape()(msa_act, msa_act_ori_shape)
            return msa_act

        msa_act = self.attn_mod(msa_act, msa_act, bias, index, nonbatched_bias)
        return msa_act


class MSAColumnAttention(nn.Cell):
    '''MSA column attention'''

    def __init__(self, config, msa_act_dim, batch_size=None, slice_num=0):
        super(MSAColumnAttention, self).__init__()
        self.config = config
        self.query_norm = P.LayerNorm(begin_norm_axis=-1, begin_params_axis=-1, epsilon=1e-5)
        self.attn_mod = Attention(self.config, msa_act_dim, msa_act_dim, msa_act_dim, batch_size)
        self.batch_size = batch_size
        self.slice_num = slice_num
        self.msa_act_dim = msa_act_dim
        self.idx = Tensor(0, mstype.int32)
        self._init_parameter()

    def _init_parameter(self):
        if self.batch_size:
            self.query_norm_gammas = Parameter(Tensor(np.zeros([self.batch_size, self.msa_act_dim]), mstype.float32))
            self.query_norm_betas = Parameter(Tensor(np.zeros([self.batch_size, self.msa_act_dim]), mstype.float32))
        else:
            self.query_norm_gammas = Parameter(Tensor(np.ones([self.msa_act_dim]), mstype.float32))
            self.query_norm_betas = Parameter(Tensor(np.zeros([self.msa_act_dim]), mstype.float32))

    def construct(self, msa_act, msa_mask, index):
        '''construct'''
        if self.batch_size:
            query_norm_gamma = P.Gather()(self.query_norm_gammas, index, 0)
            query_norm_beta = P.Gather()(self.query_norm_betas, index, 0)
        else:
            query_norm_gamma = self.query_norm_gammas
            query_norm_beta = self.query_norm_betas
        msa_act = P.Transpose()(msa_act, (1, 0, 2))
        msa_mask = P.Transpose()(msa_mask, (1, 0))

        msa_mask = P.Cast()(msa_mask, mstype.float32)
        bias = 1e9 * (msa_mask - 1.)
        bias = P.ExpandDims()(P.ExpandDims()(bias, 1), 2)
        # bias = (1e9 * (msa_mask - 1.))[:, None, None, :]
        msa_act = P.Cast()(msa_act, mstype.float32)
        query_norm_gamma = P.Cast()(query_norm_gamma, mstype.float32)
        query_norm_beta = P.Cast()(query_norm_beta, mstype.float32)
        msa_act, _, _ = self.query_norm(msa_act, query_norm_gamma, query_norm_beta)
        msa_act = P.Cast()(msa_act, mstype.float16)

        if self.slice_num:
            msa_act_ori_shape = P.Shape()(msa_act)
            slice_shape = (self.slice_num, -1) + msa_act_ori_shape[1:]
            msa_act = P.Reshape()(msa_act, slice_shape)
            bias_shape = P.Shape()(bias)
            bias = P.Reshape()(bias, slice_shape[:2] + bias_shape[1:])

            slice_idx = 0
            slice_idx_tensor = self.idx
            msa_act_tuple = ()

            msa_act_slice = P.Gather()(msa_act, slice_idx_tensor, 0)
            bias_slice = P.Gather()(bias, slice_idx_tensor, 0)
            msa_act_slice = self.attn_mod(msa_act_slice, msa_act_slice, bias_slice, index)
            msa_act_slice = P.Reshape()(msa_act_slice, ((1,) + P.Shape()(msa_act_slice)))
            msa_act_tuple = msa_act_tuple + (msa_act_slice,)
            slice_idx += 1
            slice_idx_tensor += 1

            while slice_idx < self.slice_num:
                msa_act_slice = P.Gather()(msa_act, slice_idx_tensor, 0)
                msa_act_slice = F.depend(msa_act_slice, msa_act_tuple[-1])
                bias_slice = P.Gather()(bias, slice_idx_tensor, 0)
                msa_act_slice = self.attn_mod(msa_act_slice, msa_act_slice, bias_slice, index)
                msa_act_slice = P.Reshape()(msa_act_slice, ((1,) + P.Shape()(msa_act_slice)))
                msa_act_tuple = msa_act_tuple + (msa_act_slice,)
                slice_idx += 1
                slice_idx_tensor += 1

            msa_act = P.Concat()(msa_act_tuple)
            msa_act = P.Reshape()(msa_act, msa_act_ori_shape)
            msa_act = mnp.swapaxes(msa_act, -2, -3)
            return msa_act

        msa_act = self.attn_mod(msa_act, msa_act, bias, index)
        msa_act = P.Transpose()(msa_act, (1, 0, 2))
        return msa_act


class GlobalAttention(nn.Cell):
    '''global attention'''

    def __init__(self, config, key_dim, value_dim, output_dim, batch_size=None):
        super(GlobalAttention, self).__init__()
        self.config = config
        self.key_dim = key_dim
        self.ori_key_dim = key_dim
        self.value_dim = value_dim
        self.ori_value_dim = value_dim
        self.num_head = self.config.num_head
        self.key_dim = self.key_dim // self.num_head
        self.value_dim = self.value_dim // self.num_head
        self.output_dim = output_dim
        self.matmul_trans_b = P.MatMul(transpose_b=True)
        self.batch_matmul = P.BatchMatMul()
        self.batch_matmul_trans_b = P.BatchMatMul(transpose_b=True)
        self.matmul = P.MatMul()
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
        self.gating = self.config.gating

        self.batch_size = batch_size
        self._init_parameter()

    def _init_parameter(self):
        '''init parameter'''
        if self.batch_size:
            self.linear_q_weights = Parameter(
                Tensor(np.zeros((self.batch_size, self.ori_key_dim, self.num_head, self.key_dim)), mstype.float32))
            self.linear_k_weights = Parameter(
                Tensor(np.zeros((self.batch_size, self.ori_value_dim, self.key_dim)), mstype.float32))
            self.linear_v_weights = Parameter(
                Tensor(np.zeros((self.batch_size, self.ori_value_dim, self.value_dim)), mstype.float32))
            self.linear_output_weights = Parameter(
                Tensor(np.zeros((self.batch_size, self.output_dim, self.num_head * self.value_dim)), mstype.float32))
            self.o_biases = Parameter(Tensor(np.zeros((self.batch_size, self.output_dim)), mstype.float32))
            if self.gating:
                self.linear_gating_weights = Parameter(
                    Tensor(np.zeros((self.batch_size, self.num_head * self.value_dim, self.ori_key_dim)),
                           mstype.float32))
                self.gating_biases = Parameter(Tensor(np.zeros((self.batch_size, self.ori_key_dim)), mstype.float32))
        else:
            self.linear_q_weights = Parameter(Tensor(
                glorot_uniform(self.num_head * self.ori_key_dim, self.key_dim * self.ori_key_dim,
                               (self.ori_key_dim, self.num_head, self.key_dim)), mstype.float32))
            self.linear_k_weights = Parameter(
                Tensor(glorot_uniform(self.ori_value_dim, self.key_dim, (self.ori_value_dim, self.key_dim)),
                       mstype.float32))
            self.linear_v_weights = Parameter(
                Tensor(glorot_uniform(self.ori_value_dim, self.value_dim, (self.ori_value_dim, self.value_dim)),
                       mstype.float32))
            self.linear_output_weights = Parameter(
                Tensor(np.zeros((self.output_dim, self.num_head * self.value_dim)), mstype.float32))
            self.o_biases = Parameter(Tensor(np.zeros((self.output_dim)), mstype.float32))
            if self.gating:
                self.linear_gating_weights = Parameter(
                    Tensor(np.zeros((self.num_head * self.value_dim, self.ori_key_dim)), mstype.float32))
                self.gating_biases = Parameter(Tensor(np.ones((self.ori_key_dim)), mstype.float32))

    def construct(self, q_data, m_data, q_mask, bias, index):
        '''construct'''
        if self.batch_size:
            q_data = P.Cast()(q_data, mstype.float16)
            q_weights = P.Cast()(P.Gather()(self.linear_q_weights, index, 0), mstype.float16)
            k_weights = P.Cast()(P.Gather()(self.linear_k_weights, index, 0), mstype.float16)
            v_weights = P.Cast()(P.Gather()(self.linear_v_weights, index, 0), mstype.float16)
            output_weights = P.Cast()(P.Gather()(self.linear_output_weights, index, 0), mstype.float16)
            output_bias = P.Cast()(P.Gather()(self.o_biases, index, 0), mstype.float16)
            gating_weights = 0
            gating_bias = 0
            if self.gating:
                gating_weights = P.Gather()(self.linear_gating_weights, index, 0)
                gating_weights = P.Cast()(gating_weights, mstype.float16)
                gating_bias = P.Cast()(P.Gather()(self.gating_biases, index, 0), mstype.float16)
        else:
            q_mask = P.Cast()(q_mask, mstype.float16)
            q_data = P.Cast()(q_data, mstype.float16)
            q_weights = P.Cast()(self.linear_q_weights, mstype.float16)
            k_weights = P.Cast()(self.linear_k_weights, mstype.float16)
            v_weights = P.Cast()(self.linear_v_weights, mstype.float16)
            output_weights = P.Cast()(self.linear_output_weights, mstype.float16)
            output_bias = P.Cast()(self.o_biases, mstype.float16)
            gating_weights = 0
            gating_bias = 0
            if self.gating:
                gating_weights = self.linear_gating_weights
                gating_weights = P.Cast()(gating_weights, mstype.float16)
                gating_bias = P.Cast()(self.gating_biases, mstype.float16)

        b, _, _ = m_data.shape
        # v_weights = v_weights[None, ...]

        v_weights = P.ExpandDims()(v_weights, 0)
        v_weights = P.BroadcastTo((b, self.value_dim * self.num_head, self.value_dim))(v_weights)
        v = self.batch_matmul(m_data, v_weights)
        q_mask = P.Cast()(q_mask, mstype.float32)
        q_data = P.Cast()(q_data, mstype.float32)
        # q_avg = mask_mean(q_mask, q_data, axis=1)

        mask_shape = q_mask.shape
        value_shape = q_data.shape
        broadcast_factor = 1.
        value_size = value_shape[1]
        mask_size = mask_shape[1]
        if mask_size == 1:
            broadcast_factor = broadcast_factor * value_size
        qa = P.ReduceSum()(q_mask * q_data, 1)
        qb = P.ReduceSum()(q_mask, 1) * broadcast_factor + 1e-10
        q_avg = P.Cast()(P.RealDiv()(qa, qb), mstype.float16)

        q_data = P.Cast()(q_data, mstype.float16)
        q_weights = P.Reshape()(q_weights, (-1, self.num_head * self.key_dim))
        q = P.Reshape()(self.matmul(q_avg, q_weights),
                        (-1, self.num_head, self.key_dim)) * (self.key_dim ** (-0.5))

        # k_weights = k_weights[None, ...]
        k_weights = P.ExpandDims()(k_weights, 0)
        # k_weights = mnp.broadcast_to(k_weights, (b, self.value_dim * self.num_head, self.key_dim))
        k_weights = P.BroadcastTo((b, self.value_dim * self.num_head, self.key_dim))(k_weights)
        k = self.batch_matmul(m_data, k_weights)
        # k = self.batch_matmul(m_data.astype(mstype.float16), k_weights.astype(mstype.float16))

        bias = 1e9 * (P.Transpose()(q_mask, (0, 2, 1)) - 1.0)
        logits = P.Add()(P.Cast()(self.batch_matmul_trans_b(q, k), mstype.float32), bias)

        weights = self.softmax(logits)
        weights = P.Cast()(weights, mstype.float16)
        weighted_avg = self.batch_matmul(weights, v)

        if self.gating:
            # gate_values = self.linear_gating(q_data).astype(mstype.float32)
            q_data_shape = P.Shape()(q_data)
            if len(q_data_shape) != 2:
                q_data = P.Reshape()(q_data, (-1, q_data_shape[-1]))
            out_shape = q_data_shape[:-1] + (-1,)
            gate_values = P.Reshape()(self.matmul_trans_b(q_data, gating_weights) + gating_bias, out_shape)

            gate_values = P.Cast()(gate_values, mstype.float32)
            gate_values = P.Reshape()(self.sigmoid(gate_values), (b, -1, self.num_head, self.value_dim))
            gate_values = P.Cast()(gate_values, mstype.float16)
            weighted_avg = P.Reshape()(P.ExpandDims()(weighted_avg, 1) * gate_values,
                                       (-1, self.num_head * self.value_dim))
            weighted_avg_shape = P.Shape()(weighted_avg)
            if len(weighted_avg_shape) != 2:
                weighted_avg = P.Reshape()(weighted_avg, (-1, weighted_avg_shape[-1]))
            output = P.Reshape()(P.Add()(self.matmul_trans_b(weighted_avg,
                                                             output_weights), output_bias),
                                 (b, -1, self.output_dim))

        else:
            weighted_avg = P.Reshape()(weighted_avg, (-1, self.num_head * self.value_dim))
            # output = self.linear_gating(weighted_avg)
            weighted_avg_shape = P.Shape()(weighted_avg)
            if len(weighted_avg_shape) != 2:
                weighted_avg = P.Reshape()(weighted_avg, (-1, weighted_avg_shape[-1]))
            out_shape = weighted_avg_shape[:-1] + (-1,)
            output = P.Reshape()(P.Add()(self.matmul_trans_b(weighted_avg,
                                                             output_weights), output_bias), out_shape)
            output = P.ExpandDims()(output, -1)
        return output


class MSAColumnGlobalAttention(nn.Cell):
    '''MSA column global attention'''

    def __init__(self, config, msa_act_dim, batch_size=None, slice_num=0):
        super(MSAColumnGlobalAttention, self).__init__()
        self.config = config
        self.attn_mod = GlobalAttention(self.config, msa_act_dim, msa_act_dim, msa_act_dim, batch_size)
        self.query_norm = P.LayerNorm(begin_norm_axis=-1, begin_params_axis=-1, epsilon=1e-5)
        self.batch_size = batch_size
        self.slice_num = slice_num
        self.msa_act_dim = msa_act_dim
        self.idx = Tensor(0, mstype.int32)
        self._init_parameter()

    def _init_parameter(self):
        '''init parameter'''
        if self.batch_size:
            self.query_norm_gammas = Parameter(Tensor(np.zeros((self.batch_size, self.msa_act_dim)), mstype.float32))
            self.query_norm_betas = Parameter(Tensor(np.zeros((self.batch_size, self.msa_act_dim)), mstype.float32))
        else:
            self.query_norm_gammas = Parameter(Tensor(np.ones((self.msa_act_dim)), mstype.float32))
            self.query_norm_betas = Parameter(Tensor(np.zeros((self.msa_act_dim)), mstype.float32))

    def construct(self, msa_act, msa_mask, index):
        '''construct'''
        if self.batch_size:
            query_norm_gamma = P.Gather()(self.query_norm_gammas, index, 0)
            query_norm_beta = P.Gather()(self.query_norm_betas, index, 0)
            msa_act = P.Transpose()(msa_act, (1, 0, 2))
            msa_mask = P.Transpose()(msa_mask, (1, 0))
        else:
            query_norm_gamma = self.query_norm_gammas
            query_norm_beta = self.query_norm_betas
            msa_act = P.Transpose()(msa_act, (1, 0, 2))
            msa_mask = P.Transpose()(msa_mask, (1, 0))

        msa_mask = P.Cast()(msa_mask, mstype.float32)
        bias = 1e9 * (msa_mask - 1.)
        bias = P.ExpandDims()(P.ExpandDims()(bias, 1), 2)

        msa_act = P.Cast()(msa_act, mstype.float32)
        query_norm_gamma = P.Cast()(query_norm_gamma, mstype.float32)
        query_norm_beta = P.Cast()(query_norm_beta, mstype.float32)

        msa_act, _, _ = self.query_norm(msa_act,
                                        query_norm_gamma,
                                        query_norm_beta)

        msa_act = P.Cast()(msa_act, mstype.float16)
        msa_mask = P.ExpandDims()(msa_mask, -1)

        if self.slice_num:
            msa_act_ori_shape = P.Shape()(msa_act)
            slice_shape = (self.slice_num, -1) + msa_act_ori_shape[1:]
            msa_act = P.Reshape()(msa_act, slice_shape)
            bias_shape = P.Shape()(bias)
            bias = P.Reshape()(bias, slice_shape[:2] + bias_shape[1:])
            msa_mask_shape = P.Shape()(msa_mask)
            msa_mask = P.Reshape()(msa_mask, slice_shape[:2] + msa_mask_shape[1:])

            slice_idx = 0
            slice_idx_tensor = self.idx
            msa_act_tuple = ()

            msa_act_slice = P.Gather()(msa_act, slice_idx_tensor, 0)
            msa_mask_slice = P.Gather()(msa_mask, slice_idx_tensor, 0)
            bias_slice = P.Gather()(bias, slice_idx_tensor, 0)
            msa_act_slice = self.attn_mod(msa_act_slice, msa_act_slice, msa_mask_slice, bias_slice, index)
            msa_act_slice = P.Reshape()(msa_act_slice, ((1,) + P.Shape()(msa_act_slice)))
            msa_act_tuple = msa_act_tuple + (msa_act_slice,)
            slice_idx += 1
            slice_idx_tensor += 1

            while slice_idx < self.slice_num:
                msa_act_slice = P.Gather()(msa_act, slice_idx_tensor, 0)
                msa_act_slice = F.depend(msa_act_slice, msa_act_tuple[-1])
                msa_mask_slice = P.Gather()(msa_mask, slice_idx_tensor, 0)
                bias_slice = P.Gather()(bias, slice_idx_tensor, 0)

                msa_act_slice = self.attn_mod(msa_act_slice, msa_act_slice, msa_mask_slice, bias_slice, index)
                msa_act_slice = P.Reshape()(msa_act_slice, ((1,) + P.Shape()(msa_act_slice)))
                msa_act_tuple = msa_act_tuple + (msa_act_slice,)
                slice_idx += 1
                slice_idx_tensor += 1

            msa_act = P.Concat()(msa_act_tuple)
            msa_act = P.Reshape()(msa_act, msa_act_ori_shape)
            msa_act = mnp.swapaxes(msa_act, -2, -3)
            return msa_act

        msa_act = self.attn_mod(msa_act, msa_act, msa_mask, bias, index)
        msa_act = P.Transpose()(msa_act, (1, 0, 2))
        return msa_act


class Transition(nn.Cell):
    '''transition'''

    def __init__(self, config, layer_norm_dim, batch_size=None, slice_num=0):
        super(Transition, self).__init__()
        self.config = config
        self.input_layer_norm = P.LayerNorm(begin_norm_axis=-1, begin_params_axis=-1, epsilon=1e-5)
        self.matmul = P.MatMul(transpose_b=True)
        self.layer_norm_dim = layer_norm_dim
        self.num_intermediate = int(layer_norm_dim * self.config.num_intermediate_factor)
        self.batch_size = batch_size
        self.slice_num = slice_num
        self.relu = nn.ReLU()
        self.idx = Tensor(0, mstype.int32)
        self._init_parameter()

    def _init_parameter(self):
        '''init parameter'''
        if self.batch_size:
            self.input_layer_norm_gammas = Parameter(
                Tensor(np.zeros((self.batch_size, self.layer_norm_dim)), mstype.float32))
            self.input_layer_norm_betas = Parameter(
                Tensor(np.zeros((self.batch_size, self.layer_norm_dim)), mstype.float32))
            self.transition1_weights = Parameter(
                Tensor(np.zeros((self.batch_size, self.num_intermediate, self.layer_norm_dim)), mstype.float32))
            self.transition1_biases = Parameter(
                Tensor(np.zeros((self.batch_size, self.num_intermediate)), mstype.float32))
            self.transition2_weights = Parameter(
                Tensor(np.zeros((self.batch_size, self.layer_norm_dim, self.num_intermediate)), mstype.float32))
            self.transition2_biases = Parameter(
                Tensor(np.zeros((self.batch_size, self.layer_norm_dim)), mstype.float32))
        else:
            self.input_layer_norm_gammas = Parameter(Tensor(np.ones((self.layer_norm_dim)), mstype.float32))
            self.input_layer_norm_betas = Parameter(Tensor(np.zeros((self.layer_norm_dim)), mstype.float32))
            self.transition1_weights = Parameter(initializer(lecun_init(self.layer_norm_dim, initializer_name='relu'),
                                                             [self.num_intermediate, self.layer_norm_dim]))
            self.transition1_biases = Parameter(Tensor(np.zeros((self.num_intermediate)), mstype.float32))
            self.transition2_weights = Parameter(
                Tensor(np.zeros((self.layer_norm_dim, self.num_intermediate)), mstype.float32))
            self.transition2_biases = Parameter(Tensor(np.zeros((self.layer_norm_dim)), mstype.float32))

    def construct(self, act, index):
        '''construct'''
        if self.batch_size:
            input_layer_norm_gamma = P.Gather()(self.input_layer_norm_gammas, index, 0)
            input_layer_norm_beta = P.Gather()(self.input_layer_norm_betas, index, 0)
            transition1_weight = P.Cast()(P.Gather()(self.transition1_weights, index, 0), mstype.float16)
            transition1_bias = P.Cast()(P.Gather()(self.transition1_biases, index, 0), mstype.float16)
            transition2_weight = P.Cast()(P.Gather()(self.transition2_weights, index, 0), mstype.float16)
            transition2_bias = P.Cast()(P.Gather()(self.transition2_biases, index, 0), mstype.float16)
        else:
            input_layer_norm_gamma = self.input_layer_norm_gammas
            input_layer_norm_beta = self.input_layer_norm_betas
            transition1_weight = P.Cast()(self.transition1_weights, mstype.float16)
            transition1_bias = P.Cast()(self.transition1_biases, mstype.float16)
            transition2_weight = P.Cast()(self.transition2_weights, mstype.float16)
            transition2_bias = P.Cast()(self.transition2_biases, mstype.float16)
        act = P.Cast()(act, mstype.float32)
        act, _, _ = self.input_layer_norm(act, input_layer_norm_gamma, input_layer_norm_beta)
        act = P.Cast()(act, mstype.float16)
        if self.slice_num:
            act_ori_shape = P.Shape()(act)
            slice_shape = (self.slice_num, -1) + act_ori_shape[1:]
            act = P.Reshape()(act, slice_shape)

            slice_idx = 0
            slice_idx_tensor = self.idx
            act_tuple = ()

            act_slice = P.Gather()(act, slice_idx_tensor, 0)
            act_shape = P.Shape()(act_slice)
            if len(act_shape) != 2:
                act_slice = P.Reshape()(act_slice, (-1, act_shape[-1]))
            act_slice = self.relu(P.BiasAdd()(self.matmul(act_slice, transition1_weight), transition1_bias))
            act_slice = P.BiasAdd()(self.matmul(act_slice, transition2_weight), transition2_bias)
            act_slice = P.Reshape()(act_slice, act_shape)
            act_slice = P.Reshape()(act_slice, ((1,) + P.Shape()(act_slice)))
            act_tuple = act_tuple + (act_slice,)
            slice_idx += 1
            slice_idx_tensor += 1

            while slice_idx < self.slice_num:
                act_slice = P.Gather()(act, slice_idx_tensor, 0)
                act_slice = F.depend(act_slice, act_tuple[-1])
                act_shape = P.Shape()(act_slice)
                if len(act_shape) != 2:
                    act_slice = P.Reshape()(act_slice, (-1, act_shape[-1]))
                act_slice = self.relu(P.BiasAdd()(self.matmul(act_slice, transition1_weight), transition1_bias))
                act_slice = P.BiasAdd()(self.matmul(act_slice, transition2_weight), transition2_bias)
                act_slice = P.Reshape()(act_slice, act_shape)
                act_slice = P.Reshape()(act_slice, ((1,) + P.Shape()(act_slice)))
                act_tuple = act_tuple + (act_slice,)
                slice_idx += 1
                slice_idx_tensor += 1

            act = P.Concat()(act_tuple)
            act = P.Reshape()(act, act_ori_shape)
            return act

        act_shape = P.Shape()(act)
        if len(act_shape) != 2:
            act = P.Reshape()(act, (-1, act_shape[-1]))
        act = self.relu(P.BiasAdd()(self.matmul(act, transition1_weight), transition1_bias))
        act = P.BiasAdd()(self.matmul(act, transition2_weight), transition2_bias)
        act = P.Reshape()(act, act_shape)
        return act


class OuterProductMean(nn.Cell):
    '''outerproduct mean'''

    def __init__(self, config, act_dim, num_output_channel, batch_size=None, slice_num=0):
        super(OuterProductMean, self).__init__()
        self.num_output_channel = num_output_channel
        self.config = config
        self.layer_norm_input = P.LayerNorm(begin_norm_axis=-1, begin_params_axis=-1, epsilon=1e-5)
        self.matmul_trans_b = P.MatMul(transpose_b=True)
        self.matmul = P.MatMul()
        self.batch_matmul_trans_b = P.BatchMatMul(transpose_b=True)
        self.act_dim = act_dim
        self.batch_size = batch_size
        self.slice_num = slice_num
        self.idx = Tensor(0, mstype.int32)
        self._init_parameter()

    def _init_parameter(self):
        '''init parameter'''
        if self.batch_size:
            self.layer_norm_input_gammas = Parameter(Tensor(np.zeros((self.batch_size, self.act_dim)), mstype.float32))
            self.layer_norm_input_betas = Parameter(Tensor(np.zeros((self.batch_size, self.act_dim)), mstype.float32))
            self.left_projection_weights = Parameter(
                Tensor(np.zeros((self.batch_size, self.config.num_outer_channel, self.act_dim)), mstype.float32))
            self.left_projection_biases = Parameter(
                Tensor(np.zeros((self.batch_size, self.config.num_outer_channel)), mstype.float32))
            self.right_projection_weights = Parameter(
                Tensor(np.zeros((self.batch_size, self.config.num_outer_channel, self.act_dim)), mstype.float32))
            self.right_projection_biases = Parameter(
                Tensor(np.zeros((self.batch_size, self.config.num_outer_channel)), mstype.float32))
            self.linear_output_weights = Parameter(Tensor(np.zeros(
                (self.batch_size, self.num_output_channel, self.config.num_outer_channel *
                 self.config.num_outer_channel)), mstype.float32))
            self.o_biases = Parameter(Tensor(np.zeros((self.batch_size, self.num_output_channel)), mstype.float32))
        else:
            self.layer_norm_input_gammas = Parameter(Tensor(np.ones((self.act_dim)), mstype.float32))
            self.layer_norm_input_betas = Parameter(Tensor(np.zeros((self.act_dim)), mstype.float32))
            self.left_projection_weights = Parameter(
                initializer(lecun_init(self.act_dim), [self.config.num_outer_channel, self.act_dim]))
            self.left_projection_biases = Parameter(Tensor(np.zeros((self.config.num_outer_channel)), mstype.float32))
            self.right_projection_weights = Parameter(
                initializer(lecun_init(self.act_dim), [self.config.num_outer_channel, self.act_dim]))
            self.right_projection_biases = Parameter(Tensor(np.zeros((self.config.num_outer_channel)), mstype.float32))
            self.linear_output_weights = Parameter(
                Tensor(
                    np.zeros((self.num_output_channel, self.config.num_outer_channel * self.config.num_outer_channel)),
                    mstype.float32))
            self.o_biases = Parameter(Tensor(np.zeros((self.num_output_channel)), mstype.float32))

    def construct(self, act, mask, extra_msa_norm, index):
        '''construct'''
        if self.batch_size:
            layer_norm_input_gamma = P.Gather()(self.layer_norm_input_gammas, index, 0)
            layer_norm_input_beta = P.Gather()(self.layer_norm_input_betas, index, 0)
            left_projection_weight = P.Cast()(P.Gather()(self.left_projection_weights, index, 0), mstype.float16)
            left_projection_bias = P.Cast()(P.Gather()(self.left_projection_biases, index, 0), mstype.float16)
            right_projection_weight = P.Cast()(P.Gather()(self.right_projection_weights, index, 0), mstype.float16)
            right_projection_bias = P.Cast()(P.Gather()(self.right_projection_biases, index, 0), mstype.float16)
            linear_output_weight = P.Cast()(P.Gather()(self.linear_output_weights, index, 0), mstype.float16)
            linear_output_bias = P.Cast()(P.Gather()(self.o_biases, index, 0), mstype.float16)
        else:
            layer_norm_input_gamma = self.layer_norm_input_gammas
            layer_norm_input_beta = self.layer_norm_input_betas
            left_projection_weight = P.Cast()(self.left_projection_weights, mstype.float16)
            left_projection_bias = P.Cast()(self.left_projection_biases, mstype.float16)
            right_projection_weight = P.Cast()(self.right_projection_weights, mstype.float16)
            right_projection_bias = P.Cast()(self.right_projection_biases, mstype.float16)
            linear_output_weight = P.Cast()(self.linear_output_weights, mstype.float16)
            linear_output_bias = P.Cast()(self.o_biases, mstype.float16)

        mask = P.Cast()(mask, mstype.float16)
        mask = P.ExpandDims()(mask, -1)
        act = P.Cast()(act, mstype.float32)
        layer_norm_input_gamma = P.Cast()(layer_norm_input_gamma, mstype.float32)
        layer_norm_input_beta = P.Cast()(layer_norm_input_beta, mstype.float32)
        act, _, _ = self.layer_norm_input(act,
                                          layer_norm_input_gamma,
                                          layer_norm_input_beta)
        act = P.Cast()(act, mstype.float16)

        act_shape = P.Shape()(act)
        if len(act_shape) != 2:
            act = P.Reshape()(act, (-1, act_shape[-1]))
        out_shape = act_shape[:-1] + (-1,)
        left_act = mask * P.Reshape()(
            P.BiasAdd()(self.matmul_trans_b(act, left_projection_weight), left_projection_bias), out_shape)
        right_act = mask * P.Reshape()(
            P.BiasAdd()(self.matmul_trans_b(act, right_projection_weight), right_projection_bias), out_shape)
        _, d, e = right_act.shape
        if self.slice_num:
            left_act_shape = P.Shape()(left_act)
            slice_shape = (left_act_shape[0],) + (self.slice_num, -1) + (left_act_shape[-1],)
            left_act = P.Reshape()(left_act, slice_shape)
            slice_idx = 0
            slice_idx_tensor = self.idx
            act_tuple = ()
            left_act_slice = P.Gather()(left_act, slice_idx_tensor, 1)
            a, b, c = left_act_slice.shape
            left_act_slice = P.Reshape()(mnp.transpose(left_act_slice, [2, 1, 0]), (-1, a))
            right_act = P.Reshape()(right_act, (a, -1))
            act_slice = P.Reshape()(P.Transpose()(P.Reshape()(self.matmul(left_act_slice,
                                                                          right_act),
                                                              (c, b, d, e)), (2, 1, 0, 3)), (d, b, c * e))
            act_slice_shape = P.Shape()(act_slice)
            if len(act_shape) != 2:
                act_slice = P.Reshape()(act_slice, (-1, act_slice_shape[-1]))
            act_slice = P.Reshape()(P.BiasAdd()(self.matmul_trans_b(act_slice,
                                                                    linear_output_weight),
                                                linear_output_bias), (d, b, -1))
            act_slice = mnp.transpose(act_slice, [1, 0, 2])
            act_slice = P.Reshape()(act_slice, ((1,) + P.Shape()(act_slice)))
            act_tuple = act_tuple + (act_slice,)
            slice_idx += 1
            slice_idx_tensor += 1
            while slice_idx < self.slice_num:
                left_act_slice = P.Gather()(left_act, slice_idx_tensor, 1)
                left_act_slice = F.depend(left_act_slice, act_tuple[-1])
                a, b, c = left_act_slice.shape
                left_act_slice = P.Reshape()(mnp.transpose(left_act_slice, [2, 1, 0]), (-1, a))
                right_act = P.Reshape()(right_act, (a, -1))
                act_slice = P.Reshape()(P.Transpose()(P.Reshape()(self.matmul(left_act_slice,
                                                                              right_act),
                                                                  (c, b, d, e)), (2, 1, 0, 3)), (d, b, c * e))
                act_slice_shape = P.Shape()(act_slice)
                if len(act_shape) != 2:
                    act_slice = P.Reshape()(act_slice, (-1, act_slice_shape[-1]))
                act_slice = P.Reshape()(P.BiasAdd()(self.matmul_trans_b(act_slice,
                                                                        linear_output_weight),
                                                    linear_output_bias), (d, b, -1))
                act_slice = mnp.transpose(act_slice, [1, 0, 2])
                act_slice = P.Reshape()(act_slice, ((1,) + P.Shape()(act_slice)))
                act_tuple = act_tuple + (act_slice,)
                slice_idx += 1
                slice_idx_tensor += 1
            act = P.Concat()(act_tuple)
            act_shape = P.Shape()(act)
            act = P.Reshape()(act, (-1, act_shape[-2], act_shape[-1]))
            epsilon = 1e-3
            tmp_mask = P.Transpose()(mask, (2, 1, 0))
            norm = P.Transpose()(self.batch_matmul_trans_b(tmp_mask,
                                                           tmp_mask),
                                 (1, 2, 0)).astype(mstype.float32)
            act = P.RealDiv()(act, epsilon + norm)
            return act

        a, b, c = left_act.shape
        left_act = P.Reshape()(P.Transpose()(left_act, (2, 1, 0)), (-1, a))
        right_act = P.Reshape()(right_act, (a, -1))
        act = P.Reshape()(P.Transpose()(P.Reshape()(self.matmul(left_act,
                                                                right_act),
                                                    (c, b, d, e)), (2, 1, 0, 3)), (d, b, c * e))
        act_shape = P.Shape()(act)
        if len(act_shape) != 2:
            act = P.Reshape()(act, (-1, act_shape[-1]))
        act = P.Reshape()(P.BiasAdd()(self.matmul_trans_b(act,
                                                          linear_output_weight),
                                      linear_output_bias), (d, b, -1))
        act = P.Transpose()(act, (1, 0, 2))
        epsilon = 1e-3
        # tmp_mask = P.Transpose()(mask, (2, 1, 0))
        # norm = P.Transpose()(self.batch_matmul_trans_b(tmp_mask,
        #                                                tmp_mask),
        #                      (1, 2, 0)).astype(mstype.float32)
        act = P.RealDiv()(act, epsilon + extra_msa_norm)
        return act


class TriangleMultiplication(nn.Cell):
    '''triangle multiplication'''

    def __init__(self, config, layer_norm_dim, batch_size):
        super(TriangleMultiplication, self).__init__()
        self.config = config
        self.layer_norm = P.LayerNorm(begin_norm_axis=-1, begin_params_axis=-1, epsilon=1e-5)
        self.matmul = P.MatMul(transpose_b=True)
        self.sigmoid = nn.Sigmoid()
        self.batch_matmul_trans_b = P.BatchMatMul(transpose_b=True)
        equation = ["ikc,jkc->ijc", "kjc,kic->ijc"]
        if self.config.equation not in equation:
            print("TriangleMultiplication Not Suppl")
        if self.config.equation == "ikc,jkc->ijc":
            self.equation = True
        elif self.config.equation == "kjc,kic->ijc":
            self.equation = False
        else:
            self.equation = None
        self.batch_size = batch_size
        self.layer_norm_dim = layer_norm_dim
        self._init_parameter()

    def _init_parameter(self):
        '''init parameter'''
        if self.batch_size:
            self.layer_norm_input_gammas = Parameter(
                Tensor(np.zeros((self.batch_size, self.layer_norm_dim)), mstype.float32))
            self.layer_norm_input_betas = Parameter(
                Tensor(np.zeros((self.batch_size, self.layer_norm_dim)), mstype.float32))
            self.left_projection_weights = Parameter(
                Tensor(np.zeros((self.batch_size, self.config.num_intermediate_channel, self.layer_norm_dim)),
                       mstype.float32))
            self.left_projection_biases = Parameter(
                Tensor(np.zeros((self.batch_size, self.config.num_intermediate_channel)), mstype.float32))
            self.right_projection_weights = Parameter(
                Tensor(np.zeros((self.batch_size, self.config.num_intermediate_channel, self.layer_norm_dim)),
                       mstype.float32))
            self.right_projection_biases = Parameter(
                Tensor(np.zeros((self.batch_size, self.config.num_intermediate_channel)), mstype.float32))
            self.left_gate_weights = Parameter(
                Tensor(np.zeros((self.batch_size, self.config.num_intermediate_channel, self.layer_norm_dim)),
                       mstype.float32))
            self.left_gate_biases = Parameter(
                Tensor(np.zeros((self.batch_size, self.config.num_intermediate_channel)), mstype.float32))
            self.right_gate_weights = Parameter(
                Tensor(np.zeros((self.batch_size, self.config.num_intermediate_channel, self.layer_norm_dim)),
                       mstype.float32))
            self.right_gate_biases = Parameter(
                Tensor(np.zeros((self.batch_size, self.config.num_intermediate_channel)), mstype.float32))
            self.center_layer_norm_gammas = Parameter(
                Tensor(np.zeros((self.batch_size, self.layer_norm_dim)), mstype.float32))
            self.center_layer_norm_betas = Parameter(
                Tensor(np.zeros((self.batch_size, self.layer_norm_dim)), mstype.float32))
            self.output_projection_weights = Parameter(
                Tensor(np.zeros((self.batch_size, self.layer_norm_dim, self.layer_norm_dim)), mstype.float32))
            self.output_projection_biases = Parameter(
                Tensor(np.zeros((self.batch_size, self.layer_norm_dim)), mstype.float32))
            self.gating_linear_weights = Parameter(
                Tensor(np.zeros((self.batch_size, self.layer_norm_dim, self.layer_norm_dim)), mstype.float32))
            self.gating_linear_biases = Parameter(
                Tensor(np.zeros((self.batch_size, self.layer_norm_dim)), mstype.float32))
        else:
            self.layer_norm_input_gammas = Parameter(Tensor(np.ones((self.layer_norm_dim)), mstype.float32))
            self.layer_norm_input_betas = Parameter(Tensor(np.zeros((self.layer_norm_dim)), mstype.float32))
            self.left_projection_weights = Parameter(initializer(lecun_init(self.config.num_intermediate_channel),
                                                                 [self.config.num_intermediate_channel,
                                                                  self.layer_norm_dim]))
            self.left_projection_biases = Parameter(
                Tensor(np.zeros((self.config.num_intermediate_channel)), mstype.float32))
            self.right_projection_weights = Parameter(initializer(lecun_init(self.config.num_intermediate_channel),
                                                                  [self.config.num_intermediate_channel,
                                                                   self.layer_norm_dim]))
            self.right_projection_biases = Parameter(
                Tensor(np.zeros((self.config.num_intermediate_channel)), mstype.float32))
            self.left_gate_weights = Parameter(
                Tensor(np.zeros((self.config.num_intermediate_channel, self.layer_norm_dim)), mstype.float32))
            self.left_gate_biases = Parameter(Tensor(np.ones((self.config.num_intermediate_channel)), mstype.float32))
            self.right_gate_weights = Parameter(
                Tensor(np.zeros((self.config.num_intermediate_channel, self.layer_norm_dim)), mstype.float32))
            self.right_gate_biases = Parameter(Tensor(np.ones((self.config.num_intermediate_channel)), mstype.float32))
            self.center_layer_norm_gammas = Parameter(Tensor(np.ones((self.layer_norm_dim)), mstype.float32))
            self.center_layer_norm_betas = Parameter(Tensor(np.zeros((self.layer_norm_dim)), mstype.float32))
            self.output_projection_weights = Parameter(
                Tensor(np.zeros((self.layer_norm_dim, self.layer_norm_dim)), mstype.float32))
            self.output_projection_biases = Parameter(Tensor(np.zeros((self.layer_norm_dim)), mstype.float32))
            self.gating_linear_weights = Parameter(
                Tensor(np.zeros((self.layer_norm_dim, self.layer_norm_dim)), mstype.float32))
            self.gating_linear_biases = Parameter(Tensor(np.ones((self.layer_norm_dim)), mstype.float32))

    def construct(self, act, mask, index):
        '''construct'''
        if self.batch_size:
            layer_norm_input_gamma = P.Gather()(self.layer_norm_input_gammas, index, 0)
            layer_norm_input_beta = P.Gather()(self.layer_norm_input_betas, index, 0)
            left_projection_weight = P.Cast()(P.Gather()(self.left_projection_weights, index, 0), mstype.float16)
            left_projection_bias = P.Cast()(P.Gather()(self.left_projection_biases, index, 0), mstype.float16)
            right_projection_weight = P.Cast()(P.Gather()(self.right_projection_weights, index, 0), mstype.float16)
            right_projection_bias = P.Cast()(P.Gather()(self.right_projection_biases, index, 0), mstype.float16)
            left_gate_weight = P.Cast()(P.Gather()(self.left_gate_weights, index, 0), mstype.float16)
            left_gate_bias = P.Cast()(P.Gather()(self.left_gate_biases, index, 0), mstype.float16)
            right_gate_weight = P.Cast()(P.Gather()(self.right_gate_weights, index, 0), mstype.float16)
            right_gate_bias = P.Cast()(P.Gather()(self.right_gate_biases, index, 0), mstype.float16)
            center_layer_norm_gamma = P.Cast()(P.Gather()(self.center_layer_norm_gammas, index, 0), mstype.float16)
            center_layer_norm_beta = P.Cast()(P.Gather()(self.center_layer_norm_betas, index, 0), mstype.float16)
            output_projection_weight = P.Cast()(P.Gather()(self.output_projection_weights, index, 0), mstype.float16)
            output_projection_bias = P.Cast()(P.Gather()(self.output_projection_biases, index, 0), mstype.float16)
            gating_linear_weight = P.Cast()(P.Gather()(self.gating_linear_weights, index, 0), mstype.float16)
            gating_linear_bias = P.Cast()(P.Gather()(self.gating_linear_biases, index, 0), mstype.float16)
        else:
            layer_norm_input_gamma = self.layer_norm_input_gammas
            layer_norm_input_beta = self.layer_norm_input_betas
            left_projection_weight = P.Cast()(self.left_projection_weights, mstype.float16)
            left_projection_bias = P.Cast()(self.left_projection_biases, mstype.float16)
            right_projection_weight = P.Cast()(self.right_projection_weights, mstype.float16)
            right_projection_bias = P.Cast()(self.right_projection_biases, mstype.float16)
            left_gate_weight = P.Cast()(self.left_gate_weights, mstype.float16)
            left_gate_bias = P.Cast()(self.left_gate_biases, mstype.float16)
            right_gate_weight = P.Cast()(self.right_gate_weights, mstype.float16)
            right_gate_bias = P.Cast()(self.right_gate_biases, mstype.float16)
            center_layer_norm_gamma = P.Cast()(self.center_layer_norm_gammas, mstype.float16)
            center_layer_norm_beta = P.Cast()(self.center_layer_norm_betas, mstype.float16)
            output_projection_weight = P.Cast()(self.output_projection_weights, mstype.float16)
            output_projection_bias = P.Cast()(self.output_projection_biases, mstype.float16)
            gating_linear_weight = P.Cast()(self.gating_linear_weights, mstype.float16)
            gating_linear_bias = P.Cast()(self.gating_linear_biases, mstype.float16)

        mask = P.Cast()(mask, mstype.float16)
        mask = P.ExpandDims()(mask, -1)
        act = P.Cast()(act, mstype.float32)
        act, _, _ = self.layer_norm(act,
                                    layer_norm_input_gamma,
                                    layer_norm_input_beta)
        act = P.Cast()(act, mstype.float16)

        input_act = act
        act_shape = P.Shape()(act)
        if len(act_shape) != 2:
            act = P.Reshape()(act, (-1, act_shape[-1]))
        out_shape = act_shape[:-1] + (-1,)
        # left_projection = mask * P.Reshape()(P.BiasAdd()(self.matmul(act, left_projection_weight), left_projection_bias), out_shape)
        left_projection = P.Reshape()(P.BiasAdd()(self.matmul(act, left_projection_weight), left_projection_bias),
                                      out_shape)
        act = F.depend(act, left_projection)

        left_gate_values = P.Reshape()(P.BiasAdd()(self.matmul(act, left_gate_weight), left_gate_bias), out_shape)
        left_gate_values = P.Cast()(left_gate_values, mstype.float32)
        left_gate_values = self.sigmoid(left_gate_values)
        left_gate_values = P.Cast()(left_gate_values, mstype.float16)

        left_proj_act = left_projection * left_gate_values
        act = F.depend(act, left_proj_act)
        right_projection = mask * P.Reshape()(
            P.BiasAdd()(self.matmul(act, right_projection_weight), right_projection_bias), out_shape)

        act = F.depend(act, right_projection)

        right_gate_values = P.Reshape()(P.BiasAdd()(self.matmul(act, right_gate_weight), right_gate_bias), out_shape)
        right_gate_values = P.Cast()(right_gate_values, mstype.float32)
        right_gate_values = self.sigmoid(right_gate_values)
        right_gate_values = P.Cast()(right_gate_values, mstype.float16)

        right_proj_act = right_projection * right_gate_values
        left_proj_act = F.depend(left_proj_act, right_proj_act)

        if self.equation is not None:
            if self.equation:
                left_proj_act_tmp = P.Transpose()(left_proj_act, (2, 0, 1))
                right_proj_act_tmp = P.Transpose()(right_proj_act, (2, 0, 1))
                act = self.batch_matmul_trans_b(left_proj_act_tmp, right_proj_act_tmp)
                act = P.Transpose()(act, (1, 2, 0))
            else:
                left_proj_act_tmp = P.Transpose()(left_proj_act, (2, 1, 0))
                right_proj_act_tmp = P.Transpose()(right_proj_act, (2, 1, 0))
                act = self.batch_matmul_trans_b(left_proj_act_tmp, right_proj_act_tmp)
                act = P.Transpose()(act, (2, 1, 0))

        act = P.Cast()(act, mstype.float32)
        center_layer_norm_gamma = P.Cast()(center_layer_norm_gamma, mstype.float32)
        center_layer_norm_beta = P.Cast()(center_layer_norm_beta, mstype.float32)
        act, _, _ = self.layer_norm(act,
                                    center_layer_norm_gamma,
                                    center_layer_norm_beta)
        act = P.Cast()(act, mstype.float16)
        act_shape = P.Shape()(act)

        if len(act_shape) != 2:
            act = P.Reshape()(act, (-1, act_shape[-1]))
        out_shape = act_shape[:-1] + (-1,)
        act = P.Reshape()(P.BiasAdd()(self.matmul(act, output_projection_weight), output_projection_bias), out_shape)
        input_act_shape = P.Shape()(input_act)
        if len(input_act_shape) != 2:
            input_act = P.Reshape()(input_act, (-1, input_act_shape[-1]))
        out_shape = input_act_shape[:-1] + (-1,)
        gate_values = P.Reshape()(P.BiasAdd()(self.matmul(input_act, gating_linear_weight), gating_linear_bias),
                                  out_shape)
        gate_values = P.Cast()(gate_values, mstype.float32)
        gate_values = self.sigmoid(gate_values)
        gate_values = P.Cast()(gate_values, mstype.float16)

        act = act * gate_values
        return act


class TriangleAttention(nn.Cell):
    '''triangle attention'''

    def __init__(self, config, layer_norm_dim, batch_size=None, slice_num=0):
        super(TriangleAttention, self).__init__()
        self.config = config
        self.orientation_is_per_column = (self.config.orientation == 'per_column')
        self.init_factor = Tensor(1. / np.sqrt(layer_norm_dim), mstype.float32)
        self.query_norm = P.LayerNorm(begin_norm_axis=-1, begin_params_axis=-1, epsilon=1e-5)
        self.matmul = P.MatMul(transpose_b=True)
        self.attn_mod = Attention(self.config, layer_norm_dim, layer_norm_dim, layer_norm_dim, batch_size)
        self.batch_size = batch_size
        self.slice_num = slice_num
        self.layer_norm_dim = layer_norm_dim
        self.idx = Tensor(0, mstype.int32)
        self._init_parameter()

    def _init_parameter(self):
        '''init parameter'''
        if self.batch_size:
            self.query_norm_gammas = Parameter(Tensor(np.zeros((self.batch_size, self.layer_norm_dim)), mstype.float32))
            self.query_norm_betas = Parameter(Tensor(np.zeros((self.batch_size, self.layer_norm_dim)), mstype.float32))
            self.feat_2d_weights = Parameter(
                Tensor(np.zeros((self.batch_size, self.config.num_head, self.layer_norm_dim)), mstype.float32))
        else:
            self.query_norm_gammas = Parameter(Tensor(np.ones((self.layer_norm_dim)), mstype.float32))
            self.query_norm_betas = Parameter(Tensor(np.zeros((self.layer_norm_dim)), mstype.float32))
            self.feat_2d_weights = Parameter(Tensor(
                np.random.normal(scale=1 / np.sqrt(self.layer_norm_dim),
                                 size=(self.config.num_head, self.layer_norm_dim)),
                mstype.float32))

    def construct(self, pair_act, pair_mask, index):
        '''construct'''
        if self.batch_size:
            query_norm_gamma = P.Gather()(self.query_norm_gammas, index, 0)
            query_norm_beta = P.Gather()(self.query_norm_betas, index, 0)
            feat_2d_weight = P.Cast()(P.Gather()(self.feat_2d_weights, index, 0), mstype.float16)
        else:
            query_norm_gamma = self.query_norm_gammas
            query_norm_beta = self.query_norm_betas
            feat_2d_weight = P.Cast()(self.feat_2d_weights, mstype.float16)
        if self.orientation_is_per_column:
            pair_act = P.Transpose()(pair_act, (1, 0, 2))
            pair_mask = P.Transpose()(pair_mask, (1, 0))

        pair_mask = P.Cast()(pair_mask, mstype.float32)
        pair_mask = 1e9 * (pair_mask - 1.)
        bias = P.ExpandDims()(P.ExpandDims()(pair_mask, 1), 2)

        pair_act = P.Cast()(pair_act, mstype.float32)
        pair_act, _, _ = self.query_norm(pair_act,
                                         query_norm_gamma,
                                         query_norm_beta)
        pair_act = P.Cast()(pair_act, mstype.float16)

        q, k, _ = pair_act.shape
        nonbatched_bias = self.matmul(P.Reshape()(pair_act, (-1, pair_act.shape[-1])), feat_2d_weight)
        nonbatched_bias = P.Transpose()(P.Reshape()(nonbatched_bias, (q, k, -1)), (2, 0, 1))
        if self.slice_num:
            pair_act_ori_shape = P.Shape()(pair_act)
            slice_shape = (self.slice_num, -1) + pair_act_ori_shape[1:]
            pair_act = P.Reshape()(pair_act, slice_shape).astype(mstype.float16)
            bias_shape = P.Shape()(bias)
            bias = P.Reshape()(bias, slice_shape[:2] + bias_shape[1:])

            slice_idx = 0
            slice_idx_tensor = self.idx
            pair_act_tuple = ()

            pair_act_slice = P.Gather()(pair_act, slice_idx_tensor, 0)
            bias_slice = P.Gather()(bias, slice_idx_tensor, 0)
            pair_act_slice = self.attn_mod(pair_act_slice, pair_act_slice, bias_slice, index, nonbatched_bias)
            pair_act_slice = P.Reshape()(pair_act_slice, ((1,) + P.Shape()(pair_act_slice)))
            pair_act_tuple = pair_act_tuple + (pair_act_slice,)
            slice_idx += 1
            slice_idx_tensor += 1

            while slice_idx < self.slice_num:
                pair_act_slice = P.Gather()(pair_act, slice_idx_tensor, 0)
                pair_act_slice = F.depend(pair_act_slice, pair_act_tuple[-1])
                bias_slice = P.Gather()(bias, slice_idx_tensor, 0)
                pair_act_slice = self.attn_mod(pair_act_slice, pair_act_slice, bias_slice, index, nonbatched_bias)
                pair_act_slice = P.Reshape()(pair_act_slice, ((1,) + P.Shape()(pair_act_slice)))
                pair_act_tuple = pair_act_tuple + (pair_act_slice,)
                slice_idx += 1
                slice_idx_tensor += 1
            pair_act = P.Concat()(pair_act_tuple)
            pair_act = P.Reshape()(pair_act, pair_act_ori_shape)

            if self.orientation_is_per_column:
                pair_act = mnp.swapaxes(pair_act, -2, -3)
            return pair_act

        pair_act = self.attn_mod(pair_act, pair_act, bias, index, nonbatched_bias)
        if self.orientation_is_per_column:
            pair_act = P.Transpose()(pair_act, (1, 0, 2))
        return pair_act
