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
"""MSA"""
import numpy as np
import mindspore.numpy as mnp
import mindspore.nn as nn
import mindspore.common.dtype as mstype
from mindspore import Parameter
from mindspore.common.tensor import Tensor
from mindspore.ops import operations as P
from mindspore.ops import function as F
from .basic import Attention, GlobalAttention


class MSARowAttentionWithPairBias(nn.Cell):
    '''MSA row attention'''

    def __init__(self, num_head, key_dim, gating, msa_act_dim, pair_act_dim, batch_size=None, slice_num=0):
        super(MSARowAttentionWithPairBias, self).__init__()
        self.num_head = num_head
        self.batch_size = batch_size
        self.norm = P.LayerNorm(begin_norm_axis=-1, begin_params_axis=-1, epsilon=1e-5)
        self.matmul = P.MatMul(transpose_b=True)
        self.attn_mod = Attention(num_head, key_dim, gating, msa_act_dim, msa_act_dim, msa_act_dim, batch_size)
        self.msa_act_dim = msa_act_dim
        self.pair_act_dim = pair_act_dim
        self.batch_size = batch_size
        self.slice_num = slice_num
        self.idx = Tensor(0, mstype.int32)
        self._init_parameter()

    def construct(self, msa_act, msa_mask, pair_act, index):
        '''construct'''
        if self.batch_size:
            query_norm_gamma = P.Gather()(self.query_norm_gammas, index, 0)
            query_norm_beta = P.Gather()(self.query_norm_betas, index, 0)
            feat_2d_norm_gamma = P.Gather()(self.feat_2d_norm_gammas, index, 0)
            feat_2d_norm_beta = P.Gather()(self.feat_2d_norm_betas, index, 0)
            feat_2d_weight = P.Gather()(self.feat_2d_weights, index, 0)
        else:
            query_norm_gamma = self.query_norm_gammas
            query_norm_beta = self.query_norm_betas
            feat_2d_norm_gamma = self.feat_2d_norm_gammas
            feat_2d_norm_beta = self.feat_2d_norm_betas
            feat_2d_weight = self.feat_2d_weights

        q, k, _ = pair_act.shape
        input_mask = 1e9 * (msa_mask - 1.0)
        input_mask = P.ExpandDims()(P.ExpandDims()(input_mask, 1), 2)

        msa_act, _, _ = self.norm(msa_act, query_norm_gamma, query_norm_beta)
        pair_act, _, _ = self.norm(pair_act, feat_2d_norm_gamma, feat_2d_norm_beta)
        pair_act = P.Reshape()(pair_act, (-1, pair_act.shape[-1]))
        nonbatched_bias = P.Transpose()(P.Reshape()(self.matmul(pair_act, feat_2d_weight), (q, k, self.num_head)),
                                        (2, 0, 1))

        if self.slice_num:
            msa_act_ori_shape = P.Shape()(msa_act)
            slice_shape = (self.slice_num, -1) + msa_act_ori_shape[1:]
            msa_act = P.Reshape()(msa_act, slice_shape)
            input_mask_shape = P.Shape()(input_mask)
            input_mask = P.Reshape()(input_mask, slice_shape[:2] + input_mask_shape[1:])
            slice_idx = 0
            slice_idx_tensor = self.idx
            msa_act_tuple = ()

            msa_act_slice = P.Gather()(msa_act, slice_idx_tensor, 0)
            input_mask_slice = P.Gather()(input_mask, slice_idx_tensor, 0)
            msa_act_slice = self.attn_mod(msa_act_slice, msa_act_slice, input_mask_slice,
                                          index, nonbatched_bias)
            msa_act_slice = P.Reshape()(msa_act_slice, ((1,) + P.Shape()(msa_act_slice)))
            msa_act_tuple = msa_act_tuple + (msa_act_slice,)
            slice_idx += 1
            slice_idx_tensor += 1

            while slice_idx < self.slice_num:
                msa_act_slice = P.Gather()(msa_act, slice_idx_tensor, 0)
                msa_act_slice = F.depend(msa_act_slice, msa_act_tuple[-1])
                input_mask_slice = P.Gather()(input_mask, slice_idx_tensor, 0)
                msa_act_slice = self.attn_mod(msa_act_slice, msa_act_slice, input_mask_slice, index, nonbatched_bias)
                msa_act_slice = P.Reshape()(msa_act_slice, ((1,) + P.Shape()(msa_act_slice)))
                msa_act_tuple = msa_act_tuple + (msa_act_slice,)
                slice_idx += 1
                slice_idx_tensor += 1

            msa_act = P.Concat()(msa_act_tuple)
            msa_act = P.Reshape()(msa_act, msa_act_ori_shape)
            return msa_act

        msa_act = self.attn_mod(msa_act, msa_act, input_mask, index, nonbatched_bias)
        return msa_act

    def _init_parameter(self):
        '''init parameter'''
        if self.batch_size:
            self.query_norm_gammas = Parameter(Tensor(np.zeros([self.batch_size, self.msa_act_dim,]), mstype.float32))
            self.query_norm_betas = Parameter(Tensor(np.zeros([self.batch_size, self.msa_act_dim,]), mstype.float32))
            self.feat_2d_norm_gammas = Parameter(
                Tensor(np.zeros([self.batch_size, self.pair_act_dim,]), mstype.float32))
            self.feat_2d_norm_betas = Parameter(
                Tensor(np.zeros([self.batch_size, self.pair_act_dim,]), mstype.float32))
            self.feat_2d_weights = Parameter(
                Tensor(np.zeros([self.batch_size, self.num_head, self.pair_act_dim]), mstype.float32))
        else:
            self.query_norm_gammas = Parameter(Tensor(np.ones([self.msa_act_dim,]), mstype.float32))
            self.query_norm_betas = Parameter(Tensor(np.zeros([self.msa_act_dim,]), mstype.float32))
            self.feat_2d_norm_gammas = Parameter(Tensor(np.ones([self.pair_act_dim,]), mstype.float32))
            self.feat_2d_norm_betas = Parameter(Tensor(np.zeros([self.pair_act_dim,]), mstype.float32))
            self.feat_2d_weights = Parameter(
                Tensor(np.random.normal(scale=1 / np.sqrt(self.pair_act_dim), size=[self.num_head, self.pair_act_dim]),
                       mstype.float32))


class MSAColumnAttention(nn.Cell):
    '''MSA column attention'''

    def __init__(self, num_head, key_dim, gating, msa_act_dim, batch_size=None, slice_num=0):
        super(MSAColumnAttention, self).__init__()
        self.query_norm = P.LayerNorm(begin_norm_axis=-1, begin_params_axis=-1, epsilon=1e-5)
        self.attn_mod = Attention(num_head, key_dim, gating, msa_act_dim, msa_act_dim, msa_act_dim, batch_size)
        self.batch_size = batch_size
        self.slice_num = slice_num
        self.msa_act_dim = msa_act_dim
        self.idx = Tensor(0, mstype.int32)
        self._init_parameter()

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

        input_mask = 1e9 * (msa_mask - 1.)
        input_mask = P.ExpandDims()(P.ExpandDims()(input_mask, 1), 2)
        msa_act, _, _ = self.query_norm(msa_act, query_norm_gamma, query_norm_beta)

        if self.slice_num:
            msa_act_ori_shape = P.Shape()(msa_act)
            slice_shape = (self.slice_num, -1) + msa_act_ori_shape[1:]
            msa_act = P.Reshape()(msa_act, slice_shape)
            input_mask_shape = P.Shape()(input_mask)
            input_mask = P.Reshape()(input_mask, slice_shape[:2] + input_mask_shape[1:])

            slice_idx = 0
            slice_idx_tensor = self.idx
            msa_act_tuple = ()

            msa_act_slice = P.Gather()(msa_act, slice_idx_tensor, 0)
            input_mask_slice = P.Gather()(input_mask, slice_idx_tensor, 0)
            msa_act_slice = self.attn_mod(msa_act_slice, msa_act_slice, input_mask_slice, index)
            msa_act_slice = P.Reshape()(msa_act_slice, ((1,) + P.Shape()(msa_act_slice)))
            msa_act_tuple = msa_act_tuple + (msa_act_slice,)
            slice_idx += 1
            slice_idx_tensor += 1

            while slice_idx < self.slice_num:
                msa_act_slice = P.Gather()(msa_act, slice_idx_tensor, 0)
                msa_act_slice = F.depend(msa_act_slice, msa_act_tuple[-1])
                input_mask_slice = P.Gather()(input_mask, slice_idx_tensor, 0)
                msa_act_slice = self.attn_mod(msa_act_slice, msa_act_slice, input_mask_slice, index)
                msa_act_slice = P.Reshape()(msa_act_slice, ((1,) + P.Shape()(msa_act_slice)))
                msa_act_tuple = msa_act_tuple + (msa_act_slice,)
                slice_idx += 1
                slice_idx_tensor += 1

            msa_act = P.Concat()(msa_act_tuple)
            msa_act = P.Reshape()(msa_act, msa_act_ori_shape)
            msa_act = mnp.swapaxes(msa_act, -2, -3)
            return msa_act

        msa_act = self.attn_mod(msa_act, msa_act, input_mask, index)
        msa_act = P.Transpose()(msa_act, (1, 0, 2))
        return msa_act

    def _init_parameter(self):
        if self.batch_size:
            self.query_norm_gammas = Parameter(Tensor(np.zeros([self.batch_size, self.msa_act_dim]), mstype.float32))
            self.query_norm_betas = Parameter(Tensor(np.zeros([self.batch_size, self.msa_act_dim]), mstype.float32))
        else:
            self.query_norm_gammas = Parameter(Tensor(np.ones([self.msa_act_dim]), mstype.float32))
            self.query_norm_betas = Parameter(Tensor(np.zeros([self.msa_act_dim]), mstype.float32))


class MSAColumnGlobalAttention(nn.Cell):
    '''MSA column global attention'''

    def __init__(self, num_head, gating, msa_act_dim, batch_size=None, slice_num=0):
        super(MSAColumnGlobalAttention, self).__init__()
        self.attn_mod = GlobalAttention(num_head, gating, msa_act_dim, msa_act_dim, batch_size)
        self.query_norm = P.LayerNorm(begin_norm_axis=-1, begin_params_axis=-1, epsilon=1e-5)
        self.batch_size = batch_size
        self.slice_num = slice_num
        self.msa_act_dim = msa_act_dim
        self.idx = Tensor(0, mstype.int32)
        self._init_parameter()

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

        input_mask = 1e9 * (msa_mask - 1.)
        input_mask = P.ExpandDims()(P.ExpandDims()(input_mask, 1), 2)

        msa_act, _, _ = self.query_norm(msa_act,
                                        query_norm_gamma,
                                        query_norm_beta)
        msa_mask = P.ExpandDims()(msa_mask, -1)

        if self.slice_num:
            msa_act_ori_shape = P.Shape()(msa_act)
            slice_shape = (self.slice_num, -1) + msa_act_ori_shape[1:]
            msa_act = P.Reshape()(msa_act, slice_shape)
            input_mask_shape = P.Shape()(input_mask)
            input_mask = P.Reshape()(input_mask, slice_shape[:2] + input_mask_shape[1:])
            msa_mask_shape = P.Shape()(msa_mask)
            msa_mask = P.Reshape()(msa_mask, slice_shape[:2] + msa_mask_shape[1:])

            slice_idx = 0
            slice_idx_tensor = self.idx
            msa_act_tuple = ()

            msa_act_slice = P.Gather()(msa_act, slice_idx_tensor, 0)
            msa_mask_slice = P.Gather()(msa_mask, slice_idx_tensor, 0)
            input_mask_slice = P.Gather()(input_mask, slice_idx_tensor, 0)
            msa_act_slice = self.attn_mod(msa_act_slice, msa_act_slice, msa_mask_slice,
                                          input_mask_slice, index)
            msa_act_slice = P.Reshape()(msa_act_slice, ((1,) + P.Shape()(msa_act_slice)))
            msa_act_tuple = msa_act_tuple + (msa_act_slice,)
            slice_idx += 1
            slice_idx_tensor += 1

            while slice_idx < self.slice_num:
                msa_act_slice = P.Gather()(msa_act, slice_idx_tensor, 0)
                msa_act_slice = F.depend(msa_act_slice, msa_act_tuple[-1])
                msa_mask_slice = P.Gather()(msa_mask, slice_idx_tensor, 0)
                input_mask_slice = P.Gather()(input_mask, slice_idx_tensor, 0)

                msa_act_slice = self.attn_mod(msa_act_slice, msa_act_slice, msa_mask_slice,
                                              input_mask_slice, index)
                msa_act_slice = P.Reshape()(msa_act_slice, ((1,) + P.Shape()(msa_act_slice)))
                msa_act_tuple = msa_act_tuple + (msa_act_slice,)
                slice_idx += 1
                slice_idx_tensor += 1

            msa_act = P.Concat()(msa_act_tuple)
            msa_act = P.Reshape()(msa_act, msa_act_ori_shape)
            msa_act = mnp.swapaxes(msa_act, -2, -3)
            return msa_act

        msa_act = self.attn_mod(msa_act, msa_act, msa_mask, input_mask, index)
        msa_act = P.Transpose()(msa_act, (1, 0, 2))
        return msa_act

    def _init_parameter(self):
        '''init parameter'''
        if self.batch_size:
            self.query_norm_gammas = Parameter(Tensor(np.zeros((self.batch_size, self.msa_act_dim)), mstype.float32))
            self.query_norm_betas = Parameter(Tensor(np.zeros((self.batch_size, self.msa_act_dim)), mstype.float32))
        else:
            self.query_norm_gammas = Parameter(Tensor(np.ones((self.msa_act_dim)), mstype.float32))
            self.query_norm_betas = Parameter(Tensor(np.zeros((self.msa_act_dim)), mstype.float32))
