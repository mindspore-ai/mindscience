# ============================================================================
# Copyright 2023 Huawei Technologies Co., Ltd
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
"""attention module"""
import numpy as np

import mindspore as ms
from mindspore import ops, nn
from mindspore.ops.function import broadcast_to
from mindspore import numpy as ms_np

from .utils import mask_fill


class FullAttention(nn.Cell):
    """Multi-headed attention with input/output transformations"""
    def __init__(self, mask_flag=True, scale=None, attention_dropout=0.1,
                 output_attention=False, args=None, d_value=64):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(keep_prob=1-attention_dropout)
        self.sqrt = ms_np.sqrt
        self.d_value = d_value
        self.ceof = 0. if args is None else args.ceof

    def construct(self, queries, keys, values, attn_mask):
        """Apply attention mechanism on queries, keys and values."""
        b, l, h, _ = queries.shape
        scores = ops.BatchMatMul()(queries.transpose(0, 2, 1, 3), keys.transpose(0, 2, 3, 1))
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = ops.stop_gradient(nn.Triu()(ops.ones((b, h, l, l), dtype=ms.bool_), k=1))
            scores = mask_fill(attn_mask, scores, -np.inf)
        if self.scale is None:
            a = self.dropout(ops.Softmax()((1./self.sqrt(self.d_value)) * scores))
        else:
            a = self.dropout(ops.Softmax()(self.scale * scores))
        value = ops.BatchMatMul()(a, values.transpose((0, 2, 1, 3))).transpose(0, 2, 1, 3)
        return value


class ProbAttention(nn.Cell):
    """Probabilistic attention module, and approximate top_k"""
    def __init__(self, mask_flag=False, factor=5, scale=None, attention_dropout=0.1,
                 output_attention=False, args=None, d_value=64):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(keep_prob=1-attention_dropout)
        self.sqrt = ms_np.sqrt
        self.ceof = 0.5
        self.args = args
        self.d_value = d_value
        self.attn_mask = None

    def _prob_qk(self, query, key, sample_k, n_top): # n_top: c*ln(L_q)
        """Calculate probabilistic sparse attention"""
        # Q [B, H, L, D]
        b, h, lk, e = key.shape
        _, _, lq, _ = query.shape
        # calculate the sampled Q_K
        k_expand = ops.BroadcastTo(shape=(b, h, lq, lk, e))(ops.expand_dims(key, -3))
        index_sample = ops.UniformInt()((lq, sample_k), ms.Tensor(0, ms.int32), ms.Tensor(lk, ms.int32)) # real U = U_part(factor*ln(L_k))*L_q
        k_sample = k_expand[:, :, ops.expand_dims(ms.numpy.arange(lq), 1), index_sample, :]
        q_k_sample = ops.Squeeze(-2)(ops.BatchMatMul()(ops.expand_dims(query, -2), k_sample.swapaxes(-2, -1)))
        # find the Top_k query with sparisty measurement
        m = ops.ArgMaxWithValue(-1)(q_k_sample)[0] - ops.div(ops.ReduceSum()(q_k_sample, -1), lk)

        m_top = ops.TopK(sorted=False)(m, n_top)[1]
         # use the reduced Q to calculate Q_K
        q_reduce = query[ms.numpy.arange(b)[:, None, None],
                         ms.numpy.arange(h)[None, :, None],
                         m_top, :] # factor*ln(L_q)
        q_k = ops.BatchMatMul()(q_reduce, key.swapaxes(-2, -1)) # factor*ln(L_q)*L_k

        return q_k, m_top

    def _get_initial_context(self, value, lq):
        b, h, lv, _ = value.shape
        if not self.mask_flag:
            v_sum = ops.ReduceMean()(value, -2)
            context = ops.BroadcastTo(shape=(b, h, lq, v_sum.shape[-1]))(ops.expand_dims(v_sum, -2)).copy()
        else: # use mask
            assert lq == lv # requires that L_Q == L_V, i.e. for self-attention only
            context = ops.cumsum(value, -2)
        return context

    def _update_context(self, context_in, value, scores, index, lq):
        """Update context based on value and attention index"""
        b, h, lv, _ = value.shape
        if self.mask_flag:
            mask = ms.numpy.triu(ops.ones((lq, scores.shape[-1]), ms.bool_))
            mask_ex = broadcast_to(mask[None, None, :], (b, h, lq, scores.shape[-1]))
            indicator = mask_ex[ms_np.arange(b)[:, None, None],
                                ms_np.arange(h)[None, :, None],
                                index, :]
            final_mask = indicator.view(scores.shape)

            scores = mask_fill(final_mask, scores, -np.inf)

        attn = ops.Softmax()(scores)
        context_in[ms.numpy.arange(b)[:, None, None],
                   ms.numpy.arange(h)[None, :, None],
                   index, :] = ops.BatchMatMul()(attn, value).astype(context_in.dtype)

        attns = (ops.ones(((b, h, lv, lv)), ms.float32) / lv).astype(attn.dtype)
        attns[ms.numpy.arange(b)[:, None, None], ms.numpy.arange(h)[None, :, None], index, :] = attn
        return context_in

    def construct(self, queries, keys, values, attn_mask):
        """Probabilistic attention construct function based on queries, keys and values."""
        _, lq, _, _ = queries.shape
        _, lk, _, _ = keys.shape
        queries = queries.swapaxes(2, 1)
        keys = keys.swapaxes(2, 1)
        values = values.swapaxes(2, 1)
        self.attn_mask = attn_mask

        u_part = self.factor * np.ceil(np.log(lk)).astype('int').item() # c*ln(L_k)
        u = self.factor * np.ceil(np.log(lq)).astype('int').item() # c*ln(L_q)

        u_part = u_part if u_part < lk else lk
        u = u if u < lq else lq

        scores_top, index = self._prob_qk(queries, keys, sample_k=u_part, n_top=u)

        # add scale factor
        if self.scale is None:
            scores_top = scores_top * (1./self.sqrt(self.D))
        else:
            scores_top = scores_top * self.scale
        # get the context
        context = self._get_initial_context(values, lq)
        # update the context with selected top_k queries
        context = self._update_context(context, values, scores_top, index, lq)

        return context.swapaxes(2, 1)


class AttentionLayer(nn.Cell):
    """Multi-head attention layer."""
    def __init__(self, attention, d_model, n_heads,
                 d_keys=None, d_values=None, mix=False):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Dense(d_model, d_keys * n_heads)
        self.key_projection = nn.Dense(d_model, d_keys * n_heads)
        self.value_projection = nn.Dense(d_model, d_values * n_heads)
        self.out_projection = nn.Dense(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix

    def construct(self, queries, keys, values, attn_mask):
        """Forward function of attention layer"""
        b, l, _ = queries.shape
        _, s, _ = keys.shape
        h = self.n_heads
        queries = self.query_projection(queries).view(b, l, h, -1)
        keys = self.key_projection(keys).view(b, s, h, -1)
        values = self.value_projection(values).view(b, s, h, -1)
        out = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
        )
        if self.mix:
            out = out.swapaxes(2, 1)
        out = out.view(b, l, -1)
        return self.out_projection(out)
