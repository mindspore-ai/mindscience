# Modified from RFdiffusion (https://github.com/RosettaCommons/RFdiffusion)
# Original license: BSD License
#
# Copyright 2025 Huawei Technologies Co., Ltd
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

import math

import mindspore as ms
from mindspore import nn, ops
from mindspore.common.initializer import HeNormal, One, XavierUniform, Zero, initializer

from .util_module import init_lecun_normal


class FeedForwardLayer(nn.Cell):
    def __init__(self, d_model, r_ff, p_drop=0.1):
        super(FeedForwardLayer, self).__init__()
        self.norm = nn.LayerNorm((d_model,), epsilon=1e-5)
        self.linear1 = nn.Linear(d_model, d_model * r_ff)
        self.dropout = nn.Dropout(p=p_drop)
        self.linear2 = nn.Linear(d_model * r_ff, d_model)

        self.reset_parameter()

    def reset_parameter(self):
        # initialize linear layer right before ReLu: He initializer (kaiming normal)
        self.linear1.weight.set_data(
            initializer(
                HeNormal(nonlinearity="relu"),
                self.linear1.weight.shape,
                self.linear1.weight.dtype,
            )
        )
        self.linear1.bias.set_data(
            initializer(Zero(), self.linear1.bias.shape, self.linear1.bias.dtype)
        )

        # initialize linear layer right before residual connection: zero initialize
        self.linear2.weight.set_data(
            initializer(Zero(), self.linear2.weight.shape, self.linear2.weight.dtype)
        )
        self.linear2.bias.set_data(
            initializer(Zero(), self.linear2.bias.shape, self.linear2.bias.dtype)
        )

    def construct(self, src):
        src = self.norm(src)
        src = self.linear2(self.dropout(ops.relu(self.linear1(src), inplace=True)))
        return src


class Attention(nn.Cell):
    # calculate multi-head attention
    def __init__(self, d_query, d_key, n_head, d_hidden, d_out):
        super(Attention, self).__init__()
        self.h = n_head
        self.dim = d_hidden

        self.to_q = nn.Linear(d_query, n_head * d_hidden, bias=False)
        self.to_k = nn.Linear(d_key, n_head * d_hidden, bias=False)
        self.to_v = nn.Linear(d_key, n_head * d_hidden, bias=False)

        self.to_out = nn.Linear(n_head * d_hidden, d_out)
        self.scaling = 1 / math.sqrt(d_hidden)

        # initialize all parameters properly
        self.reset_parameter()

    def reset_parameter(self):
        # query/key/value projection: Glorot uniform / Xavier uniform
        self.to_q.weight.set_data(
            initializer(XavierUniform(), self.to_q.weight.shape, self.to_q.weight.dtype)
        )
        self.to_k.weight.set_data(
            initializer(XavierUniform(), self.to_k.weight.shape, self.to_k.weight.dtype)
        )
        self.to_v.weight.set_data(
            initializer(XavierUniform(), self.to_v.weight.shape, self.to_v.weight.dtype)
        )

        # to_out: right before residual connection: zero initialize -- to make it sure residual operation is same to the Identity at the beginning
        self.to_out.weight.set_data(
            initializer(Zero(), self.to_out.weight.shape, self.to_out.weight.dtype)
        )
        self.to_out.bias.set_data(
            initializer(Zero(), self.to_out.bias.shape, self.to_out.bias.dtype)
        )

    def construct(self, query, key, value):
        B, Q = query.shape[:2]
        B, K = key.shape[:2]

        query = self.to_q(query).reshape(B, Q, self.h, self.dim)
        key = self.to_k(key).reshape(B, K, self.h, self.dim)
        value = self.to_v(value).reshape(B, K, self.h, self.dim)

        query = query * self.scaling
        attn = ms.mint.einsum("bqhd,bkhd->bhqk", query, key)
        attn = ops.softmax(attn, axis=-1)

        out = ms.mint.einsum("bhqk,bkhd->bqhd", attn, value)
        out = out.reshape(B, Q, self.h * self.dim)

        out = self.to_out(out)

        return out


class AttentionWithBias(nn.Cell):
    def __init__(self, d_in=256, d_bias=128, n_head=8, d_hidden=32):
        super(AttentionWithBias, self).__init__()
        self.norm_in = nn.LayerNorm((d_in,), epsilon=1e-5)
        self.norm_bias = nn.LayerNorm((d_bias,), epsilon=1e-5)

        self.to_q = nn.Linear(d_in, n_head * d_hidden, bias=False)
        self.to_k = nn.Linear(d_in, n_head * d_hidden, bias=False)
        self.to_v = nn.Linear(d_in, n_head * d_hidden, bias=False)
        self.to_b = nn.Linear(d_bias, n_head, bias=False)
        self.to_g = nn.Linear(d_in, n_head * d_hidden)
        self.to_out = nn.Linear(n_head * d_hidden, d_in)

        self.scaling = 1 / math.sqrt(d_hidden)
        self.h = n_head
        self.dim = d_hidden

        self.reset_parameter()

    def reset_parameter(self):
        # query/key/value projection: Glorot uniform / Xavier uniform
        self.to_q.weight.set_data(
            initializer(XavierUniform(), self.to_q.weight.shape, self.to_q.weight.dtype)
        )
        self.to_k.weight.set_data(
            initializer(XavierUniform(), self.to_k.weight.shape, self.to_k.weight.dtype)
        )
        self.to_v.weight.set_data(
            initializer(XavierUniform(), self.to_v.weight.shape, self.to_v.weight.dtype)
        )

        # bias: normal distribution
        self.to_b = init_lecun_normal(self.to_b)

        # gating: zero weights, one biases (mostly open gate at the beginning)
        self.to_g.weight.set_data(
            initializer(Zero(), self.to_g.weight.shape, self.to_g.weight.dtype)
        )
        self.to_g.bias.set_data(
            initializer(One(), self.to_g.bias.shape, self.to_g.bias.dtype)
        )

        # to_out: right before residual connection: zero initialize -- to make it sure residual operation is same to the Identity at the beginning
        self.to_out.weight.set_data(
            initializer(Zero(), self.to_out.weight.shape, self.to_out.weight.dtype)
        )
        self.to_out.bias.set_data(
            initializer(Zero(), self.to_out.bias.shape, self.to_out.bias.dtype)
        )

    def construct(self, x, bias):
        B, L = x.shape[:2]

        x = self.norm_in(x)
        bias = self.norm_bias(bias)

        query = self.to_q(x).reshape(B, L, self.h, self.dim)
        key = self.to_k(x).reshape(B, L, self.h, self.dim)
        value = self.to_v(x).reshape(B, L, self.h, self.dim)
        bias = self.to_b(bias)  # (B, L, L, h)
        gate = ops.sigmoid(self.to_g(x))

        key = key * self.scaling
        attn = ms.mint.einsum("bqhd,bkhd->bqkh", query, key)
        attn = attn + bias
        attn = ops.softmax(attn, axis=-2)

        out = ms.mint.einsum("bqkh,bkhd->bqhd", attn, value).reshape(B, L, -1)
        out = gate * out

        out = self.to_out(out)
        return out


# MSA Attention (row/column) from AlphaFold architecture
class SequenceWeight(nn.Cell):
    def __init__(self, d_msa, n_head, d_hidden, p_drop=0.1):
        super(SequenceWeight, self).__init__()
        self.h = n_head
        self.dim = d_hidden
        self.scale = 1.0 / math.sqrt(self.dim)

        self.to_query = nn.Linear(d_msa, n_head * d_hidden)
        self.to_key = nn.Linear(d_msa, n_head * d_hidden)
        self.dropout = nn.Dropout(p=p_drop)

        self.reset_parameter()

    def reset_parameter(self):
        # query/key/value projection: Glorot uniform / Xavier uniform
        self.to_query.weight.set_data(
            initializer(
                XavierUniform(), self.to_query.weight.shape, self.to_query.weight.dtype
            )
        )
        self.to_key.weight.set_data(
            initializer(
                XavierUniform(), self.to_key.weight.shape, self.to_key.weight.dtype
            )
        )

    def construct(self, msa):
        B, N, L = msa.shape[:3]

        tar_seq = msa[:, 0]

        q = self.to_query(tar_seq).view(B, 1, L, self.h, self.dim)
        k = self.to_key(msa).view(B, N, L, self.h, self.dim)

        q = q * self.scale
        attn = ms.mint.einsum("bqihd,bkihd->bkihq", q, k)
        attn = ops.softmax(attn, axis=1)
        return self.dropout(attn)


class MSARowAttentionWithBias(nn.Cell):
    def __init__(self, d_msa=256, d_pair=128, n_head=8, d_hidden=32):
        super(MSARowAttentionWithBias, self).__init__()
        self.norm_msa = nn.LayerNorm((d_msa,), epsilon=1e-5)
        self.norm_pair = nn.LayerNorm((d_pair,), epsilon=1e-5)

        self.seq_weight = SequenceWeight(d_msa, n_head, d_hidden, p_drop=0.1)
        self.to_q = nn.Linear(d_msa, n_head * d_hidden, bias=False)
        self.to_k = nn.Linear(d_msa, n_head * d_hidden, bias=False)
        self.to_v = nn.Linear(d_msa, n_head * d_hidden, bias=False)
        self.to_b = nn.Linear(d_pair, n_head, bias=False)
        self.to_g = nn.Linear(d_msa, n_head * d_hidden)
        self.to_out = nn.Linear(n_head * d_hidden, d_msa)

        self.scaling = 1 / math.sqrt(d_hidden)
        self.h = n_head
        self.dim = d_hidden

        self.reset_parameter()

    def reset_parameter(self):
        # query/key/value projection: Glorot uniform / Xavier uniform
        self.to_q.weight.set_data(
            initializer(XavierUniform(), self.to_q.weight.shape, self.to_q.weight.dtype)
        )
        self.to_k.weight.set_data(
            initializer(XavierUniform(), self.to_k.weight.shape, self.to_k.weight.dtype)
        )
        self.to_v.weight.set_data(
            initializer(XavierUniform(), self.to_v.weight.shape, self.to_v.weight.dtype)
        )

        # bias: normal distribution
        self.to_b = init_lecun_normal(self.to_b)

        # gating: zero weights, one biases (mostly open gate at the beginning)
        self.to_g.weight.set_data(
            initializer(Zero(), self.to_g.weight.shape, self.to_g.weight.dtype)
        )
        self.to_g.bias.set_data(
            initializer(One(), self.to_g.bias.shape, self.to_g.bias.dtype)
        )

        # to_out: right before residual connection: zero initialize -- to make it sure residual operation is same to the Identity at the beginning
        self.to_out.weight.set_data(
            initializer(Zero(), self.to_out.weight.shape, self.to_out.weight.dtype)
        )
        self.to_out.bias.set_data(
            initializer(Zero(), self.to_out.bias.shape, self.to_out.bias.dtype)
        )

    def construct(self, msa, pair):  # TODO: make this as tied-attention
        B, N, L = msa.shape[:3]

        msa = self.norm_msa(msa)
        pair = self.norm_pair(pair)

        seq_weight = self.seq_weight(msa)  # (B, N, L, h, 1)
        query = self.to_q(msa).reshape(B, N, L, self.h, self.dim)
        key = self.to_k(msa).reshape(B, N, L, self.h, self.dim)
        value = self.to_v(msa).reshape(B, N, L, self.h, self.dim)
        bias = self.to_b(pair)  # (B, L, L, h)
        gate = ops.sigmoid(self.to_g(msa))

        query = query * seq_weight.expand((-1, -1, -1, -1, self.dim))
        key = key * self.scaling
        attn = ms.mint.einsum("bsqhd,bskhd->bqkh", query, key)
        attn = attn + bias
        attn = ops.softmax(attn, axis=-2)

        out = ms.mint.einsum("bqkh,bskhd->bsqhd", attn, value).reshape(B, N, L, -1)
        out = gate * out

        out = self.to_out(out)
        return out


class MSAColAttention(nn.Cell):
    def __init__(self, d_msa=256, n_head=8, d_hidden=32):
        super(MSAColAttention, self).__init__()
        self.norm_msa = nn.LayerNorm((d_msa,), epsilon=1e-5)

        self.to_q = nn.Linear(d_msa, n_head * d_hidden, bias=False)
        self.to_k = nn.Linear(d_msa, n_head * d_hidden, bias=False)
        self.to_v = nn.Linear(d_msa, n_head * d_hidden, bias=False)
        self.to_g = nn.Linear(d_msa, n_head * d_hidden)
        self.to_out = nn.Linear(n_head * d_hidden, d_msa)

        self.scaling = 1 / math.sqrt(d_hidden)
        self.h = n_head
        self.dim = d_hidden

        self.reset_parameter()

    def reset_parameter(self):
        # query/key/value projection: Glorot uniform / Xavier uniform
        self.to_q.weight.set_data(
            initializer(XavierUniform(), self.to_q.weight.shape, self.to_q.weight.dtype)
        )
        self.to_k.weight.set_data(
            initializer(XavierUniform(), self.to_k.weight.shape, self.to_k.weight.dtype)
        )
        self.to_v.weight.set_data(
            initializer(XavierUniform(), self.to_v.weight.shape, self.to_v.weight.dtype)
        )

        # gating: zero weights, one biases (mostly open gate at the beginning)
        self.to_g.weight.set_data(
            initializer(Zero(), self.to_g.weight.shape, self.to_g.weight.dtype)
        )
        self.to_g.bias.set_data(
            initializer(One(), self.to_g.bias.shape, self.to_g.bias.dtype)
        )

        # to_out: right before residual connection: zero initialize -- to make it sure residual operation is same to the Identity at the beginning
        self.to_out.weight.set_data(
            initializer(Zero(), self.to_out.weight.shape, self.to_out.weight.dtype)
        )
        self.to_out.bias.set_data(
            initializer(Zero(), self.to_out.bias.shape, self.to_out.bias.dtype)
        )

    def construct(self, msa):
        B, N, L = msa.shape[:3]

        msa = self.norm_msa(msa)

        query = self.to_q(msa).reshape(B, N, L, self.h, self.dim)
        key = self.to_k(msa).reshape(B, N, L, self.h, self.dim)
        value = self.to_v(msa).reshape(B, N, L, self.h, self.dim)
        gate = ops.sigmoid(self.to_g(msa))

        query = query * self.scaling
        attn = ms.mint.einsum("bqihd,bkihd->bihqk", query, key)
        attn = ops.softmax(attn, axis=-1)

        out = ms.mint.einsum("bihqk,bkihd->bqihd", attn, value).reshape(B, N, L, -1)
        out = gate * out

        out = self.to_out(out)
        return out


class MSAColGlobalAttention(nn.Cell):
    def __init__(self, d_msa=64, n_head=8, d_hidden=8):
        super(MSAColGlobalAttention, self).__init__()
        self.norm_msa = nn.LayerNorm((d_msa,), epsilon=1e-5)

        self.to_q = nn.Linear(d_msa, n_head * d_hidden, bias=False)
        self.to_k = nn.Linear(d_msa, d_hidden, bias=False)
        self.to_v = nn.Linear(d_msa, d_hidden, bias=False)
        self.to_g = nn.Linear(d_msa, n_head * d_hidden)
        self.to_out = nn.Linear(n_head * d_hidden, d_msa)

        self.scaling = 1 / math.sqrt(d_hidden)
        self.h = n_head
        self.dim = d_hidden

        self.reset_parameter()

    def reset_parameter(self):
        # query/key/value projection: Glorot uniform / Xavier uniform
        self.to_q.weight.set_data(
            initializer(XavierUniform(), self.to_q.weight.shape, self.to_q.weight.dtype)
        )
        self.to_k.weight.set_data(
            initializer(XavierUniform(), self.to_k.weight.shape, self.to_k.weight.dtype)
        )
        self.to_v.weight.set_data(
            initializer(XavierUniform(), self.to_v.weight.shape, self.to_v.weight.dtype)
        )

        # gating: zero weights, one biases (mostly open gate at the beginning)
        self.to_g.weight.set_data(
            initializer(Zero(), self.to_g.weight.shape, self.to_g.weight.dtype)
        )
        self.to_g.bias.set_data(
            initializer(One(), self.to_g.bias.shape, self.to_g.bias.dtype)
        )

        # to_out: right before residual connection: zero initialize -- to make it sure residual operation is same to the Identity at the beginning
        self.to_out.weight.set_data(
            initializer(Zero(), self.to_out.weight.shape, self.to_out.weight.dtype)
        )
        self.to_out.bias.set_data(
            initializer(Zero(), self.to_out.bias.shape, self.to_out.bias.dtype)
        )

    def construct(self, msa):
        B, N, L = msa.shape[:3]

        msa = self.norm_msa(msa)

        query = self.to_q(msa).reshape(B, N, L, self.h, self.dim)
        query = query.mean(dim=1)  # (B, L, h, dim)
        key = self.to_k(msa)  # (B, N, L, dim)
        value = self.to_v(msa)  # (B, N, L, dim)
        gate = ops.sigmoid(self.to_g(msa))  # (B, N, L, h*dim)

        query = query * self.scaling
        attn = ms.mint.einsum("bihd,bkid->bihk", query, key)  # (B, L, h, N)
        attn = ops.softmax(attn, axis=-1)

        out = ms.mint.einsum("bihk,bkid->bihd", attn, value).reshape(
            B, 1, L, -1
        )  # (B, 1, L, h*dim)
        out = gate * out  # (B, N, L, h*dim)

        out = self.to_out(out)
        return out


# Instead of triangle attention, use Tied axail attention with bias from coordinates..?
class BiasedAxialAttention(nn.Cell):
    def __init__(self, d_pair, d_bias, n_head, d_hidden, p_drop=0.1, is_row=True):
        super(BiasedAxialAttention, self).__init__()

        self.is_row = is_row
        self.norm_pair = nn.LayerNorm((d_pair,), epsilon=1e-5)
        self.norm_bias = nn.LayerNorm((d_bias,), epsilon=1e-5)

        self.to_q = nn.Linear(d_pair, n_head * d_hidden, bias=False)
        self.to_k = nn.Linear(d_pair, n_head * d_hidden, bias=False)
        self.to_v = nn.Linear(d_pair, n_head * d_hidden, bias=False)
        self.to_b = nn.Linear(d_bias, n_head, bias=False)
        self.to_g = nn.Linear(d_pair, n_head * d_hidden)
        self.to_out = nn.Linear(n_head * d_hidden, d_pair)

        self.scaling = 1 / math.sqrt(d_hidden)
        self.h = n_head
        self.dim = d_hidden

        # initialize all parameters properly
        self.reset_parameter()

    def reset_parameter(self):
        # query/key/value projection: Glorot uniform / Xavier uniform
        self.to_q.weight.set_data(
            initializer(XavierUniform(), self.to_q.weight.shape, self.to_q.weight.dtype)
        )
        self.to_k.weight.set_data(
            initializer(XavierUniform(), self.to_k.weight.shape, self.to_k.weight.dtype)
        )
        self.to_v.weight.set_data(
            initializer(XavierUniform(), self.to_v.weight.shape, self.to_v.weight.dtype)
        )

        # bias: normal distribution
        self.to_b = init_lecun_normal(self.to_b)

        # gating: zero weights, one biases (mostly open gate at the beginning)
        self.to_g.weight.set_data(
            initializer(Zero(), self.to_g.weight.shape, self.to_g.weight.dtype)
        )
        self.to_g.bias.set_data(
            initializer(One(), self.to_g.bias.shape, self.to_g.bias.dtype)
        )

        # to_out: right before residual connection: zero initialize -- to make it sure residual operation is same to the Identity at the beginning
        self.to_out.weight.set_data(
            initializer(Zero(), self.to_out.weight.shape, self.to_out.weight.dtype)
        )
        self.to_out.bias.set_data(
            initializer(Zero(), self.to_out.bias.shape, self.to_out.bias.dtype)
        )

    def construct(self, pair, bias):
        # pair: (B, L, L, d_pair)
        B, L = pair.shape[:2]

        if self.is_row:
            pair = pair.permute(0, 2, 1, 3)
            bias = bias.permute(0, 2, 1, 3)

        pair = self.norm_pair(pair)
        bias = self.norm_bias(bias)

        query = self.to_q(pair).reshape(B, L, L, self.h, self.dim)
        key = self.to_k(pair).reshape(B, L, L, self.h, self.dim)
        value = self.to_v(pair).reshape(B, L, L, self.h, self.dim)
        bias = self.to_b(bias)  # (B, L, L, h)
        gate = ops.sigmoid(self.to_g(pair))  # (B, L, L, h*dim)

        query = query * self.scaling
        key = key / math.sqrt(L)  # normalize for tied attention
        attn = ms.mint.einsum("bnihk,bnjhk->bijh", query, key)  # tied attention
        attn = attn + bias  # apply bias
        attn = ops.softmax(attn, axis=-2)  # (B, L, L, h)

        out = ms.mint.einsum("bijh,bkjhd->bikhd", attn, value).reshape(B, L, L, -1)
        out = gate * out

        out = self.to_out(out)
        if self.is_row:
            out = out.permute(0, 2, 1, 3)
        return out
