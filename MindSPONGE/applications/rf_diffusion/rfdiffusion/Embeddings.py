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


import mindspore as ms
import numpy as np
from mindspore import nn, ops
from mindspore.common.initializer import HeNormal, Zero, initializer

from .Attention_module import Attention, AttentionWithBias, FeedForwardLayer
from .Track_module import PairStr2Pair
from .util_module import find_breaks, init_lecun_normal, rbf

# Module contains classes and functions to generate initial embeddings


class PositionalEncoding2D(nn.Cell):
    # Add relative positional encoding to pair features
    def __init__(self, d_model, minpos=-32, maxpos=32, p_drop=0.1):
        super(PositionalEncoding2D, self).__init__()
        self.minpos = minpos
        self.maxpos = maxpos
        self.nbin = abs(minpos) + maxpos + 1
        self.emb = nn.Embedding(self.nbin, d_model)
        self.drop = nn.Dropout(p=p_drop)

    def construct(self, x, idx, cyclize=None):
        bins = np.arange(self.minpos, self.maxpos).tolist()
        seqsep = idx[:, None, :] - idx[:, :, None]  # (B, L, L)

        # adding support for multi-chain cyclic
        # find chain breaks and label chain ids
        breaks = find_breaks(
            idx.squeeze().numpy(), thresh=35
        )  # NOTE: Hard coded threshold for defining chain breaks here
        #       Typical jump for chainbreaks is +200
        #       Assumes monotonically increasing absolute IDX

        chainids = np.zeros_like(idx.squeeze().numpy())
        for i, b in enumerate(breaks):
            chainids[b:] = i + 1
        chainids = ms.from_numpy(chainids)

        # cyclic peptide
        if cyclize is not None:
            for chid in ms.mint.unique(chainids):
                is_chid = chainids == chid
                cur_cyclize = cyclize * is_chid
                cur_mask = cur_cyclize[:, None] * cur_cyclize[None, :]  # (L,L)
                cur_ncyc = ms.mint.sum(cur_cyclize)

                seqsep[:, cur_mask * (seqsep[0] > cur_ncyc // 2)] -= cur_ncyc
                seqsep[:, cur_mask * (seqsep[0] < -cur_ncyc // 2)] += cur_ncyc

        ib = ops.bucketize(seqsep, bins).long()  # (B, L, L)
        emb = self.emb(ib)  # (B, L, L, d_model)
        x = x + emb  # add relative positional encoding
        return self.drop(x)


class MSA_emb(nn.Cell):
    # Get initial seed MSA embedding
    def __init__(
        self,
        d_msa=256,
        d_pair=128,
        d_state=32,
        d_init=22 + 22 + 2 + 2,
        minpos=-32,
        maxpos=32,
        p_drop=0.1,
        input_seq_onehot=False,
    ):
        super(MSA_emb, self).__init__()
        self.emb = nn.Linear(d_init, d_msa)  # embedding for general MSA
        self.emb_q = nn.Embedding(
            22, d_msa
        )  # embedding for query sequence -- used for MSA embedding
        self.emb_left = nn.Embedding(
            22, d_pair
        )  # embedding for query sequence -- used for pair embedding
        self.emb_right = nn.Embedding(
            22, d_pair
        )  # embedding for query sequence -- used for pair embedding
        self.emb_state = nn.Embedding(22, d_state)
        self.drop = nn.Dropout(p=p_drop)
        self.pos = PositionalEncoding2D(
            d_pair, minpos=minpos, maxpos=maxpos, p_drop=p_drop
        )

        self.input_seq_onehot = input_seq_onehot

        self.reset_parameter()

    def reset_parameter(self):
        self.emb = init_lecun_normal(self.emb)
        self.emb_q = init_lecun_normal(self.emb_q)
        self.emb_left = init_lecun_normal(self.emb_left)
        self.emb_right = init_lecun_normal(self.emb_right)
        self.emb_state = init_lecun_normal(self.emb_state)

        self.emb.bias.set_data(
            initializer(Zero(), self.emb.bias.shape, self.emb.bias.dtype)
        )

    def construct(self, msa, seq, idx, cyclize):
        # Inputs:
        #   - msa: Input MSA (B, N, L, d_init)
        #   - seq: Input Sequence (B, L)
        #   - idx: Residue index
        # Outputs:
        #   - msa: Initial MSA embedding (B, N, L, d_msa)
        #   - pair: Initial Pair embedding (B, L, L, d_pair)

        N = msa.shape[1]  # number of sequenes in MSA

        # msa embedding
        msa = self.emb(msa)  # (B, N, L, d_model) # MSA embedding

        # Sergey's one hot trick
        tmp = (seq @ self.emb_q.embedding_table).unsqueeze(
            1
        )  # (B, 1, L, d_model) -- query embedding

        # adding query embedding to MSA
        msa = msa + tmp.expand((-1, N, -1, -1))
        msa = self.drop(msa)

        # pair embedding
        # Sergey's one hot trick
        # (B, 1, L, d_pair)
        left = (seq @ self.emb_left.embedding_table)[:, None]
        # (B, L, 1, d_pair)
        right = (seq @ self.emb_right.embedding_table)[:, :, None]

        pair = left + right  # (B, L, L, d_pair)
        pair = self.pos(pair, idx, cyclize)  # add relative position

        # state embedding
        # Sergey's one hot trick
        state = self.drop(seq @ self.emb_state.embedding_table)
        return msa, pair, state


class Extra_emb(nn.Cell):
    # Get initial seed MSA embedding
    def __init__(
        self, d_msa=256, d_init=22 + 1 + 2, p_drop=0.1, input_seq_onehot=False
    ):
        super(Extra_emb, self).__init__()
        self.emb = nn.Linear(d_init, d_msa)  # embedding for general MSA
        self.emb_q = nn.Embedding(22, d_msa)  # embedding for query sequence
        self.drop = nn.Dropout(p=p_drop)

        self.input_seq_onehot = input_seq_onehot

        self.reset_parameter()

    def reset_parameter(self):
        self.emb = init_lecun_normal(self.emb)
        self.emb.bias.set_data(
            initializer(Zero(), self.emb.bias.shape, self.emb.bias.dtype)
        )

    def construct(self, msa, seq, idx):
        # Inputs:
        #   - msa: Input MSA (B, N, L, d_init)
        #   - seq: Input Sequence (B, L)
        #   - idx: Residue index
        # Outputs:
        #   - msa: Initial MSA embedding (B, N, L, d_msa)
        N = msa.shape[1]  # number of sequenes in MSA
        msa = self.emb(msa)  # (B, N, L, d_model) # MSA embedding

        # Sergey's one hot trick
        seq = (seq @ self.emb_q.embedding_table).unsqueeze(
            1
        )  # (B, 1, L, d_model) -- query embedding
        # adding query embedding to MSA
        msa = msa + seq.expand((-1, N, -1, -1))
        return self.drop(msa)


class TemplatePairStack(nn.Cell):
    # process template pairwise features
    # use structure-biased attention
    def __init__(self, n_block=2, d_templ=64, n_head=4, d_hidden=16, p_drop=0.25):
        super(TemplatePairStack, self).__init__()
        self.n_block = n_block
        proc_s = [
            PairStr2Pair(
                d_pair=d_templ, n_head=n_head, d_hidden=d_hidden, p_drop=p_drop
            )
            for i in range(n_block)
        ]
        self.block = nn.CellList(proc_s)
        self.norm = nn.LayerNorm((d_templ,), epsilon=1e-5)

    def construct(self, templ, rbf_feat, use_checkpoint=False):
        B, T, L = templ.shape[:3]
        templ = templ.reshape(B * T, L, L, -1)

        for i_block in range(self.n_block):
            if use_checkpoint:
                templ = ms.recompute(self.block[i_block], templ, rbf_feat)
            else:
                templ = self.block[i_block](templ, rbf_feat)
        return self.norm(templ).reshape(B, T, L, L, -1)


class TemplateTorsionStack(nn.Cell):
    def __init__(self, n_block=2, d_templ=64, n_head=4, d_hidden=16, p_drop=0.15):
        super(TemplateTorsionStack, self).__init__()
        self.n_block = n_block
        self.proj_pair = nn.Linear(d_templ + 36, d_templ)
        proc_s = [
            AttentionWithBias(
                d_in=d_templ, d_bias=d_templ, n_head=n_head, d_hidden=d_hidden
            )
            for i in range(n_block)
        ]
        self.row_attn = nn.CellList(proc_s)
        proc_s = [FeedForwardLayer(d_templ, 4, p_drop=p_drop) for i in range(n_block)]
        self.ff = nn.CellList(proc_s)
        self.norm = nn.LayerNorm((d_templ,), epsilon=1e-5)

    def reset_parameter(self):
        self.proj_pair = init_lecun_normal(self.proj_pair)
        self.proj_pair.bias.set_data(
            initializer(Zero(), self.proj_pair.bias.shape, self.proj_pair.bias.dtype)
        )

    def construct(self, tors, pair, rbf_feat, use_checkpoint=False):
        B, T, L = tors.shape[:3]
        tors = tors.reshape(B * T, L, -1)
        pair = pair.reshape(B * T, L, L, -1)
        pair = ms.mint.cat((pair, rbf_feat), dim=-1)
        pair = self.proj_pair(pair)

        for i_block in range(self.n_block):
            if use_checkpoint:
                tors = tors + ms.recompute(self.row_attn[i_block], tors, pair)
            else:
                tors = tors + self.row_attn[i_block](tors, pair)
            tors = tors + self.ff[i_block](tors)
        return self.norm(tors).reshape(B, T, L, -1)


class Templ_emb(nn.Cell):
    # Get template embedding
    # Features are
    #   t2d:
    #   - 37 distogram bins + 6 orientations (43)
    #   - Mask (missing/unaligned) (1)
    #   t1d:
    #   - tiled AA sequence (20 standard aa + gap)
    #   - confidence (1)
    #   - contacting or note (1). NB this is added for diffusion model. Used only in complex training examples - 1 signifies that a residue in the non-diffused chain\
    #     i.e. the context, is in contact with the diffused chain.
    #
    # Added extra t1d dimension for contacting or not
    def __init__(
        self,
        d_t1d=21 + 1 + 1,
        d_t2d=43 + 1,
        d_tor=30,
        d_pair=128,
        d_state=32,
        n_block=2,
        d_templ=64,
        n_head=4,
        d_hidden=16,
        p_drop=0.25,
    ):
        super(Templ_emb, self).__init__()
        # process 2D features
        self.emb = nn.Linear(d_t1d * 2 + d_t2d, d_templ)
        self.templ_stack = TemplatePairStack(
            n_block=n_block,
            d_templ=d_templ,
            n_head=n_head,
            d_hidden=d_hidden,
            p_drop=p_drop,
        )

        self.attn = Attention(d_pair, d_templ, n_head, d_hidden, d_pair)

        # process torsion angles
        self.emb_t1d = nn.Linear(d_t1d + d_tor, d_templ)
        self.proj_t1d = nn.Linear(d_templ, d_templ)
        # self.tor_stack = TemplateTorsionStack(n_block=n_block, d_templ=d_templ, n_head=n_head,
        #                                      d_hidden=d_hidden, p_drop=p_drop)
        self.attn_tor = Attention(d_state, d_templ, n_head, d_hidden, d_state)

        self.reset_parameter()

    def reset_parameter(self):
        self.emb = init_lecun_normal(self.emb)
        self.emb.bias.set_data(
            initializer(Zero(), self.emb.bias.shape, self.emb.bias.dtype)
        )

        self.emb_t1d.weight.set_data(
            initializer(
                HeNormal(nonlinearity="relu"),
                self.emb_t1d.weight.shape,
                self.emb_t1d.weight.dtype,
            )
        )
        self.emb_t1d.bias.set_data(
            initializer(Zero(), self.emb_t1d.bias.shape, self.emb_t1d.bias.dtype)
        )

        self.proj_t1d = init_lecun_normal(self.proj_t1d)
        self.proj_t1d.bias.set_data(
            initializer(Zero(), self.proj_t1d.bias.shape, self.proj_t1d.bias.dtype)
        )

    def construct(self, t1d, t2d, alpha_t, xyz_t, pair, state, use_checkpoint=False):
        # Input
        #   - t1d: 1D template info (B, T, L, 23)
        #   - t2d: 2D template info (B, T, L, L, 44)
        B, T, L, _ = t1d.shape

        # Prepare 2D template features
        left = t1d.unsqueeze(3).expand((-1, -1, -1, L, -1))
        right = t1d.unsqueeze(2).expand((-1, -1, L, -1, -1))
        #
        templ = ms.mint.cat((t2d, left, right), -1)  # (B, T, L, L, 90)
        templ = self.emb(templ)  # Template templures (B, T, L, L, d_templ)
        # process each template features
        xyz_t = xyz_t.reshape(B * T, L, -1, 3)
        rbf_feat = rbf(ms.mint.cdist(xyz_t[:, :, 1], xyz_t[:, :, 1]))
        templ = self.templ_stack(
            templ, rbf_feat, use_checkpoint=use_checkpoint
        )  # (B, T, L,L, d_templ)

        # Prepare 1D template torsion angle features
        t1d = ms.mint.cat((t1d, alpha_t), dim=-1)  # (B, T, L, 23+30)

        # process each template features
        t1d = self.proj_t1d(ops.relu(self.emb_t1d(t1d), inplace=True))

        # mixing query state features to template state features
        state = state.reshape(B * L, 1, -1)
        t1d = t1d.permute(0, 2, 1, 3).reshape(B * L, T, -1)
        if use_checkpoint:
            out = ms.recompute(self.attn_tor, state, t1d, t1d)
            out = out.reshape(B, L, -1)
        else:
            out = self.attn_tor(state, t1d, t1d).reshape(B, L, -1)
        state = state.reshape(B, L, -1)
        state = state + out

        # mixing query pair features to template information (Template pointwise attention)
        pair = pair.reshape(B * L * L, 1, -1)
        templ = templ.permute(0, 2, 3, 1, 4).reshape(B * L * L, T, -1)
        if use_checkpoint:
            out = ms.recompute(self.attn, pair, templ, templ)
            out = out.reshape(B, L, L, -1)
        else:
            out = self.attn(pair, templ, templ).reshape(B, L, L, -1)

        pair = pair.reshape(B, L, L, -1)
        pair = pair + out

        return pair, state


class Recycling(nn.Cell):
    def __init__(self, d_msa=256, d_pair=128, d_state=32):
        super(Recycling, self).__init__()
        self.proj_dist = nn.Linear(36 + d_state * 2, d_pair)
        self.norm_state = nn.LayerNorm((d_state,), epsilon=1e-5)
        self.norm_pair = nn.LayerNorm((d_pair,), epsilon=1e-5)
        self.norm_msa = nn.LayerNorm((d_msa,), epsilon=1e-5)

        self.reset_parameter()

    def reset_parameter(self):
        self.proj_dist = init_lecun_normal(self.proj_dist)
        self.proj_dist.bias.set_data(
            initializer(Zero(), self.proj_dist.bias.shape, self.proj_dist.bias.dtype)
        )

    def construct(self, seq, msa, pair, xyz, state):
        B, L = pair.shape[:2]
        state = self.norm_state(state)

        left = state.unsqueeze(2).expand((-1, -1, L, -1))
        right = state.unsqueeze(1).expand((-1, L, -1, -1))

        # three anchor atoms
        N = xyz[:, :, 0]
        Ca = xyz[:, :, 1]
        C = xyz[:, :, 2]

        # recreate Cb given N,Ca,C
        b = Ca - N
        c = C - Ca
        a = ms.mint.cross(b, c, dim=-1)
        Cb = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + Ca

        dist = rbf(ms.mint.cdist(Cb, Cb))
        dist = ms.mint.cat((dist, left, right), dim=-1)
        dist = self.proj_dist(dist)
        pair = dist + self.norm_pair(pair)
        msa = self.norm_msa(msa)
        return msa, pair, state
