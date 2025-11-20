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
from mindspore import nn, ops
from mindspore.common.initializer import HeNormal, Zero, initializer

from .Attention_module import *
from .SE3_network import SE3TransformerWrapper
from .util_module import *

# Components for three-track blocks
# 1. MSA -> MSA update (biased attention. bias from pair & structure)
# 2. Pair -> Pair update (biased attention. bias from structure)
# 3. MSA -> Pair update (extract coevolution signal)
# 4. Str -> Str update (node from MSA, edge from Pair)


# Update MSA with biased self-attention. bias from Pair & Str
class MSAPairStr2MSA(nn.Cell):
    def __init__(
        self,
        d_msa=256,
        d_pair=128,
        n_head=8,
        d_state=16,
        d_hidden=32,
        p_drop=0.15,
        use_global_attn=False,
    ):
        super(MSAPairStr2MSA, self).__init__()
        self.norm_pair = nn.LayerNorm((d_pair,), epsilon=1e-5)
        self.proj_pair = nn.Linear(d_pair + 36, d_pair)
        self.norm_state = nn.LayerNorm((d_state,), epsilon=1e-5)
        self.proj_state = nn.Linear(d_state, d_msa)
        self.drop_row = Dropout(broadcast_dim=1, p_drop=p_drop)
        self.row_attn = MSARowAttentionWithBias(
            d_msa=d_msa, d_pair=d_pair, n_head=n_head, d_hidden=d_hidden
        )
        if use_global_attn:
            self.col_attn = MSAColGlobalAttention(
                d_msa=d_msa, n_head=n_head, d_hidden=d_hidden
            )
        else:
            self.col_attn = MSAColAttention(
                d_msa=d_msa, n_head=n_head, d_hidden=d_hidden
            )
        self.ff = FeedForwardLayer(d_msa, 4, p_drop=p_drop)

        # Do proper initialization
        self.reset_parameter()

    def reset_parameter(self):
        # initialize weights to normal distrib
        self.proj_pair = init_lecun_normal(self.proj_pair)
        self.proj_state = init_lecun_normal(self.proj_state)

        # initialize bias to zeros
        self.proj_pair.bias.set_data(
            initializer(Zero(), self.proj_pair.bias.shape, self.proj_pair.bias.dtype)
        )
        self.proj_state.bias.set_data(
            initializer(Zero(), self.proj_state.bias.shape, self.proj_state.bias.dtype)
        )

    def construct(self, msa, pair, rbf_feat, state):
        """
        Inputs:
            - msa: MSA feature (B, N, L, d_msa)
            - pair: Pair feature (B, L, L, d_pair)
            - rbf_feat: Ca-Ca distance feature calculated from xyz coordinates (B, L, L, 36)
            - xyz: xyz coordinates (B, L, n_atom, 3)
            - state: updated node features after SE(3)-Transformer layer (B, L, d_state)
        Output:
            - msa: Updated MSA feature (B, N, L, d_msa)
        """
        B, N, L = msa.shape[:3]

        # prepare input bias feature by combining pair & coordinate info
        pair = self.norm_pair(pair)
        pair = ms.mint.cat((pair, rbf_feat), dim=-1)
        pair = self.proj_pair(pair)  # (B, L, L, d_pair)

        # update query sequence feature (first sequence in the MSA) with feedbacks (state) from SE3
        state = self.norm_state(state)
        state = self.proj_state(state).reshape(B, 1, L, -1)
        msa = msa.index_add(
            1,
            ms.tensor(
                [
                    0,
                ],
                dtype=ms.int32,
            ),
            state,
        )

        # Apply row/column attention to msa & transform
        msa = msa + self.drop_row(self.row_attn(msa, pair))
        msa = msa + self.col_attn(msa)
        msa = msa + self.ff(msa)

        return msa


class PairStr2Pair(nn.Cell):
    def __init__(self, d_pair=128, n_head=4, d_hidden=32, d_rbf=36, p_drop=0.15):
        super(PairStr2Pair, self).__init__()

        self.emb_rbf = nn.Linear(d_rbf, d_hidden)
        self.proj_rbf = nn.Linear(d_hidden, d_pair)

        self.drop_row = Dropout(broadcast_dim=1, p_drop=p_drop)
        self.drop_col = Dropout(broadcast_dim=2, p_drop=p_drop)

        self.row_attn = BiasedAxialAttention(
            d_pair, d_pair, n_head, d_hidden, p_drop=p_drop, is_row=True
        )
        self.col_attn = BiasedAxialAttention(
            d_pair, d_pair, n_head, d_hidden, p_drop=p_drop, is_row=False
        )

        self.ff = FeedForwardLayer(d_pair, 2)

        self.reset_parameter()

    def reset_parameter(self):
        self.emb_rbf.weight.set_data(
            initializer(
                HeNormal(nonlinearity="relu"),
                self.emb_rbf.weight.shape,
                self.emb_rbf.weight.dtype,
            )
        )

        self.emb_rbf.bias.set_data(
            initializer(Zero(), self.emb_rbf.bias.shape, self.emb_rbf.bias.dtype)
        )

        self.proj_rbf = init_lecun_normal(self.proj_rbf)
        self.proj_rbf.bias.set_data(
            initializer(Zero(), self.proj_rbf.bias.shape, self.proj_rbf.bias.dtype)
        )

    def construct(self, pair, rbf_feat):
        rbf_feat = self.proj_rbf(ops.relu(self.emb_rbf(rbf_feat), inplace=True))

        pair = pair + self.drop_row(self.row_attn(pair, rbf_feat))
        pair = pair + self.drop_col(self.col_attn(pair, rbf_feat))
        pair = pair + self.ff(pair)
        return pair


class MSA2Pair(nn.Cell):
    def __init__(self, d_msa=256, d_pair=128, d_hidden=32, p_drop=0.15):
        super(MSA2Pair, self).__init__()
        self.norm = nn.LayerNorm((d_msa,), epsilon=1e-5)
        self.proj_left = nn.Linear(d_msa, d_hidden)
        self.proj_right = nn.Linear(d_msa, d_hidden)
        self.proj_out = nn.Linear(d_hidden * d_hidden, d_pair)

        self.reset_parameter()

    def reset_parameter(self):
        # normal initialization
        self.proj_left = init_lecun_normal(self.proj_left)
        self.proj_right = init_lecun_normal(self.proj_right)
        self.proj_left.bias.set_data(
            initializer(Zero(), self.proj_left.bias.shape, self.proj_left.bias.dtype)
        )
        self.proj_right.bias.set_data(
            initializer(Zero(), self.proj_right.bias.shape, self.proj_right.bias.dtype)
        )

        # zero initialize output
        self.proj_out.weight.set_data(
            initializer(Zero(), self.proj_out.weight.shape, self.proj_out.weight.dtype)
        )
        self.proj_out.bias.set_data(
            initializer(Zero(), self.proj_out.bias.shape, self.proj_out.bias.dtype)
        )

    def construct(self, msa, pair):
        B, N, L = msa.shape[:3]
        msa = self.norm(msa)
        left = self.proj_left(msa)
        right = self.proj_right(msa)
        right = right / float(N)
        out = ms.mint.einsum("bsli,bsmj->blmij", left, right).reshape(B, L, L, -1)
        out = self.proj_out(out)

        pair = pair + out

        return pair


class SCPred(nn.Cell):
    def __init__(self, d_msa=256, d_state=32, d_hidden=128, p_drop=0.15):
        super(SCPred, self).__init__()
        self.norm_s0 = nn.LayerNorm((d_msa,), epsilon=1e-5)
        self.norm_si = nn.LayerNorm((d_state,), epsilon=1e-5)
        self.linear_s0 = nn.Linear(d_msa, d_hidden)
        self.linear_si = nn.Linear(d_state, d_hidden)

        # ResNet layers
        self.linear_1 = nn.Linear(d_hidden, d_hidden)
        self.linear_2 = nn.Linear(d_hidden, d_hidden)
        self.linear_3 = nn.Linear(d_hidden, d_hidden)
        self.linear_4 = nn.Linear(d_hidden, d_hidden)

        # Final outputs
        self.linear_out = nn.Linear(d_hidden, 20)

        self.reset_parameter()

    def reset_parameter(self):
        # normal initialization
        self.linear_s0 = init_lecun_normal(self.linear_s0)
        self.linear_si = init_lecun_normal(self.linear_si)
        self.linear_out = init_lecun_normal(self.linear_out)
        self.linear_s0.bias.set_data(
            initializer(Zero(), self.linear_s0.bias.shape, self.linear_s0.bias.dtype)
        )
        self.linear_si.bias.set_data(
            initializer(Zero(), self.linear_si.bias.shape, self.linear_si.bias.dtype)
        )
        self.linear_out.bias.set_data(
            initializer(Zero(), self.linear_out.bias.shape, self.linear_out.bias.dtype)
        )

        # right before relu activation: He initializer (kaiming normal)
        self.linear_1.weight.set_data(
            initializer(
                HeNormal(nonlinearity="relu"),
                self.linear_1.weight.shape,
                self.linear_1.weight.dtype,
            )
        )
        self.linear_1.bias.set_data(
            initializer(Zero(), self.linear_1.bias.shape, self.linear_1.bias.dtype)
        )

        self.linear_3.weight.set_data(
            initializer(
                HeNormal(nonlinearity="relu"),
                self.linear_3.weight.shape,
                self.linear_3.weight.dtype,
            )
        )
        self.linear_3.bias.set_data(
            initializer(Zero(), self.linear_3.bias.shape, self.linear_3.bias.dtype)
        )

        # right before residual connection: zero initialize
        self.linear_2.weight.set_data(
            initializer(Zero(), self.linear_2.weight.shape, self.linear_2.weight.dtype)
        )
        self.linear_2.bias.set_data(
            initializer(Zero(), self.linear_2.bias.shape, self.linear_2.bias.dtype)
        )

        self.linear_4.weight.set_data(
            initializer(Zero(), self.linear_4.weight.shape, self.linear_4.weight.dtype)
        )
        self.linear_4.bias.set_data(
            initializer(Zero(), self.linear_4.bias.shape, self.linear_4.bias.dtype)
        )

    def construct(self, seq, state):
        """
        Predict side-chain torsion angles along with backbone torsions
        Inputs:
            - seq: hidden embeddings corresponding to query sequence (B, L, d_msa)
            - state: state feature (output l0 feature) from previous SE3 layer (B, L, d_state)
        Outputs:
            - si: predicted torsion angles (phi, psi, omega, chi1~4 with cos/sin, Cb bend, Cb twist, CG) (B, L, 10, 2)
        """
        B, L = seq.shape[:2]
        seq = self.norm_s0(seq)
        state = self.norm_si(state)
        si = self.linear_s0(seq) + self.linear_si(state)

        si = si + self.linear_2(
            ops.relu(self.linear_1(ops.relu(si, inplace=True)), inplace=True)
        )
        si = si + self.linear_4(
            ops.relu(self.linear_3(ops.relu(si, inplace=True)), inplace=True)
        )

        si = self.linear_out(ops.relu(si, inplace=True))
        return si.view(B, L, 10, 2)


class Str2Str(nn.Cell):
    def __init__(
        self,
        d_msa=256,
        d_pair=128,
        d_state=16,
        SE3_param={
            "l0_in_features": 32,
            "l0_out_features": 16,
            "num_edge_features": 32,
        },
        p_drop=0.1,
    ):
        super(Str2Str, self).__init__()

        # initial node & pair feature process
        self.norm_msa = nn.LayerNorm((d_msa,), epsilon=1e-5)
        self.norm_pair = nn.LayerNorm((d_pair,), epsilon=1e-5)
        self.norm_state = nn.LayerNorm((d_state,), epsilon=1e-5)

        self.embed_x = nn.Linear(d_msa + d_state, SE3_param["l0_in_features"])
        self.embed_e1 = nn.Linear(d_pair, SE3_param["num_edge_features"])
        self.embed_e2 = nn.Linear(
            SE3_param["num_edge_features"] + 36 + 1, SE3_param["num_edge_features"]
        )

        self.norm_node = nn.LayerNorm((SE3_param["l0_in_features"],), epsilon=1e-5)
        self.norm_edge1 = nn.LayerNorm((SE3_param["num_edge_features"],), epsilon=1e-5)
        self.norm_edge2 = nn.LayerNorm((SE3_param["num_edge_features"],), epsilon=1e-5)

        self.se3 = SE3TransformerWrapper(**SE3_param)
        self.sc_predictor = SCPred(
            d_msa=d_msa, d_state=SE3_param["l0_out_features"], p_drop=p_drop
        )

        self.reset_parameter()

    def reset_parameter(self):
        # initialize weights to normal distribution
        self.embed_x = init_lecun_normal(self.embed_x)
        self.embed_e1 = init_lecun_normal(self.embed_e1)
        self.embed_e2 = init_lecun_normal(self.embed_e2)

        # initialize bias to zeros
        self.embed_x.bias.set_data(
            initializer(Zero(), self.embed_x.bias.shape, self.embed_x.bias.dtype)
        )
        self.embed_e1.bias.set_data(
            initializer(Zero(), self.embed_e1.bias.shape, self.embed_e1.bias.dtype)
        )
        self.embed_e2.bias.set_data(
            initializer(Zero(), self.embed_e2.bias.shape, self.embed_e2.bias.dtype)
        )

    def construct(
        self,
        msa,
        pair,
        R_in,
        T_in,
        xyz,
        state,
        idx,
        motif_mask,
        cyclic_reses=None,
        top_k=64,
        eps=1e-5,
    ):
        B, N, L = msa.shape[:3]

        if motif_mask is None:
            motif_mask = ms.mint.zeros(L).bool()

        # process msa & pair features
        node = self.norm_msa(msa[:, 0])
        pair = self.norm_pair(pair)
        state = self.norm_state(state)

        node = ms.mint.cat((node, state), dim=-1)
        node = self.norm_node(self.embed_x(node))
        pair = self.norm_edge1(self.embed_e1(pair))

        neighbor = get_seqsep(idx, cyclic_reses)
        rbf_feat = rbf(ms.mint.cdist(xyz[:, :, 1], xyz[:, :, 1]))

        pair = ms.mint.cat((pair, rbf_feat, neighbor.to(dtype=ms.float32)), dim=-1)
        pair = self.norm_edge2(self.embed_e2(pair))

        # define graph
        if top_k != 0:
            G, edge_feats = make_topk_graph(xyz[:, :, 1, :], pair, idx, top_k=top_k)
        else:
            G, edge_feats = make_full_graph(xyz[:, :, 1, :], pair, idx, top_k=top_k)
        l1_feats = xyz - xyz[:, :, 1, :].unsqueeze(2)
        l1_feats = l1_feats.reshape(B * L, -1, 3)

        # apply SE(3) Transformer & update coordinates
        shift = self.se3(G, node.reshape(B * L, -1, 1), l1_feats, edge_feats)

        state = shift["0"].reshape(B, L, -1)  # (B, L, C)

        offset = shift["1"].reshape(B, L, 2, 3)
        offset[:, motif_mask, ...] = (
            0  # NOTE: motif mask is all zeros if not freeezing the motif
        )

        delTi = offset[:, :, 0, :] / 10.0  # translation
        R = offset[:, :, 1, :] / 100.0  # rotation

        Qnorm = ms.mint.sqrt(1 + ms.mint.sum(R * R, dim=-1))
        qA, qB, qC, qD = (
            1 / Qnorm,
            R[:, :, 0] / Qnorm,
            R[:, :, 1] / Qnorm,
            R[:, :, 2] / Qnorm,
        )

        delRi = ms.mint.zeros((B, L, 3, 3))
        delRi[:, :, 0, 0] = qA * qA + qB * qB - qC * qC - qD * qD
        delRi[:, :, 0, 1] = 2 * qB * qC - 2 * qA * qD
        delRi[:, :, 0, 2] = 2 * qB * qD + 2 * qA * qC
        delRi[:, :, 1, 0] = 2 * qB * qC + 2 * qA * qD
        delRi[:, :, 1, 1] = qA * qA - qB * qB + qC * qC - qD * qD
        delRi[:, :, 1, 2] = 2 * qC * qD - 2 * qA * qB
        delRi[:, :, 2, 0] = 2 * qB * qD - 2 * qA * qC
        delRi[:, :, 2, 1] = 2 * qC * qD + 2 * qA * qB
        delRi[:, :, 2, 2] = qA * qA - qB * qB - qC * qC + qD * qD

        Ri = ms.mint.einsum("bnij,bnjk->bnik", delRi, R_in)
        Ti = delTi + T_in  # einsum('bnij,bnj->bni', delRi, T_in) + delTi

        alpha = self.sc_predictor(msa[:, 0], state)
        return Ri, Ti, state, alpha


class IterBlock(nn.Cell):
    def __init__(
        self,
        d_msa=256,
        d_pair=128,
        n_head_msa=8,
        n_head_pair=4,
        use_global_attn=False,
        d_hidden=32,
        d_hidden_msa=None,
        p_drop=0.15,
        SE3_param={
            "l0_in_features": 32,
            "l0_out_features": 16,
            "num_edge_features": 32,
        },
    ):
        super(IterBlock, self).__init__()
        if d_hidden_msa == None:
            d_hidden_msa = d_hidden

        self.msa2msa = MSAPairStr2MSA(
            d_msa=d_msa,
            d_pair=d_pair,
            n_head=n_head_msa,
            d_state=SE3_param["l0_out_features"],
            use_global_attn=use_global_attn,
            d_hidden=d_hidden_msa,
            p_drop=p_drop,
        )
        self.msa2pair = MSA2Pair(
            d_msa=d_msa, d_pair=d_pair, d_hidden=d_hidden // 2, p_drop=p_drop
        )
        # d_hidden=d_hidden, p_drop=p_drop)
        self.pair2pair = PairStr2Pair(
            d_pair=d_pair, n_head=n_head_pair, d_hidden=d_hidden, p_drop=p_drop
        )
        self.str2str = Str2Str(
            d_msa=d_msa,
            d_pair=d_pair,
            d_state=SE3_param["l0_out_features"],
            SE3_param=SE3_param,
            p_drop=p_drop,
        )

    def construct(
        self,
        msa,
        pair,
        R_in,
        T_in,
        xyz,
        state,
        idx,
        motif_mask,
        use_checkpoint=False,
        cyclic_reses=None,
    ):
        rbf_feat = rbf(ms.mint.cdist(xyz[:, :, 1, :], xyz[:, :, 1, :]))
        if use_checkpoint:
            msa = ms.recompute(self.msa2msa, msa, pair, rbf_feat, state)
            pair = ms.recompute(self.msa2pair, msa, pair)
            pair = ms.recompute(self.pair2pair, pair, rbf_feat)
            R, T, state, alpha = ms.recompute(
                self.str2str,
                msa,
                pair,
                R_in,
                T_in,
                xyz,
                state,
                idx,
                motif_mask,
                cyclic_reses,
                top_k=0,
            )
        else:
            msa = self.msa2msa(msa, pair, rbf_feat, state)
            pair = self.msa2pair(msa, pair)
            pair = self.pair2pair(pair, rbf_feat)
            R, T, state, alpha = self.str2str(
                msa,
                pair,
                R_in,
                T_in,
                xyz,
                state,
                idx,
                motif_mask=motif_mask,
                cyclic_reses=cyclic_reses,
                top_k=0,
            )

        return msa, pair, R, T, state, alpha


class IterativeSimulator(nn.Cell):
    def __init__(
        self,
        n_extra_block=4,
        n_main_block=12,
        n_ref_block=4,
        d_msa=256,
        d_msa_full=64,
        d_pair=128,
        d_hidden=32,
        n_head_msa=8,
        n_head_pair=4,
        SE3_param_full={
            "l0_in_features": 32,
            "l0_out_features": 16,
            "num_edge_features": 32,
        },
        SE3_param_topk={
            "l0_in_features": 32,
            "l0_out_features": 16,
            "num_edge_features": 32,
        },
        p_drop=0.15,
    ):
        super(IterativeSimulator, self).__init__()
        self.n_extra_block = n_extra_block
        self.n_main_block = n_main_block
        self.n_ref_block = n_ref_block

        self.proj_state = nn.Linear(
            SE3_param_topk["l0_out_features"], SE3_param_full["l0_out_features"]
        )
        # Update with extra sequences
        if n_extra_block > 0:
            self.extra_block = nn.CellList(
                [
                    IterBlock(
                        d_msa=d_msa_full,
                        d_pair=d_pair,
                        n_head_msa=n_head_msa,
                        n_head_pair=n_head_pair,
                        d_hidden_msa=8,
                        d_hidden=d_hidden,
                        p_drop=p_drop,
                        use_global_attn=True,
                        SE3_param=SE3_param_full,
                    )
                    for i in range(n_extra_block)
                ]
            )

        # Update with seed sequences
        if n_main_block > 0:
            self.main_block = nn.CellList(
                [
                    IterBlock(
                        d_msa=d_msa,
                        d_pair=d_pair,
                        n_head_msa=n_head_msa,
                        n_head_pair=n_head_pair,
                        d_hidden=d_hidden,
                        p_drop=p_drop,
                        use_global_attn=False,
                        SE3_param=SE3_param_full,
                    )
                    for i in range(n_main_block)
                ]
            )

        self.proj_state2 = nn.Linear(
            SE3_param_full["l0_out_features"], SE3_param_topk["l0_out_features"]
        )
        # Final SE(3) refinement
        if n_ref_block > 0:
            self.str_refiner = Str2Str(
                d_msa=d_msa,
                d_pair=d_pair,
                d_state=SE3_param_topk["l0_out_features"],
                SE3_param=SE3_param_topk,
                p_drop=p_drop,
            )

        self.reset_parameter()

    def reset_parameter(self):
        self.proj_state = init_lecun_normal(self.proj_state)
        self.proj_state.bias.set_data(
            initializer(Zero(), self.proj_state.bias.shape, self.proj_state.bias.dtype)
        )
        self.proj_state2 = init_lecun_normal(self.proj_state2)
        self.proj_state2.bias.set_data(
            initializer(
                Zero(), self.proj_state2.bias.shape, self.proj_state2.bias.dtype
            )
        )

    def construct(
        self,
        seq,
        msa,
        msa_full,
        pair,
        xyz_in,
        state,
        idx,
        cyclic_reses=None,
        use_checkpoint=False,
        motif_mask=None,
    ):
        """
        input:
           seq: query sequence (B, L)
           msa: seed MSA embeddings (B, N, L, d_msa)
           msa_full: extra MSA embeddings (B, N, L, d_msa_full)
           pair: initial residue pair embeddings (B, L, L, d_pair)
           xyz_in: initial BB coordinates (B, L, n_atom, 3)
           state: initial state features containing mixture of query seq, sidechain, accuracy info (B, L, d_state)
           idx: residue index
           motif_mask: bool tensor, True if motif position that is frozen, else False(L,)
        """

        B, L = pair.shape[:2]

        if motif_mask is None:
            motif_mask = ms.mint.zeros(L).bool()

        R_in = ms.mint.eye(3).reshape(1, 1, 3, 3).expand((B, L, -1, -1))
        T_in = xyz_in[:, :, 1].clone()
        xyz_in = xyz_in - T_in.unsqueeze(-2)

        state = self.proj_state(state)

        R_s = list()
        T_s = list()
        alpha_s = list()
        for i_m in range(self.n_extra_block):
            # Get current BB structure
            xyz = ms.mint.einsum("bnij,bnaj->bnai", R_in, xyz_in) + T_in.unsqueeze(-2)

            msa_full, pair, R_in, T_in, state, alpha = self.extra_block[i_m](
                msa_full,
                pair,
                R_in,
                T_in,
                xyz,
                state,
                idx,
                motif_mask=motif_mask,
                use_checkpoint=use_checkpoint,
                cyclic_reses=cyclic_reses,
            )
            R_s.append(R_in)
            T_s.append(T_in)
            alpha_s.append(alpha)

        for i_m in range(self.n_main_block):
            R_in = R_in
            T_in = T_in
            # Get current BB structure
            xyz = ms.mint.einsum("bnij,bnaj->bnai", R_in, xyz_in) + T_in.unsqueeze(-2)

            msa, pair, R_in, T_in, state, alpha = self.main_block[i_m](
                msa,
                pair,
                R_in,
                T_in,
                xyz,
                state,
                idx,
                motif_mask=motif_mask,
                use_checkpoint=use_checkpoint,
                cyclic_reses=cyclic_reses,
            )
            R_s.append(R_in)
            T_s.append(T_in)
            alpha_s.append(alpha)

        state = self.proj_state2(state)
        for i_m in range(self.n_ref_block):
            R_in = R_in
            T_in = T_in
            xyz = ms.mint.einsum("bnij,bnaj->bnai", R_in, xyz_in) + T_in.unsqueeze(-2)
            R_in, T_in, state, alpha = self.str_refiner(
                msa,
                pair,
                R_in,
                T_in,
                xyz,
                state,
                idx,
                top_k=64,
                motif_mask=motif_mask,
                cyclic_reses=cyclic_reses,
            )
            R_s.append(R_in)
            T_s.append(T_in)
            alpha_s.append(alpha)

        R_s = ms.mint.stack(R_s, dim=0)
        T_s = ms.mint.stack(T_s, dim=0)
        alpha_s = ms.mint.stack(alpha_s, dim=0)

        return msa, pair, R_s, T_s, alpha_s, state
