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
"""model"""
import itertools
import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, context
from mindspore.common.initializer import initializer, XavierUniform

from .utils import ProcessLinspace


def gather_edges(edges, neighbor_idx):
    # Features [B,N,N,C] at Neighbor indices [B,N,K] => Neighbor features [B,N,K,C]
    neighbors = ops.broadcast_to(ops.expand_dims(neighbor_idx, -1),
                                 (neighbor_idx.shape[0], neighbor_idx.shape[1], neighbor_idx.shape[2], edges.shape[-1]))
    edge_features = ops.GatherD()(edges, 2, neighbors)
    return edge_features


def gather_nodes(nodes, neighbor_idx):
    # Features [B,N,C] at Neighbor indices [B,N,K] => [B,N,K,C]
    # Flatten and expand indices per batch [B,N,K] => [B,NK] => [B,NK,C]
    neighbors_flat = neighbor_idx.view((neighbor_idx.shape[0], -1))
    neighbors_flat = ops.broadcast_to(ops.expand_dims(neighbors_flat, -1),
                                      (neighbors_flat.shape[0], neighbors_flat.shape[1], nodes.shape[2]))
    # Gather and re-pack
    neighbor_features = ops.GatherD()(nodes, 1, neighbors_flat)
    neighbor_features = neighbor_features.view(tuple(list(neighbor_idx.shape)[:3] + [-1]))
    return neighbor_features


def gather_nodes_t(nodes, neighbor_idx):
    # Features [B,N,C] at Neighbor index [B,K] => Neighbor features[B,K,C]
    idx_flat = ops.broadcast_to(ops.expand_dims(neighbor_idx, -1),
                                (neighbor_idx.shape[0], neighbor_idx.shape[1], nodes.shape[2]))
    neighbor_features = ops.GatherD()(nodes, 1, idx_flat)
    return neighbor_features


def cat_neighbors_nodes(h_nodes, h_neighbors, e_idx):
    """cat_neighbors_nodes"""
    h_nodes = gather_nodes(h_nodes, e_idx)
    h_nn = ops.Concat(axis=-1)((h_neighbors, h_nodes))
    return h_nn


class EncLayer(nn.Cell):
    """Encoder"""

    def __init__(self, num_hidden, num_in, dropout=0.1, scale=30):
        super(EncLayer, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)
        self.dropout3 = nn.Dropout(p=dropout)
        self.norm1 = nn.LayerNorm([num_hidden])
        self.norm2 = nn.LayerNorm([num_hidden])
        self.norm3 = nn.LayerNorm([num_hidden])

        self.w1 = nn.Dense(num_hidden + num_in, num_hidden, weight_init=XavierUniform(), has_bias=True)
        self.w2 = nn.Dense(num_hidden, num_hidden, weight_init=XavierUniform(), has_bias=True)
        self.w3 = nn.Dense(num_hidden, num_hidden, weight_init=XavierUniform(), has_bias=True)
        self.w11 = nn.Dense(num_hidden + num_in, num_hidden, weight_init=XavierUniform(), has_bias=True)
        self.w12 = nn.Dense(num_hidden, num_hidden, weight_init=XavierUniform(), has_bias=True)
        self.w13 = nn.Dense(num_hidden, num_hidden, weight_init=XavierUniform(), has_bias=True)
        self.act = nn.GELU()
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def construct(self, h_v, h_e, e_idx, mask_v=None, mask_attend=None):
        """ Parallel computation of full transformer layer """
        h_ev = cat_neighbors_nodes(h_v, h_e, e_idx)
        h_v_expand = ops.broadcast_to(ops.expand_dims(h_v, -2),
                                      (h_v.shape[0], h_v.shape[1], h_ev.shape[-2], h_v.shape[2]))
        h_ev = ops.Concat(axis=-1)((h_v_expand, h_ev))
        h_message = self.w3(self.act(self.w2(self.act(self.w1(h_ev)))))
        if mask_attend is not None:
            h_message = ops.expand_dims(mask_attend, -1) * h_message
        dh = ops.ReduceSum()(h_message, -2) / self.scale
        h_v = self.norm1(h_v + self.dropout1(dh))

        dh = self.dense(h_v)
        h_v = self.norm2(h_v + self.dropout2(dh))
        if mask_v is not None:
            mask_v = ops.expand_dims(mask_v, -1)
            h_v = mask_v * h_v

        h_ev = cat_neighbors_nodes(h_v, h_e, e_idx)
        h_v_expand = ops.broadcast_to(ops.expand_dims(h_v, -2),
                                      (h_v.shape[0], h_v.shape[1], h_ev.shape[-2], h_v.shape[2]))
        h_ev = ops.Concat(axis=-1)((h_v_expand, h_ev))
        h_message = self.w13(self.act(self.w12(self.act(self.w11(h_ev)))))
        h_e = self.norm3(h_e + self.dropout3(h_message))
        return h_v, h_e


class DecLayer(nn.Cell):
    """Decoder"""

    def __init__(self, num_hidden, num_in, dropout=0.1, scale=30):  # dropout=0.1
        super(DecLayer, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)
        self.norm1 = nn.LayerNorm([num_hidden])
        self.norm2 = nn.LayerNorm([num_hidden])

        self.w1 = nn.Dense(num_hidden + num_in, num_hidden, weight_init=XavierUniform(), has_bias=True)
        self.w2 = nn.Dense(num_hidden, num_hidden, weight_init=XavierUniform(), has_bias=True)
        self.w3 = nn.Dense(num_hidden, num_hidden, weight_init=XavierUniform(), has_bias=True)
        self.act = nn.GELU()
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def construct(self, h_v, h_e, mask_v=None, mask_attend=None):
        """ Parallel computation of full transformer layer """
        # Concatenate h_v_i to h_e_ij
        h_v_expand = ops.broadcast_to(ops.expand_dims(h_v, -2),
                                      (h_v.shape[0], h_v.shape[1], h_e.shape[-2], h_v.shape[2]))
        h_ev = ops.Concat(axis=-1)((h_v_expand, h_e))

        h_message = self.w3(self.act(self.w2(self.act(self.w1(h_ev)))))
        if mask_attend is not None:
            h_message = ops.expand_dims(mask_attend, -1) * h_message
        dh = ops.ReduceSum()(h_message, -2) / self.scale

        h_v = self.norm1(h_v + self.dropout1(dh))

        # Position-wise feedforward
        dh = self.dense(h_v)
        h_v = self.norm2(h_v + self.dropout2(dh))

        if mask_v is not None:
            mask_v = ops.expand_dims(mask_v, -1)
            h_v = mask_v * h_v
        return h_v


class PositionWiseFeedForward(nn.Cell):
    """PositionWiseFeedForward"""

    def __init__(self, num_hidden, num_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.w_in = nn.Dense(num_hidden, num_ff, weight_init=XavierUniform(), has_bias=True)
        self.w_out = nn.Dense(num_ff, num_hidden, weight_init=XavierUniform(), has_bias=True)
        self.act = nn.GELU()

    def construct(self, h_v):
        h = self.act(self.w_in(h_v))
        h = self.w_out(h)
        return h


class PositionalEncodings(nn.Cell):
    """PositionalEncodings"""

    def __init__(self, num_embeddings, max_relative_feature=32):
        super(PositionalEncodings, self).__init__()
        self.num_embeddings = num_embeddings
        self.max_relative_feature = max_relative_feature
        self.linear = nn.Dense(2 * max_relative_feature + 1 + 1, num_embeddings, weight_init=XavierUniform())

    def construct(self, offset, mask):
        d = ops.clip_by_value(offset + self.max_relative_feature, ms.Tensor(0), \
                              ms.Tensor(2 * self.max_relative_feature)) * mask \
            + (1 - mask) * (2 * self.max_relative_feature + 1)
        d_onehot = nn.OneHot(depth=2 * self.max_relative_feature + 1 + 1)(d)
        e = self.linear(d_onehot)
        return e


class ProteinFeatures(nn.Cell):
    """ProteinFeatures"""

    def __init__(self, edge_features, node_features, num_positional_embeddings=16,
                 num_rbf=16, top_k=30, augment_eps=0.):
        """ Extract protein features """
        super(ProteinFeatures, self).__init__()
        self.edge_features = edge_features
        self.node_features = node_features
        self.top_k = top_k
        self.augment_eps = augment_eps
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings

        self.embeddings = PositionalEncodings(num_positional_embeddings)
        edge_in = num_positional_embeddings + num_rbf * 25
        self.edge_embedding = nn.Dense(edge_in, edge_features, weight_init=XavierUniform(), has_bias=False)
        self.norm_edges = nn.LayerNorm([edge_features])
        if context.get_context("device_target") == "GPU":
            self.dtypes = ms.float32
        else:
            self.dtypes = ms.float16

    def construct(self, x, mask, residue_idx, chain_labels):
        """construct"""
        if self.augment_eps > 0:
            x = x + self.augment_eps * ops.StandardNormal()(x.shape)

        b = x[:, :, 1, :] - x[:, :, 0, :]
        c = x[:, :, 2, :] - x[:, :, 1, :]
        a = ms.numpy.cross(b, c)
        cb = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + x[:, :, 1, :]
        ca = x[:, :, 1, :]
        n = x[:, :, 0, :]
        c = x[:, :, 2, :]
        o = x[:, :, 3, :]

        d_neighbors, e_idx = self._dist(ca, mask)

        rbf_all = []
        rbf_all.append(self._rbf(d_neighbors))  # Ca-Ca
        rbf_all.append(self._get_rbf(n, n, e_idx))  # N-N
        rbf_all.append(self._get_rbf(c, c, e_idx))  # C-C
        rbf_all.append(self._get_rbf(o, o, e_idx))  # o-o
        rbf_all.append(self._get_rbf(cb, cb, e_idx))  # Cb-Cb
        rbf_all.append(self._get_rbf(ca, n, e_idx))  # Ca-N
        rbf_all.append(self._get_rbf(ca, c, e_idx))  # ca-C
        rbf_all.append(self._get_rbf(ca, o, e_idx))  # ca-o
        rbf_all.append(self._get_rbf(ca, cb, e_idx))  # ca-Cb
        rbf_all.append(self._get_rbf(n, c, e_idx))  # N-C
        rbf_all.append(self._get_rbf(n, o, e_idx))  # N-o
        rbf_all.append(self._get_rbf(n, cb, e_idx))  # N-Cb
        rbf_all.append(self._get_rbf(cb, c, e_idx))  # cb-C
        rbf_all.append(self._get_rbf(cb, o, e_idx))  # cb-o
        rbf_all.append(self._get_rbf(o, c, e_idx))  # o-C
        rbf_all.append(self._get_rbf(n, ca, e_idx))  # N-ca
        rbf_all.append(self._get_rbf(c, ca, e_idx))  # C-ca
        rbf_all.append(self._get_rbf(o, ca, e_idx))  # o-ca
        rbf_all.append(self._get_rbf(cb, ca, e_idx))  # cb-ca
        rbf_all.append(self._get_rbf(c, n, e_idx))  # C-N
        rbf_all.append(self._get_rbf(o, n, e_idx))  # o-N
        rbf_all.append(self._get_rbf(cb, n, e_idx))  # cb-N
        rbf_all.append(self._get_rbf(c, cb, e_idx))  # C-cb
        rbf_all.append(self._get_rbf(o, cb, e_idx))  # o-cb
        rbf_all.append(self._get_rbf(c, o, e_idx))  # C-O
        rbf_all = ops.Concat(axis=-1)(tuple(rbf_all))

        offset = residue_idx[:, :, None] - residue_idx[:, None, :]
        offset = gather_edges(offset[:, :, :, None], e_idx)[:, :, :, 0]  # [B, L, K]

        d_chains = ops.Cast()(((chain_labels[:, :, None] - chain_labels[:, None, \
                                                           :]) == 0), ms.int32)  # find self vs non-self interaction
        e_chains = gather_edges(d_chains[:, :, :, None], e_idx)[:, :, :, 0]
        e_positional = self.embeddings(ops.Cast()(offset, ms.int32), e_chains)

        e = ops.Concat(axis=-1)((e_positional, rbf_all.astype(self.dtypes)))
        e = self.edge_embedding(e)
        e = self.norm_edges(e)
        return e, e_idx

    def _dist(self, x, mask, eps=1E-6):
        """_dist"""
        mask_2d = ops.expand_dims(mask, 1) * ops.expand_dims(mask, 2)
        dx = ops.expand_dims(x, 1) - ops.expand_dims(x, 2)
        d = mask_2d * ops.Sqrt()(ops.ReduceSum()(dx ** 2, 3) + eps)
        _, d_max = ops.ArgMaxWithValue(keep_dims=True, axis=-1)(d)
        d_adjust = d + (1. - mask_2d) * d_max
        shape = x.shape[1]
        d_neighbors, e_idx = ops.TopK(sorted=True)(d_adjust, x.shape[1])
        d_neighbors = d_neighbors[..., ::-1]
        e_idx = e_idx[..., ::-1]
        if self.top_k > shape:
            slice_index = shape
        else:
            slice_index = self.top_k
        d_neighbors = d_neighbors[..., :slice_index]
        e_idx = e_idx[..., :slice_index]
        return d_neighbors, e_idx

    def _rbf(self, d):
        d_min, d_max, d_count = 2., 22., self.num_rbf
        d_mu = ProcessLinspace()(Tensor(d_min, ms.float32), Tensor(d_max, ms.float32), d_count)
        d_mu = d_mu.view((1, 1, 1, -1))
        d_sigma = (d_max - d_min) / d_count
        d_expand = ops.expand_dims(d, -1)
        rbf = ops.exp(-((d_expand - d_mu) / d_sigma) ** 2)
        return rbf

    def _get_rbf(self, a, b, e_idx):
        """_get_rbf"""
        d_a_b = ops.Sqrt()(ops.ReduceSum()((a[:, :, None, :] - b[:, None, :, :]) ** 2, -1) + 1e-6)  # [B, L, L]
        d_a_b_neighbors = gather_edges(d_a_b[:, :, :, None], e_idx)[:, :, :, 0]  # [B,L,K]
        rbf_a_b = self._rbf(d_a_b_neighbors)
        return rbf_a_b


def broadcast(src: ms.Tensor, axis: int):
    src = src.asnumpy()
    ix = np.argwhere(src == src.copy())
    src = src.reshape(-1)
    ix[:, axis] = src
    return ms.Tensor(ix)


def scatter_(src: ms.Tensor, index: ms.Tensor, out: ms.Tensor, axis: int = -1):
    index = broadcast(index, axis)
    op = ops.TensorScatterUpdate()
    return op(out, index, src.reshape(-1))


class ProteinMPNN(nn.Cell):
    """ProteinMPNN"""

    def __init__(self, config):
        super(ProteinMPNN, self).__init__()

        # Hyperparameters
        self.node_features = config.node_features
        self.edge_features = config.edge_features
        self.hidden_dim = config.hidden_dim

        # Featurization layers
        self.features = ProteinFeatures(config.node_features, config.edge_features, top_k=config.k_neighbors,
                                        augment_eps=config.augment_eps)

        self.w_e = nn.Dense(config.edge_features, config.hidden_dim, weight_init=XavierUniform(), has_bias=True)
        self.w_s = nn.Embedding(config.vocab, config.hidden_dim)

        # Encoder layers
        self.encoder_layers = nn.CellList([
            EncLayer(config.hidden_dim, config.hidden_dim * 2, dropout=config.dropout)
            for _ in range(config.num_encoder_layers)
        ])

        # Decoder layers
        self.decoder_layers = nn.CellList([
            DecLayer(config.hidden_dim, config.hidden_dim * 3, dropout=config.dropout)
            for _ in range(config.num_decoder_layers)
        ])
        self.w_out = nn.Dense(config.hidden_dim, config.num_letters, weight_init=XavierUniform(), has_bias=True)

        for p in self.get_parameters():
            if p.dim() > 1:
                p.set_data(initializer(XavierUniform(), p.shape, ms.float32))

    def my_einsum(self, x1, x2, x3):
        b, _, q = x2.shape
        b, j, _ = x3.shape
        x2 = ops.transpose(x2, (0, 2, 1)).reshape(-1, q)
        x2 = ops.MatMul()(x2, x1)  # bqi * ij ==> bqj
        out = ops.BatchMatMul()(x2.reshape(b, q, j), x3)  # bqj * bjp ==> bqp
        return out

    def construct(self, x, s, mask, chain_m, residue_idx, chain_encoding_all, randn=None, use_input_decoding_order=True,
                  decoding_order=None):
        """ Graph-conditioned sequence model """
        # Prepare node and edge embeddings
        e, e_idx = self.features(x, mask, residue_idx, chain_encoding_all)
        h_v = ops.Zeros()((e.shape[0], e.shape[1], e.shape[-1]), ms.float32)
        h_e = self.w_e(e)

        # Encoder is unmasked self-attention
        mask_attend = gather_nodes(ops.expand_dims(mask, -1), e_idx).squeeze(-1)
        mask_attend = ops.expand_dims(mask, -1) * mask_attend
        for layer in self.encoder_layers:
            h_v, h_e = layer(h_v, h_e, e_idx, mask, mask_attend)

        # Concatenate sequence embeddings for autoregressive decoder
        s = ops.Cast()(s, ms.int32)
        h_s = self.w_s(s)
        h_es = cat_neighbors_nodes(h_s, h_e, e_idx)

        # Build encoder embeddings
        h_ex_encoder = cat_neighbors_nodes(ops.ZerosLike()(h_s), h_e, e_idx)
        h_exv_encoder = cat_neighbors_nodes(h_v, h_ex_encoder, e_idx)

        chain_m = chain_m * mask  # update chain_m to include missing regions
        if not use_input_decoding_order:
            _, decoding_order = ops.Sort()(ops.Mul()((chain_m + 0.0001), (ops.Abs()(
                randn))))
        else:
            _, decoding_order = ops.Sort()(ops.Mul()((chain_m + 0.0001), (ops.Abs()(
                ops.StandardNormal()(chain_m.shape)))))
        mask_size = e_idx.shape[1]
        permutation_matrix_reverse = ops.Cast()(nn.OneHot(depth=mask_size)(decoding_order), ms.float32)
        mask_matrix = 1 - nn.Triu()(ops.Ones()((mask_size, mask_size), ms.float32))
        order_mask_backward = self.my_einsum(mask_matrix, permutation_matrix_reverse, permutation_matrix_reverse)
        mask_attend = ops.expand_dims(ops.GatherD()(order_mask_backward, 2, e_idx), -1)
        mask_d1 = mask.view((mask.shape[0], mask.shape[1], 1, 1))
        mask_bw = mask_d1 * mask_attend
        mask_fw = mask_d1 * (1. - mask_attend)

        h_exv_encoder_fw = mask_fw * h_exv_encoder
        for layer in self.decoder_layers:
            # Masked positions attend to encoder information, unmasked see.
            h_esv = cat_neighbors_nodes(h_v, h_es, e_idx)
            h_esv = mask_bw * h_esv + h_exv_encoder_fw
            h_v = layer(h_v, h_esv, mask)

        logits = self.w_out(h_v)
        log_probs = nn.LogSoftmax(axis=-1)(logits)

        return log_probs

    def sample(self, x, randn, s_true, chain_mask, chain_encoding_all, residue_idx, mask=None, temperature=1.0,
               omit_aas_np=None, bias_aas_np=None, chain_m_pos=None, omit_aa_mask=None, pssm_coef=None, pssm_bias=None,
               pssm_multi=None, pssm_log_odds_flag=None, pssm_log_odds_mask=None, pssm_bias_flag=None,
               bias_by_res=None):
        """sample"""
        # Prepare node and edge embeddings
        e, e_idx = self.features(x, mask, residue_idx, chain_encoding_all)
        h_v = ops.Zeros()((e.shape[0], e.shape[1], e.shape[-1]), ms.float32)
        h_e = self.w_e(e)

        # Encoder is unmasked self-attention
        mask_attend = gather_nodes(ops.expand_dims(mask, -1), e_idx).squeeze(-1)
        mask_attend = ops.expand_dims(mask, -1) * mask_attend
        for layer in self.encoder_layers:
            h_v, h_e = layer(h_v, h_e, e_idx, mask, mask_attend)

        # Decoder uses masked self-attention
        chain_mask = chain_mask * chain_m_pos * mask  # update chain_m to include missing regions
        _, decoding_order = ops.Sort()((chain_mask + 0.0001) * (ops.Abs()(
            randn)))
        mask_size = e_idx.shape[1]
        permutation_matrix_reverse = ops.Cast()(nn.OneHot(depth=mask_size)(decoding_order), ms.float32)
        mask_matrix = 1 - nn.Triu()(ops.Ones()((mask_size, mask_size), ms.float32))
        order_mask_backward = self.my_einsum(mask_matrix, permutation_matrix_reverse, permutation_matrix_reverse)
        mask_attend = ops.expand_dims(ops.GatherD()(order_mask_backward, 2, e_idx), -1)
        mask_d1 = mask.view((mask.shape[0], mask.shape[1], 1, 1))
        mask_bw = mask_d1 * mask_attend
        mask_fw = mask_d1 * (1. - mask_attend)

        n_batch, n_nodes = x.shape[0], x.shape[1]
        all_probs = ops.Zeros()((n_batch, n_nodes, 21), ms.float32)
        h_s = ops.ZerosLike()(h_v)
        s = ops.Zeros()((n_batch, n_nodes), ms.int32)
        h_v_stack = [h_v] + [ops.ZerosLike()(h_v) for _ in range(len(self.decoder_layers))]
        constant = ms.Tensor(omit_aas_np)
        constant_bias = ms.Tensor(bias_aas_np, ms.float32)
        omit_aa_mask_flag = omit_aa_mask is not None

        h_ex_encoder = cat_neighbors_nodes(ops.ZerosLike()(h_s), h_e, e_idx)
        h_exv_encoder = cat_neighbors_nodes(h_v, h_ex_encoder, e_idx)
        h_exv_encoder_fw = mask_fw * h_exv_encoder
        for t_ in range(n_nodes):
            t = decoding_order[:, t_]  # [B]
            chain_mask_gathered = ops.GatherD()(chain_mask, 1, t[:, None])  # [B]
            bias_by_res_gathered = ops.GatherD()(bias_by_res, 1, ms.numpy.tile(t[:, None, None], (1, 1, 21)))[:, 0, \
                                   :]  # [B, 21]
            if (chain_mask_gathered == 0).all():
                s_t = ops.GatherD()(s_true, 1, t[:, None])
            else:
                # Hidden layers
                e_idx_t = ops.GatherD()(e_idx, 1, ms.numpy.tile(t[:, None, None], (1, 1, e_idx.shape[-1])))
                h_e_t = ops.GatherD()(h_e, 1,
                                      ms.numpy.tile(t[:, None, None, None], (1, 1, h_e.shape[-2], h_e.shape[-1])))
                h_es_t = cat_neighbors_nodes(h_s, h_e_t, e_idx_t)
                h_exv_encoder_t = ops.GatherD()(h_exv_encoder_fw, 1,
                                                ms.numpy.tile(t[:, None, None, None], (1, 1, h_exv_encoder_fw.shape[-2],
                                                                                       h_exv_encoder_fw.shape[-1])))
                mask_t = ops.GatherD()(mask, 1, t[:, None])
                for l, layer in enumerate(self.decoder_layers):
                    # Updated relational features for future states
                    h_esv_decoder_t = cat_neighbors_nodes(h_v_stack[l], h_es_t, e_idx_t)
                    h_v_t = ops.GatherD()(h_v_stack[l], 1,
                                          ms.numpy.tile(t[:, None, None], (1, 1, h_v_stack[l].shape[-1])))
                    h_esv_t = ops.GatherD()(mask_bw, 1, ms.numpy.tile(t[:, None, None, None], \
                                                                      (1, 1, mask_bw.shape[-2], mask_bw.shape[
                                                                          -1]))) * h_esv_decoder_t + h_exv_encoder_t
                    h_v_stack[l + 1] = scatter_(layer(h_v_t, h_esv_t, mask_v=mask_t),
                                                ms.numpy.tile(t[:, None, None], (1, 1, h_v.shape[-1])),
                                                h_v_stack[l + 1], axis=1)
                # Sampling step
                h_v_t = ops.GatherD()(h_v_stack[-1], 1,
                                      ms.numpy.tile(t[:, None, None], (1, 1, h_v_stack[-1].shape[-1])))[:, 0]
                logits = self.w_out(h_v_t) / temperature
                probs = ops.Softmax(axis=-1)((logits - constant[None, :] * 1e8 + constant_bias[None, \
                                :] / temperature + bias_by_res_gathered / temperature).astype(ms.float32))
                if pssm_bias_flag:
                    pssm_coef_gathered = ops.GatherD()(pssm_coef, 1, t[:, None])[:, 0]
                    pssm_bias_gathered = ops.GatherD()(pssm_bias, 1, ms.numpy.tile(t[:, None, None], \
                                                     (1, 1, pssm_bias.shape[-1])))[:, 0]
                    probs = (1 - pssm_multi * pssm_coef_gathered[:, None]) * probs + \
                            pssm_multi * pssm_coef_gathered[:, None] * pssm_bias_gathered
                if pssm_log_odds_flag:
                    pssm_log_odds_mask_gathered = ops.GatherD()(pssm_log_odds_mask, 1, \
                                                                ms.numpy.tile(t[:, None, None], \
                                                                 (1, 1, pssm_log_odds_mask.shape[-1])))[:, 0]
                    probs_masked = probs * pssm_log_odds_mask_gathered
                    probs_masked += probs * 0.001
                    probs = probs_masked / ops.ReduceSum(keep_dims=True)(probs_masked, axis=-1)
                if omit_aa_mask_flag:
                    omit_aa_mask_gathered = ops.GatherD()(omit_aa_mask, 1, ms.numpy.tile(t[:, None, None], \
                                                         (1, 1, omit_aa_mask.shape[-1])))[:, 0]
                    probs_masked = probs * (1.0 - omit_aa_mask_gathered)
                    probs = probs_masked / ops.ReduceSum(keep_dims=True)(probs_masked, axis=-1)  # [B, 21]
                probs_ = np.squeeze(probs.asnumpy(), axis=0).astype("float64")
                p = np.array([i / np.sum(probs_) for i in probs_])
                s_t = np.random.multinomial(1, p, size=1)
                s_t = ms.Tensor(np.where(s_t == 1)[1])
                all_probs = scatter_(ops.Cast()((chain_mask_gathered[:, :, None] * probs[:, None, :]), ms.float32), \
                                     ms.numpy.tile(t[:, None, None], (1, 1, 21)), all_probs, axis=1)
            s_true_gathered = ops.GatherD()(s_true, 1, t[:, None])
            s_t = ops.Cast()((s_t * chain_mask_gathered + s_true_gathered * (1.0 - chain_mask_gathered)), ms.int32)
            temp1 = self.w_s(s_t)
            h_s = scatter_(temp1, ms.numpy.tile(t[:, None, None], (1, 1, temp1.shape[-1])), h_s, axis=1)
            s_t = ops.Cast()(s_t, ms.float32)
            s = ops.Cast()(s, ms.float32)
            s = scatter_(s_t, t[:, None], s, axis=1)
        output_dict = {"s": s, "probs": all_probs, "decoding_order": decoding_order}
        return output_dict

    def tied_sample(self, x, randn, s_true, chain_mask, chain_encoding_all, residue_idx, mask=None, temperature=1.0,
                    omit_aas_np=None, bias_aas_np=None, chain_m_pos=None, omit_aa_mask=None, pssm_coef=None,
                    pssm_bias=None, pssm_multi=None, pssm_log_odds_flag=None, pssm_log_odds_mask=None,
                    pssm_bias_flag=None, tied_pos=None, tied_beta=None, bias_by_res=None):
        """tied_sample"""
        # Prepare node and edge embeddings
        e, e_idx = self.features(x, mask, residue_idx, chain_encoding_all)
        h_v = ops.Zeros()((e.shape[0], e.shape[1], e.shape[-1]), ms.float32)
        h_e = self.w_e(e)
        # Encoder is unmasked self-attention
        mask_attend = gather_nodes(ops.expand_dims(mask, -1), e_idx).squeeze(-1)
        mask_attend = ops.expand_dims(mask, -1) * mask_attend
        for layer in self.encoder_layers:
            h_v, h_e = layer(h_v, h_e, e_idx, mask, mask_attend)

        # Decoder uses masked self-attention
        chain_mask = chain_mask * chain_m_pos * mask
        _, decoding_order = ops.Sort()((chain_mask + 0.0001) * (ops.Abs()(
            randn)))

        new_decoding_order = []
        for t_dec in list(decoding_order[0,]):
            if t_dec not in list(itertools.chain(*new_decoding_order)):
                list_a = [item for item in tied_pos if t_dec in item]
                if list_a:
                    new_decoding_order.append(list_a[0])
                else:
                    new_decoding_order.append([t_dec])
        decoding_order = ms.numpy.tile(ms.Tensor(list(itertools.chain(*new_decoding_order)))[None,], (
            x.shape[0], 1))

        mask_size = e_idx.shape[1]
        permutation_matrix_reverse = ops.Cast()(nn.OneHot(depth=mask_size)(decoding_order), ms.float32)
        mask_matrix = 1 - nn.Triu()(ops.Ones()((mask_size, mask_size), ms.float32))
        order_mask_backward = self.my_einsum(mask_matrix, permutation_matrix_reverse, permutation_matrix_reverse)
        mask_attend = ops.expand_dims(ops.GatherD()(order_mask_backward, 2, e_idx), -1)
        mask_d1 = mask.view((mask.shape[0], mask.shape[1], 1, 1))
        mask_bw = mask_d1 * mask_attend
        mask_fw = mask_d1 * (1. - mask_attend)

        n_batch, n_nodes = x.shape[0], x.shape[1]
        all_probs = ops.Zeros()((n_batch, n_nodes, 21), ms.float32)
        h_s = ops.ZerosLike()(h_v)
        s = ops.Zeros()((n_batch, n_nodes), ms.int32)
        h_v_stack = [h_v] + [ops.ZerosLike()(h_v) for _ in range(len(self.decoder_layers))]
        constant = ms.Tensor(omit_aas_np)
        constant_bias = ms.Tensor(bias_aas_np)
        omit_aa_mask_flag = omit_aa_mask is not None

        h_ex_encoder = cat_neighbors_nodes(ops.ZerosLike()(h_s), h_e, e_idx)
        h_exv_encoder = cat_neighbors_nodes(h_v, h_ex_encoder, e_idx)
        h_exv_encoder_fw = mask_fw * h_exv_encoder
        for t_list in new_decoding_order:
            logits = 0.0
            logit_list = []
            done_flag = False
            t = None
            for t in t_list:
                t = int(t)
                if (chain_mask[:, t] == 0).all():
                    s_t = s_true[:, t]
                    for t1 in t_list:
                        t1 = int(t1)
                        h_s[:, t1, :] = self.w_s(s_t)
                        s[:, t1] = s_t
                    done_flag = True
                    break
                else:
                    e_idx_t = e_idx[:, t:t + 1, :]
                    h_e_t = h_e[:, t:t + 1, :, :]
                    h_es_t = cat_neighbors_nodes(h_s, h_e_t, e_idx_t)
                    h_exv_encoder_t = h_exv_encoder_fw[:, t:t + 1, :, :]
                    mask_t = mask[:, t:t + 1]
                    for l, layer in enumerate(self.decoder_layers):
                        h_esv_decoder_t = cat_neighbors_nodes(h_v_stack[l], h_es_t, e_idx_t)
                        h_v_t = h_v_stack[l][:, t:t + 1, :]
                        h_esv_t = mask_bw[:, t:t + 1, :, :] * h_esv_decoder_t + h_exv_encoder_t
                        h_v_stack[l + 1][:, t, :] = layer(h_v_t, h_esv_t, mask_v=mask_t).squeeze(1)
                    h_v_t = h_v_stack[-1][:, t, :]
                    logit_list.append((self.w_out(h_v_t) / temperature) / len(t_list))
                    logits += tied_beta[t] * (self.w_out(h_v_t) / temperature) / len(t_list)
            if done_flag:
                pass
            else:
                bias_by_res_gathered = bias_by_res[:, t, :]  # [B, 21]
                probs = ops.Softmax(axis=-1)((logits - constant[None, :] * 1e8 + constant_bias[None, \
                            :] / temperature + bias_by_res_gathered / temperature).astype(ms.float32))
                if pssm_bias_flag:
                    pssm_coef_gathered = pssm_coef[:, t]
                    pssm_bias_gathered = pssm_bias[:, t]
                    probs = (1 - pssm_multi * pssm_coef_gathered[:, None]) * probs + pssm_multi * \
                            pssm_coef_gathered[:, None] * pssm_bias_gathered
                if pssm_log_odds_flag:
                    pssm_log_odds_mask_gathered = pssm_log_odds_mask[:, t]
                    probs_masked = probs * pssm_log_odds_mask_gathered
                    probs_masked += probs * 0.001
                    probs = probs_masked / ops.ReduceSum(keep_dims=True)(probs_masked, axis=-1)  # [B, 21]
                if omit_aa_mask_flag:
                    omit_aa_mask_gathered = omit_aa_mask[:, t]
                    probs_masked = probs * (1.0 - omit_aa_mask_gathered)
                    probs = probs_masked / ops.ReduceSum(keep_dims=True)(probs_masked, axis=-1)  # [B, 21]
                s_t_repeat = ops.multinomial(probs, 1).squeeze(-1)
                for t in t_list:
                    t = int(t)
                    h_s[:, t, :] = self.w_s(s_t_repeat)
                    s[:, t] = s_t_repeat
                    all_probs[:, t, :] = ops.Cast()(probs, ms.float32)
        output_dict = {"s": s, "probs": all_probs, "decoding_order": decoding_order}
        return output_dict

    def conditional_probs(self, x, s, mask, chain_m, residue_idx, chain_encoding_all, randn, backbone_only=False):
        """ Graph-conditioned sequence model """
        # Prepare node and edge embeddings
        e, e_idx = self.features(x, mask, residue_idx, chain_encoding_all)
        h_v_enc = ops.Zeros()((e.shape[0], e.shape[1], e.shape[-1]), ms.float32)
        h_e = self.w_e(e)

        # Encoder is unmasked self-attention
        mask_attend = gather_nodes(ops.expand_dims(mask, -1), e_idx).squeeze(-1)
        mask_attend = ops.expand_dims(mask, -1) * mask_attend
        for layer in self.encoder_layers:
            h_v_enc, h_e = layer(h_v_enc, h_e, e_idx, mask, mask_attend)

        # Concatenate sequence embeddings for autoregressive decoder
        h_s = self.w_s(s)
        h_es = cat_neighbors_nodes(h_s, h_e, e_idx)

        # Build encoder embeddings
        h_ex_encoder = cat_neighbors_nodes(ops.ZerosLike()(h_s), h_e, e_idx)
        h_exv_encoder = cat_neighbors_nodes(h_v_enc, h_ex_encoder, e_idx)

        chain_m = chain_m * mask  # update chain_m to include missing regions

        chain_m_np = chain_m.asnumpy()
        idx_to_loop = np.argwhere(chain_m_np[0, :] == 1)[:, 0]
        log_conditional_probs = ops.Cast()(ops.Zeros()([x.shape[0], chain_m.shape[1], 21]), ms.float32)

        for idx in idx_to_loop:
            h_v = Tensor.copy(h_v_enc)
            order_mask = ops.Cast()(ops.Zeros()(chain_m.shape[1]), ms.float32)
            if backbone_only:
                order_mask = ops.Cast()(ops.Ones()(chain_m.shape[1]), ms.float32)
                order_mask[idx] = 0.
            else:
                order_mask = ops.Cast()(ops.Zeros()(chain_m.shape[1]), ms.float32)
                order_mask[idx] = 1.
            _, decoding_order = ops.Sort()((order_mask[None,] + 0.0001) * (ops.Abs()(
                randn)))
            mask_size = e_idx.shape[1]
            permutation_matrix_reverse = ops.Cast()(nn.OneHot(depth=mask_size)(decoding_order), ms.float32)
            mask_matrix = 1 - nn.Triu()(ops.Ones()((mask_size, mask_size), ms.float32))
            order_mask_backward = self.my_einsum(mask_matrix, permutation_matrix_reverse, permutation_matrix_reverse)
            mask_attend = ops.expand_dims(ops.GatherD()(order_mask_backward, 2, e_idx), -1)
            mask_d1 = mask.view((mask.shape[0], mask.shape[1], 1, 1))
            mask_bw = mask_d1 * mask_attend
            mask_fw = mask_d1 * (1. - mask_attend)

            h_exv_encoder_fw = mask_fw * h_exv_encoder
            for layer in self.decoder_layers:
                # Masked positions attend to encoder information, unmasked see.
                h_esv = cat_neighbors_nodes(h_v, h_es, e_idx)
                h_esv = mask_bw * h_esv + h_exv_encoder_fw
                h_v = layer(h_v, h_esv, mask)

            logits = self.w_out(h_v)
            log_probs = nn.LogSoftmax(axis=-1)(logits)
            log_conditional_probs[:, idx, :] = log_probs[:, idx, :]
        return log_conditional_probs

    def unconditional_probs(self, x, mask, residue_idx, chain_encoding_all):
        """ Graph-conditioned sequence model """
        # Prepare node and edge embeddings
        e, e_idx = self.features(x, mask, residue_idx, chain_encoding_all)
        h_v = ops.Zeros()((e.shape[0], e.shape[1], e.shape[-1]), ms.float32)
        h_e = self.w_e(e)

        # Encoder is unmasked self-attention
        mask_attend = gather_nodes(ops.expand_dims(mask, -1), e_idx).squeeze(-1)
        mask_attend = ops.expand_dims(mask, -1) * mask_attend
        for layer in self.encoder_layers:
            h_v, h_e = layer(h_v, h_e, e_idx, mask, mask_attend)

        # Build encoder embeddings
        h_ex_encoder = cat_neighbors_nodes(ops.ZerosLike()(h_v), h_e, e_idx)
        h_exv_encoder = cat_neighbors_nodes(h_v, h_ex_encoder, e_idx)

        order_mask_backward = ops.Zeros()((x.shape[0], x.shape[1], x.shape[1]), ms.float32)
        mask_attend = ops.expand_dims(ops.GatherD()(order_mask_backward, 2, e_idx), -1)
        mask_d1 = mask.view((mask.shape[0], mask.shape[1], 1, 1))
        mask_fw = mask_d1 * (1. - mask_attend)

        h_exv_encoder_fw = mask_fw * h_exv_encoder
        for layer in self.decoder_layers:
            h_v = layer(h_v, h_exv_encoder_fw, mask)

        logits = self.w_out(h_v)
        log_probs = nn.LogSoftmax(axis=-1)(logits)
        return log_probs
