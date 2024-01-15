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
"""Feature extraction"""

import math
import numpy as np
import mindspore as ms
import mindspore.ops as ops
import mindspore.nn as nn
from mindspore import context
# pylint: disable=relative-beyond-top-level
from .basic_modules import GVP, LayerNorm, Dense
from .util import normalize, norm, nan_to_num, rbf, flatten_graph, ms_transpose, ms_padding_without_val


class GVPInputFeaturizer(nn.Cell):
    """Input feature extraction for GVP"""

    @staticmethod
    def get_node_features(coords, coord_mask, with_coord_mask=True):
        """Get node features"""
        node_scalar_features = GVPInputFeaturizer._dihedrals(coords)
        if with_coord_mask:
            coord_mask = ops.ExpandDims()(ops.Cast()(coord_mask, ms.float32), -1)
            node_scalar_features = ops.Concat(axis=-1)([node_scalar_features, coord_mask])
        x_ca = coords[:, :, 1]
        orientations = GVPInputFeaturizer._orientations(x_ca)
        sidechains = GVPInputFeaturizer._sidechains(coords)
        node_vector_features = ops.Concat(axis=-2)([orientations, ops.ExpandDims()(sidechains, -2)])
        return node_scalar_features, node_vector_features

    @staticmethod
    def _orientations(x):

        forward = normalize(x[:, 1:] - x[:, :-1])
        backward = normalize(x[:, :-1] - x[:, 1:])
        forward = ops.concat((forward, ops.Zeros()((forward.shape[0], 1, forward.shape[2]), ms.float32)), 1)
        backward = ops.concat((ops.Zeros()((backward.shape[0], 1, backward.shape[2]), ms.float32), backward), 1)

        output = ops.Concat(axis=-2)([ops.ExpandDims()(forward, -2), ops.ExpandDims()(backward, -2)])
        return output

    @staticmethod
    def _sidechains(x):
        n, origin, c = x[:, :, 0], x[:, :, 1], x[:, :, 2]
        c, n = normalize(c - origin), normalize(n - origin)
        bisector = normalize(c + n)
        perp = normalize(ms.numpy.cross(c, n))
        vec = -bisector * math.sqrt(1 / 3) - perp * math.sqrt(2 / 3)
        return vec

    @staticmethod
    def _dihedrals(x, eps=1e-7):
        """Dihedron"""

        y = x[:, :, :3].reshape((x.shape[0], (x.shape[1] * x.shape[2]), x.shape[3]))
        bsz = x.shape[0]
        dx = y[:, 1:] - y[:, :-1]
        u = normalize(dx, dim=-1)
        u_2 = u[:, :-2]
        u_1 = u[:, 1:-1]
        u_0 = u[:, 2:]

        # Backbone normals
        n_2 = normalize(ms.numpy.cross(u_2, u_1), dim=-1)
        n_1 = normalize(ms.numpy.cross(u_1, u_0), dim=-1)

        # Angle between normals
        cosd = ops.ReduceSum()(n_2 * n_1, -1)

        min_value = ms.Tensor((-1 + eps), ms.float32)
        max_value = ms.Tensor((1 - eps), ms.float32)
        cosd = ops.clip_by_value(cosd, clip_value_min=min_value, clip_value_max=max_value)
        d = ops.Sign()((u_2 * n_1).sum(-1)) * ops.ACos()(cosd)

        # This scheme will remove phi[0], psi[-1], omega[-1]
        d = ms_padding_without_val(d, [1, 2])
        d = ops.Reshape()(d, (bsz, -1, 3))
        # Lift angle representations to the circle
        d_features = ops.Concat(axis=-1)([ops.Cos()(d), ops.Sin()(d)])
        return d_features

    @staticmethod
    def _positional_embeddings(edge_index,
                               num_embeddings=None,
                               num_positional_embeddings=16):
        """Positional embeddings"""

        num_embeddings = num_embeddings or num_positional_embeddings or []
        d = edge_index[0] - edge_index[1]

        frequency = ops.Exp()(
            ms.numpy.arange(0, num_embeddings, 2, dtype=ms.float32)
            * -(np.log(10000.0) / num_embeddings)
        )
        angles = ops.ExpandDims()(d, -1) * frequency
        e = ops.Concat(-1)((ops.Cos()(angles), ops.Sin()(angles)))
        return e

    @staticmethod
    def _dist(x, coord_mask, padding_mask, top_k_neighbors):
        """ Pairwise euclidean distances """
        bsz, maxlen = x.shape[0], x.shape[1]
        coord_mask = ops.Cast()(coord_mask, ms.float32)
        coord_mask_2d = ops.ExpandDims()(coord_mask, 1) * ops.ExpandDims()(coord_mask, 2)
        residue_mask = ~padding_mask
        residue_mask = ops.Cast()(residue_mask, ms.float32)
        residue_mask_2d = ops.ExpandDims()(residue_mask, 1) * ops.ExpandDims()(residue_mask, 2)
        dx = ops.ExpandDims()(x, 1) - ops.ExpandDims()(x, 2)
        d = coord_mask_2d * norm(dx, dim=-1)

        # sorting preference: first those with coords, then among the residues that
        # exist but are masked use distance in sequence as tie breaker, and then the
        # residues that came from padding are last
        seqpos = ms.numpy.arange(maxlen)
        seqpos_1 = ops.ExpandDims()(seqpos, 1)
        seqpos_0 = ops.ExpandDims()(seqpos, 0)
        d_seq = ops.Abs()(seqpos_1 - seqpos_0)
        if bsz != 1:
            d_seq = ms.numpy.tile(d_seq, (bsz, 1, 1))
        coord_mask_2d = ops.Cast()(coord_mask_2d, ms.bool_)
        residue_mask_2d = ops.Cast()(residue_mask_2d, ms.bool_)
        verse_coord_mask_2d = ops.Cast()(~coord_mask_2d, ms.float32)
        verse_residue_mask_2d = ops.Cast()(~residue_mask_2d, ms.float32)
        d_adjust = nan_to_num(d) + (verse_coord_mask_2d) * (1e8 + d_seq * 1e6) + (
            verse_residue_mask_2d) * (1e10)

        if top_k_neighbors == -1:
            d_neighbors = d_adjust / 1e4
            e_idx = seqpos.repeat(
                *d_neighbors.shape[:-1], 1)
        else:
            d_adjust = d_adjust / 1e4
            if context.get_context("device_target") == "GPU":
                d_neighbors, e_idx = ops.Sort(axis=-1, descending=True)(d_adjust)
            else:
                d_neighbors, e_idx = ops.TopK(sorted=True)(d_adjust, d_adjust.shape[-1])
            d_neighbors, e_idx = d_neighbors[..., ::-1], e_idx[..., ::-1]
            d_neighbors, e_idx = d_neighbors[:, :, 0:int(min(top_k_neighbors, x.shape[1]))], \
                                 e_idx[:, :, 0:int(min(top_k_neighbors, x.shape[1]))]
        d_neighbors = ms.Tensor(d_neighbors, ms.float32)*1e4
        coord_mask_neighbors = (d_neighbors < 5e7)
        residue_mask_neighbors = (d_neighbors < 5e9)
        output = [d_neighbors, e_idx, coord_mask_neighbors, residue_mask_neighbors]
        return output


class Normalize(nn.Cell):
    """Normalization"""

    def __init__(self, features, epsilon=1e-6):
        super(Normalize, self).__init__()
        self.gain = ms.Parameter(ops.Ones()(features, ms.float32))
        self.bias = ms.Parameter(ops.Zeros()(features, ms.float32))
        self.epsilon = epsilon

    def construct(self, x, dim=-1):
        """Normalization construction"""

        mu = x.mean(dim, keep_dims=True)
        sigma = ops.Sqrt()(x.var(dim, keepdims=True) + self.epsilon)
        gain = self.gain
        bias = self.bias
        # Reshape
        if dim != -1:
            shape = [1] * len(mu.size())
            shape[dim] = self.gain.size()[0]
            gain = gain.view(shape)
            bias = bias.view(shape)
        return gain * (x - mu) / (sigma + self.epsilon) + bias


class DihedralFeatures(nn.Cell):
    """Dihedral features"""

    def __init__(self, node_embed_dim):
        """ Embed dihedral angle features. """
        super(DihedralFeatures, self).__init__()
        # 3 dihedral angles; sin and cos of each angle
        node_in = 6
        # Normalization and embedding
        self.node_embedding = Dense(node_in, node_embed_dim, has_bias=True)
        self.norm_nodes = Normalize(node_embed_dim)

    @staticmethod
    def _dihedrals(x, eps=1e-7, return_angles=False):
        """Dihedron in DihedralFeatures"""

        # First 3 coordinates are N, CA, C
        x = x[:, :, :3, :].reshape(x.shape[0], 3 * x.shape[1], 3)

        # Shifted slices of unit vectors
        dx = x[:, 1:, :] - x[:, :-1, :]
        u = ops.L2Normalize(axis=-1)(dx)
        u_2 = u[:, :-2, :]
        u_1 = u[:, 1:-1, :]
        u_0 = u[:, 2:, :]
        # Backbone normals
        n_2 = ops.L2Normalize(axis=-1)(ms.numpy.cross(u_2, u_1))
        n_1 = ops.L2Normalize(axis=-1)(ms.numpy.cross(u_1, u_0))

        # Angle between normals
        cosd = (n_2 * n_1).sum(-1)
        min_value = ms.Tensor((-1 + eps), ms.float32)
        max_value = ms.Tensor((1 - eps), ms.float32)
        cosd = ops.clip_by_value(cosd, clip_value_min=min_value, clip_value_max=max_value)
        d = ops.Sign()((u_2 * n_1).sum(-1)) * ops.ACos()(cosd)

        # This scheme will remove phi[0], psi[-1], omega[-1]
        d = ms_padding_without_val(d, [1, 2])
        d = d.view((d.shape[0], int(d.shape[1] / 3), 3))
        phi, psi, omega = ops.Unstack(axis=-1)(d)

        if return_angles:
            return phi, psi, omega

        # Lift angle representations to the circle
        d_features = ops.Concat(axis=2)((ops.Cos()(d), ops.Sin()(d)))
        return d_features

    def construct(self, x):
        """ Featurize coordinates as an attributed graph """
        v = self._dihedrals(x)
        v = self.node_embedding(v)
        v = self.norm_nodes(v)
        return v


class GVPGraphEmbedding(GVPInputFeaturizer):
    """GVP graph embedding"""

    def __init__(self, args):
        super().__init__()
        self.top_k_neighbors = args.top_k_neighbors
        self.num_positional_embeddings = 16
        self.remove_edges_without_coords = True
        node_input_dim = (7, 3)
        edge_input_dim = (34, 1)
        node_hidden_dim = (args.node_hidden_dim_scalar,
                           args.node_hidden_dim_vector)
        edge_hidden_dim = (args.edge_hidden_dim_scalar,
                           args.edge_hidden_dim_vector)
        self.embed_node = nn.SequentialCell(
            [GVP(node_input_dim, node_hidden_dim, activations=(None, None)),
             LayerNorm(node_hidden_dim, eps=1e-4)]
        )
        self.embed_edge = nn.SequentialCell(
            [GVP(edge_input_dim, edge_hidden_dim, activations=(None, None)),
             LayerNorm(edge_hidden_dim, eps=1e-4)]
        )
        self.embed_confidence = Dense(16, args.node_hidden_dim_scalar)

    def construct(self, coords, coord_mask, padding_mask, confidence):
        """GVP graph embedding construction"""

        node_features = self.get_node_features(coords, coord_mask)

        edge_features, edge_index = self.get_edge_features(
            coords, coord_mask, padding_mask)
        node_embeddings_scalar, node_embeddings_vector = self.embed_node(node_features)
        edge_embeddings = self.embed_edge(edge_features)

        rbf_rep = rbf(confidence, 0., 1.)

        node_embeddings = (
            node_embeddings_scalar + self.embed_confidence(rbf_rep),
            node_embeddings_vector
        )


        node_embeddings, edge_embeddings, edge_index = flatten_graph(
            node_embeddings, edge_embeddings, edge_index)
        return node_embeddings, edge_embeddings, edge_index

    def get_edge_features(self, coords, coord_mask, padding_mask):
        """Get edge features"""

        x_ca = coords[:, :, 1]

        # Get distances to the top k neighbors
        e_dist, e_idx, e_coord_mask, e_residue_mask = GVPInputFeaturizer._dist(
            x_ca, coord_mask, padding_mask, self.top_k_neighbors)
        # Flatten the graph to be batch size 1 for torch_geometric package
        dest = e_idx
        e_idx_b, e_idx_l, k = e_idx.shape[:3]

        src = ms.numpy.arange(e_idx_l).view((1, e_idx_l, 1))
        src = ops.BroadcastTo((e_idx_b, e_idx_l, k))(src)


        edge_index = ops.Stack(axis=0)([src, dest])

        edge_index = edge_index.reshape((edge_index.shape[0], edge_index.shape[1],
                                         (edge_index.shape[2] * edge_index.shape[3])))

        # After flattening, [B, E]
        e_dist = e_dist.reshape((e_dist.shape[0], (e_dist.shape[1] * e_dist.shape[2])))

        e_coord_mask = e_coord_mask.reshape((e_coord_mask.shape[0], (e_coord_mask.shape[1] * e_coord_mask.shape[2])))
        e_coord_mask = ops.ExpandDims()(e_coord_mask, -1)
        e_residue_mask = e_residue_mask.reshape((e_residue_mask.shape[0],
                                                 (e_residue_mask.shape[1] * e_residue_mask.shape[2])))

        # Calculate relative positional embeddings and distance RBF
        pos_embeddings = GVPInputFeaturizer._positional_embeddings(
            edge_index,
            num_positional_embeddings=self.num_positional_embeddings,
        )
        d_rbf = rbf(e_dist, 0., 20.)

        # Calculate relative orientation
        x_src = ops.ExpandDims()(x_ca, 2)
        x_src = ops.BroadcastTo((-1, -1, k, -1))(x_src)
        x_src = x_src.reshape((x_src.shape[0], (x_src.shape[1] * x_src.shape[2]), x_src.shape[3]))

        a = ops.ExpandDims()(edge_index[1, :, :], -1)
        a = ops.BroadcastTo((e_idx_b, e_idx_l * k, 3))(a)
        x_dest = ops.GatherD()(
            x_ca,
            1,
            a
        )
        coord_mask_src = ops.ExpandDims()(coord_mask, 2)
        coord_mask_src = ops.BroadcastTo((-1, -1, k))(coord_mask_src)
        coord_mask_src = coord_mask_src.reshape((coord_mask_src.shape[0],
                                                 (coord_mask_src.shape[1] * coord_mask_src.shape[2])))

        b = ops.BroadcastTo((e_idx_b, e_idx_l * k))(edge_index[1, :, :])

        coord_mask_dest = ops.GatherD()(
            coord_mask,
            1,
            b
        )
        e_vectors = x_src - x_dest
        # For the ones without coordinates, substitute in the average vector
        e_coord_mask = ops.Cast()(e_coord_mask, ms.float32)
        e_vector_mean = ops.ReduceSum(keep_dims=True) \
                            (e_vectors * e_coord_mask, axis=1) / ops.ReduceSum(keep_dims=True)(e_coord_mask, axis=1)
        e_coord_mask = ops.Cast()(e_coord_mask, ms.bool_)
        e_vectors = e_vectors * e_coord_mask + e_vector_mean * ~(e_coord_mask)
        # Normalize and remove nans
        edge_s = ops.Concat(axis=-1)([d_rbf, pos_embeddings])
        edge_v = ops.ExpandDims()(normalize(e_vectors), -2)
        edge_s, edge_v = map(nan_to_num, (edge_s, edge_v))
        # Also add indications of whether the coordinates are present

        edge_s = ops.Concat(axis=-1)([
            edge_s,
            ops.ExpandDims()((~coord_mask_src).astype(np.float32), -1),
            ops.ExpandDims()((~coord_mask_dest).astype(np.float32), -1)])
        e_residue_mask = ops.Cast()(e_residue_mask, ms.bool_)
        fill_value = ms.Tensor(-1, dtype=edge_index.dtype)
        edge_index = edge_index.masked_fill(~e_residue_mask, fill_value)

        if self.remove_edges_without_coords:
            edge_index = ops.masked_fill(edge_index, ~e_coord_mask.squeeze(-1), fill_value)

        return (edge_s, edge_v), ms_transpose(edge_index, 0, 1)
