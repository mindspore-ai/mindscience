# Copyright 2021-2022 @ Shenzhen Bay Laboratory &
#                       Peking University &
#                       Huawei Technologies Co., Ltd
#
# This code is a part of AIchemist package.
#
# The AIchemist is open-source software based on the AI-framework:
# MindSpore (https://www.mindspore.cn/)
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
"""
embedding
"""

import math

import mindspore as ms
from mindspore import nn
from mindspore import ops
import mindspore.nn.probability.distribution as msd

from .. import core
from .. import layers
from ..data import feature
from ..core import Registry as R


def get_mask(n_lig_nodes, n_rec_nodes):
    """_summary_

    Args:
        n_lig_nodes (_type_): _description_
        n_rec_nodes (_type_): _description_

    Returns:
        _type_: _description_
    """
    rows = int(n_lig_nodes.sum())
    cols = int(n_rec_nodes.sum())
    mask = ops.zeros([rows, cols])
    partial_l = 0
    partial_r = 0
    for l_n, r_n in zip(n_lig_nodes, n_rec_nodes):
        mask[partial_l: partial_l + l_n, partial_r: partial_r + r_n] = 1
        partial_l = partial_l + l_n
        partial_r = partial_r + r_n
    return mask


@R.register('nn.EquiBind')
class EquiBind(core.Cell):
    """_summary_

    Args:
        core (_type_): _description_
    """

    def __init__(self, n_lays, use_rec_atoms, shared_layers, noise_decay_rate, cross_msgs, noise_initial,
                 use_edge_features_in_gmn, use_mean_node_features, residue_emb_dim, iegmn_lay_hid_dim, num_att_heads,
                 dropout, random_vec_dim=0, random_vec_std=1, use_scalar_features=True,
                 num_lig_feats=None, move_keypts_back=False, normalize_z_lig_directions=False,
                 unnormalized_kpt_weights=False, centroid_keypts_construction_rec=False,
                 centroid_keypts_construction_lig=False, rec_no_softmax=False, lig_no_softmax=False,
                 normalize_z_rec_directions=False, separate_lig=False, use_evolved_lig=False,
                 centroid_keypts_construction=False, evolve_only=False, save_trajectories=False, **kwargs):
        super().__init__()
        self._kwargs = kwargs
        self.n_lays = n_lays
        self.cross_msgs = cross_msgs
        self.shared_layers = shared_layers
        self.save_trajectories = save_trajectories
        self.unnormalized_kpt_weights = unnormalized_kpt_weights
        self.use_rec_atoms = use_rec_atoms
        self.noise_decay_rate = noise_decay_rate
        self.noise_initial = noise_initial
        self.use_edge_features_in_gmn = use_edge_features_in_gmn
        self.use_mean_node_features = use_mean_node_features
        self.random_vec_dim = random_vec_dim
        self.random_vec_std = random_vec_std
        self.move_keypts_back = move_keypts_back
        self.normalize_z_lig_directions = normalize_z_lig_directions
        self.centroid_keypts_construction = centroid_keypts_construction
        self.centroid_keypts_construction_rec = centroid_keypts_construction_rec
        self.centroid_keypts_construction_lig = centroid_keypts_construction_lig
        self.normalize_z_rec_directions = normalize_z_rec_directions
        self.rec_no_softmax = rec_no_softmax
        self.lig_no_softmax = lig_no_softmax
        self.evolve_only = evolve_only
        self.use_evolved_lig = use_evolved_lig
        self.separate_lig = separate_lig
        self.residue_emb_dim = residue_emb_dim
        self.iegmn_lay_hid_dim = iegmn_lay_hid_dim
        self.dropout = dropout

        self.lig_atom_embedder = layers.AtomEncoder(
            emb_dim=residue_emb_dim - self.random_vec_dim,
            feature_dims=feature.lig_feature_dims,
            use_scalar_feat=use_scalar_features,
            n_feats_to_use=num_lig_feats
        )
        if self.separate_lig:
            self.lig_separate_atom_embedder = layers.AtomEncoder(
                emb_dim=residue_emb_dim - self.random_vec_dim,
                feature_dims=feature.lig_feature_dims,
                use_scalar_feat=use_scalar_features,
                n_feats_to_use=num_lig_feats
            )
        if self.use_rec_atoms:
            self.rec_embedder = layers.AtomEncoder(
                emb_dim=residue_emb_dim - self.random_vec_dim,
                feature_dims=feature.rec_atom_feature_dims,
                use_scalar_feat=use_scalar_features)
        else:
            self.rec_embedder = layers.AtomEncoder(
                emb_dim=residue_emb_dim - self.random_vec_dim,
                feature_dims=feature.rec_residue_feature_dims,
                use_scalar_feat=use_scalar_features
            )

        # attn layers
        self.n_att_heads = num_att_heads
        self.out_feats_dim = iegmn_lay_hid_dim
        self.keypts_attn_lig = nn.SequentialCell([
            nn.Dense(self.out_feats_dim, self.n_att_heads * self.out_feats_dim, has_bias=False)
        ])
        self.keypts_queries_lig = nn.SequentialCell([
            nn.Dense(self.out_feats_dim, self.n_att_heads * self.out_feats_dim, has_bias=False)
        ])
        self.keypts_attn_rec = nn.SequentialCell([
            nn.Dense(self.out_feats_dim, self.n_att_heads * self.out_feats_dim, has_bias=False)
        ])
        self.keypts_queries_rec = nn.SequentialCell([
            nn.Dense(self.out_feats_dim, self.n_att_heads * self.out_feats_dim, has_bias=False)
        ])
        self.activation = nn.LeakyReLU()

        self.h_mean_lig = nn.SequentialCell([
            nn.Dense(self.out_feats_dim, self.out_feats_dim),
            nn.Dropout(dropout),
            self.activation,
        ])
        self.h_mean_rec = nn.SequentialCell([
            nn.Dense(self.out_feats_dim, self.out_feats_dim),
            nn.Dropout(dropout),
            self.activation,
        ])

        if self.unnormalized_kpt_weights:
            self.scale_lig = nn.Dense(self.out_feats_dim, 1)
            self.scale_rec = nn.Dense(self.out_feats_dim, 1)

        if self.normalize_z_lig_directions:
            self.z_lig_dir_norm = layers.CoordsNorm()
        if self.normalize_z_rec_directions:
            self.z_rec_dir_norm = layers.CoordsNorm()
        self.iegmn_layers = nn.CellList()

    def __repr__(self):
        return "IEGMN " + str(self.__dict__)

    def initialize(self, **kwargs):
        """_summary_
        """
        kwargs.update(self._kwargs)
        input_node_feats_dim = self.residue_emb_dim
        if self.use_mean_node_features:
            input_node_feats_dim += 5  # Additional features from mu_r_norm

        self.iegmn_layers.append(
            layers.IEGMN(orig_h_feats_dim=input_node_feats_dim,
                         h_feats_dim=input_node_feats_dim,
                         out_feats_dim=self.iegmn_lay_hid_dim,
                         cross_msgs=self.cross_msgs,
                         dropout=self.dropout,
                         save_trajectories=self.save_trajectories, **kwargs))

        if self.shared_layers:
            lay = layers.IEGMN(orig_h_feats_dim=input_node_feats_dim,
                               h_feats_dim=self.iegmn_lay_hid_dim,
                               out_feats_dim=self.iegmn_lay_hid_dim,
                               cross_msgs=self.cross_msgs,
                               dropout=self.dropout,
                               save_trajectories=self.save_trajectories, **kwargs)
            for _ in range(self.n_lays):
                self.iegmn_layers.append(lay)
        else:
            for _ in range(self.n_lays):
                self.iegmn_layers.append(
                    layers.IEGMN(orig_h_feats_dim=input_node_feats_dim,
                                 h_feats_dim=self.iegmn_lay_hid_dim,
                                 out_feats_dim=self.iegmn_lay_hid_dim,
                                 cross_msgs=self.cross_msgs,
                                 save_trajectories=self.save_trajectories, **kwargs))

    def construct(self, lig_graph, rec_graph, geo_graph, epoch=0):
        """_summary_

        Args:
            lig_graph (_type_): _description_
            rec_graph (_type_): _description_
            geo_graph (_type_): _description_
            epoch (int, optional): _description_. Defaults to 0.

        Raises:
            RuntimeError: _description_

        Returns:
            _type_: _description_
        """
        rec_coord = rec_graph.coord()
        lig_coord = lig_graph.coord()
        lig_h_feat = self.lig_atom_embedder(lig_graph.node_feat)

        if self.use_rec_atoms:
            rec_h_feat = self.rec_embedder(rec_graph.node_feat)
        else:
            rec_h_feat = self.rec_embedder(rec_graph.node_feat)  # (N_res, emb_dim)

        rand_dist = msd.Normal(mean=0, sd=self.random_vec_std)
        if self.random_vec_dim != 0:
            rand_h_lig = rand_dist.sample((lig_h_feat.shape[0], self.random_vec_dim))
            rand_h_rec = rand_dist.sample((rec_h_feat.shape[0], self.random_vec_dim))
            lig_h_feat = ops.concat([lig_h_feat, rand_h_lig], axis=1)
            rec_h_feat = ops.concat([rec_h_feat, rand_h_rec], axis=1)

        # random noise:
        if self.noise_initial > 0:
            noise_level = self.noise_initial * self.noise_decay_rate ** (epoch + 1)
            lig_h_feat += noise_level * ops.randn(lig_h_feat.shape)
            rec_h_feat += noise_level * ops.randn(rec_h_feat.shape)
            lig_coord += noise_level * ops.randn(lig_coord.shape)
            rec_coord += noise_level * ops.randn(rec_coord.shape)

        lig_new_feat = lig_h_feat
        rec_new_feat = rec_h_feat
        lig_graph.edge_feat *= self.use_edge_features_in_gmn
        rec_graph.edge_feat *= self.use_edge_features_in_gmn

        mask = None
        if self.cross_msgs:
            mask = get_mask(lig_graph.n_nodes, rec_graph.n_nodes)
        full_trajectory = [lig_coord.asnumpy()]
        geom_losses = 0
        for layer in self.iegmn_layers:
            lig_coord, lig_h_feat, rec_coord, rec_h_feat, trajectory, geom_loss = \
                layer(lig_graph=lig_graph,
                      rec_graph=rec_graph,
                      lig_coord=lig_coord,
                      rec_coord=rec_coord,
                      h_lig_feat=lig_h_feat,
                      ori_lig_node_feat=lig_new_feat,
                      h_rec_feat=rec_h_feat,
                      ori_rec_node_feat=rec_new_feat,
                      mask=mask,
                      geo_graph=geo_graph
                      )
            geom_losses += geom_loss
            full_trajectory.extend(trajectory)

        rotations = []
        translations = []
        recs_keypts = []
        ligs_keypts = []
        ligs_evolve = []
        lig_coord_ = []

        if self.evolve_only:
            for idx in range(len(lig_graph)):
                start = lig_graph.cum_nodes[idx] - lig_graph.n_nodes[idx]
                end = lig_graph.cum_nodes[idx]
                z_lig_coords = lig_coord[start:end]
                ligs_evolve.append(z_lig_coords)
            return [rotations, translations, ligs_keypts, recs_keypts, ligs_evolve, geom_losses]

        # TODO: run SVD in batches, if possible
        for idx in range(len(lig_graph)):
            rec_keypts, lig_keypts, rotation, translation = \
                self.batch_svd(idx, lig_graph, rec_graph, lig_h_feat, rec_h_feat, lig_coord, rec_coord)
            recs_keypts.append(rec_keypts)
            ligs_keypts.append(lig_keypts)
            rotations.append(rotation)
            translations.append(translation)
            ligs_evolve.append(z_lig_coords)

            orig_coords_lig = lig_graph.new_coord[start:end]
            rotation = rotations[idx]
            translation = translations[idx]
            assert translation.shape[0] == 1 and translation.shape[1] == 3

            if self.use_evolved_lig:
                predicted_coords = ligs_evolve[idx] @ rotation.T + translation
            else:
                predicted_coords = orig_coords_lig @ rotation.T + translation
            lig_coord_.append(predicted_coords)
        out_ = [lig_coord_, ligs_keypts, recs_keypts, rotations, translations, geom_losses]
        return out_

    def batch_svd(self, idx, lig_graph, rec_graph, lig_h_feat, rec_h_feat, lig_coord, rec_coord):
        """_summary_

        Args:
            idx (_type_): _description_
            lig_graph (_type_): _description_
            rec_graph (_type_): _description_
            lig_h_feat (_type_): _description_
            rec_h_feat (_type_): _description_
            lig_coord (_type_): _description_
            rec_coord (_type_): _description_

        Raises:
            RuntimeError: _description_

        Returns:
            _type_: _description_
        """
        start = lig_graph.cum_nodes[idx] - lig_graph.n_nodes[idx]
        end = lig_graph.cum_nodes[idx]
        rec_start = rec_graph.cum_nodes[idx] - rec_graph.n_nodes[idx]
        rec_end = rec_graph.cum_nodes[idx]

        # Get H vectors
        rec_feat = rec_h_feat[rec_start:rec_end]  # (m, d)
        rec_feat_mean = self.h_mean_rec(rec_feat).mean(axis=0, keepdims=True)  # (1, d)
        lig_feat = lig_h_feat[start:end]  # (n, d)
        lig_feat_mean = self.h_mean_lig(lig_feat).mean(axis=0, keepdims=True)  # (1, d)

        d = lig_feat.shape[1]
        assert d == self.out_feats_dim
        # Z coordinates
        z_rec_coords = rec_coord[rec_start:rec_end]
        z_lig_coords = lig_coord[start:end]

        # Att weights to compute the receptor centroid. They query is the average_h_ligand.
        # Keys are each h_receptor_j
        rec_attn = self.keypts_attn_rec(rec_feat).view(-1, self.n_att_heads, d).transpose(1, 0, 2)
        rec_query = self.keypts_queries_rec(lig_feat_mean).view(1, self.n_att_heads, d).transpose((1, 2, 0))
        att_weights_rec = rec_attn @ rec_query / math.sqrt(d)
        if not self.rec_no_softmax:
            att_weights_rec = ops.softmax(att_weights_rec, axis=1)
        att_weights_rec = att_weights_rec.view(self.n_att_heads, -1)

        # Att weights to compute the ligand centroid. They query is the average_h_receptor. Keys are each h_ligand_i
        lig_attn = self.keypts_attn_lig(lig_feat).view(-1, self.n_att_heads, d).transpose(1, 0, 2)
        lig_query = self.keypts_queries_lig(rec_feat_mean).view(1, self.n_att_heads, d).transpose((1, 2, 0))
        att_weights_lig = lig_attn @ lig_query / math.sqrt(d)
        if not self.lig_no_softmax:
            att_weights_lig = ops.softmax(att_weights_lig, axis=1)
        att_weights_lig = att_weights_lig.view(self.n_att_heads, -1)

        if self.unnormalized_kpt_weights:
            lig_scales = self.scale_lig(lig_feat)
            rec_scales = self.scale_rec(rec_feat)
            z_lig_coords *= lig_scales
            z_rec_coords *= rec_scales

        if self.centroid_keypts_construction_rec:
            z_rec_mean = z_rec_coords.mean(axis=0)
            z_rec_directions = z_rec_coords - z_rec_mean
            if self.normalize_z_rec_directions:
                z_rec_directions = self.z_rec_dir_norm(z_rec_directions)
            rec_keypts = att_weights_rec @ z_rec_directions  # K_heads, 3
            if self.move_keypts_back:
                rec_keypts += z_rec_mean
        else:
            rec_keypts = att_weights_rec @ z_rec_coords  # K_heads, 3

        if self.centroid_keypts_construction or self.centroid_keypts_construction_lig:
            z_lig_mean = z_lig_coords.mean(axis=0)
            z_lig_directions = z_lig_coords - z_lig_mean
            if self.normalize_z_lig_directions:
                z_lig_directions = self.z_lig_dir_norm(z_lig_directions)
            lig_keypts = att_weights_lig @ z_lig_directions  # K_heads, 3
            if self.move_keypts_back:
                lig_keypts += z_lig_mean
        else:
            lig_keypts = att_weights_lig @ z_lig_coords  # K_heads, 3

        # Apply Kabsch algorithm
        rec_keypts_mean = rec_keypts.mean(axis=0, keepdims=True)  # (1,3)
        lig_keypts_mean = lig_keypts.mean(axis=0, keepdims=True)  # (1,3)

        mat_a = (rec_keypts - rec_keypts_mean).T @ (lig_keypts - lig_keypts_mean) / self.n_att_heads  # 3, 3

        st, ut, vt = mat_a.svd()
        num_it = 0
        while st.min() < 1e-3 or (ops.abs(st.view(1, 3) ** 2 - st.view(3, 1) ** 2 + ops.eye(3))).min() < 1e-2:
            mat_a += ops.rand([3, 3]) * ops.eye(3)
            st, ut, vt = mat_a.svd()
            num_it += 1
            if num_it > 100:
                raise RuntimeError('SVD was consitantly unstable')

        corr_mat = ops.diag(ms.Tensor([1, 1, int(mat_a.det().sign())]))
        rotation = (ut @ corr_mat) @ vt

        translation = rec_keypts_mean - lig_keypts_mean @ rotation.T  # (1,3)
        # end AP 1
        out_ = [rec_keypts, lig_keypts, rotation, translation]
        return out_
