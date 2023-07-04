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
IEGMN layers
"""

from mindspore import ops
from mindspore import nn
from .common import GraphNorm, CoordsNorm
from ..util import scatter
from .transformer import Attention
from .. import core


def get_layer_norm(layer_norm_type, dim):
    if layer_norm_type == 'BN':
        lay = nn.BatchNorm1d(dim)
    elif layer_norm_type == 'LN':
        lay = nn.LayerNorm(dim)
    else:
        lay = nn.Identity()
    return lay


def get_norm(layer_norm_type, dim):
    if layer_norm_type == 'BN':
        return nn.BatchNorm1d(dim)
    if layer_norm_type == 'LN':
        return nn.LayerNorm(dim)
    if layer_norm_type == 'GN':
        return GraphNorm(dim)
    assert layer_norm_type in ['0', 0]
    return nn.Identity()


def apply_norm(g, h, norm_type, norm_layer):
    if norm_type == 'GN':
        return norm_layer(g, h)
    return norm_layer(h)


class IEGMN(core.Cell):
    """_summary_

    Args:
        core (_type_): _description_
    """

    def __init__(
            self,
            orig_h_feats_dim,
            h_feats_dim,  # in dim of h
            out_feats_dim,  # out dim of h
            cross_msgs,
            final_h_layer_norm,
            use_dist_in_layers,
            skip_weight_h,
            x_connection_init,
            lig_input_edge_feats_dim,
            rec_input_edge_feats_dim,
            layer_norm='0',
            coord_norm='0',
            dropout=0.1,
            save_trajectories=False,
            rec_square_distance_scale=1,
            standard_norm_order=False,
            normalize_coordinate_update=False,
            lig_evolve=True,
            rec_evolve=True,
            fine_tune=False,
            geometry_regularization=False,
            norm_cross_coords_update=False,
            loss_geometry_regularization=False,
            geom_reg_steps=1,
            geometry_reg_step_size=0.1,
            **kwargs):

        super().__init__(**kwargs)
        self.fine_tune = fine_tune
        self.cross_msgs = cross_msgs
        self.normalize_coordinate_update = normalize_coordinate_update
        self.final_h_layer_norm = final_h_layer_norm
        self.use_dist_in_layers = use_dist_in_layers
        self.skip_weight_h = skip_weight_h
        self.x_connection_init = x_connection_init
        self.layer_norm = layer_norm
        self.coord_norm = coord_norm
        self.rec_square_distance_scale = rec_square_distance_scale
        self.geometry_reg_step_size = geometry_reg_step_size
        self.norm_cross_coords_update = norm_cross_coords_update
        self.loss_geometry_regularization = loss_geometry_regularization

        self.lig_evolve = lig_evolve
        self.rec_evolve = rec_evolve
        self.h_feats_dim = h_feats_dim
        self.out_feats_dim = out_feats_dim
        self.standard_norm_order = standard_norm_order
        self.all_sigmas_dist = [1.5 ** x for x in range(15)]
        self.geometry_regularization = geometry_regularization
        self.geom_reg_steps = geom_reg_steps
        self.save_trajectories = save_trajectories
        self.activation = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)

        self.lig_attn = Attention(embed_dim=h_feats_dim)
        self.rec_attn = Attention(embed_dim=h_feats_dim)

        # NODES
        self.lig_node_input = nn.Dense(orig_h_feats_dim + 2 * h_feats_dim + self.out_feats_dim, h_feats_dim)
        self.lig_node_output = nn.Dense(h_feats_dim, out_feats_dim)
        self.rec_node_input = nn.Dense(orig_h_feats_dim + 2 * h_feats_dim + self.out_feats_dim, h_feats_dim)
        self.rec_node_output = nn.Dense(h_feats_dim, out_feats_dim)
        self.node_norm = nn.Identity()  # nn.LayerNorm(h_feats_dim)

        # Coordinate
        self.lig_coords_input = nn.Dense(self.out_feats_dim, self.out_feats_dim)
        self.lig_coords_output = nn.Dense(self.out_feats_dim, 1)
        self.rec_coords_input = nn.Dense(self.out_feats_dim, self.out_feats_dim)
        self.rec_coords_output = nn.Dense(self.out_feats_dim, 1)

        # EDGES
        lig_edge_mlp_input_dim = (self.h_feats_dim * 2) + lig_input_edge_feats_dim
        rec_edge_mlp_input_dim = (self.h_feats_dim * 2) + rec_input_edge_feats_dim
        if self.use_dist_in_layers:
            if self.lig_evolve:
                lig_edge_mlp_input_dim += len(self.all_sigmas_dist)
            if self.rec_evolve:
                rec_edge_mlp_input_dim += len(self.all_sigmas_dist)
        self.lig_input = nn.Dense(lig_edge_mlp_input_dim, self.out_feats_dim)
        self.lig_output = nn.Dense(self.out_feats_dim, self.out_feats_dim)
        self.rec_input = nn.Dense(rec_edge_mlp_input_dim, self.out_feats_dim)
        self.rec_output = nn.Dense(self.out_feats_dim, self.out_feats_dim)

        if self.standard_norm_order:
            self.lig_edge_mlp = nn.SequentialCell([
                self.lig_input, get_layer_norm(self.layer_norm, self.out_feats_dim),
                self.activation, self.dropout,
                self.lig_output, get_layer_norm(self.layer_norm, self.out_feats_dim),
            ])
            self.rec_edge_mlp = nn.SequentialCell([
                self.rec_input, get_layer_norm(self.layer_norm, self.out_feats_dim),
                self.activation, self.dropout,
                self.rec_output, get_layer_norm(self.layer_norm, self.out_feats_dim),
            ])
            self.lig_node_mlp = nn.SequentialCell([
                self.lig_node_input, get_layer_norm(self.layer_norm, self.out_feats_dim),
                self.activation, self.dropout,
                self.lig_node_output, get_layer_norm(self.layer_norm, self.out_feats_dim),
            ])
            self.rec_node_mlp = nn.SequentialCell([
                self.rec_node_input, get_layer_norm(self.layer_norm, self.out_feats_dim),
                self.activation, self.dropout,
                self.rec_node_output, get_layer_norm(self.layer_norm, self.out_feats_dim),
            ])
            self.lig_coords_mlp = nn.SequentialCell([
                self.lig_coords_input,
                get_layer_norm(self.coord_norm, self.out_feats_dim),
                self.activation, self.dropout,
                self.lig_coords_output
            ])
            self.rec_coords_mlp = nn.SequentialCell([
                self.rec_coords_input,
                get_layer_norm(self.coord_norm, self.out_feats_dim),
                self.activation, self.dropout,
                self.rec_coords_output
            ])
        else:
            self.lig_edge_mlp = nn.SequentialCell([
                self.lig_input, self.dropout,
                self.activation, get_layer_norm(self.layer_norm, self.out_feats_dim),
                self.lig_output,
            ])
            self.rec_edge_mlp = nn.SequentialCell(
                self.rec_input, self.dropout,
                self.activation, get_layer_norm(self.layer_norm, self.out_feats_dim),
                self.rec_output,
            )
            self.lig_node_mlp = nn.SequentialCell(
                self.lig_node_input, self.dropout,
                self.activation, get_layer_norm(self.layer_norm, self.out_feats_dim),
                self.lig_node_output,
            )
            self.rec_node_mlp = nn.SequentialCell(
                self.rec_node_input, self.dropout,
                self.activation, get_layer_norm(self.layer_norm, self.out_feats_dim),
                self.rec_node_output,
            )
            self.lig_coords_mlp = nn.SequentialCell(
                self.lig_coords_input,
                self.dropout, self.activation,
                get_layer_norm(self.coord_norm, self.out_feats_dim),
                self.lig_coords_output
            )
            self.rec_coords_mlp = nn.SequentialCell(
                self.rec_coords_input,
                self.dropout, self.activation,
                get_layer_norm(self.coord_norm, self.out_feats_dim),
                self.rec_coords_output
            )

        # normalization of x_i - x_j is not currently used
        if self.normalize_coordinate_update:
            self.lig_coords_norm = CoordsNorm(scale_init=1e-2)
            self.rec_coords_norm = CoordsNorm(scale_init=1e-2)
        if self.fine_tune:
            if self.norm_cross_coords_update:
                self.lig_cross_coords_norm = CoordsNorm(scale_init=1e-2)
                self.rec_cross_coords_norm = CoordsNorm(scale_init=1e-2)
            else:
                self.lig_cross_coords_norm = nn.Identity()
                self.rec_cross_coords_norm = nn.Identity()

        if self.fine_tune:
            self.rec_fine_tune_attn = Attention(self.h_feats_dim)
            self.lig_fine_tune_attn = Attention(self.h_feats_dim)

        self.final_h_norm_lig = get_norm(self.final_h_layer_norm, self.out_feats_dim)
        self.final_h_norm_rec = get_norm(self.final_h_layer_norm, self.out_feats_dim)

    def __repr__(self):
        return "IEGMN Layer " + str(self.__dict__)

    def apply_lig_edges(self, src_feat, trg_feat, edge_feat, x_rel=None):
        if x_rel is not None:
            x_rel_msg = (x_rel ** 2).sum(axis=1, keepdims=True)
            x_rel_msg = ops.concat([ops.exp(-x_rel_msg / sigma) for sigma in self.all_sigmas_dist], axis=-1)
            x_feat = ops.concat([src_feat, trg_feat, edge_feat, x_rel_msg], axis=1)
        else:
            x_feat = ops.concat([src_feat, trg_feat, edge_feat], axis=1)
        return self.lig_edge_mlp(x_feat)

    def apply_rec_edges(self, src_feat, trg_feat, edge_feat, x_rel=None):
        if x_rel is not None:
            x_rel_msg = (x_rel ** 2).sum(axis=1, keepdims=True)
            # divide square distance by 10 to have a nicer separation instead of many 0.00000
            x_rel_msg = ops.concat([ops.exp(-x_rel_msg / self.rec_square_distance_scale / sigma) for sigma in
                                    self.all_sigmas_dist], axis=-1)
            x_feat = ops.concat([src_feat, trg_feat, edge_feat, x_rel_msg], axis=1)
        else:
            x_feat = ops.concat([src_feat, trg_feat, edge_feat], axis=1)
        return self.rec_edge_mlp(x_feat)

    def construct(self, lig_graph, rec_graph, lig_coord, rec_coord, h_lig_feat,
                  ori_lig_node_feat, h_rec_feat, ori_rec_node_feat, mask, geo_graph):
        """_summary_

        Args:
            lig_graph (_type_): _description_
            rec_graph (_type_): _description_
            lig_coord (_type_): _description_
            rec_coord (_type_): _description_
            h_lig_feat (_type_): _description_
            ori_lig_node_feat (_type_): _description_
            h_rec_feat (_type_): _description_
            ori_rec_node_feat (_type_): _description_
            mask (_type_): _description_
            geo_graph (_type_): _description_

        Returns:
            _type_: _description_
        """
        lig_src, lig_trg = lig_graph.edge_list.T
        rec_src, rec_trg = rec_graph.edge_list.T
        lig_node_out = ops.concat([lig_trg, ops.arange(lig_graph.n_node)])
        rec_node_out = ops.concat([rec_trg, ops.arange(rec_graph.n_node)])
        ori_lig_coord = lig_graph.new_coord
        ori_rec_coord = rec_graph.coord

        if self.lig_evolve:
            lig_rel = lig_coord[lig_src] - lig_coord[lig_trg]
            lig_msg = self.apply_lig_edges(ori_lig_node_feat[lig_src],
                                           ori_lig_node_feat[lig_trg], lig_graph.edge_feat, lig_rel)
        else:
            lig_msg = self.apply_lig_edges(ori_lig_node_feat[lig_src], ori_lig_node_feat[lig_trg], lig_graph.edge_feat)
        if self.rec_evolve:
            rec_rel = rec_coord[rec_src] - rec_coord[rec_trg]
            rec_msg = self.apply_rec_edges(ori_rec_node_feat[rec_src],
                                           ori_rec_node_feat[rec_trg], rec_graph.edge_feat, rec_rel)
        else:
            rec_msg = self.apply_lig_edges(ori_rec_node_feat[rec_src], ori_rec_node_feat[rec_trg], rec_graph.edge_feat)

        norm_lig_feat = apply_norm(lig_graph, h_lig_feat, self.final_h_layer_norm, self.final_h_norm_lig)
        norm_rec_feat = apply_norm(rec_graph, h_rec_feat, self.final_h_layer_norm, self.final_h_norm_rec)

        attn_lig_feats = self.lig_attn(norm_lig_feat, norm_rec_feat, norm_rec_feat, mask)
        attn_rec_feats = self.rec_attn(norm_rec_feat, norm_lig_feat, norm_lig_feat, mask)

        norm_lig_feat = apply_norm(lig_graph, attn_lig_feats, self.final_h_layer_norm, self.final_h_norm_lig)
        norm_rec_feat = apply_norm(lig_graph, attn_rec_feats, self.final_h_layer_norm, self.final_h_norm_rec)

        if self.lig_evolve:
            lig_coef = self.lig_coords_mlp(lig_msg)
            if self.normalize_coordinate_update:
                lig_rel = self.lig_coords_norm(lig_rel)
            lig_m = lig_rel * lig_coef
            lig_m = ops.concat([lig_m, ops.zeros([int(lig_graph.n_node), lig_m.shape[1]])])
            lig_x_update = scatter(lig_m, lig_node_out, axis=0, n_axis=lig_graph.n_node, reduce='mean')
            # Inspired by https://arxiv.org/pdf/2108.10521.pdf we use original X and not only graph.ndata['x_now']
            lig_x_evolve = self.x_connection_init * ori_lig_coord + \
                (1. - self.x_connection_init) * lig_coord + lig_x_update
        else:
            lig_x_evolve = lig_coord
        if self.rec_evolve:
            rec_coef = self.rec_coords_mlp(rec_msg)
            if self.normalize_coordinate_update:
                rec_rel = self.rec_coords_norm(rec_rel)
            rec_m = rec_rel * rec_coef
            rec_m = ops.concat([rec_m, ops.zeros([int(rec_graph.n_node), rec_m.shape[1]])])
            rec_x_update = scatter(rec_m, rec_node_out, axis=0, n_axis=rec_graph.n_node, reduce='mean')
            # Inspired by https://arxiv.org/pdf/2108.10521.pdf we use original X and not only graph.ndata['x_now']
            rec_x_evolve = self.x_connection_init * ori_rec_coord + \
                (1. - self.x_connection_init) * rec_coord + rec_x_update
        else:
            rec_x_evolve = rec_coord

        lig_m = ops.concat([lig_msg, ops.zeros([int(lig_graph.n_node), lig_msg.shape[1]])])
        lig_aggr_msg = scatter(lig_m, lig_node_out, axis=0, n_axis=lig_graph.n_node, reduce='mean')
        rec_m = ops.concat([rec_msg, ops.zeros([int(rec_graph.n_node), rec_msg.shape[1]])])
        rec_aggr_msg = scatter(rec_m, rec_node_out, axis=0, n_axis=rec_graph.n_node, reduce='mean')

        if self.fine_tune:
            lig_x_evolve += self.lig_fine_tune_attn(lig_coord, h_rec_feat, h_lig_feat, mask)
            rec_x_evolve += self.rec_fine_tune_attn(rec_coord, h_lig_feat, h_lig_feat, mask)

        trajectory = []
        if self.save_trajectories:
            trajectory.append(lig_x_evolve.asnumpy())
        if self.loss_geometry_regularization:
            src, trg = geo_graph.edge_list.T
            geo_feat = geo_graph.edge_feats
            d_squared = ((lig_x_evolve[src] - lig_x_evolve[trg]) ** 2).sum(axis=1)
            geom_loss = ops.sum((d_squared - geo_feat ** 2) ** 2)
        else:
            geom_loss = 0

        if self.geometry_regularization:
            src, trg = geo_graph.edge_list.T
            geo_feat = geo_graph.edge_feat
            for _ in range(self.geom_reg_steps):
                d_squared = ((lig_x_evolve[src] - lig_x_evolve[trg]) ** 2).sum(axis=1)

                # this is the loss whose gradient we are calculating here
                grad_d_squared = 2 * (lig_x_evolve[src] - lig_x_evolve[trg])
                geo_partial_grads = 2 * (d_squared - geo_feat ** 2)[:, None] * grad_d_squared
                grad_x_evolve = scatter(geo_partial_grads, trg, n_axis=geo_graph.n_node, axis=0, reduce='add')
                lig_x_evolve += self.geometry_reg_step_size * grad_x_evolve
                if self.save_trajectories:
                    trajectory.append(lig_x_evolve.asnumpy())

        lig_input_node_upd = ops.concat([self.node_norm(h_lig_feat), lig_aggr_msg,
                                         attn_lig_feats, ori_lig_node_feat], axis=-1)

        rec_input_node_upd = ops.concat([self.node_norm(h_rec_feat), rec_aggr_msg,
                                         attn_rec_feats, ori_rec_node_feat], axis=-1)

        # Skip connections
        if self.h_feats_dim == self.out_feats_dim:
            lig_node_upd = self.skip_weight_h * self.lig_node_mlp(lig_input_node_upd) + (
                1. - self.skip_weight_h) * h_lig_feat
            rec_node_upd = self.skip_weight_h * self.rec_node_mlp(rec_input_node_upd) + (
                1. - self.skip_weight_h) * h_rec_feat
        else:
            lig_node_upd = self.lig_node_mlp(lig_input_node_upd)
            rec_node_upd = self.rec_node_mlp(rec_input_node_upd)

        lig_node_upd = apply_norm(lig_graph, lig_node_upd, self.final_h_layer_norm, self.final_h_norm_lig)
        rec_node_upd = apply_norm(rec_graph, rec_node_upd, self.final_h_layer_norm, self.final_h_norm_rec)
        out_ = [lig_x_evolve, lig_node_upd, rec_x_evolve, rec_node_upd, trajectory, geom_loss]
        return out_
