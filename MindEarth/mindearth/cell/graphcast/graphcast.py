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
# ==============================================================================
"""basic"""
from __future__ import absolute_import

import numpy as np

import mindspore as ms
import mindspore.nn as nn
from mindspore import ops, set_seed, Tensor

set_seed(0)
np.random.seed(0)


class MLPNet(nn.Cell):
    """
    The MLPNet Network. Applies a series of fully connected layers to the incoming data among which hidden layers have
        same number of dims.

    Args:
        in_channels (int): the number of input layer channel.
        out_channels (int): the number of output layer channel.
        latent_dims (int): the number of dims of hidden layers.
        has_layernorm (Union[bool, List]): The switch for whether linear block has a layer normalization layer.
            if has_layernorm was List, each element corresponds to each layer.

    Inputs:
        - **input** (Tensor) - Tensor of shape :math:`(*, dims[0])

    Outputs:
        - **output** (Tensor) - Tensor of shape :math:`(*, dims[-1])

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> inputs = Tensor(np.array([[180, 234, 154], [244, 48, 247]], np.float32))
        >>> net = MLPNet(in_channels=3, out_channels=8, latent_dims=32)
        >>> output = net(inputs)
        >>> print(output.shape)
        (2, 8)

    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 latent_dims,
                 has_layernorm=True):
        super(MLPNet, self).__init__()
        cell_list = [nn.Dense(in_channels,
                              latent_dims,
                              has_bias=False,
                              activation=None),
                     nn.SiLU(),
                     nn.Dense(latent_dims,
                              out_channels,
                              has_bias=False,
                              activation=None),
                     ]
        if has_layernorm:
            cell_list.append(nn.LayerNorm([out_channels]))
        self.network = nn.SequentialCell(cell_list)

    def construct(self, x: Tensor):
        '''MLPNet forward function

        Args:
            x (Tensor): Input Tensor.
        '''
        return self.network(x)


class Embedder(nn.Cell):
    """
    Embed raw features of the grid nodes, mesh nodes, multi-mesh edges, grid2mesh edges and mesh2grid edges.
    """
    def __init__(self,
                 vg_in_channels,
                 vm_in_channels,
                 em_in_channels,
                 eg2m_in_channels,
                 em2g_in_channels,
                 latent_dims):
        super(Embedder, self).__init__()
        self.v_g_embedder = MLPNet(in_channels=vg_in_channels, out_channels=latent_dims, latent_dims=latent_dims)
        self.v_m_embedder = MLPNet(in_channels=vm_in_channels, out_channels=latent_dims, latent_dims=latent_dims)
        self.e_m_embedder = MLPNet(in_channels=em_in_channels, out_channels=latent_dims, latent_dims=latent_dims)
        self.e_g2m_embedder = MLPNet(in_channels=eg2m_in_channels, out_channels=latent_dims,
                                     latent_dims=latent_dims)
        self.e_m2g_embedder = MLPNet(in_channels=em2g_in_channels, out_channels=latent_dims,
                                     latent_dims=latent_dims)

    def construct(self,
                  grid_node_feats,
                  mesh_node_feats,
                  mesh_edge_feats,
                  g2m_edge_feats,
                  m2g_edge_feats):
        '''Embedder forward function'''
        v_g = self.v_g_embedder(grid_node_feats)
        v_m = self.v_m_embedder(mesh_node_feats)
        e_m = self.e_m_embedder(mesh_edge_feats)
        e_g2m = self.e_g2m_embedder(g2m_edge_feats)
        e_m2g = self.e_m2g_embedder(m2g_edge_feats)
        return v_g, v_m, e_m, e_g2m, e_m2g


class G2MGnn(nn.Cell):
    """
    Transfer data for the grid to the mesh.
    """
    def __init__(self,
                 node_in_channels,
                 node_out_channels,
                 edge_in_channels,
                 edge_out_channels,
                 latent_dims,
                 src_idx,
                 dst_idx):
        super(G2MGnn, self).__init__()
        self.interaction = InteractionLayer(node_in_channels,
                                            node_out_channels,
                                            edge_in_channels,
                                            edge_out_channels,
                                            latent_dims,
                                            src_idx,
                                            dst_idx,
                                            is_homo=False)
        self.grid_node_mlp = MLPNet(in_channels=node_in_channels,
                                    out_channels=node_out_channels,
                                    latent_dims=latent_dims)

    def construct(self, g2m_edge_feats, mesh_node_feats, grid_node_feats):
        '''G2MGnn forward function'''
        mesh_node_aggr, g2m_edge_aggr = self.interaction((grid_node_feats, mesh_node_feats, g2m_edge_feats))
        grid_node_new = self.grid_node_mlp(grid_node_feats)
        return g2m_edge_aggr, mesh_node_aggr, grid_node_new + grid_node_feats


class Encoder(nn.Cell):
    """
    Encoder, which moves data from the grid to the mesh with a single message passing step.
    """
    def __init__(self,
                 vg_in_channels,
                 vm_in_channels,
                 em_in_channels,
                 eg2m_in_channels,
                 em2g_in_channels,
                 latent_dims,
                 src_idx,
                 dst_idx):
        super(Encoder, self).__init__()
        self.feature_embedder = Embedder(vg_in_channels,
                                         vm_in_channels,
                                         em_in_channels,
                                         eg2m_in_channels,
                                         em2g_in_channels,
                                         latent_dims)
        self.g2m_gnn = G2MGnn(node_in_channels=latent_dims,
                              node_out_channels=latent_dims,
                              edge_in_channels=latent_dims,
                              edge_out_channels=latent_dims,
                              latent_dims=latent_dims,
                              src_idx=src_idx,
                              dst_idx=dst_idx)

    def construct(self,
                  grid_node_feats,
                  mesh_node_feats,
                  mesh_edge_feats,
                  g2m_edge_feats,
                  m2g_edge_feats):
        '''Encoder forward function'''
        vg, vm, em, eg2m, em2g = self.feature_embedder(grid_node_feats,
                                                       mesh_node_feats,
                                                       mesh_edge_feats,
                                                       g2m_edge_feats,
                                                       m2g_edge_feats)
        eg2m, vm, vg = self.g2m_gnn(eg2m, vm, vg)
        return vg, vm, em, eg2m, em2g


class InteractionLayer(nn.Cell):
    """
    Run message passing in the multimesh.
    """
    def __init__(self,
                 node_in_channels,
                 node_out_channels,
                 edge_in_channels,
                 edge_out_channels,
                 latent_dims,
                 src_idx,
                 dst_idx,
                 is_homo):
        super(InteractionLayer, self).__init__()

        # process node
        self.node_fn = MLPNet(in_channels=node_in_channels + edge_out_channels,
                              out_channels=node_out_channels,
                              latent_dims=latent_dims)

        # process edge
        self.edge_fn = MLPNet(in_channels=2 * node_in_channels + edge_in_channels,
                              out_channels=edge_out_channels,
                              latent_dims=latent_dims)

        self.src_idx = src_idx
        self.dst_idx = dst_idx
        self.is_homo = is_homo

    def construct(self, feats):
        '''InteractionLayer forward function'''
        if self.is_homo:
            src_node_feats, dst_node_feats, edge_feats = feats[0], feats[0], feats[1]
        else:
            src_node_feats, dst_node_feats, edge_feats = feats[0], feats[1], feats[2]
        src_feats = ops.gather(src_node_feats, self.src_idx, axis=0)
        dst_feats = ops.gather(dst_node_feats, self.dst_idx, axis=0)
        updated_edge_feats = self.edge_fn(
            ops.Concat(-1)((src_feats, dst_feats, edge_feats)))
        temp_node = ms.ops.Zeros()((dst_node_feats.shape[0], edge_feats.shape[-1]), updated_edge_feats.dtype)
        scattered_edge_feats = ops.TensorScatterAdd()(
            temp_node, self.dst_idx.reshape(-1, 1), updated_edge_feats)
        updated_dst_feats = self.node_fn(
            ops.Concat(-1)((dst_node_feats, scattered_edge_feats)))
        return (updated_dst_feats + dst_node_feats, updated_edge_feats + edge_feats)


class Processor(nn.Cell):
    """
    Processor, which performs message passing on the multi-mesh.
    """
    def __init__(self,
                 node_in_channels,
                 node_out_channels,
                 edge_in_channels,
                 edge_out_channels,
                 processing_steps,
                 latent_dims,
                 src_idx,
                 dst_idx):
        super(Processor, self).__init__()
        self.processing_steps = processing_steps
        self.cell_list = nn.SequentialCell()
        for _ in range(self.processing_steps):
            self.cell_list.append(InteractionLayer(node_in_channels,
                                                   node_out_channels,
                                                   edge_in_channels,
                                                   edge_out_channels,
                                                   latent_dims,
                                                   src_idx,
                                                   dst_idx,
                                                   is_homo=True))

    def construct(self, node_feats, edge_feats):
        '''Processor forward function'''
        node_feats, edge_feats = self.cell_list((node_feats, edge_feats))
        return node_feats, edge_feats


class M2GGnn(nn.Cell):
    """
    Transfer data from the mesh to the grid.
    """
    def __init__(self,
                 node_in_channels,
                 node_out_channels,
                 edge_in_channels,
                 edge_out_channels,
                 latent_dims,
                 src_idx,
                 dst_idx):
        super(M2GGnn, self).__init__()
        self.interaction = InteractionLayer(node_in_channels,
                                            node_out_channels,
                                            edge_in_channels,
                                            edge_out_channels,
                                            latent_dims,
                                            src_idx,
                                            dst_idx,
                                            is_homo=False)
        self.mesh_node_mlp = MLPNet(in_channels=node_in_channels,
                                    out_channels=node_out_channels,
                                    latent_dims=latent_dims)

    def construct(self, m2g_edge_feats, mesh_node_feats, grid_node_feats):
        '''M2GGnn forward function'''
        grid_node_aggr, m2g_edge_aggr = self.interaction((mesh_node_feats, grid_node_feats, m2g_edge_feats))
        mesh_node_aggr = self.mesh_node_mlp(mesh_node_feats)
        return m2g_edge_aggr, mesh_node_aggr + mesh_node_feats, grid_node_aggr


class Decoder(nn.Cell):
    """
    Decoder, which moves data from the mesh back into the grid with a single message passing step.
    """
    def __init__(self,
                 node_in_channels,
                 node_out_channels,
                 edge_in_channels,
                 edge_out_channels,
                 node_final_dims,
                 latent_dims,
                 src_idx,
                 dst_idx):
        super(Decoder, self).__init__()

        self.m2g_gnn = M2GGnn(node_in_channels,
                              node_out_channels,
                              edge_in_channels,
                              edge_out_channels,
                              latent_dims,
                              src_idx,
                              dst_idx)
        self.node_fn = MLPNet(in_channels=node_in_channels,
                              out_channels=node_final_dims,
                              latent_dims=latent_dims,
                              has_layernorm=False)

    def construct(self, m2g_edge_feats, mesh_node_feats, grid_node_feats):
        '''Decoder forward function'''
        m2g_edge_feats, mesh_node_feats, grid_node_feats = self.m2g_gnn(m2g_edge_feats, mesh_node_feats,
                                                                        grid_node_feats)
        return self.node_fn(grid_node_feats)
