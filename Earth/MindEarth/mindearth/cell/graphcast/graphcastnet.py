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
"""GraphCastNet base class"""

from mindspore import nn, ops, Tensor
from .graphcast import Encoder, Processor, Decoder


class GraphCastNet(nn.Cell):
    r"""
    The GraphCast is based on graph neural networks and a novel high-resolution
    multi-scale mesh representation autoregressive model.
    The details can be found in `GraphCast: Learning skillful medium-range
    global weather forecasting <https://arxiv.org/pdf/2212.12794.pdf>`_.

    Args:
         vg_in_channels (int): The grid node dimensions.
         vg_out_channels (int): The grid node final dimensions.
         vm_in_channels (int): The mesh node dimensions.
         em_in_channels (int): The mesh edge dimensions.
         eg2m_in_channels (int): The grid to mesh edge dimensions.
         em2g_in_channels (int): The mesh to grid edge dimensions.
         latent_dims (int): The number of dims of hidden layers.
         processing_steps (int): The number of processing steps.
         g2m_src_idx (Tensor): The source node index of grid to mesh edges.
         g2m_dst_idx (Tensor): The destination node index of grid to mesh edges.
         m2m_src_idx (Tensor): The source node index of mesh to mesh edges.
         m2m_dst_idx (Tensor): The destination node index of mesh to mesh edges.
         m2g_src_idx (Tensor): The source node index of mesh to grid edges.
         m2g_dst_idx (Tensor): The destination node index of mesh to grid edges.
         mesh_node_feats (Tensor): The features of mesh nodes.
         mesh_edge_feats (Tensor): The features of mesh edges.
         g2m_edge_feats (Tensor): The features of grid to mesh edges.
         m2g_edge_feats (Tensor): The features of mesh to grid edges.
         per_variable_level_mean (Tensor): The mean of the per-variable-level inverse variance of time differences.
         per_variable_level_std (Tensor): The standard deviation of the per-variable-level inverse variance of time
                                          differences.
         recompute (bool, optional): Determine whether to recompute. Default: ``False`` .

    Inputs:
        - **input** (Tensor) - Tensor of shape :math:`(batch\_size, height\_size * width\_size, feature\_size)` .

    Outputs:
        - **output** (Tensor) - Tensor of shape :math:`(height\_size * width\_size, feature\_size)` .

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore as ms
        >>> from mindspore import context, Tensor
        >>> from mindearth.cell.graphcast.graphcastnet import GraphCastNet
        >>>
        >>> mesh_node_num = 2562
        >>> grid_node_num = 32768
        >>> mesh_edge_num = 20460
        >>> g2m_edge_num = 50184
        >>> m2g_edge_num = 98304
        >>> vm_in_channels = 3
        >>> em_in_channels = 4
        >>> eg2m_in_channels = 4
        >>> em2g_in_channels = 4
        >>> feature_num = 69
        >>> g2m_src_idx = Tensor(np.random.randint(0, grid_node_num, size=[g2m_edge_num]), ms.int32)
        >>> g2m_dst_idx = Tensor(np.random.randint(0, mesh_node_num, size=[g2m_edge_num]), ms.int32)
        >>> m2m_src_idx = Tensor(np.random.randint(0, mesh_node_num, size=[mesh_edge_num]), ms.int32)
        >>> m2m_dst_idx = Tensor(np.random.randint(0, mesh_node_num, size=[mesh_edge_num]), ms.int32)
        >>> m2g_src_idx = Tensor(np.random.randint(0, mesh_node_num, size=[m2g_edge_num]), ms.int32)
        >>> m2g_dst_idx = Tensor(np.random.randint(0, grid_node_num, size=[m2g_edge_num]), ms.int32)
        >>> mesh_node_feats = Tensor(np.random.rand(mesh_node_num, vm_in_channels).astype(np.float32), ms.float32)
        >>> mesh_edge_feats = Tensor(np.random.rand(mesh_edge_num, em_in_channels).astype(np.float32), ms.float32)
        >>> g2m_edge_feats = Tensor(np.random.rand(g2m_edge_num, eg2m_in_channels).astype(np.float32), ms.float32)
        >>> m2g_edge_feats = Tensor(np.random.rand(m2g_edge_num, em2g_in_channels).astype(np.float32), ms.float32)
        >>> per_variable_level_mean = Tensor(np.random.rand(feature_num,).astype(np.float32), ms.float32)
        >>> per_variable_level_std = Tensor(np.random.rand(feature_num,).astype(np.float32), ms.float32)
        >>> grid_node_feats = Tensor(np.random.rand(grid_node_num, feature_num).astype(np.float32), ms.float32)
        >>> graphcast_model = GraphCastNet(vg_in_channels=feature_num,
        >>>                                vg_out_channels=feature_num,
        >>>                                vm_in_channels=vm_in_channels,
        >>>                                em_in_channels=em_in_channels,
        >>>                                eg2m_in_channels=eg2m_in_channels,
        >>>                                em2g_in_channels=em2g_in_channels,
        >>>                                latent_dims=512,
        >>>                                processing_steps=4,
        >>>                                g2m_src_idx=g2m_src_idx,
        >>>                                g2m_dst_idx=g2m_dst_idx,
        >>>                                m2m_src_idx=m2m_src_idx,
        >>>                                m2m_dst_idx=m2m_dst_idx,
        >>>                                m2g_src_idx=m2g_src_idx,
        >>>                                m2g_dst_idx=m2g_dst_idx,
        >>>                                mesh_node_feats=mesh_node_feats,
        >>>                                mesh_edge_feats=mesh_edge_feats,
        >>>                                g2m_edge_feats=g2m_edge_feats,
        >>>                                m2g_edge_feats=m2g_edge_feats,
        >>>                                per_variable_level_mean=per_variable_level_mean,
        >>>                                per_variable_level_std=per_variable_level_std)
        >>> out = graphcast_model(Tensor(grid_node_feats, ms.float32))
        >>> print(out.shape)
        (32768, 69))

    """

    def __init__(self,
                 vg_in_channels,
                 vg_out_channels,
                 vm_in_channels,
                 em_in_channels,
                 eg2m_in_channels,
                 em2g_in_channels,
                 latent_dims,
                 processing_steps,
                 g2m_src_idx,
                 g2m_dst_idx,
                 m2m_src_idx,
                 m2m_dst_idx,
                 m2g_src_idx,
                 m2g_dst_idx,
                 mesh_node_feats,
                 mesh_edge_feats,
                 g2m_edge_feats,
                 m2g_edge_feats,
                 per_variable_level_mean,
                 per_variable_level_std,
                 recompute=False):
        super(GraphCastNet, self).__init__()
        self.vg_out_channels = vg_out_channels
        self.mesh_node_feats = mesh_node_feats
        self.mesh_edge_feats = mesh_edge_feats
        self.g2m_edge_feats = g2m_edge_feats
        self.m2g_edge_feats = m2g_edge_feats
        self.per_variable_level_mean = per_variable_level_mean
        self.per_variable_level_std = per_variable_level_std
        self.encoder = Encoder(vg_in_channels=vg_in_channels,
                               vm_in_channels=vm_in_channels,
                               em_in_channels=em_in_channels,
                               eg2m_in_channels=eg2m_in_channels,
                               em2g_in_channels=em2g_in_channels,
                               latent_dims=latent_dims,
                               src_idx=g2m_src_idx,
                               dst_idx=g2m_dst_idx,
                               )

        self.processor = Processor(node_in_channels=latent_dims,
                                   node_out_channels=latent_dims,
                                   edge_in_channels=latent_dims,
                                   edge_out_channels=latent_dims,
                                   processing_steps=processing_steps,
                                   latent_dims=latent_dims,
                                   src_idx=m2m_src_idx,
                                   dst_idx=m2m_dst_idx)

        self.decoder = Decoder(node_in_channels=latent_dims,
                               node_out_channels=latent_dims,
                               edge_in_channels=latent_dims,
                               edge_out_channels=latent_dims,
                               node_final_dims=vg_out_channels,
                               latent_dims=latent_dims,
                               src_idx=m2g_src_idx,
                               dst_idx=m2g_dst_idx)
        if recompute:
            self.encoder.recompute()
            self.processor.recompute()
            self.decoder.recompute()

    def construct(self, grid_node_feats: Tensor):
        """GraphCast forward function.

        Args:
            grid_node_feats (Tensor): Input Tensor.
        """
        grid_node_feats = ops.squeeze(grid_node_feats)
        vg, vm, em, _, em2g = self.encoder(grid_node_feats, self.mesh_node_feats, self.mesh_edge_feats,
                                           self.g2m_edge_feats,
                                           self.m2g_edge_feats)
        updated_vm, _ = self.processor(vm, em)
        node_feats = self.decoder(em2g, updated_vm, vg)
        output = (node_feats * self.per_variable_level_std + self.per_variable_level_mean) +\
                 grid_node_feats[:, -self.vg_out_channels:]
        return output
