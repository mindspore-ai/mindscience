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
"""test mindearth GraphCastNet"""

import numpy as np
import pytest

import mindspore as ms
from mindspore import context, Tensor

from mindearth.cell import GraphCastNet

@pytest.mark.level0
@platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_graphcastnet():
    """
    Feature: Test GraphCastNet in platform gpu and ascend.
    Description: The forward output should has expected shape.
    Expectation: Success or throw AssertionError.
    """
    context.set_context(mode=context.GRAPH_MODE)
    mesh_node_num = 2562
    grid_node_num = 32768
    mesh_edge_num = 20460
    g2m_edge_num = 50184
    m2g_edge_num = 98304
    vm_in_channels = 3
    em_in_channels = 4
    eg2m_in_channels = 4
    em2g_in_channels = 4
    feature_num = 69
    g2m_src_idx = Tensor(np.random.randint(0, grid_node_num, size=[g2m_edge_num]), ms.int32)
    g2m_dst_idx = Tensor(np.random.randint(0, mesh_node_num, size=[g2m_edge_num]), ms.int32)
    m2m_src_idx = Tensor(np.random.randint(0, mesh_node_num, size=[mesh_edge_num]), ms.int32)
    m2m_dst_idx = Tensor(np.random.randint(0, mesh_node_num, size=[mesh_edge_num]), ms.int32)
    m2g_src_idx = Tensor(np.random.randint(0, mesh_node_num, size=[m2g_edge_num]), ms.int32)
    m2g_dst_idx = Tensor(np.random.randint(0, grid_node_num, size=[m2g_edge_num]), ms.int32)
    mesh_node_feats = Tensor(np.random.rand(mesh_node_num, vm_in_channels).astype(np.float32), ms.float32)
    mesh_edge_feats = Tensor(np.random.rand(mesh_edge_num, em_in_channels).astype(np.float32), ms.float32)
    g2m_edge_feats = Tensor(np.random.rand(g2m_edge_num, eg2m_in_channels).astype(np.float32), ms.float32)
    m2g_edge_feats = Tensor(np.random.rand(m2g_edge_num, em2g_in_channels).astype(np.float32), ms.float32)
    per_variable_level_mean = Tensor(np.random.rand(feature_num,).astype(np.float32), ms.float32)
    per_variable_level_std = Tensor(np.random.rand(feature_num,).astype(np.float32), ms.float32)
    grid_node_feats = Tensor(np.random.rand(grid_node_num, feature_num).astype(np.float32), ms.float32)
    graphcast_model = GraphCastNet(vg_in_channels=feature_num,
                                   vg_out_channels=feature_num,
                                   vm_in_channels=vm_in_channels,
                                   em_in_channels=em_in_channels,
                                   eg2m_in_channels=eg2m_in_channels,
                                   em2g_in_channels=em2g_in_channels,
                                   latent_dims=512,
                                   processing_steps=4,
                                   g2m_src_idx=g2m_src_idx,
                                   g2m_dst_idx=g2m_dst_idx,
                                   m2m_src_idx=m2m_src_idx,
                                   m2m_dst_idx=m2m_dst_idx,
                                   m2g_src_idx=m2g_src_idx,
                                   m2g_dst_idx=m2g_dst_idx,
                                   mesh_node_feats=mesh_node_feats,
                                   mesh_edge_feats=mesh_edge_feats,
                                   g2m_edge_feats=g2m_edge_feats,
                                   m2g_edge_feats=m2g_edge_feats,
                                   per_variable_level_mean=per_variable_level_mean,
                                   per_variable_level_std=per_variable_level_std)
    out = graphcast_model(Tensor(grid_node_feats, ms.float32))
    assert out.shape == (grid_node_num, feature_num), f"For `GraphCastNet`, the output should be\
         ({grid_node_num}, {feature_num}), but got {out.shape}."
