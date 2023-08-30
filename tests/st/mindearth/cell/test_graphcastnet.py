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
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_graphcastnet():
    """
    Feature: Test GraphCastNet in platform gpu and ascend.
    Description: The forward output should has expected shape.
    Expectation: Success or throw AssertionError.
    """
    context.set_context(mode=context.GRAPH_MODE)
    g2m_src_idx = Tensor(np.random.randint(0, 32768, size=[50184]), ms.int32)
    g2m_dst_idx = Tensor(np.random.randint(0, 32768, size=[50184]), ms.int32)
    m2m_src_idx = Tensor(np.random.randint(0, 2562, size=[20460]), ms.int32)
    m2m_dst_idx = Tensor(np.random.randint(0, 2562, size=[20460]), ms.int32)
    m2g_src_idx = Tensor(np.random.randint(0, 2562, size=[98304]), ms.int32)
    m2g_dst_idx = Tensor(np.random.randint(0, 2562, size=[98304]), ms.int32)
    mesh_node_feats = Tensor(np.random.rand(2560, 3).astype(np.float32), ms.float32)
    mesh_edge_feats = Tensor(np.random.rand(20460, 4).astype(np.float32), ms.float32)
    g2m_edge_feats = Tensor(np.random.rand(50184, 4).astype(np.float32), ms.float32)
    m2g_edge_feats = Tensor(np.random.rand(98304, 4).astype(np.float32), ms.float32)
    per_variable_level_mean = Tensor(np.random.rand(69,).astype(np.float32), ms.float32)
    per_variable_level_std = Tensor(np.random.rand(69,).astype(np.float32), ms.float32)
    grid_node_feats = Tensor(np.random.rand(32768, 69).astype(np.float32), ms.float32)
    context.set_context(mode=context.GRAPH_MODE)
    graphcast_model = GraphCastNet(vg_in_channels=69,
                                   vg_out_channels=69,
                                   vm_in_channels=3,
                                   em_in_channels=4,
                                   eg2m_in_channels=4,
                                   em2g_in_channels=4,
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
    assert out.shape == (32768, 69), f"For `GraphCastNet`, the output should be (32768, 69), but got {out.shape}."
