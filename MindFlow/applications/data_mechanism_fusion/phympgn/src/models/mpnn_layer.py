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
# ==============================================================================
"""mpnn layer"""
from mindspore import nn, ops

from .utils import build_net


class MPNNLayer(nn.Cell):
    """MPNNLayer"""
    def __init__(self, phi_layers, gamma_layers):
        super().__init__()
        self.phi = build_net(phi_layers)
        self.gamma = build_net(gamma_layers)

    def construct(self, edge_index, node_features, edge_features):
        # (n_edge, h_dim)
        m_ij = self.message(edge_index, node_features, edge_features)
        # (n_node, h_dim)
        aggr = self.aggregate(edge_index, m_ij, node_features.shape[0])
        # (n_node, h_dim)
        node_features_new = self.update(aggr, node_features)
        return node_features_new

    def message(self, edge_index, node_features, edge_features):
        sender, receiver = edge_index[0], edge_index[1]
        # (n_edge, node_h_dim * 2 + edge_h_dim)
        phi_input = ops.cat([node_features[sender],
                             node_features[receiver] - node_features[sender],
                             edge_features], axis=1)
        return self.phi(phi_input)

    def aggregate(self, edge_index, messages, node_num):
        aggr = scatter_mean(messages, edge_index[0, :].reshape(-1, 1),
                            dim_size=node_num)
        return aggr

    def update(self, aggr, node_features):
        # (n_node, node_h_dim + h_dim)
        gamma_input = ops.cat([node_features, aggr], axis=1)
        return self.gamma(gamma_input)  # (n_node, gamma_out_dim)


def scatter_sum(src, index, dim_size):
    """scatter sum"""
    assert len(index.shape) == 2
    assert index.shape[-1] == 1
    assert src.shape[0] == index.shape[0]
    assert len(src.shape) == 2

    tmp_node = ops.zeros((dim_size, src.shape[1]), dtype=src.dtype)
    out = ops.tensor_scatter_add(tmp_node, index, src)
    return out


def scatter_mean(src, index, dim_size):
    """scatter mean"""
    assert len(index.shape) == 2
    assert index.shape[-1] == 1
    assert src.shape[0] == index.shape[0]
    assert len(src.shape) == 2

    ones = ops.ones((index.shape[0], 1), dtype=src.dtype)
    cnt = scatter_sum(ones, index, dim_size)
    total = scatter_sum(src, index, dim_size)
    out = total / cnt
    return out
