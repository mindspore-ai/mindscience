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
"""
Graph Convolutional Network (GCN) Implementation for PowerFlowNet
MindSpore Version - CPU and NPU Compatible
"""

from mindspore import nn, ops

from src.gnn_ops import GCNConv


class GCNNet(nn.Cell):
    """Simple GCN Network for Power Flow Prediction"""

    def __init__(self, nfeature_dim, output_dim, hidden_dim, n_gnn_layers, dropout_rate):
        super().__init__()
        self.nfeature_dim = nfeature_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_gnn_layers = n_gnn_layers
        self.dropout_rate = dropout_rate

        # Input layer
        self.input_layer = nn.Dense(nfeature_dim, hidden_dim)

        # GCN layers
        self.convs = nn.CellList()
        for _ in range(n_gnn_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.convs.append(GCNConv(hidden_dim, output_dim))

        self.dropout = nn.Dropout(p=dropout_rate)

    def construct(self, data):
        """Forward pass of the GCN network.

        Args:
            data (Data): Data object containing:
                x (Tensor): Node features tensor with shape (num_nodes, nfeature_dim).
                edge_index (Tensor): Edge indices tensor with shape (2, num_edges).

        Returns:
            Tensor: Output features with shape (num_nodes, output_dim).
        """
        x = self.input_layer(data.x)
        edge_index = data.edge_index

        num_convs = len(self.convs)
        for i in range(num_convs - 1):
            x = self.convs[i](x, edge_index)
            x = ops.relu(x)
            x = self.dropout(x)

        x = self.convs[-1](x, edge_index)
        return x
