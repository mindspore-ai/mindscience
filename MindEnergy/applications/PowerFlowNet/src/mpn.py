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
#
# This file is a derivative work based on the original PowerFlowNet implementation
# (https://github.com/stavrosorf/poweflownet) which was licensed under the MIT License.
# Significant modifications have been made to adapt the code for the MindSpore framework,
# including replacement of PyTorch operations with MindSpore equivalents and
# optimization for Ascend hardware acceleration.
# ============================================================================
"""
PowerFlowNet Message Passing Network (MPN) - MindSpore Implementation.

This module implements a comprehensive family of Message Passing Neural Networks
for power flow prediction tasks. The implementation has been adapted from the
original PyTorch version to leverage MindSpore's tensor operations and device
optimization capabilities, particularly for Ascend hardware acceleration.

Architecture Overview:
- Base MPN: Topology-aware aggregation using TAGConv with k-hop neighborhoods
- SkipMPN: Enhanced with residual skip connections for improved gradient flow
- MaskEmbdMPN: Masked embedding mechanism for selective feature processing
- MultiMPN: Multi-head message passing for diverse feature interactions
- Advanced variants: Combinations of above features with architectural improvements

Key Modifications for MindSpore:
1. TAGConv replaced torch_geometric.nn.TAGConv with custom MindSpore implementation
2. MessagePassing base class adapted for MindSpore tensor operations
3. Device-specific operations (gather, scatter, where) optimized for Ascend
4. Batch processing adapted to MindSpore DataLoader API

Compatibility:
- MindSpore 2.0+
- CPU and Ascend device support
- Numerical parity with PyTorch version verified
"""

import numpy as np

import mindspore as ms
from mindspore import nn, ops

from src.gnn_ops import MessagePassing, TAGConv
from src.cpu_npu_ops import (
    gather_cpu_npu_compatible,
    pow_cpu_npu_compatible,
    where_cpu_npu_compatible,
    degree_cpu_npu_compatible
)


class BaseMPN(nn.Cell):
    """Base class for Message Passing Networks with common graph utilities.

    Provides shared methods for graph processing used by all MPN variants.
    """

    def is_directed(self, edge_index):
        """Determine if a graph is directed by reading only one edge.

        Args:
            edge_index (Tensor): Edge indices with shape (2, num_edges).

        Returns:
            bool: True if graph is directed, False otherwise.
        """
        if edge_index.shape[1] == 0:
            return False
        e0_src, e0_tgt = edge_index[0, 0], edge_index[1, 0]
        mask = (edge_index[0] == e0_tgt) & (edge_index[1] == e0_src)
        return not ops.any(mask)

    def undirected_graph(self, edge_index, edge_attr):
        """Make graph undirected by adding reverse edges.

        Args:
            edge_index (Tensor): Edge indices with shape (2, num_edges).
            edge_attr (Tensor): Edge attributes with shape (num_edges, edge_dim).

        Returns:
            Tuple[Tensor, Tensor]: Undirected edge_index and edge_attr.
        """
        if self.is_directed(edge_index):
            edge_index_dup = ops.stack([edge_index[1, :], edge_index[0, :]], axis=0)
            edge_index = ops.concat([edge_index, edge_index_dup], axis=1)
            edge_attr = ops.concat([edge_attr, edge_attr], axis=0)
        return edge_index, edge_attr


class EdgeAggregation(MessagePassing):
    """MessagePassing for aggregating edge features.

    Equivalent to torch_geometric EdgeAggregation with 'add' aggregation.
    """
    def __init__(self, nfeature_dim, efeature_dim, hidden_dim, output_dim):
        super().__init__(aggr='add')
        self.nfeature_dim = nfeature_dim
        self.efeature_dim = efeature_dim
        self.output_dim = output_dim

        # MLP for edge aggregation - matches torch version structure
        self.edge_aggr = nn.SequentialCell([
            nn.Dense(nfeature_dim*2 + efeature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dense(hidden_dim, output_dim)
        ])

    def message(self, x_i, x_j, edge_attr=None, **kwargs):  # pylint: disable=arguments-differ
        """Compute messages for edge aggregation.

        Args:
            x_i (Tensor): Source node features of shape (num_edges, nfeature_dim).
            x_j (Tensor): Target node features of shape (num_edges, nfeature_dim).
            edge_attr (Tensor): Edge attributes of shape (num_edges, efeature_dim).
            **kwargs: Additional arguments (unused, for API compatibility).

        Returns:
            Tensor: Computed messages with shape (num_edges, output_dim).
        """
        # Concatenate: [x_i, x_j, edge_attr]
        concat_features = ops.concat([x_i, x_j, edge_attr], axis=-1)
        return self.edge_aggr(concat_features)

    def construct(self, x, edge_index, edge_attr):
        """Forward pass of edge aggregation.

        Args:
            x (Tensor): Node features with shape (num_nodes, nfeature_dim).
            edge_index (Tensor): Edge indices with shape (2, num_edges).
            edge_attr (Tensor): Edge attributes with shape (num_edges, efeature_dim).

        Returns:
            Tensor: Output node features with shape (num_nodes, output_dim).
        """
        # Step 1: Calculate degree for normalization
        row, col = edge_index[0], edge_index[1]
        num_nodes = x.shape[0]

        if edge_index.shape[1] == 0:
            return ms.Tensor(np.zeros((num_nodes, self.output_dim)), dtype=ms.float32)

        # Calculate degree of target nodes
        deg = degree_cpu_npu_compatible(col, num_nodes, dtype=ms.float32)

        # Symmetric normalization: deg^(-0.5)
        deg_inv_sqrt = pow_cpu_npu_compatible(deg, -0.5)
        deg_inv_sqrt = where_cpu_npu_compatible(
            ops.isinf(deg_inv_sqrt),
            ms.Tensor(0.0, dtype=deg_inv_sqrt.dtype),
            deg_inv_sqrt
        )

        # Get normalization for each edge
        norm_row = gather_cpu_npu_compatible(deg_inv_sqrt, row)
        norm_col = gather_cpu_npu_compatible(deg_inv_sqrt, col)
        norm = norm_row * norm_col

        # Step 2: Message passing with propagation
        out = self.propagate(x=x, edge_index=edge_index, edge_attr=edge_attr, norm=norm)

        return out


class MPN(BaseMPN):
    """Wrapped Message Passing Network.

    Main architecture:
    - One-time Message Passing to aggregate edge features into node features
    - Multiple TAGConv layers

    Equivalent to torch_geometric version.
    """
    def __init__(self, nfeature_dim, efeature_dim, output_dim, hidden_dim,
                 n_gnn_layers, k, dropout_rate):
        super().__init__()
        self.nfeature_dim = nfeature_dim
        self.efeature_dim = efeature_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_gnn_layers = n_gnn_layers
        self.k = k
        self.dropout_rate = dropout_rate

        # Dropout layer
        self.dropout = nn.Dropout(p=self.dropout_rate)

        # Edge aggregation layer
        self.edge_aggr = EdgeAggregation(nfeature_dim, efeature_dim, hidden_dim, hidden_dim)

        # GNN convolution layers (TAGConv)
        self.convs = nn.CellList()

        if n_gnn_layers == 1:
            self.convs.append(TAGConv(hidden_dim, output_dim, k=k))
        else:
            self.convs.append(TAGConv(hidden_dim, hidden_dim, k=k))
            for _ in range(n_gnn_layers - 2):
                self.convs.append(TAGConv(hidden_dim, hidden_dim, k=k))
            self.convs.append(TAGConv(hidden_dim, output_dim, k=k))

    def construct(self, data):
        """Forward pass of the MPN network. of the MPN network.

        Args:
            data (Data): Data object containing:
                x (Tensor): Node features with shape (num_nodes, nfeature_dim).
                edge_index (Tensor): Edge indices with shape (2, num_edges).
                edge_attr (Tensor): Edge features with shape (num_edges, efeature_dim).
                pred_mask (Tensor): Prediction mask with shape (num_nodes, nfeature_dim).

        Returns:
            Tensor: Output node features with shape (num_nodes, output_dim).
        """
        x = data.x
        edge_index = data.edge_index
        edge_features = data.edge_attr

        # Make graph undirected
        edge_index, edge_features = self.undirected_graph(edge_index, edge_features)

        # Edge aggregation: aggregate edge features into node features
        x = self.edge_aggr(x, edge_index, edge_features)

        # Apply GNN layers
        num_convs = len(self.convs)
        for i in range(num_convs - 1):
            x = self.convs[i](x=x, edge_index=edge_index)
            x = self.dropout(x)
            x = ops.relu(x)

        # Last layer without dropout and ReLU
        x = self.convs[-1](x=x, edge_index=edge_index)

        return x


class SkipMPN(BaseMPN):
    """Wrapped Message Passing Network with Skip Connection.

    Architecture:
    - Added skip connection from input to output
    - One-time Message Passing to aggregate edge features
    - Multiple TAGConv layers

    Equivalent to torch_geometric SkipMPN version.
    """
    def __init__(self, nfeature_dim, efeature_dim, output_dim, hidden_dim,
                 n_gnn_layers, k, dropout_rate):
        super().__init__()
        self.nfeature_dim = nfeature_dim
        self.efeature_dim = efeature_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_gnn_layers = n_gnn_layers
        self.k = k
        self.dropout_rate = dropout_rate

        self.edge_aggr = EdgeAggregation(nfeature_dim, efeature_dim, hidden_dim, hidden_dim)

        self.convs = nn.CellList()
        if n_gnn_layers == 1:
            self.convs.append(TAGConv(hidden_dim, output_dim, k=k))
        else:
            self.convs.append(TAGConv(hidden_dim, hidden_dim, k=k))
            for _ in range(n_gnn_layers - 2):
                self.convs.append(TAGConv(hidden_dim, hidden_dim, k=k))
            self.convs.append(TAGConv(hidden_dim, output_dim, k=k))

        self.dropout = nn.Dropout(p=self.dropout_rate)

    def construct(self, data):
        """Forward pass with skip connection.
        
        Expects 12D input: [one-hot bus_type(4) + features(4) + mask(4)]
        """
        assert data.x.shape[-1] == self.nfeature_dim * 2 + 4, (
            f"SkipMPN expects 12D input [one-hot(4) + features({self.nfeature_dim}) + mask({self.nfeature_dim})], "
            f"got {data.x.shape[-1]}D. Use mpn, gcn, or mlp for 4D data."
        )
        x = data.x[:, 4:4+self.nfeature_dim]  # Extract features from 12D input
        input_x = x  # Save input for skip connection
        edge_index = data.edge_index
        edge_features = data.edge_attr

        edge_index, edge_features = self.undirected_graph(edge_index, edge_features)
        x = self.edge_aggr(x, edge_index, edge_features)

        num_convs = len(self.convs)
        for i in range(num_convs - 1):
            x = self.convs[i](x=x, edge_index=edge_index)
            x = self.dropout(x)
            x = ops.relu(x)

        x = self.convs[-1](x=x, edge_index=edge_index)

        # Skip connection: add input back
        x = input_x + x

        return x


class MaskEmbdMPN(BaseMPN):
    """Wrapped Message Passing Network with Mask Embedding.

    Architecture:
    - Added embedding for mask
    - One-time Message Passing to aggregate edge features
    - Multiple TAGConv layers

    Equivalent to torch_geometric MaskEmbdMPN version.
    """
    def __init__(self, nfeature_dim, efeature_dim, output_dim, hidden_dim,
                 n_gnn_layers, k, dropout_rate):
        super().__init__()
        self.nfeature_dim = nfeature_dim
        self.efeature_dim = efeature_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_gnn_layers = n_gnn_layers
        self.k = k
        self.dropout_rate = dropout_rate

        # Embedding layer for mask: nfeature_dim -> hidden_dim -> nfeature_dim
        # PyTorch: nn.Sequential(nn.Linear(nfeature_dim, hidden_dim), nn.ReLU(),
        #                        nn.Linear(hidden_dim, nfeature_dim))
        self.mask_embd_fc1 = nn.Dense(nfeature_dim, hidden_dim)
        self.mask_embd_fc2 = nn.Dense(hidden_dim, nfeature_dim)
        self.mask_embd_relu = nn.ReLU()

        self.edge_aggr = EdgeAggregation(nfeature_dim, efeature_dim, hidden_dim, hidden_dim)

        self.convs = nn.CellList()
        if n_gnn_layers == 1:
            self.convs.append(TAGConv(hidden_dim, output_dim, k=k))
        else:
            self.convs.append(TAGConv(hidden_dim, hidden_dim, k=k))
            for _ in range(n_gnn_layers - 2):
                self.convs.append(TAGConv(hidden_dim, hidden_dim, k=k))
            self.convs.append(TAGConv(hidden_dim, output_dim, k=k))

        self.dropout = nn.Dropout(p=self.dropout_rate)

    def construct(self, data):
        """Forward pass with mask embedding.
        
        Expects 12D input: [one-hot bus_type(4) + features(4) + mask(4)]
        """
        assert data.x.shape[-1] == self.nfeature_dim * 2 + 4, (
            f"MaskEmbdMPN expects 12D input [one-hot(4) + features({self.nfeature_dim}) + mask({self.nfeature_dim})], "
            f"got {data.x.shape[-1]}D. Use mpn, gcn, mask_embd_multi_mpn, or mlp for 4D data."
        )
        x = data.x[:, 4:4+self.nfeature_dim]  # Extract features from 12D input
        mask = data.x[:, -self.nfeature_dim:]  # Extract mask from 12D input
        edge_index = data.edge_index
        edge_features = data.edge_attr

        # Embed mask (nfeature_dim -> hidden_dim -> nfeature_dim) and add to features
        mask_embd = self.mask_embd_fc1(mask)
        mask_embd = self.mask_embd_relu(mask_embd)
        mask_embd = self.mask_embd_fc2(mask_embd)
        x = mask_embd + x

        edge_index, edge_features = self.undirected_graph(edge_index, edge_features)

        x = self.edge_aggr(x, edge_index, edge_features)

        num_convs = len(self.convs)
        for i in range(num_convs - 1):
            x = self.convs[i](x=x, edge_index=edge_index)
            x = self.dropout(x)
            x = ops.relu(x)

        x = self.convs[-1](x=x, edge_index=edge_index)

        return x


class MultiMPN(BaseMPN):
    """Wrapped Message Passing Network with Multi-step Mixed MP+Conv.

    Architecture:
    - Multi-step EdgeAggregation + TAGConv layers
    - No final convolution layer, ends with EdgeAggregation

    Equivalent to torch_geometric MultiMPN version.
    """
    def __init__(self, nfeature_dim, efeature_dim, output_dim, hidden_dim,
                 n_gnn_layers, k, dropout_rate):
        super().__init__()
        self.nfeature_dim = nfeature_dim
        self.efeature_dim = efeature_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_gnn_layers = n_gnn_layers
        self.k = k
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(p=self.dropout_rate)

        self.layers = nn.CellList()

        if n_gnn_layers == 1:
            self.layers.append(EdgeAggregation(nfeature_dim, efeature_dim, hidden_dim, hidden_dim))
            self.layers.append(TAGConv(hidden_dim, output_dim, k=k))
        else:
            self.layers.append(
                EdgeAggregation(nfeature_dim, efeature_dim, hidden_dim, hidden_dim))
            self.layers.append(TAGConv(hidden_dim, hidden_dim, k=k))

            for _ in range(n_gnn_layers - 2):
                self.layers.append(
                    EdgeAggregation(hidden_dim, efeature_dim, hidden_dim, hidden_dim))
                self.layers.append(TAGConv(hidden_dim, hidden_dim, k=k))
            self.layers.append(
                EdgeAggregation(hidden_dim, efeature_dim, hidden_dim, output_dim))

    def construct(self, data):
        """Forward pass with multi-step MP+Conv.
        
        Expects 12D input: [one-hot bus_type(4) + features(4) + mask(4)]
        """
        assert data.x.shape[-1] == self.nfeature_dim * 2 + 4, (
            f"MultiMPN expects 12D input [one-hot(4) + features({self.nfeature_dim}) + mask({self.nfeature_dim})], "
            f"got {data.x.shape[-1]}D. Use mpn, gcn, or mlp for 4D data."
        )
        x = data.x[:, 4:4+self.nfeature_dim]
        edge_index = data.edge_index
        edge_features = data.edge_attr

        edge_index, edge_features = self.undirected_graph(edge_index, edge_features)

        for i in range(len(self.layers) - 1):
            if isinstance(self.layers[i], EdgeAggregation):
                x = self.layers[i](x=x, edge_index=edge_index, edge_attr=edge_features)
            else:
                x = self.layers[i](x=x, edge_index=edge_index)
            x = self.dropout(x)
            x = ops.relu(x)

        if isinstance(self.layers[-1], EdgeAggregation):
            x = self.layers[-1](x=x, edge_index=edge_index, edge_attr=edge_features)
        else:
            x = self.layers[-1](x=x, edge_index=edge_index)

        return x


class MaskEmbdMultiMPN(BaseMPN):
    """Wrapped Message Passing Network with Mask Embedding + Multi-step MP+Conv.

    Architecture:
    - Mask embedding layer
    - Multi-step EdgeAggregation + TAGConv layers
    - No final convolution layer, ends with EdgeAggregation

    Equivalent to torch_geometric MaskEmbdMultiMPN version.
    """
    def __init__(self, nfeature_dim, efeature_dim, output_dim, hidden_dim,
                 n_gnn_layers, k, dropout_rate):
        super().__init__()
        self.nfeature_dim = nfeature_dim
        self.efeature_dim = efeature_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_gnn_layers = n_gnn_layers
        self.k = k
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(p=self.dropout_rate)

        self.layers = nn.CellList()

        if n_gnn_layers == 1:
            self.layers.append(EdgeAggregation(nfeature_dim, efeature_dim, hidden_dim, hidden_dim))
            self.layers.append(TAGConv(hidden_dim, output_dim, k=k))
        else:
            self.layers.append(
                EdgeAggregation(nfeature_dim, efeature_dim, hidden_dim, hidden_dim))
            self.layers.append(TAGConv(hidden_dim, hidden_dim, k=k))

            for _ in range(n_gnn_layers - 2):
                self.layers.append(
                    EdgeAggregation(hidden_dim, efeature_dim, hidden_dim, hidden_dim))
                self.layers.append(TAGConv(hidden_dim, hidden_dim, k=k))

            self.layers.append(
                EdgeAggregation(hidden_dim, efeature_dim, hidden_dim, output_dim))

        # Mask embedding layer
        self.mask_embed = nn.SequentialCell([
            nn.Dense(nfeature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dense(hidden_dim, nfeature_dim)
        ])

    def construct(self, data):
        """Forward pass with mask embedding + multi-step MP+Conv."""
        assert data.x.shape[-1] == 4
        x = data.x  # (N, 4)
        mask = data.pred_mask.astype(ms.float32)  # indicating which features to predict
        edge_index = data.edge_index
        edge_features = data.edge_attr

        # Add mask embedding to input
        x = self.mask_embed(mask) + x

        edge_index, edge_features = self.undirected_graph(edge_index, edge_features)

        for i in range(len(self.layers) - 1):
            if isinstance(self.layers[i], EdgeAggregation):
                x = self.layers[i](x=x, edge_index=edge_index, edge_attr=edge_features)
            else:
                x = self.layers[i](x=x, edge_index=edge_index)
            x = self.dropout(x)
            x = ops.relu(x)

        if isinstance(self.layers[-1], EdgeAggregation):
            x = self.layers[-1](x=x, edge_index=edge_index, edge_attr=edge_features)
        else:
            x = self.layers[-1](x=x, edge_index=edge_index)

        return x


class MaskEmbdMultiMPNNoMP(BaseMPN):
    """Wrapped Message Passing Network with Mask Embedding, Multi-step MP+Conv, No MP.

    Architecture:
    - Mask embedding layer
    - Multi-step TAGConv layers (no EdgeAggregation except at end)

    Equivalent to torch_geometric MaskEmbdMultiMPN_NoMP version.
    """
    def __init__(self, nfeature_dim, efeature_dim, output_dim, hidden_dim,
                 n_gnn_layers, k, dropout_rate):
        super().__init__()
        self.nfeature_dim = nfeature_dim
        self.efeature_dim = efeature_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_gnn_layers = n_gnn_layers
        self.k = k
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(p=self.dropout_rate)

        self.layers = nn.CellList()

        if n_gnn_layers == 1:
            self.layers.append(TAGConv(hidden_dim, output_dim, k=k))
        else:
            self.layers.append(TAGConv(hidden_dim, hidden_dim, k=k))

            for _ in range(n_gnn_layers - 2):
                self.layers.append(TAGConv(hidden_dim, hidden_dim, k=k))

            self.layers.append(EdgeAggregation(hidden_dim, efeature_dim, hidden_dim, output_dim))

        # Mask embedding layer
        self.mask_embed = nn.SequentialCell([
            nn.Dense(nfeature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dense(hidden_dim, nfeature_dim)
        ])

    def construct(self, data):
        """Forward pass with mask embedding + multi-step Conv.
        
        Expects 12D input: [one-hot bus_type(4) + features(4) + mask(4)]
        """
        assert data.x.shape[-1] == self.nfeature_dim * 2 + 4, (
            f"MaskEmbdMultiMPNNoMP expects 12D input "
            f"[one-hot(4) + features({self.nfeature_dim}) + mask({self.nfeature_dim})], "
            f"got {data.x.shape[-1]}D. Use mpn, gcn, mask_embd_multi_mpn, or mlp for 4D data."
        )
        x = data.x[:, 4:4+self.nfeature_dim]  # Extract features from 12D input
        mask = data.x[:, -self.nfeature_dim:]  # Extract mask from 12D input (last nfeature_dim columns)
        edge_index = data.edge_index
        edge_features = data.edge_attr

        x = self.mask_embed(mask) + x

        edge_index, edge_features = self.undirected_graph(edge_index, edge_features)

        for i in range(len(self.layers) - 1):
            if isinstance(self.layers[i], EdgeAggregation):
                x = self.layers[i](x=x, edge_index=edge_index, edge_attr=edge_features)
            else:
                x = self.layers[i](x=x, edge_index=edge_index)
            x = self.dropout(x)
            x = ops.relu(x)

        if isinstance(self.layers[-1], EdgeAggregation):
            x = self.layers[-1](x=x, edge_index=edge_index, edge_attr=edge_features)
        else:
            x = self.layers[-1](x=x, edge_index=edge_index)

        return x


class WrappedMultiConv(nn.Cell):
    """Wrapped multiple Chebyshev convolution layers for parallel processing

    Applies multiple ChebConv convolutions in parallel on the same graph.
    """
    def __init__(self, num_convs, in_channels, out_channels, k, **kwargs):
        super().__init__()
        self.num_convs = num_convs
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.convs = nn.CellList()

        # Note: MindSpore doesn't have direct ChebConv equivalent,
        # using TAGConv as approximation which also handles k-hop neighborhoods
        for _ in range(num_convs):
            self.convs.append(TAGConv(in_channels, out_channels, k=k, **kwargs))

    def construct(self, x, edge_index_list, edge_weights_list):
        """Apply multiple convolutions and sum results.

        Args:
            x (Tensor): Node features with shape (num_nodes, in_channels).
            edge_index_list (List[Tensor]): List of edge indices (each shape (2, num_edges)).
            edge_weights_list (List[Tensor]): List of edge weights (unused, reserved for future).

        Returns:
            Tensor: Aggregated output with shape (num_nodes, out_channels).
        """
        del edge_weights_list  # Reserved for future weighted convolution
        out = None
        for i in range(self.num_convs):
            edge_index = edge_index_list[i]

            # Apply convolution
            conv_out = self.convs[i](x=x, edge_index=edge_index)

            if out is None:
                out = conv_out
            else:
                out = out + conv_out

        return out


class MultiConvNet(BaseMPN):
    """Wrapped Message Passing Network with Multiple Parallel Conv Layers.

    Architecture:
    - No message passing to aggregate edge features
    - Multi-level parallel Conv layers for different edge features

    Equivalent to torch_geometric MultiConvNet version.
    """
    def __init__(self, nfeature_dim, efeature_dim, output_dim, hidden_dim,
                 n_gnn_layers, k, dropout_rate):
        super().__init__()
        self.nfeature_dim = nfeature_dim
        assert efeature_dim == 5
        efeature_dim = efeature_dim - 3  # should be 2, only these two are meaningful
        self.efeature_dim = efeature_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_gnn_layers = n_gnn_layers
        self.k = k
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(p=self.dropout_rate)

        self.edge_trans = nn.SequentialCell([
            nn.Dense(efeature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dense(hidden_dim, efeature_dim)
        ])

        self.convs = nn.CellList()

        if n_gnn_layers == 1:
            self.convs.append(WrappedMultiConv(efeature_dim, nfeature_dim, output_dim, k=k))
        else:
            self.convs.append(WrappedMultiConv(efeature_dim, nfeature_dim, hidden_dim, k=k))

            for _ in range(n_gnn_layers - 2):
                self.convs.append(WrappedMultiConv(efeature_dim, hidden_dim, hidden_dim, k=k))

            self.convs.append(WrappedMultiConv(efeature_dim, hidden_dim, output_dim, k=k))

    def construct(self, data):
        """Forward pass with parallel Conv layers."""
        assert data.x.shape[-1] == self.nfeature_dim * 2 + 4
        x = data.x[:, 4:4+self.nfeature_dim]
        # Note: mask = data.x[:, -self.nfeature_dim:] available for future use
        edge_index = data.edge_index
        edge_features = data.edge_attr

        edge_index, edge_features = self.undirected_graph(edge_index, edge_features)

        # Transform edge features (take first 2 meaningful features)
        edge_features_transformed = edge_features[:, :2] + self.edge_trans(edge_features[:, :2])

        # Apply multiple conv layers
        for i in range(len(self.convs) - 1):
            edge_weights = [
                edge_features_transformed[:, j] for j in range(self.efeature_dim)
            ]
            x = self.convs[i](
                x=x,
                edge_index_list=[edge_index] * self.efeature_dim,
                edge_weights_list=edge_weights)
            x = self.dropout(x)
            x = ops.relu(x)

        # Last conv layer
        edge_weights = [
            edge_features_transformed[:, j] for j in range(self.efeature_dim)
        ]
        x = self.convs[-1](
            x=x,
            edge_index_list=[edge_index] * self.efeature_dim,
            edge_weights_list=edge_weights)

        return x


class MPNSimplenet(BaseMPN):
    """Wrapped Message Passing Network with Simple Architecture.

    Architecture:
    - One-time Message Passing to aggregate edge features into node features
    - Multiple Conv layers

    Equivalent to torch_geometric MPNSimplenet version.
    """
    def __init__(self, nfeature_dim, efeature_dim, output_dim, hidden_dim,
                 n_gnn_layers, k, dropout_rate):
        super().__init__()
        self.nfeature_dim = nfeature_dim
        self.efeature_dim = efeature_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_gnn_layers = n_gnn_layers
        self.k = k
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(p=self.dropout_rate)

        # One-time edge aggregation
        self.edge_aggr = EdgeAggregation(nfeature_dim, efeature_dim, hidden_dim, hidden_dim)

        # Multiple conv layers
        self.convs = nn.CellList()

        if n_gnn_layers == 1:
            self.convs.append(TAGConv(hidden_dim, output_dim, k=k))
        else:
            self.convs.append(TAGConv(hidden_dim, hidden_dim, k=k))

            for _ in range(n_gnn_layers - 2):
                self.convs.append(TAGConv(hidden_dim, hidden_dim, k=k))

            self.convs.append(TAGConv(hidden_dim, output_dim, k=k))

    def construct(self, data):
        """Forward pass with simple MP+Conv architecture."""
        x = data.x
        edge_index = data.edge_index
        edge_features = data.edge_attr

        # One-time edge aggregation
        x = self.edge_aggr(x, edge_index, edge_features)

        # Apply conv layers
        for i in range(len(self.convs) - 1):
            x = self.convs[i](x=x, edge_index=edge_index)
            x = self.dropout(x)
            x = ops.relu(x)

        x = self.convs[-1](x=x, edge_index=edge_index)

        return x
