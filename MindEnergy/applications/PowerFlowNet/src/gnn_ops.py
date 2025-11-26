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
Lightweight MindSpore GNN operations
Minimal implementation of torch_geometric-like functionality for MindSpore
Optimized for CPU and NPU compatibility
"""

import inspect

import mindspore as ms
from mindspore import Tensor, ops, nn, mint


class MessagePassing(nn.Cell):
    """
    Lightweight MessagePassing base class for GNN layers
    Simplified version compatible with both CPU and NPU
    """

    def __init__(self, aggr: str = 'add', flow: str = 'src_to_trg'):
        super().__init__()
        self.aggr = aggr
        self.flow = flow

    def aggregate(self, x_i, aggr_index, num_nodes):
        """
        Aggregate messages based on aggregation type

        Args:
            x_i: Messages to aggregate (num_edges, feature_dim)
            aggr_index: Target node indices (num_edges,)
            num_nodes: Total number of nodes

        Returns:
            Aggregated values (num_nodes, feature_dim)
        """
        if self.aggr in ('add', 'sum'):
            return self._segment_sum(x_i, aggr_index, num_nodes)
        if self.aggr == 'mean':
            return self._segment_mean(x_i, aggr_index, num_nodes)
        if self.aggr == 'max':
            return self._segment_max(x_i, aggr_index, num_nodes)
        raise ValueError(f"Unknown aggregation type: {self.aggr}")

    @staticmethod
    def _segment_sum(values, indices, num_nodes):
        """CPU/NPU compatible segment sum"""
        result = mint.zeros((num_nodes, values.shape[-1]), dtype=values.dtype)

        for i in range(num_nodes):
            mask = indices == i
            if ops.any(mask):
                result[i] = ops.sum(values[mask], dim=0)

        return result

    @staticmethod
    def _segment_mean(values, indices, num_nodes):
        """CPU/NPU compatible segment mean"""
        result_sum = MessagePassing._segment_sum(values, indices, num_nodes)

        # Count occurrences
        count = mint.zeros((num_nodes,), dtype=ms.float32)
        for i in range(num_nodes):
            mask = indices == i
            count[i] = ops.sum(mask.astype(ms.float32))

        # Avoid division by zero
        count = ops.where(count > 0, count, mint.ones_like(count))

        return result_sum / count.expand_dims(-1)

    @staticmethod
    def _segment_max(values, indices, num_nodes):
        """CPU/NPU compatible segment max"""
        result = mint.full((num_nodes, values.shape[-1]), float('-inf'), dtype=values.dtype)

        for i in range(num_nodes):
            mask = indices == i
            if ops.any(mask):
                result[i] = ops.max(values[mask], axis=0)

        return result

    def propagate(self, x, edge_index, edge_attr=None, **kwargs):
        """
        Execute message passing

        Args:
            x: Node features (num_nodes, feature_dim)
            edge_index: Edge indices (2, num_edges)
            edge_attr: Edge attributes (num_edges, edge_dim)
            **kwargs: Additional arguments passed to message()

        Returns:
            Aggregated messages (num_nodes, feature_dim)
        """
        src, dst = edge_index[0], edge_index[1]
        num_nodes = x.shape[0]

        # Get source and target node features
        x_src = x[src]  # (num_edges, feature_dim) - features of source nodes
        x_dst = x[dst]  # (num_edges, feature_dim) - features of target nodes

        # Prepare message arguments
        # torch_geometric naming: x_i is target, x_j is source
        message_kwargs = {'x_i': x_dst, 'x_j': x_src}
        if edge_attr is not None:
            message_kwargs['edge_attr'] = edge_attr
        message_kwargs.update(kwargs)

        # Get the signature of the message method to filter arguments
        message_sig = inspect.signature(self.message)
        message_params = set(message_sig.parameters.keys())

        # Filter to only pass arguments that the message method accepts
        filtered_kwargs = {
            k: v for k, v in message_kwargs.items()
            if k in message_params
        }

        # Compute messages
        messages = self.message(**filtered_kwargs)  # (num_edges, feature_dim)

        # Aggregate messages
        out = self.aggregate(messages, dst, num_nodes)

        # Update (default: identity)
        out = self.update(out, x)

        return out

    def message(self, **kwargs):
        """Compute messages for message passing.

        This method should be overridden in subclasses to define custom message computation.

        Args:
            **kwargs: Keyword arguments that may include:
                x_i (Tensor): Features of target nodes (num_edges, feature_dim).
                x_j (Tensor): Features of source nodes (num_edges, feature_dim).
                edge_attr (Tensor): Edge attributes if available (num_edges, edge_dim).

        Returns:
            Tensor: Computed messages with shape (num_edges, feature_dim).
        """
        return kwargs.get('x', None)

    def update(self, aggr_out, x):
        """Update node features. Override in subclass."""
        del x  # Unused in base class
        return aggr_out

class TAGConv(MessagePassing):
    """
    Topology Adaptive Graph Convolutional Network layer
    Equivalent to torch_geometric TAGConv implementation

    Reference: "Topology Adaptive Graph Convolutional Networks"
    https://arxiv.org/abs/1710.10370
    """

    def __init__(self, in_channels: int, out_channels: int, k: int = 3,
                 bias: bool = True, normalize: bool = True):
        super().__init__(aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k = k
        self.normalize = normalize

        # Linear transformations for each hop (matching torch_geometric)
        # Note: k+1 linear layers for k-hop aggregation + identity
        self.lins = nn.CellList([
            nn.Dense(in_channels, out_channels, has_bias=False)
            for _ in range(k + 1)
        ])

        if bias:
            self.bias = ms.Parameter(mint.zeros(out_channels, dtype=ms.float32))
        else:
            self.bias = None

    def construct(self, x, edge_index, edge_weight=None):
        """
        Forward pass - matches torch_geometric TAGConv exactly

        Args:
            x: Node features (num_nodes, in_channels)
            edge_index: Edge indices (2, num_edges)
            edge_weight: Edge weights (num_edges,) optional

        Returns:
            Output features (num_nodes, out_channels)
        """
        # Step 1: Apply GCN normalization (symmetric: D^-0.5 A D^-0.5)
        if self.normalize:
            edge_weight = self._gcn_norm(x.shape[0], edge_index, edge_weight)

        # Step 2: Initial linear transformation (K=0 term)
        out = self.lins[0](x)

        # Step 3: k-hop propagation with cumulative aggregation
        x_k = x
        for hop in range(1, self.k + 1):
            # One-hop propagation using edge_weight
            x_k = self._propagate_k(x_k, edge_index, edge_weight)

            # Add transformed aggregation to output
            out = out + self.lins[hop](x_k)

        # Step 4: Add bias
        if self.bias is not None:
            out = out + self.bias

        return out

    def _gcn_norm(self, num_nodes, edge_index, edge_weight):
        """
        Apply symmetric normalization: D^-0.5 A D^-0.5
        Using scatter_add equivalent for degree computation
        """
        num_edges = edge_index.shape[1]
        col = edge_index[1]  # destination nodes

        # Initialize edge weight if not provided
        if edge_weight is None:
            edge_weight = mint.ones(num_edges, dtype=ms.float32)

        # Compute degree: scatter_add(edge_weight, col)
        # deg[col[i]] += edge_weight[i]
        deg = mint.zeros(num_nodes, dtype=ms.float32)

        # Use scatter_add via matrix multiplication trick
        # Create a matrix where column i has all edge weights pointing to node i
        for_scatter = ops.zeros((num_edges, num_nodes), dtype=ms.float32)
        for i in range(num_edges):
            for_scatter[i, col[i]] = edge_weight[i]
        deg = ops.sum(for_scatter, dim=0)

        # Compute D^-0.5: handle zero degree nodes like PyTorch
        # deg_inv_sqrt[deg == 0] = 0 (not inf)
        # Use where to avoid inf: if deg > 0, compute 1/sqrt(deg), else 0
        deg_inv_sqrt = ops.where(
            deg > 0,
            ops.rsqrt(ops.maximum(deg, ms.Tensor(1e-10, ms.float32))),
            ms.Tensor(0.0, ms.float32)
        )

        # Apply normalization: norm[i] = D^-0.5[src[i]] * weight[i] * D^-0.5[dst[i]]
        src = edge_index[0]
        norm = deg_inv_sqrt[src] * edge_weight * deg_inv_sqrt[col]

        return norm

    def _propagate_k(self, x, edge_index, edge_weight):
        """
        Single hop propagation with proper normalization
        Equivalent to torch_geometric message passing
        """
        src = edge_index[0]  # source nodes
        dst = edge_index[1]  # destination nodes
        num_nodes = x.shape[0]
        num_edges = src.shape[0]

        # Gather source features
        x_src = x[src]  # (num_edges, feature_dim)

        # Apply normalized edge weight
        if edge_weight is not None:
            x_src = x_src * edge_weight.expand_dims(-1)  # (num_edges, feature_dim)

        # Aggregate using matrix multiplication trick
        # Create aggregation matrix: agg[i,j] = 1 if dst[i] == j, 0 otherwise
        agg_matrix = ops.zeros((num_edges, num_nodes), dtype=ms.float32)
        for i in range(num_edges):
            agg_matrix[i, dst[i]] = 1.0

        # out[j] = sum_i(agg_matrix[i,j] * x_src[i])
        out = ops.matmul(agg_matrix.t(), x_src)

        return out


class GCNConv(MessagePassing):
    """
    Graph Convolutional Network layer
    Uses MessagePassing base class for proper aggregation
    """

    def __init__(self, in_channels: int, out_channels: int, bias: bool = True):
        super().__init__(aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.lin = nn.Dense(in_channels, out_channels, has_bias=bias)

    def message(self, x_j, norm=None, **kwargs):  # pylint: disable=arguments-differ
        """Compute messages - GCN uses neighbor features scaled by norm"""
        if norm is not None:
            return x_j * norm.expand_dims(-1)
        return x_j

    def construct(self, x, edge_index, edge_weight=None):  # pylint: disable=unused-argument
        """
        Forward pass

        Args:
            x: Node features (num_nodes, in_channels)
            edge_index: Edge indices (2, num_edges)
            edge_weight: Edge weights (num_edges,) optional, currently unused

        Returns:
            Output features (num_nodes, out_channels)
        """
        # Transform features first
        x = self.lin(x)

        # Compute symmetric normalization
        row, col = edge_index[0], edge_index[1]
        num_nodes = x.shape[0]
        num_edges = edge_index.shape[1]

        # Compute degrees using a simpler method
        # Count how many edges point TO each node
        deg = mint.zeros((num_nodes,), dtype=ms.float32)
        for idx in range(num_edges):
            node_idx = int(col[idx].asnumpy())
            deg[node_idx] = deg[node_idx] + 1.0

        # Add 1 for self-loop (GCN convention)
        deg = deg + 1.0

        # Symmetric normalization: D^(-1/2)
        deg_inv_sqrt = ops.pow(deg + 1e-8, -0.5)

        # Edge weights for normalization
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Use propagate for message passing
        out = self.propagate(x=x, edge_index=edge_index, norm=norm)

        return out


def degree(index: Tensor, num_nodes=None, dtype=None) -> Tensor:
    """
    Compute node degrees from edge index
    CPU/NPU compatible implementation

    Args:
        index: Node indices (num_edges,)
        num_nodes: Total number of nodes
        dtype: Output dtype

    Returns:
        Degree tensor (num_nodes,)
    """
    if num_nodes is None:
        num_nodes = int(ops.max(index).asnumpy()) + 1

    if dtype is None:
        dtype = ms.float32

    result = mint.zeros((num_nodes,), dtype=dtype)

    for i in range(num_nodes):
        mask = index == i
        result[i] = ops.sum(mask.astype(dtype))

    return result


def to_undirected(edge_index: Tensor, num_nodes=None) -> Tensor:
    """
    Convert directed graph to undirected

    Args:
        edge_index: Edge indices (2, num_edges)
        num_nodes: Total number of nodes (unused, kept for API compatibility)

    Returns:
        Undirected edge index (2, 2*num_edges)
    """
    del num_nodes  # Unused, kept for API compatibility
    src, dst = edge_index[0], edge_index[1]

    # Create reverse edges
    reverse_edges = ops.stack([dst, src], axis=0)

    # Concatenate
    undirected = ops.concat([edge_index, reverse_edges], axis=1)

    return undirected
