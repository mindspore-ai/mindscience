# Copyright 2022 Huawei Technologies Co., Ltd
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
r"""Graphormer layer"""
from mindspore import dtype as mstype
from mindspore import Tensor, nn
from mindspore.common.initializer import initializer, Normal


class GraphNodeFeature(nn.Cell):
    r"""
    Compute node features for each node in the graph.

    Args:
        num_heads (int): Number of attention heads.
        num_node_type (int): Number of node types.
        num_in_degree (int): The maximum in-degree of all nodes.
        num_out_degree (int): The maximum out-degree of all nodes.
        embed_dim (int): The dimension of embedding.
        compute_dtype (mstype.dtype): The computation type. Default: mstype.float16.

    Inputs:
        - **node_type** (Tensor) - The type of each node, shape :math:`(n\_graph, n\_node, 1)`.
        - **in_degree** (Tensor) - The in-degree of each node, shape :math:`(n\_graph, n\_node)`.
        - **out_degree** (Tensor) - The out-degree of each node, shape :math:`(n\_graph, n\_node)`.


    Outputs:
        Tensor of shape :math:`(n\_graph, n\_node, n\_hidden)`.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import mindspore.nn as nn
        >>> from mindspore import Tensor
        >>> from src.cell.pdeformer.graphormer.graphormer_layer import GraphNodeFeature
        >>> num_heads = 8
        >>> num_node_type = 10
        >>> num_in_degree = 10
        >>> num_out_degree = 10
        >>> embed_dim = 64
        >>> node_type = Tensor(np.random.randint(0, num_node_type, size=(2, 16, 1)), dtype=mstype.int32)
        >>> in_degree = Tensor(np.random.randint(0, num_in_degree, size=(2, 16)), dtype=mstype.int32)
        >>> out_degree = Tensor(np.random.randint(0, num_out_degree, size=(2, 16)), dtype=mstype.int32)
        >>> feature_layer = GraphNodeFeature(num_heads,
        >>>                                  num_node_type,
        >>>                                  num_in_degree,
        >>>                                  num_out_degree,
        >>>                                  embed_dim,
        >>>                                  mstype.float16)
        >>> output = feature_layer(node_type, in_degree, out_degree)
        >>> print(output.shape)
        (2, 16, 64)
    """

    def __init__(
            self,
            num_heads: int,
            num_node_type: int,
            num_in_degree: int,
            num_out_degree: int,
            embed_dim: int,
            compute_dtype=mstype.float16) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.num_node_type = num_node_type

        # 1 for graph token
        self.node_encoder = nn.Embedding(num_node_type + 1, embed_dim, padding_idx=0).to_float(compute_dtype)
        self.in_degree_encoder = nn.Embedding(num_in_degree, embed_dim, padding_idx=0).to_float(compute_dtype)
        self.out_degree_encoder = nn.Embedding(
            num_out_degree, embed_dim, padding_idx=0
        ).to_float(compute_dtype)

        self.init_params()

    def init_params(self) -> None:
        r"""Initialize all embedding_table."""
        self.node_encoder.embedding_table.set_data(
            initializer(
                Normal(sigma=0.02, mean=0.0),
                self.node_encoder.embedding_table.shape, self.node_encoder.embedding_table.dtype))
        self.in_degree_encoder.embedding_table.set_data(
            initializer(
                Normal(sigma=0.02, mean=0.0),
                self.in_degree_encoder.embedding_table.shape, self.in_degree_encoder.embedding_table.dtype))
        self.out_degree_encoder.embedding_table.set_data(
            initializer(
                Normal(sigma=0.02, mean=0.0),
                self.out_degree_encoder.embedding_table.shape, self.out_degree_encoder.embedding_table.dtype))

    def construct(self,
                  node_type: Tensor,
                  in_degree: Tensor,
                  out_degree: Tensor) -> Tensor:
        r"""construct"""
        node_feature = self.node_encoder(node_type).sum(axis=-2)  # [n_graph, n_node, n_hidden]
        node_feature += self.in_degree_encoder(in_degree)  # [n_graph, n_node, n_hidden]
        node_feature += self.out_degree_encoder(out_degree)  # [n_graph, n_node, n_hidden]

        return node_feature


class GraphAttnBias(nn.Cell):
    r"""
    Compute attention bias for each head.

    Args:
        num_heads (int): Number of attention heads.
        num_spatial (int): The maximum number of hops in the shortest path between any two nodes.
        compute_dtype (mstype.dtype): The computation type. Default: mstype.float16.

    Inputs:
        - **attn_bias** (Tensor) - The attention bias of the graph, shape :math:`(n\_graph, n\_node, n\_node)`.
        - **spatial_pos** (Tensor) - The spatial position from each node to each other node,
          shape :math:`(n\_graph, n\_node, n\_node)`.

    Outputs:
        Tensor of shape :math:`(n\_graph, n\_head, n\_node, n\_node)`.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import mindspore.nn as nn
        >>> from mindspore import Tensor
        >>> from src.cell.pdeformer.graphormer.graphormer_layer import GraphAttnBias
        >>> num_heads = 8
        >>> num_spatial = 10
        >>> attn_bias = Tensor(np.random.randint(0, 1, size=(2, 16, 16)), dtype=mstype.float32)
        >>> spatial_pos = Tensor(np.random.randint(0, num_spatial, size=(2, 16, 16)), dtype=mstype.int32)
        >>> bias_layer = GraphAttnBias(num_heads, num_spatial, mstype.float16)
        >>> output = bias_layer(attn_bias, spatial_pos)
        >>> print(output.shape)
        (2, 8, 16, 16)
    """

    def __init__(
            self,
            num_heads: int,
            num_spatial: int,
            compute_dtype=mstype.float16) -> None:
        super().__init__()
        self.num_heads = num_heads

        self.spatial_pos_encoder = nn.Embedding(num_spatial, num_heads, padding_idx=0).to_float(compute_dtype)
        self.spatial_pos_encoder_rev = nn.Embedding(num_spatial, num_heads, padding_idx=0).to_float(compute_dtype)

        self.init_params()

    def init_params(self) -> None:
        r"""Initialize all embedding_table"""
        self.spatial_pos_encoder.embedding_table.set_data(
            initializer(
                Normal(sigma=0.02, mean=0.0),
                self.spatial_pos_encoder.embedding_table.shape, self.spatial_pos_encoder.embedding_table.dtype))
        self.spatial_pos_encoder_rev.embedding_table.set_data(
            initializer(
                Normal(sigma=0.02, mean=0.0),
                self.spatial_pos_encoder_rev.embedding_table.shape, self.spatial_pos_encoder_rev.embedding_table.dtype))

    def construct(self,
                  attn_bias: Tensor,
                  spatial_pos: Tensor) -> Tensor:
        r"""construct"""
        graph_attn_bias = attn_bias.copy()  # [n_graph, n_node, n_node]

        graph_attn_bias = graph_attn_bias.unsqueeze(1)  # [n_graph, n_node, n_node] -> [n_graph, 1, n_node, n_node]
        multiples = (1, self.num_heads, 1, 1)
        # [n_graph, 1, n_node, n_node] -> [n_graph, n_head, n_node, n_node]
        graph_attn_bias = graph_attn_bias.tile(multiples)

        # The original implementation of Graphormer tackles with undirected graphs,
        # and uses A_ij = .. + b_phi(i,j) (see eqn(6) of Graphormer paper). For DAGs,
        # we use A_ij = .. + b_phi(i,j) + d_phi(j,i).
        spatial_pos_bias_0 = self.spatial_pos_encoder(spatial_pos)  # [n_graph, n_node, n_node, n_head]
        spatial_pos_rev = spatial_pos.permute(0, 2, 1)  # [n_graph, n_node, n_node]
        spatial_pos_bias_1 = self.spatial_pos_encoder_rev(spatial_pos_rev)  # [n_graph, n_node, n_node, n_head]
        spatial_pos_bias = spatial_pos_bias_0 + spatial_pos_bias_1  # [n_graph, n_node, n_node, n_head]
        # [n_graph, n_node, n_node, n_head] -> [n_graph, n_head, n_node, n_node]
        spatial_pos_bias = spatial_pos_bias.permute(0, 3, 1, 2)

        graph_attn_bias = spatial_pos_bias + attn_bias.unsqueeze(1)  # reset
        return graph_attn_bias  # [n_graph, n_head, n_node, n_node]
