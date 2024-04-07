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
r"""Graphormer encoder"""
from typing import Optional

from mindspore import dtype as mstype
from mindspore import Tensor, nn, ops

from .graphormer_layer import GraphNodeFeature, GraphAttnBias
from .graphormer_encoder_layer import GraphormerEncoderLayer


class GraphormerEncoder(nn.Cell):
    r"""
    Graphormer encoder.

    Args:
        num_node_type (int): Number of node types (excluding the global node type).
        num_in_degree (int): The maximum in-degree of all nodes.
        num_out_degree (int): The maximum out-degree of all nodes.
        num_edge_type (int): Number of edge types.
        num_spatial (int): The maximum number of hops in the shortest path between any two nodes.
        num_edge_dist (int): For a set composed of the hops of the shortest path between any two
            nodes, the number of set elements.
        multi_hop_max_dist (int): When encoding edges, the maximum number of hops for the
            shortest path.
        num_encoder_layers (int): The number of encoder layers. Default: ``12``.
        embed_dim (int): The dimension of embedding. Default: ``768``.
        ffn_embed_dim (int): The dimension of FFN's embedding (the number of output neurons
            corresponding to the first nn.Dense layer of the FFN). Default: ``3072``.
        num_heads (int): The number of heads in MHA. Default: ``8``.
        dropout (float): The discard rate of dropout layer. Default: ``0.1``.
        attention_dropout (float): The discard rate of dropout layer in MHA. Default: ``0.1``.
        activation_dropout (float): The discard rate of dropout layer in FFN. Default: ``0.1``.
        encoder_normalize_before (bool): The embedding corresponding to the node performs or does
            not perform layernorm before entering the encoder of the Transformer.
            Default: ``False``.
        pre_layernorm (bool): LayerNorm is applied either before or after the self-attention/ffn
            modules. Default: ``False``.
        activation_fn (str): The type of activation function. Default: ``gelu``.
        compute_dtype (mstype.dtype): The computation type of the layer. Default: ``mstype.float16``.

    Inputs:
        - **node_type** (Tensor) - The type of each node, shape :math:`(n\_graph, n\_node, 1)`.
        - **node_input_feature** (Tensor) - The input feature of each node,
          shape :math:`(n\_graph, n\_node, embed\_dim)`.
        - **in_degree** (Tensor) - The in-degree of each node, shape :math:`(n\_graph, n\_node)`.
        - **out_degree** (Tensor) - The out-degree of each node, shape :math:`(n\_graph, n\_node)`.
        - **attn_bias** (Tensor) - The attention bias of the graph, shape :math:`(n\_graph, n\_node, n\_node)`.
        - **spatial_pos** (Tensor) - The spatial position from each node to each other node,
          shape :math:`(n\_graph, n\_node, n\_node)`.
        - **token_embeddings** (Tensor, optional) - The token embedding of each node,
          shape :math:`(n\_graph, n\_node, embed\_dim)`. Default: None.
        - **attn_mask** (Tensor, optional) - The attention mask of the graph,
          shape :math:`(n\_graph, n\_node, n\_node)`. Default: None.

    Outputs:
        Tensor of shape :math:`(n\_node, n\_graph, embed\_dim)`.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> from src.cell.pdeformer.graphormer.graphormer_encoder import GraphormerEncoder
        >>> num_node_type = 5
        >>> num_in_degree = 10
        >>> num_out_degree = 10
        >>> num_spatial = 10
        >>> num_encoder_layers = 12
        >>> embed_dim = 768
        >>> ffn_embed_dim = 3072
        >>> num_heads = 8
        >>> dropout = 0.1
        >>> attention_dropout = 0.1
        >>> activation_dropout = 0.1
        >>> encoder_normalize_before = False
        >>> pre_layernorm = False
        >>> activation_fn = "gelu"
        >>> compute_dtype = mstype.float16
        >>> node_type = Tensor(np.random.randint(0, num_node_type, size=(2, 10, 1)), mstype.int32)
        >>> node_input_feature = Tensor(np.random.randn(2, 10, embed_dim), mstype.float32)
        >>> in_degree = Tensor(np.random.randint(0, num_in_degree, size=(2, 10)), mstype.int32)
        >>> out_degree = Tensor(np.random.randint(0, num_out_degree, size=(2, 10)), mstype.int32)
        >>> attn_bias = Tensor(np.random.randn(2, 10, 10), mstype.float32)
        >>> spatial_pos = Tensor(np.random.randint(0, num_spatial, size=(2, 10, 10)), mstype.int32)
        >>> encoder = GraphormerEncoder(num_node_type,
        >>>                             num_in_degree,
        >>>                             num_out_degree,
        >>>                             num_spatial,
        >>>                             num_encoder_layers,
        >>>                             embed_dim,
        >>>                             ffn_embed_dim,
        >>>                             num_heads,
        >>>                             dropout,
        >>>                             attention_dropout,
        >>>                             activation_dropout,
        >>>                             encoder_normalize_before,
        >>>                             pre_layernorm,
        >>>                             activation_fn,
        >>>                             compute_dtype)
        >>> output = encoder(node_type, node_input_feature, in_degree, out_degree, attn_bias, spatial_pos)
        >>> print(output.shape)
        (10, 2, 768)
    """

    def __init__(
            self,
            num_node_type: int,
            num_in_degree: int,
            num_out_degree: int,
            num_spatial: int,
            num_encoder_layers: int = 12,
            embed_dim: int = 768,
            ffn_embed_dim: int = 768,
            num_heads: int = 32,
            dropout: float = 0.1,
            attention_dropout: float = 0.1,
            activation_dropout: float = 0.1,
            encoder_normalize_before: bool = False,
            pre_layernorm: bool = False,
            activation_fn: str = "gelu",
            compute_dtype=mstype.float16) -> None:
        super().__init__()

        self.dropout_module = nn.Dropout(p=dropout)
        self.embed_dim = embed_dim

        self.graph_node_feature = GraphNodeFeature(
            num_heads=num_heads,
            num_node_type=num_node_type,
            num_in_degree=num_in_degree,
            num_out_degree=num_out_degree,
            embed_dim=embed_dim,
            compute_dtype=compute_dtype
        )

        self.graph_attn_bias = GraphAttnBias(
            num_heads=num_heads,
            num_spatial=num_spatial,
            compute_dtype=compute_dtype
        )

        if encoder_normalize_before:
            self.emb_layer_norm = nn.LayerNorm([self.embed_dim], epsilon=1e-5).to_float(mstype.float32)
        else:
            self.emb_layer_norm = None

        self.layers = nn.CellList()
        for _ in range(num_encoder_layers):
            self.layers.append(GraphormerEncoderLayer(
                embed_dim,
                ffn_embed_dim,
                num_heads,
                dropout,
                attention_dropout,
                activation_dropout,
                activation_fn,
                pre_layernorm,
                compute_dtype=compute_dtype
            ))

    def construct(
            self,
            node_type: Tensor,
            node_input_feature: Tensor,
            in_degree: Tensor,
            out_degree: Tensor,
            attn_bias: Tensor,
            spatial_pos: Tensor,
            token_embeddings: Optional[Tensor] = None,
            attn_mask: Optional[Tensor] = None) -> Tensor:
        r"""construct"""
        # Compute padding mask which is needed for multihead attention
        node_type_ = node_type.squeeze(-1)
        padding_mask = ops.equal(node_type_, 0)  # [n_graph, n_node]

        if token_embeddings is not None:
            x = token_embeddings  # [n_graph, n_node, embed_dim]
        else:
            x = self.graph_node_feature(node_type, in_degree, out_degree)  # [n_graph, n_node, embed_dim]

        x = x + node_input_feature  # [n_graph, n_node, embed_dim]

        # Shape is [n_graph, n_head, n_node, n_node].
        attn_bias = self.graph_attn_bias(attn_bias, spatial_pos)

        if self.emb_layer_norm is not None:
            x = self.emb_layer_norm(x)

        # account for padding while computing the representation
        x = x.transpose(1, 0, 2)  # [n_graph, n_node, embed_dim] -> [n_node, n_graph, embed_dim]

        for layer in self.layers:
            x = layer(x, attn_bias, attn_mask, padding_mask)  # [n_node, n_graph, embed_dim]

        return x  # [n_node, n_graph, embed_dim]
