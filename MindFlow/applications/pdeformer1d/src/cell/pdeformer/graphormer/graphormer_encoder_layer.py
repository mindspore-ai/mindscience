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
r"""Graphormer encoder layer."""
from typing import Optional

from mindspore import dtype as mstype
from mindspore import Tensor, nn

from ...basic_block import UniformInitDense
from .multihead_attention import MultiheadAttention


def get_activation_fn(activation_fn: str = "gelu") -> nn.Cell:
    r"""Get activation function."""
    if activation_fn.lower() == "relu":
        return nn.ReLU()
    if activation_fn.lower() == "gelu":
        return nn.GELU()
    raise NotImplementedError


class GraphormerEncoderLayer(nn.Cell):
    r"""
    Basic module in Transformer encoder, including multihead-attention (MHA)
    and feed-forward-network (FFN).

    Args:
        embed_dim (int): The dimension of embedding. Default: ``768``.
        ffn_embed_dim (int): The dimension of FFN's embedding. Default: ``3072``.
        num_heads (int): The number of heads in MHA. Default: ``8``.
        dropout (float): The discard rate of dropout layer. Default: ``0.1``.
        attention_dropout (float): The discard rate of dropout layer in MHA. Default: ``0.1``.
        activation_dropout (float): The discard rate of dropout layer in FFN. Default: ``0.1``.
        activation_fn (str): The activation function in FFN. Default: ``"gelu"``.
        pre_layernorm (bool): LayerNorm is applied either before or after the self-attention/ffn
            modules. Default: ``False``.
        compute_dtype (mstype.dtype): The computation type of the layer. Default: ``mstype.float16``.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(n\_node, n\_graph, embed\_dim)`.
        - **attn_bias** (Tensor, optional) - Graphormer's self-attention bias for encoding graph
          structure information, shape :math:`(n\_graph * num\_heads,, n\_node, n\_node)`.
        - **attn_mask** (ByteTensor, optional) - Used to implement causal attention, where the mask
          prevents the attention from looking forward in time, shape : math:`(n\_node, n\_node)`.
        - **attn_padding_mask** (ByteTensor, optional) - Mask to exclude keys that are pads where
          padding elements are indicated by 1s, shape : math:`(n\_graph, n\_node)`.

    Outputs:
        Tensor of shape :math:`(n\_node, n\_graph, embed\_dim)`.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> from src.cell.pdeformer.graphormer.graphormer_encoder_layer import GraphormerEncoderLayer
        >>> x = Tensor(np.random.randn(16, 32, 768), mstype.float32)
        >>> encoder_layer = GraphormerEncoderLayer(embed_dim=768, ffn_embed_dim=3072, num_heads=8, dropout=0.1,
        >>>                                       attention_dropout=0.1, activation_dropout=0.1, activation_fn="gelu",
        >>>                                       pre_layernorm=False, compute_dtype=mstype.float16)
        >>> output = encoder_layer(x)
        >>> print(output.shape)
        (16, 32, 768)
    """

    def __init__(
            self,
            embed_dim: int = 768,
            ffn_embed_dim: int = 3072,
            num_heads: int = 8,
            dropout: float = 0.1,
            attention_dropout: float = 0.1,
            activation_dropout: float = 0.1,
            activation_fn: str = "gelu",
            pre_layernorm: bool = False,
            compute_dtype=mstype.float16) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_embed_dim = ffn_embed_dim
        self.num_attention_heads = num_heads
        self.attention_dropout = attention_dropout
        self.pre_layernorm = pre_layernorm

        self.dropout_module = nn.Dropout(p=dropout)
        self.activation_dropout_module = nn.Dropout(p=activation_dropout)
        self.activation_fn = get_activation_fn(activation_fn)
        self.multihead_attn = MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            compute_dtype=compute_dtype
        )
        self.attn_layer_norm = nn.LayerNorm([embed_dim], epsilon=1e-5).to_float(mstype.float32)
        self.fc1 = UniformInitDense(embed_dim, ffn_embed_dim, has_bias=True).to_float(compute_dtype)
        self.fc2 = UniformInitDense(ffn_embed_dim, embed_dim, has_bias=True).to_float(compute_dtype)
        self.ffn_layer_norm = nn.LayerNorm([embed_dim], epsilon=1e-5).to_float(mstype.float32)

    def construct(
            self,
            x: Tensor,
            attn_bias: Optional[Tensor] = None,
            attn_mask: Optional[Tensor] = None,
            attn_padding_mask: Optional[Tensor] = None) -> Tensor:
        '''construct'''
        residual = x  # [n_node, n_graph, embed_dim]
        if self.pre_layernorm:
            x = self.attn_layer_norm(x)  # [n_node, n_graph, embed_dim]

        # Multihead attention (MHA)
        x = self.multihead_attn(
            x,
            attn_bias=attn_bias,
            key_padding_mask=attn_padding_mask,
            attn_mask=attn_mask,
        )  # [n_node, n_graph, embed_dim]
        x = residual + x  # [n_node, n_graph, embed_dim]
        if not self.pre_layernorm:
            x = self.attn_layer_norm(x)  # [n_node, n_graph, embed_dim]

        # Feed forward network (FFN)
        residual = x  # [n_node, n_graph, embed_dim]
        if self.pre_layernorm:
            x = self.ffn_layer_norm(x)  # [n_node, n_graph, embed_dim]
        x = self.activation_fn(self.fc1(x))  # [n_node, n_graph, ffn_embed_dim]
        x = self.fc2(x)  # [n_node, n_graph, embed_dim]
        x = residual + x  # [n_node, n_graph, embed_dim]
        if not self.pre_layernorm:
            x = self.ffn_layer_norm(x)  # [n_node, n_graph, embed_dim]

        return x  # [n_node, n_graph, embed_dim]
