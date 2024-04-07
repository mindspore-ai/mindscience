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
r"""Multi-headed attention."""
from typing import Optional
import math

import mindspore as ms
from mindspore import dtype as mstype
from mindspore import Tensor, nn, ops
from mindspore.common.initializer import initializer, XavierUniform, Zero, Uniform


class MultiheadAttention(nn.Cell):
    r"""
    Multi-headed attention. See "Attention Is All You Need" paper for more details.

    Args:
        embed_dim (int): The dimension of embedding.
        num_heads (int): The number of heads.
        dropout (float): The discard rate of dropout layer. Default: ``0.0``.
        bias (bool): Determine whether bias is included in the nn.Dense layer. Default: ``True``.
        compute_dtype (mstype.dtype): The computation type. Default: mstype.float16.

    Inputs:
        - **x** (Tensor) - Input Tensor, shape is : math:`(n\_node, n\_graph, embed\_dim)`.
        - **attn_bias** (Tensor, optional) - Graphormer's self-attention bias for encoding graph
          structure information, shape is : math:`(n\_graph * num\_heads,, n\_node, n\_node)`.
        - **key_padding_mask** (ByteTensor, optional) - Mask to exclude keys that are pads where
          padding elements are indicated by 1s, shape is : math:`(n\_graph, n\_node)`.
        - **attn_mask** (ByteTensor, optional) - Used to implement causal attention, where the mask
          prevents the attention from looking forward in time, shape is : math:`(n\_node, n\_node)`.

    Outputs:
        Tensor of shape :math:`(n\_node, n\_graph, embed\_dim)`.

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> from src.cell.pdeformer.graphormer.multihead_attention import MultiheadAttention
        >>> x = Tensor(np.random.randn(16, 8, 128), dtype=mstype.float32)
        >>> mha = MultiheadAttention(embed_dim=128, num_heads=8)
        >>> output = mha(x)
        >>> print(output.shape)
        (16, 8, 128)
    """

    def __init__(
            self,
            embed_dim,
            num_heads,
            dropout=0.0,
            bias=True,
            compute_dtype=mstype.float16) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        if self.embed_dim <= 0:
            raise ValueError("'embed_dim' must be a positive integer.")
        self.compute_dtype = compute_dtype
        self.num_heads = num_heads

        self.dropout_module = nn.Dropout(p=dropout)

        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != self.embed_dim:
            raise ValueError("'embed_dim' must be divisible by 'num_heads'")
        self.scaling = self.head_dim ** -0.5

        self.k_proj = nn.Dense(embed_dim, embed_dim, has_bias=bias).to_float(compute_dtype)
        self.v_proj = nn.Dense(embed_dim, embed_dim, has_bias=bias).to_float(compute_dtype)
        self.q_proj = nn.Dense(embed_dim, embed_dim, has_bias=bias).to_float(compute_dtype)
        self.out_proj = nn.Dense(embed_dim, embed_dim, has_bias=bias).to_float(compute_dtype)

        self.init_params()

        self.cast = ops.Cast()

    def init_params(self) -> None:
        """
        Set initializer to parameters, Empirically observed the convergence to be much
        better with the scaled initialization.
        """

        scale = math.sqrt(1 / self.embed_dim)
        self.k_proj.weight.set_data(
            initializer(XavierUniform(gain=1 / math.sqrt(2)), self.k_proj.weight.shape, self.k_proj.weight.dtype))
        if self.k_proj.bias is not None:
            self.k_proj.bias.set_data(
                initializer(Uniform(scale), self.k_proj.bias.shape, self.k_proj.bias.dtype))

        self.v_proj.weight.set_data(
            initializer(XavierUniform(gain=1 / math.sqrt(2)), self.v_proj.weight.shape, self.v_proj.weight.dtype))
        if self.v_proj.bias is not None:
            self.v_proj.bias.set_data(
                initializer(Uniform(scale), self.v_proj.bias.shape, self.v_proj.bias.dtype))

        self.q_proj.weight.set_data(
            initializer(XavierUniform(gain=1 / math.sqrt(2)), self.q_proj.weight.shape, self.q_proj.weight.dtype))
        if self.q_proj.bias is not None:
            self.q_proj.bias.set_data(
                initializer(Uniform(scale), self.q_proj.bias.shape, self.q_proj.weight.dtype))

        self.out_proj.weight.set_data(
            initializer(XavierUniform(gain=1), self.out_proj.weight.shape, self.out_proj.weight.dtype))
        if self.out_proj.bias is not None:
            self.out_proj.bias.set_data(
                initializer(Zero(), self.out_proj.bias.shape, self.out_proj.bias.dtype))

    def construct(
            self,
            x: Tensor,
            attn_bias: Optional[Tensor],
            key_padding_mask: Optional[Tensor] = None,
            attn_mask: Optional[Tensor] = None) -> Tensor:
        r"""construct"""
        n_node, n_graph, embed_dim = x.shape

        # [n_node, n_graph, embed_dim] * [embed_dim, embed_dim] -> [n_node, n_graph, embed_dim]
        query = self.q_proj(x)

        # [n_node, n_graph, embed_dim] * [embed_dim, embed_dim] -> [n_node, n_graph, embed_dim]
        key = self.k_proj(x)

        # [n_node, n_graph, embed_dim] * [embed_dim, embed_dim] -> [n_node, n_graph, embed_dim]
        value = self.v_proj(x)

        query *= self.scaling

        # [n_node, n_graph, embed_dim] -> [n_graph*num_heads, n_node, head_dim]
        query = query.flatten().view(n_node, n_graph * self.num_heads,
                                     self.head_dim).transpose(1, 0, 2)

        # [n_node, n_graph, embed_dim] -> [n_graph*num_heads, n_node, head_dim]
        key = key.flatten().view(n_node, n_graph * self.num_heads,
                                 self.head_dim).transpose(1, 0, 2)

        # [n_node, n_graph, embed_dim] -> [n_graph*num_heads, n_node, head_dim]
        value = value.flatten().view(n_node, n_graph * self.num_heads,
                                     self.head_dim).transpose(1, 0, 2)

        # [n_graph*num_heads, n_node, head_dim] x [n_graph*num_heads, head_dim, n_node]
        # -> [n_graph*num_heads, n_node, n_node]
        attn_weights = ops.bmm(query, key.transpose(0, 2, 1))

        # Core code of Graphormer
        if attn_bias is not None:
            # Shape is [n_graph*num_heads, n_node, n_node].
            attn_weights += attn_bias.view(n_graph * self.num_heads, n_node, n_node)

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(dim=0)  # [n_node, n_node] -> [1, n_node, n_node]
            attn_weights += attn_mask  # [n_graph*num_heads, n_node, n_node]

        if key_padding_mask is not None and key_padding_mask.ndim == 0:
            key_padding_mask = None

        if key_padding_mask is not None:
            if key_padding_mask.shape[0] != n_graph or key_padding_mask.shape[1] != n_node:
                raise ValueError(
                    f"'key_padding_mask' shape error: Expected ({n_graph}, {n_node}), "
                    f"but got {key_padding_mask.shape}.")

            # don't attend to padding symbols
            # [n_graph*num_heads, n_node, n_node] -> [n_graph, num_heads, n_node, n_node]
            attn_weights = attn_weights.view(n_graph, self.num_heads, n_node, n_node)

            # [n_graph, n_node] -> [n_graph, 1, 1, n_node]
            key_padding_mask = key_padding_mask.unsqueeze(dim=1).unsqueeze(dim=2).to(ms.bool_)
            attn_weights = attn_weights.masked_fill(
                key_padding_mask, float("-inf"))  # [n_graph, num_heads, n_node, n_node]

            # [n_graph, num_heads, n_node, n_node] -> [n_graph*num_heads, n_node, n_node]
            attn_weights = attn_weights.view(n_graph * self.num_heads, n_node, n_node)

        attn_weights = self.cast(attn_weights, mstype.float32)
        attn_probs = ops.softmax(attn_weights, axis=-1)  # [n_graph*num_heads, n_node, n_node]
        attn_probs = self.cast(attn_probs, self.compute_dtype)

        # [n_graph*num_heads, n_node, n_node] x [n_graph*num_heads, n_node, head_dim]
        # -> [n_graph*num_heads, n_node, head_dim]
        attn = ops.bmm(attn_probs, value)

        # [n_graph*num_heads, n_node, head_dim] -> [n_node, n_graph, embed_dim]
        attn = attn.transpose(1, 0, 2).flatten().view(n_node, n_graph, embed_dim)

        # [n_node, n_graph, embed_dim] * [embed_dim, embed_dim] -> [n_node, n_graph, embed_dim]
        attn = self.out_proj(attn)

        return attn  # [n_node, n_graph, embed_dim]
