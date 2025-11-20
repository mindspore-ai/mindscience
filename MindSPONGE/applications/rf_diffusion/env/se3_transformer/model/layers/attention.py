# Modified from se3-transformer-public (https://github.com/FabianFuchsML/se3-transformer-public)
# Original license: MIT License
#
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

from typing import Dict, Optional, Union

import mindspore as ms
import mindspore.nn as nn
import numpy as np
import sharker
from mindspore import Tensor
from se3_transformer.model.fiber import Fiber
from se3_transformer.model.layers.convolution import ConvSE3, ConvSE3FuseLevel
from se3_transformer.model.layers.linear import LinearSE3
from se3_transformer.runtime.utils import (
    aggregate_residual,
    degree_to_dim,
    unfuse_features,
)
from sharker.data import Graph
from sharker.utils import softmax


class AttentionSE3(nn.Cell):
    """Multi-headed sparse graph self-attention (SE(3)-equivariant)"""

    def __init__(self, num_heads: int, key_fiber: Fiber, value_fiber: Fiber):
        """
        :param num_heads:     Number of attention heads
        :param key_fiber:     Fiber for the keys (and also for the queries)
        :param value_fiber:   Fiber for the values
        """
        super().__init__()
        self.num_heads = num_heads
        self.key_fiber = key_fiber
        self.value_fiber = value_fiber

    def construct(
        self,
        # edge features (may be fused)
        value: Union[Tensor, Dict[str, Tensor]],
        # edge features (may be fused)
        key: Union[Tensor, Dict[str, Tensor]],
        # node features
        query: Dict[str, Tensor],
        graph: Graph,
    ):

        if isinstance(key, Tensor):
            # case where features of all types are fused
            key = key.reshape(key.shape[0], self.num_heads, -1)
            # need to reshape queries that way to keep the same layout as keys
            out = ms.mint.cat([query[str(d)] for d in self.key_fiber.degrees], dim=-1)
            query = out.reshape(list(query.values())[0].shape[0], self.num_heads, -1)
        else:
            # features are not fused, need to fuse and reshape them
            key = self.key_fiber.to_attention_heads(key, self.num_heads)
            query = self.key_fiber.to_attention_heads(query, self.num_heads)

        # Compute attention weights (softmax of inner product between key and query)
        dst = graph.edge_index[1]
        query_dst = query[dst]
        edge_weights = (key * query_dst).sum(dim=-1)
        edge_weights /= np.sqrt(self.key_fiber.num_features)
        edge_weights = softmax(edge_weights, dst)
        edge_weights = edge_weights[..., None, None]

        if isinstance(value, Tensor):
            # features of all types are fused
            v = value.view(value.shape[0], self.num_heads, -1, value.shape[-1])
            weights = edge_weights * v
            # row,col = graph.edge_index
            feat_out = sharker.utils.scatter(weights, dst, dim=0, reduce="sum")
            # merge heads
            feat_out = feat_out.view(feat_out.shape[0], -1, feat_out.shape[-1])
            out = unfuse_features(feat_out, self.value_fiber.degrees)
        else:
            out = {}
            for degree, channels in self.value_fiber:
                v = value[str(degree)].view(
                    -1,
                    self.num_heads,
                    channels // self.num_heads,
                    degree_to_dim(degree),
                )
                weights = edge_weights * v
                res = sharker.utils.scatter(weights, dst, dim=0, reduce="sum")
                # merge heads
                out[str(degree)] = res.view(-1, channels, degree_to_dim(degree))

        return out


class AttentionBlockSE3(nn.Cell):
    """Multi-headed sparse graph self-attention block with skip connection, linear projection (SE(3)-equivariant)"""

    def __init__(
        self,
        fiber_in: Fiber,
        fiber_out: Fiber,
        fiber_edge: Optional[Fiber] = None,
        num_heads: int = 4,
        channels_div: int = 2,
        use_layer_norm: bool = False,
        max_degree: bool = 4,
        fuse_level: ConvSE3FuseLevel = ConvSE3FuseLevel.FULL,
        **kwargs
    ):
        """
        :param fiber_in:         Fiber describing the input features
        :param fiber_out:        Fiber describing the output features
        :param fiber_edge:       Fiber describing the edge features (node distances excluded)
        :param num_heads:        Number of attention heads
        :param channels_div:     Divide the channels by this integer for computing values
        :param use_layer_norm:   Apply layer normalization between MLP layers
        :param max_degree:       Maximum degree used in the bases computation
        :param fuse_level:       Maximum fuse level to use in TFN convolutions
        """
        super().__init__()
        if fiber_edge is None:
            fiber_edge = Fiber({})
        self.fiber_in = fiber_in
        # value_fiber has same structure as fiber_out but #channels divided by 'channels_div'
        value_fiber = Fiber(
            [(degree, channels // channels_div) for degree, channels in fiber_out]
        )
        # key_query_fiber has the same structure as fiber_out, but only degrees which are in in_fiber
        # (queries are merely projected, hence degrees have to match input)
        key_query_fiber = Fiber(
            [
                (fe.degree, fe.channels)
                for fe in value_fiber
                if fe.degree in fiber_in.degrees
            ]
        )

        self.to_key_value = ConvSE3(
            fiber_in,
            value_fiber + key_query_fiber,
            pool=False,
            fiber_edge=fiber_edge,
            use_layer_norm=use_layer_norm,
            max_degree=max_degree,
            fuse_level=fuse_level,
            allow_fused_output=True,
        )
        self.to_query = LinearSE3(fiber_in, key_query_fiber)
        self.attention = AttentionSE3(num_heads, key_query_fiber, value_fiber)
        self.project = LinearSE3(value_fiber + fiber_in, fiber_out)

    def construct(
        self,
        node_features: Dict[str, Tensor],
        edge_features: Dict[str, Tensor],
        graph: Graph,
        basis: Dict[str, Tensor],
    ):
        fused_key_value = self.to_key_value(node_features, edge_features, graph, basis)
        key, value = self._get_key_value_from_fused(fused_key_value)
        query = self.to_query(node_features)
        z = self.attention(value, key, query, graph)
        z_concat = aggregate_residual(node_features, z, "cat")
        return self.project(z_concat)

    def _get_key_value_from_fused(self, fused_key_value):
        # Extract keys and queries features from fused features
        if isinstance(fused_key_value, Tensor):
            # Previous layer was a fully fused convolution
            value, key = ms.mint.chunk(fused_key_value, chunks=2, dim=-2)
        else:
            key, value = {}, {}
            for degree, feat in fused_key_value.items():
                if int(degree) in self.fiber_in.degrees:
                    value[degree], key[degree] = ms.mint.chunk(feat, chunks=2, dim=-2)
                else:
                    value[degree] = feat

        return key, value
