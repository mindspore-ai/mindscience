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
"""GVP encoder"""

import mindspore.ops as ops
import mindspore.nn as nn

from src.features import GVPGraphEmbedding
from src.gvp_modules import GVPConvLayer
from src.gvp_utils import unflatten_graph


class GVPEncoder(nn.Cell):
    """GVP encoder"""
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.embed_graph = GVPGraphEmbedding(args)

        node_hidden_dim = (args.node_hidden_dim_scalar,
                           args.node_hidden_dim_vector)
        edge_hidden_dim = (args.edge_hidden_dim_scalar,
                           args.edge_hidden_dim_vector)

        conv_activations = (ops.ReLU(), ops.Sigmoid())
        self.encoder_layers = nn.CellList(
            [GVPConvLayer(
                node_hidden_dim,
                edge_hidden_dim,
                drop_rate=args.dropout,
                vector_gate=True,
                attention_heads=0,
                n_message=3,
                conv_activations=conv_activations,
                n_edge_gvps=0,
                eps=1e-4,
                layernorm=True,
            )
             for i in range(args.num_encoder_layers)]
        )

    def construct(self, coords, coord_mask, padding_mask, confidence):
        node_embeddings, edge_embeddings, edge_index = self.embed_graph(
            coords, coord_mask, padding_mask, confidence)

        for _, layer in enumerate(self.encoder_layers):
            node_embeddings, edge_embeddings = layer(node_embeddings,
                                                     edge_index, edge_embeddings)

        node_embeddings = unflatten_graph(node_embeddings, coords.shape[0])
        return node_embeddings
