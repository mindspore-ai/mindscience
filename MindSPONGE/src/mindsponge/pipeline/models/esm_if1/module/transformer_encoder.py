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
"""Constructing encoder in transformer network"""

import argparse
import math
from typing import Optional
import mindspore as ms
import mindspore.ops as ops
from mindspore import nn, Tensor
# pylint: disable=relative-beyond-top-level
from .features import GVPInputFeaturizer, DihedralFeatures, GVPGraphEmbedding
from .util import nan_to_num, get_rotation_frames, rotate, rbf, unflatten_graph, ms_transpose, ms_flatten
from .basic_modules import GVPConvLayer, MultiheadAttention, Dense, SinusoidalPositionalEmbedding


class TransformerEncoderLayer(nn.Cell):
    """Transformer encoder layer"""

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.embed_dim = args.encoder_embed_dim
        self.self_attn = self.build_self_attention(self.embed_dim, args)
        self.self_attn_layer_norm = nn.LayerNorm([self.embed_dim])
        self.dropout_module = nn.Dropout(1 - args.dropout)
        self.activation_fn = ops.ReLU()
        self.fc1 = self.build_fc1(
            self.embed_dim,
            args.encoder_ffn_embed_dim,
        )
        self.fc2 = self.build_fc2(
            args.encoder_ffn_embed_dim,
            self.embed_dim,
        )

        self.final_layer_norm = nn.LayerNorm([self.embed_dim])

    def build_fc1(self, input_dim, output_dim):
        return Dense(input_dim, output_dim)

    def build_fc2(self, input_dim, output_dim):
        return Dense(input_dim, output_dim)

    def build_self_attention(self, embed_dim, args):
        return MultiheadAttention(
            embed_dim,
            args.encoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True,
        )

    def residual_connection(self, x, residual):
        return residual + x

    def construct(
            self,
            x,
            encoder_padding_mask: Optional[Tensor],
            attn_mask: Optional[Tensor] = None,
    ):
        """Transformer encoder layer construction"""

        if attn_mask is not None:
            attn_mask = ms.ops.MaskedFill()(attn_mask, attn_mask.to(bool()), -1e8 if x.dtype == ms.float32 else -1e4)

        residual = x
        x = self.self_attn_layer_norm(x)
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=encoder_padding_mask,
            need_weights=False,
            attn_mask=attn_mask,
        )
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)

        residual = x
        x = self.final_layer_norm(x)
        x = self.activation_fn(self.fc1(x))
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        return x


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


class GVPTransformerEncoder(nn.Cell):
    """GVP transformer encoder"""

    def __init__(self, args, dictionary, embed_tokens):
        super().__init__()
        self.args = args
        self.dictionary = dictionary

        self.dropout_module = nn.Dropout(1 - args.dropout)

        embed_dim = embed_tokens.embedding_size
        self.padding_idx = embed_tokens.padding_idx

        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(embed_dim)
        self.embed_positions = SinusoidalPositionalEmbedding(
            embed_dim,
            self.padding_idx,
        )
        self.embed_gvp_input_features = Dense(15, embed_dim)
        self.embed_confidence = Dense(16, embed_dim)
        self.embed_dihedrals = DihedralFeatures(embed_dim)

        gvp_args = argparse.Namespace()
        for k, v in vars(args).items():
            if k.startswith("gvp_"):
                setattr(gvp_args, k[4:], v)
        self.gvp_encoder = GVPEncoder(gvp_args)
        gvp_out_dim = gvp_args.node_hidden_dim_scalar + \
                      (3 * gvp_args.node_hidden_dim_vector)
        self.embed_gvp_output = Dense(gvp_out_dim, embed_dim)

        self.layers = nn.CellList([])
        self.layers.extend(
            [self.build_encoder_layer(args) for i in range(args.encoder_layers)]
        )
        self.num_layers = len(self.layers)
        self.layer_norm = nn.LayerNorm([embed_dim])

    def build_encoder_layer(self, args):
        return TransformerEncoderLayer(args)

    def forward_embedding(self, coords, padding_mask, confidence):
        """GVP transformer encoder embedding"""

        components = dict()
        coord_mask = ops.IsFinite()(coords).all(axis=-1).all(axis=-1)
        coords = nan_to_num(coords)
        padding_mask_cast = ops.cast(padding_mask, ms.int32)
        mask_tokens = padding_mask_cast * self.dictionary.padding_idx + \
            (1 - padding_mask_cast) * self.dictionary.get_idx("<mask>")
        components["tokens"] = self.embed_tokens(mask_tokens) * self.embed_scale
        components["diherals"] = self.embed_dihedrals(coords)

        # GVP encoder
        gvp_out_scalars, gvp_out_vectors = \
            self.gvp_encoder(coords, coord_mask, padding_mask, confidence)
        r = get_rotation_frames(coords)
        # Rotate to local rotation frame for rotation-invariance
        gvp_out_features = ops.Concat(-1)([
            gvp_out_scalars,
            ms_flatten(rotate(gvp_out_vectors, ms_transpose(r, r.dim()-2, r.dim()-1)), -2, -1),
        ])
        components["gvp_out"] = self.embed_gvp_output(gvp_out_features)

        components["confidence"] = self.embed_confidence(
            rbf(confidence, 0., 1.))

        # In addition to GVP encoder outputs, also directly embed GVP input node
        # features to the Transformer
        scalar_features, vector_features = GVPInputFeaturizer.get_node_features(
            coords, coord_mask, with_coord_mask=False)
        features = ops.Concat(-1)([
            scalar_features,
            ms_flatten(rotate(vector_features, ms_transpose(r, r.dim()-2, r.dim()-1)), -2, -1),
        ])
        components["gvp_input_features"] = self.embed_gvp_input_features(features)

        embed = sum(components.values())

        x = embed
        x = x + self.embed_positions(mask_tokens)
        x = self.dropout_module(x)
        return x, components

    def construct(
            self,
            coords,
            encoder_padding_mask,
            confidence,
            return_all_hiddens: bool = False,
    ):
        """GVP transformer encoder construction"""

        x, encoder_embedding = \
            self.forward_embedding(coords, encoder_padding_mask, confidence)
        # account for padding while computing the representation
        unsqueeze = ops.ExpandDims()
        x = x * (1 - unsqueeze(encoder_padding_mask, -1).astype(x.dtype))

        # B x T x C -> T x B x C
        x = ms_transpose(x, 0, 1)

        encoder_states = []

        if return_all_hiddens:
            encoder_states.append(x)

        # encoder layers
        for layer in self.layers:
            x = layer(
                x, encoder_padding_mask=encoder_padding_mask
            )
            if return_all_hiddens:
                if encoder_states is None:
                    raise ValueError("'encoder_states' should not be None")
                encoder_states.append(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask],  # B x T
            "encoder_embedding": [encoder_embedding],  # dictionary
            "encoder_states": encoder_states,  # List[T x B x C]
        }
