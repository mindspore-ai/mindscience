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
import mindspore as ms
import mindspore.ops as ops
import mindspore.nn as nn

from src.modules import SinusoidalPositionalEmbedding
from src.features import GVPInputFeaturizer, DihedralFeatures
from src.gvp_encoder import GVPEncoder
from src.transformer_layer import TransformerEncoderLayer
from src.util import nan_to_num, get_rotation_frames, rotate, rbf
from src.util import Dense


def ms_transpose(x, index_a, index_b):
    """Transpose"""
    index = list(i for i in range(len(x.shape)))
    index[index_a] = index_b
    index[index_b] = index_a
    input_trans = x.transpose(index)
    return input_trans


def ms_flatten(input_tensor, start_dim, end_dim):
    """Flatten"""
    if start_dim == 0:
        shape_list = list(input_tensor.shape[end_dim+1:])
        dim = 1
        for i in range(start_dim, end_dim+1):
            dim = input_tensor.shape[i] * dim
        shape_list.insert(0, dim)
        shape_list = tuple(shape_list)
        flatten = ms.ops.Reshape()
        output = flatten(input_tensor, shape_list)
        return output
    if end_dim in (-1, input_tensor.dim() - 1):
        shape_list = list(input_tensor.shape[:start_dim])
        dim = 1
        for i in range(start_dim, end_dim + 1):
            dim = input_tensor.shape[i] * dim
        shape_list.append(dim)
        shape_list = tuple(shape_list)
        flatten = ms.ops.Reshape()
        output = flatten(input_tensor, shape_list)
        return output
    raise ValueError


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
        mask_tokens = (
            padding_mask * self.dictionary.padding_idx +
            ~padding_mask * self.dictionary.get_idx("<mask>")
        )
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
                assert encoder_states is not None
                encoder_states.append(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask],  # B x T
            "encoder_embedding": [encoder_embedding],  # dictionary
            "encoder_states": encoder_states,  # List[T x B x C]
        }
