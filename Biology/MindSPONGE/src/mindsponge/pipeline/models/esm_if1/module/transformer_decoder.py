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
"""Constructing decoder in transformer network"""

import math
from typing import Dict, List, Optional
import numpy as np
import mindspore as ms
import mindspore.ops as ops
import mindspore.nn as nn
from mindspore.common.initializer import Normal, initializer
from mindspore import Tensor
# pylint: disable=relative-beyond-top-level
from .basic_modules import Dense, SinusoidalPositionalEmbedding, \
MultiheadAttention, _set_input_buffer, _get_input_buffer
from .util import ms_transpose


def fill_with_neg_inf(t):
    """FP16-compatible function that fills a tensor with -inf."""
    return ops.Fill()(ms.float32, t.shape, float("-inf"))


class TransformerDecoder(nn.Cell):
    """Transformer decoder"""

    def __init__(
            self,
            args,
            dictionary,
            embed_tokens,
    ):
        super().__init__()
        self.args = args
        self.dictionary = dictionary
        self._future_mask = np.empty((0))

        self.dropout_module = nn.Dropout(1 - args.dropout)

        input_embed_dim = embed_tokens.embedding_size
        embed_dim = args.decoder_embed_dim
        self.embed_dim = embed_dim
        self.padding_idx = embed_tokens.padding_idx
        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(embed_dim)

        self.project_in_dim = (
            Dense(input_embed_dim, embed_dim, has_bias=False)
            if embed_dim != input_embed_dim
            else None
        )
        self.embed_positions = SinusoidalPositionalEmbedding(
            embed_dim,
            self.padding_idx,
        )

        self.layers = nn.CellList([])
        self.layers.extend(
            [
                self.build_decoder_layer(args)
                for _ in range(args.decoder_layers)
            ]
        )
        self.num_layers = len(self.layers)
        self.layer_norm = nn.LayerNorm([embed_dim])

        self.build_output_projection(args, dictionary)

    def build_output_projection(self, args, dictionary):
        self.output_projection = Dense(
            args.decoder_embed_dim, len(dictionary), has_bias=False
        )
        self.output_projection.weight = initializer(Normal(sigma=args.decoder_embed_dim ** -0.5, mean=0),
                                                    shape=self.output_projection.weight.shape,
                                                    dtype=self.output_projection.weight.dtype)

    def build_decoder_layer(self, args):
        return TransformerDecoderLayer(args)

    def construct(
            self,
            prev_output_tokens,
            encoder_out: Optional[Dict[str, List[ms.Tensor]]] = None,
            incremental_state: Optional[Dict[str, Dict[str, Optional[ms.Tensor]]]] = None,
            features_only: bool = False,
            return_all_hiddens: bool = False,
    ):
        """Transformer decoder construction"""

        x, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
        )
        _ = return_all_hiddens
        if not features_only:
            x = self.output_layer(x)
        x = ms_transpose(x, 1, 2) # B x T x C -> B x C x T
        return x, extra

    def extract_features(
            self,
            prev_output_tokens,
            encoder_out: Optional[Dict[str, List[ms.Tensor]]],
            incremental_state: Optional[Dict[str, Dict[str, Optional[ms.Tensor]]]] = None,
    ):
        """Extract features"""

        bs, _ = prev_output_tokens.shape

        enc: Optional[ms.float32] = None
        padding_mask: Optional[ms.float32] = None
        if encoder_out is not None and encoder_out["encoder_out"]:
            enc = encoder_out["encoder_out"][0]
            if enc.shape[1] != bs:
                raise ValueError(f"Expected enc.shape == (t, {bs}, c) got {enc.shape}")
        if encoder_out is not None and encoder_out["encoder_padding_mask"]:
            padding_mask = encoder_out["encoder_padding_mask"][0]

        # embed positions
        positions = self.embed_positions(
            prev_output_tokens
        )

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            positions = positions[:, -1:]

        # embed tokens and positions
        prev_output_tokens = ops.Cast()(prev_output_tokens, ms.int32)
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        x += positions

        x = self.dropout_module(x)

        # B x T x C -> T x B x C
        x = ms_transpose(x, 0, 1)

        self_attn_padding_mask: Optional[ms.Tensor] = None
        if ops.Equal()(prev_output_tokens, self.padding_idx).any():
            self_attn_padding_mask = ops.Equal()(prev_output_tokens, self.padding_idx)

        # decoder layers
        inner_states: List[Optional[ms.Tensor]] = [x]
        for _, layer in enumerate(self.layers):
            if incremental_state is None:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None

            x, _, _ = layer(
                x,
                enc,
                padding_mask,
                incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                need_attn=False,
                need_head_weights=False,
            )
            inner_states.append(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # T x B x C -> B x C x T
        x = ms_transpose(x, 0, 1)

        return x, {"inner_states": inner_states}

    def output_layer(self, features):
        """Project features to the vocabulary size."""
        return self.output_projection(features)

    def buffered_future_mask(self, tensor):
        """Buffered future mask"""

        dim = tensor.shape[0]
        if (
                self._future_mask.shape[0] == 0
                or self._future_mask.shape[0] < dim
        ):
            self._future_mask = fill_with_neg_inf(ops.Zeros()((dim, dim), ms.float32))
            mask = ms.nn.Triu()(ms.ops.ones(self._future_mask.shape, ms.float32), 1)
            self._future_mask[ms.numpy.logical_not(mask)] = 0

        self._future_mask = self._future_mask.astype(tensor.dtype)
        return self._future_mask[:dim, :dim]


class TransformerDecoderLayer(nn.Cell):
    """Transformer decoder layer"""

    def __init__(
            self, args, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False
    ):
        super().__init__()
        self.embed_dim = args.decoder_embed_dim
        self.dropout_module = nn.Dropout(1 - args.dropout)

        self.self_attn = self.build_self_attention(
            self.embed_dim,
            args,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
        )
        self.nh = self.self_attn.num_heads
        self.head_dim = self.self_attn.head_dim

        self.activation_fn = ops.ReLU()

        self.self_attn_layer_norm = nn.LayerNorm([self.embed_dim])

        if no_encoder_attn:
            self.encoder_attn = None
            self.encoder_attn_layer_norm = None
        else:
            self.encoder_attn = self.build_encoder_attention(self.embed_dim, args)
            self.encoder_attn_layer_norm = nn.LayerNorm([self.embed_dim])

        self.ffn_layernorm = (
            nn.LayerNorm([args.decoder_ffn_embed_dim])
            if getattr(args, "scale_fc", False)
            else None
        )
        self.w_resid = (
            ms.Parameter(
                ops.Ones()(
                    self.embed_dim,
                ),
                requires_grad=True,
            )
            if getattr(args, "scale_resids", False)
            else None
        )

        self.fc1 = self.build_fc1(
            self.embed_dim,
            args.decoder_ffn_embed_dim,
        )
        self.fc2 = self.build_fc2(
            args.decoder_ffn_embed_dim,
            self.embed_dim,
        )

        self.final_layer_norm = nn.LayerNorm([self.embed_dim])
        self.need_attn = True

    def build_fc1(self, input_dim, output_dim):
        return Dense(input_dim, output_dim)

    def build_fc2(self, input_dim, output_dim):
        return Dense(input_dim, output_dim)

    def build_self_attention(
            self, embed_dim, args, add_bias_kv=False, add_zero_attn=False
    ):
        return MultiheadAttention(
            embed_dim,
            args.decoder_attention_heads,
            dropout=args.attention_dropout,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            self_attention=True,
        )

    def build_encoder_attention(self, embed_dim, args):
        return MultiheadAttention(
            embed_dim,
            args.decoder_attention_heads,
            kdim=args.encoder_embed_dim,
            vdim=args.encoder_embed_dim,
            dropout=args.attention_dropout,
            encoder_decoder_attention=True,
        )

    def residual_connection(self, x, residual):
        return residual + x

    def construct(
            self,
            x,
            encoder_out: Optional[ms.Tensor] = None,
            encoder_padding_mask: Optional[ms.Tensor] = None,
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            prev_self_attn_state: Optional[List[ms.Tensor]] = None,
            prev_attn_state: Optional[List[ms.Tensor]] = None,
            self_attn_mask: Optional[ms.Tensor] = None,
            self_attn_padding_mask: Optional[ms.Tensor] = None,
            need_attn: bool = False,
            need_head_weights: bool = False,
    ):
        """Transformer decoder layer construction"""

        if need_head_weights:
            need_attn = True

        residual = x
        x = self.self_attn_layer_norm(x)
        if prev_self_attn_state is not None:
            prev_key, prev_value = prev_self_attn_state[:2]
            saved_state: Dict[str, Optional[Tensor]] = {
                "prev_key": prev_key,
                "prev_value": prev_value,
            }
            if len(prev_self_attn_state) >= 3:
                saved_state["prev_key_padding_mask"] = prev_self_attn_state[2]
            if incremental_state is None:
                raise ValueError("'incremental_state' should not be None")
            _set_input_buffer(self.self_attn, incremental_state, saved_state)
        _ = _get_input_buffer(self.self_attn, incremental_state)
        y = x

        x, attn = self.self_attn(
            query=x,
            key=y,
            value=y,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)

        if self.encoder_attn is not None and encoder_out is not None:
            residual = x
            x = self.encoder_attn_layer_norm(x)
            if prev_attn_state is not None:
                prev_key, prev_value = prev_attn_state[:2]
                saved_state: Dict[str, Optional[Tensor]] = {
                    "prev_key": prev_key,
                    "prev_value": prev_value,
                }
                if len(prev_attn_state) >= 3:
                    saved_state["prev_key_padding_mask"] = prev_attn_state[2]
                if incremental_state is None:
                    raise ValueError("'incremental_state' should not be None")
                _set_input_buffer(self.encoder_attn, incremental_state, saved_state)

            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=need_attn or (not self.training and self.need_attn),
                need_head_weights=need_head_weights,
            )
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)

        residual = x
        x = self.final_layer_norm(x)

        x = self.activation_fn(self.fc1(x))
        if self.ffn_layernorm is not None:
            x = self.ffn_layernorm(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        if self.w_resid is not None:
            residual = ops.Mul()(self.w_resid, residual)
        x = self.residual_connection(x, residual)
        return x, attn, None
