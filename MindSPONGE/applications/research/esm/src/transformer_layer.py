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
"""Transformer layer"""

from typing import Dict, List, Optional
import mindspore as ms
import mindspore.ops as ops
import mindspore.nn as nn
from mindspore import Tensor
from src.multihead_attention import MultiheadAttention
from src.util import Dense


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


def _set_input_buffer(selfattention: MultiheadAttention,
                      incremental_state: Dict[str, Dict[str, Optional[ms.Tensor]]],
                      buffer: Dict[str, Optional[ms.Tensor]],
                      ):
    """Set input buffer"""
    return selfattention.set_incremental_state(incremental_state, "attn_state", buffer)


def _get_input_buffer(
        selfattention: MultiheadAttention, incremental_state: Optional[Dict[str, Dict[str, Optional[ms.Tensor]]]]
) -> Dict[str, Optional[ms.Tensor]]:
    """Get input buffer"""
    result = selfattention.get_incremental_state(incremental_state, "attn_state")
    if result is not None:
        return result
    empty_result: Dict[str, Optional[ms.Tensor]] = {}
    return empty_result


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
            assert incremental_state is not None
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
                assert incremental_state is not None
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
