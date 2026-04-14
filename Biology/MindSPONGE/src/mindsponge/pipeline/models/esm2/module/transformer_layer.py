# Copyright 2023 Huawei Technologies Co., Ltd
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
"""Transformer Layer"""
import mindspore as ms
import mindspore.ops as ops
import mindspore.nn as nn
from mindspore.nn import LayerNorm as ESM1bLayerNorm
# pylint: disable=relative-beyond-top-level
from .basic_modules import MultiheadAttention


def gelu(x):
    """Implementation of the gelu activation function.
    For information: OpenAI GPT's gelu is slightly different
    (and gives slightly different results):
    0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    tmp = x / ops.sqrt(ms.Tensor(2, ms.float32))
    x = ops.mul(x, ops.mul(0.5, ops.add(1.0, ops.erf(tmp))))
    return x


class TransformerLayer(nn.Cell):
    """Transformer layer block."""
    def __init__(self, embed_dim, ffn_embed_dim, attention_heads, add_bias_kv=True,
                 use_rotary_embeddings: bool = False):
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_embed_dim = ffn_embed_dim
        self.attention_heads = attention_heads
        self.use_rotary_embeddings = use_rotary_embeddings
        self._init_submodules(add_bias_kv)

    def construct(self, x, self_attn_padding_mask=None, need_head_weights=False):
        """Transformer layer block."""
        residual = x
        x = self.self_attn_layer_norm(x)
        x, attn = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=self_attn_padding_mask,
            need_weights=True,
            need_head_weights=need_head_weights,
        )
        x = residual + x
        residual = x
        x = self.final_layer_norm(x)
        x = gelu(self.fc1(x))
        x = self.fc2(x)
        x = residual + x
        return x, attn

    def _init_submodules(self, add_bias_kv):
        """init submodules"""
        # pylint: disable=C0103
        BertLayerNorm = ESM1bLayerNorm

        self.self_attn = MultiheadAttention(
            self.embed_dim,
            self.attention_heads,
            add_bias_kv=add_bias_kv,
            add_zero_attn=False,
            use_rotary_embeddings=self.use_rotary_embeddings,
        )
        self.self_attn_layer_norm = BertLayerNorm((self.embed_dim,), begin_norm_axis=-1, begin_params_axis=-1)

        self.fc1 = nn.Dense(self.embed_dim, self.ffn_embed_dim)
        self.fc2 = nn.Dense(self.ffn_embed_dim, self.embed_dim)

        self.final_layer_norm = BertLayerNorm((self.embed_dim,), begin_norm_axis=-1, begin_params_axis=-1)
