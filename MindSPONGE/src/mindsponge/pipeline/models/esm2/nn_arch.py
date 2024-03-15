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
"""esm2 model"""
from typing import Union
import mindspore as ms
from mindspore import ops, nn
from mindspore.nn import LayerNorm
import mindspore.numpy as mnp
from mindspore import context
# pylint: disable=relative-beyond-top-level
from ..esm_if1.module.util import Alphabet
from .module.transformer_layer import TransformerLayer
from .module.contact_prediction_head import ContactPredictionHead
from .module.roberta_lm_head import RobertaLMHead


class ESM2(nn.Cell):
    """ESM2 Model structure"""
    def __init__(
            self,
            num_layers: int = 33,
            embed_dim: int = 1280,
            attention_heads: int = 20,
            alphabet: Union[Alphabet, str] = "ESM-1b",
            token_dropout: bool = True,
            return_contacts=False,
            need_head_weights=False
    ):
        super().__init__()
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.attention_heads = attention_heads
        if not isinstance(alphabet, Alphabet):
            alphabet = Alphabet.from_architecture(alphabet)
        self.alphabet = alphabet
        self.alphabet_size = len(alphabet)
        self.padding_idx = alphabet.padding_idx
        self.mask_idx = alphabet.mask_idx
        self.cls_idx = alphabet.cls_idx
        self.eos_idx = alphabet.eos_idx
        self.prepend_bos = alphabet.prepend_bos
        self.append_eos = alphabet.append_eos
        self.token_dropout = token_dropout
        self.return_contacts = return_contacts
        self.need_head_weights = need_head_weights
        self._init_submodules()
        if context.get_context("device_target") == "GPU":
            self.fill_value = ms.Tensor(0, dtype=ms.float32)
        else:
            self.fill_value = ms.Tensor(0, dtype=ms.float16)

    def construct(self, tokens):
        """ESM2 Model structure"""
        need_head_weights = self.need_head_weights
        if self.return_contacts:
            need_head_weights = True
        padding_mask = ops.equal(tokens, self.padding_idx)
        x = self.embed_scale * self.embed_tokens(tokens)
        if self.token_dropout:
            mask = tokens == self.mask_idx
            mask = ops.unsqueeze(mask, dim=-1)
            x = ops.masked_fill(x, mask, self.fill_value)
            # x: B x T x C
            mask_ratio_train = 0.15 * 0.8
            src_lengths = (~padding_mask).sum(-1)
            mask_ratio_observed = (tokens == self.mask_idx).sum(-1).to(x.dtype) / src_lengths
            x = x * (1 - mask_ratio_train) / (1 - mask_ratio_observed)[:, None, None]
        x = x * (1 - ops.unsqueeze(padding_mask, dim=-1))
        attn_weights = []
        # (B, T, E) => (T, B, E)
        x = ops.transpose(x, (1, 0, 2))
        layer_count = 0
        for _, layer in enumerate(self.layers):
            x, attn = layer(
                x,
                self_attn_padding_mask=padding_mask,
                need_head_weights=need_head_weights,
            )
            if need_head_weights:
                attn_weights.append(attn.transpose(1, 0, 2, 3))
            layer_count += 1

        x = self.emb_layer_norm_after(x)
        x = ops.transpose(x, (1, 0, 2))  # (T, B, E) => (B, T, E)

        # last hidden representation should have layer norm applied
        hidden_representations = x
        x = self.lm_head(x)
        if need_head_weights:
            attentions = mnp.stack(attn_weights, 1)
            attention_mask = 1 - padding_mask
            attention_mask = attention_mask.unsqueeze(1) * attention_mask.unsqueeze(2)
            attentions = attentions * attention_mask[:, None, None, :, :]
            contacts = self.contact_head(tokens, attentions)
            result = x, hidden_representations, attentions, contacts
        else:
            result = x, hidden_representations
        return result

    def _init_submodules(self):
        """init submodules"""
        self.embed_scale = 1
        self.embed_tokens = nn.Embedding(
            self.alphabet_size,
            self.embed_dim,
            padding_idx=self.padding_idx,
        )

        self.layers = nn.CellList(
            [
                TransformerLayer(
                    self.embed_dim,
                    4 * self.embed_dim,
                    self.attention_heads,
                    add_bias_kv=False,
                    use_rotary_embeddings=True,
                )
                for _ in range(self.num_layers)
            ]
        )

        self.contact_head = ContactPredictionHead(
            self.num_layers * self.attention_heads,
            self.prepend_bos,
            self.append_eos,
            eos_idx=self.eos_idx,
        )
        self.emb_layer_norm_after = LayerNorm((self.embed_dim,), begin_norm_axis=-1, begin_params_axis=-1)

        self.lm_head = RobertaLMHead(
            embed_dim=self.embed_dim,
            output_dim=self.alphabet_size,
            weight=self.embed_tokens.embedding_table,
        )
        