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
"""gvp transformer model"""

import mindspore as ms
import mindspore.ops as ops
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.common.initializer import Normal, initializer
from .module.transformer_encoder import GVPTransformerEncoder
from .module.transformer_decoder import TransformerDecoder
from .module.util import CoordBatchConverter


class GVPTransformerModel(nn.Cell):
    """GVP transformer model"""

    def __init__(self, args, alphabet):
        super(GVPTransformerModel, self).__init__()
        encoder_embed_tokens = self.build_embedding(
            alphabet, args.encoder_embed_dim,
        )
        decoder_embed_tokens = self.build_embedding(
            alphabet, args.decoder_embed_dim,
        )
        encoder = self.build_encoder(args, alphabet, encoder_embed_tokens)
        decoder = self.build_decoder(args, alphabet, decoder_embed_tokens)
        self.args = args
        self.encoder = encoder
        self.decoder = decoder

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        encoder = GVPTransformerEncoder(args, src_dict, embed_tokens)
        return encoder

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        decoder = TransformerDecoder(
            args,
            tgt_dict,
            embed_tokens,
        )
        return decoder

    @classmethod
    def build_embedding(cls, dictionary, embed_dim):
        num_embeddings = len(dictionary)
        padding_idx = dictionary.padding_idx
        emb = nn.Embedding(num_embeddings, embed_dim, padding_idx=padding_idx)
        emb.embedding_table = initializer(Normal(mean=0, sigma=embed_dim ** -0.5), emb.embedding_table.shape,
                                          dtype=ms.float32)
        emb.embedding_table[padding_idx] = 0
        return emb

    def construct(self, net_input):
        """Transformer construction"""

        coords, padding_mask, confidence, prev_output_tokens = net_input
        return_all_hiddens: bool = False
        features_only: bool = False
        encoder_out = self.encoder(coords, padding_mask, confidence, return_all_hiddens=return_all_hiddens)
        logits, _ = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            return_all_hiddens=return_all_hiddens,
        )
        return logits

    def sample(self, coords, temperature=1.0, confidence=None):
        """Sample sequence designs for a given structure"""

        l_coords = len(coords)
        # Convert to batch format
        batch_converter = CoordBatchConverter(self.decoder.dictionary)
        batch_coords, confidence, _, _, padding_mask = (
            batch_converter([(coords, confidence, None)])
        )
        batch_coords = Tensor(batch_coords)
        confidence = Tensor(confidence)
        padding_mask = Tensor(padding_mask)

        # Start with prepend token
        sampled_tokens = ops.Zeros()((1, 1 + l_coords), ms.float32)
        sampled_tokens[0, 0] = self.decoder.dictionary.get_idx('<cath>')

        # Save incremental states for faster sampling
        incremental_state = dict()

        # Run encoder only once
        encoder_out = self.encoder(batch_coords, padding_mask, confidence)

        # Decode one token at a time
        for i in range(1, l_coords + 1):
            logits, _ = self.decoder(sampled_tokens[:, :i], encoder_out, incremental_state=incremental_state)
            logits = logits[0].reshape(1, -1)
            logits /= temperature
            softmax = ops.Softmax(axis=-1)
            probs = softmax(logits)
            probs = probs.reshape(1, -1)
            tokens = ops.Argmax()(probs)
            sampled_tokens[:, i] = tokens
        sampled_seq = sampled_tokens[0, 1:]
        sampled_seq = ops.Cast()(sampled_seq, ms.int32)

        # Convert back to string via lookup
        output = ''.join([self.decoder.dictionary.get_tok(a) for a in sampled_seq])
        return output
