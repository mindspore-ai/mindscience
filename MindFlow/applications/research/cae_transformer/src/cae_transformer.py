# ============================================================================
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
"""cae transformer model"""
import mindspore as ms
from mindspore import ops, nn

from .model import Encoder, EncoderLayer, ConvLayer, Decoder, DecoderLayer
from .attn import FullAttention, ProbAttention, AttentionLayer
from .embedding import DataEmbedding
from .cae import CaeDecoder, CaeEncoder


class CaeInformer(nn.Cell):
    """Cae-Transformer flow field prediction model"""
    def __init__(self, enc_in, dec_in, c_out, pred_len, label_len,
                 data_dimension, conv_kernel, maxpool_kernel, maxpool_stride,
                 channels_encoder, channels_decoder, channels_dense,
                 d_model=128, n_heads=2, e_layers=1, d_layers=1, d_ff=256, attn='full'):
        """
        Args:
        enc_in: int, input dimension of encoder
        dec_in: int, input dimension of decoder
        c_out: int, output dimension of decoder
        pred_len: int, length of prediction
        label_len: int, length of label sequence of decoder
        data_dimension: int, dimension of input data
        conv_kernel: list, kernel size of convolution layer
        maxpool_kernel: list, kernel size of maxpooling layer
        maxpool_stride: list, stride of maxpooling layer
        """
        super(CaeInformer, self).__init__()
        self.pred_len = pred_len
        self.label_len = label_len
        self.attn = attn
        self.reshape = ops.Reshape()

        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, 'fixed', 'h', 0.1)
        self.dec_embedding = DataEmbedding(dec_in, d_model, 'fixed', 'h', 0.1)
        # Attention
        attn = ProbAttention if attn == 'prob' else FullAttention
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(attn(False, 5, attention_dropout=0.1,
                                        output_attention=False, args=None),
                                   d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=0.1,
                    activation='gelu'
                ) for _ in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for _ in range(e_layers-1)
            ],
            norm_layer=nn.LayerNorm([d_model])
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(attn(True, 5, attention_dropout=0.1, output_attention=False,
                                        args=None, d_value=d_model/n_heads),
                                   d_model, n_heads, mix=False),
                    AttentionLayer(FullAttention(False, 5, attention_dropout=0.1, output_attention=False),
                                   d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=0.1,
                    activation='gelu',
                )
                for l in range(d_layers)
            ],
            norm_layer=nn.LayerNorm([d_model])
        )
        self.projection = nn.Dense(d_model, c_out, has_bias=True)
        # CAE
        self.cae_encoder = CaeEncoder(conv_kernel, maxpool_kernel, maxpool_stride, channels_encoder, channels_dense)
        self.cae_decoder = CaeDecoder(data_dimension, conv_kernel, channels_decoder, channels_dense)

    def construct(self, x,
                  enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        """predict flow field based on input data"""
        cast = ops.Cast()
        batch_size = ops.shape(x)[0]
        seq_len = ops.shape(x)[1]
        x0 = x[:, -1]
        x = self.reshape(x, (-1, 1, 256, 256))

        # CAE Encoder
        x_enc = self.cae_encoder(x)
        x_enc = self.reshape(x_enc, (batch_size, seq_len, -1))
        latent_size = ops.shape(x_enc)[-1]
        # print(x_enc)

        # Transformer Encoder
        enc_inp = self.enc_embedding(x_enc)
        enc_out = self.encoder(enc_inp, attn_mask=enc_self_mask)

        # Transformer Decoder
        dec_inp = ops.Zeros()(
            (batch_size, self.pred_len, x_enc.shape[-1]), ms.float32
        )
        dec_inp = cast(
            ops.concat([x_enc[:, - self.label_len :, :], dec_inp], axis=1),
            ms.float32,
        )
        dec_inp = self.dec_embedding(dec_inp)
        dec_out = self.decoder(dec_inp, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)[:, -self.pred_len:, :]

        # CAE Decoder
        dec_out = self.reshape(dec_out, (-1, 1, latent_size))
        cae_out = self.cae_decoder(dec_out)
        cae_out = self.reshape(cae_out, (batch_size, self.pred_len, 256, 256)) + x0

        return cae_out
    