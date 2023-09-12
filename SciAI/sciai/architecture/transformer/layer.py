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
# ==============================================================================
"""transformer layers"""
import mindspore as ms
from mindspore import Parameter, Tensor, ops, nn
from mindspore.common.initializer import initializer

from sciai.utils.math_utils import get_2d_sin_cos_pos_embed

__all__ = ['Decoder', 'Encoder']


class Attention(nn.Cell):
    """Attention modules"""

    def __init__(self,
                 embed_dim,
                 num_heads,
                 mlp_ratio,
                 dropout_rate=1.0,
                 dtype=ms.float16):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.embed_dim_per_head = embed_dim // num_heads
        self.embed_dim_per_head_fp32 = Tensor(self.embed_dim_per_head, ms.float32)
        self.mlp_ratio = mlp_ratio
        self.dtype = dtype

        self.layer_norm = nn.LayerNorm([embed_dim], epsilon=1e-6).to_float(ms.float32)

        self.query = nn.Dense(self.embed_dim, self.embed_dim, weight_init='XavierUniform').to_float(dtype)
        self.key = nn.Dense(self.embed_dim, self.embed_dim, weight_init='XavierUniform').to_float(dtype)
        self.value = nn.Dense(self.embed_dim, self.embed_dim, weight_init='XavierUniform').to_float(dtype)

        self.proj = nn.Dense(self.embed_dim, self.embed_dim, weight_init='XavierUniform').to_float(dtype)
        self.attention_dropout_rate = nn.Dropout(dropout_rate)
        self.proj_dropout = nn.Dropout(dropout_rate)

        self.softmax = nn.Softmax(axis=-1)
        self.shape = ops.Shape()
        self.reshape = ops.Reshape()
        self.transpose = ops.Transpose()
        self.sqrt = ops.Sqrt()
        self.cast = ops.Cast()

    def transpose_for_scores(self, x):
        """transpose_for_scores"""
        new_x_shape = self.shape(x)[:-1] + (self.num_heads, self.embed_dim_per_head)
        x = self.reshape(x, new_x_shape)
        return self.transpose(x, (0, 2, 1, 3))

    def construct(self, x):
        """construct"""
        mixed_query_layer = self.query(x)
        mixed_key_layer = self.key(x)
        mixed_value_layer = self.value(x)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores_dtype = ops.matmul(query_layer, self.transpose(key_layer, (0, 1, 3, 2)))
        attention_scores_fp32 = self.cast(attention_scores_dtype, ms.float32)

        scaled_attention_scores_fp32 = attention_scores_fp32 / self.sqrt(self.embed_dim_per_head_fp32)
        attention_probs_fp32 = self.softmax(scaled_attention_scores_fp32)
        attention_probs_dtype = self.cast(attention_probs_fp32, self.dtype)
        attention_probs = self.attention_dropout_rate(attention_probs_dtype)

        context_layer = ops.matmul(attention_probs, value_layer)
        context_layer = self.transpose(context_layer, (0, 2, 1, 3))
        new_context_layer_shape = self.shape(context_layer)[:-2] + (self.embed_dim,)
        context_layer = self.reshape(context_layer, new_context_layer_shape)
        attention_output = self.proj(context_layer)
        attention_output = self.proj_dropout(attention_output)

        return attention_output


class Mlp(nn.Cell):
    """Mlp"""

    def __init__(self,
                 embed_dim,
                 mlp_ratio,
                 dropout_rate=1.0,
                 dtype=ms.float16):
        super(Mlp, self).__init__()
        self.fc1 = nn.Dense(embed_dim, embed_dim * mlp_ratio, weight_init='XavierUniform').to_float(dtype)
        self.fc2 = nn.Dense(embed_dim * mlp_ratio, embed_dim, weight_init='XavierUniform').to_float(dtype)
        self.act_fn = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)

    def construct(self, x):
        """construct"""
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Block(nn.Cell):
    """Block"""

    def __init__(self,
                 embed_dim,
                 num_heads,
                 mlp_ratio,
                 dropout_rate=1.0,
                 dtype=ms.float16):
        super(Block, self).__init__()
        self.embed_dim = embed_dim
        self.attention_norm = nn.LayerNorm([embed_dim], epsilon=1e-6).to_float(ms.float32)
        self.ffn_norm = nn.LayerNorm([embed_dim], epsilon=1e-6).to_float(ms.float32)
        self.ffn = Mlp(embed_dim, mlp_ratio, dropout_rate, dtype=dtype)
        self.attn = Attention(embed_dim, num_heads, mlp_ratio, dropout_rate, dtype=dtype)

    def construct(self, x):
        """construct"""
        h = x
        x = self.attention_norm(x)
        x = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x


class Embedding(nn.Cell):
    """Construct patch embeddings with positional embeddings"""

    def __init__(self,
                 input_dims,
                 embed_dim,
                 patch_size=16,
                 dtype=ms.float16):
        super(Embedding, self).__init__()
        self.dtype = dtype
        self.patch_embedding = nn.Conv2d(in_channels=input_dims,
                                         out_channels=embed_dim,
                                         kernel_size=patch_size,
                                         stride=patch_size,
                                         has_bias=True).to_float(dtype)
        self.init_weights()
        self.reshape = ops.Reshape()
        self.transpose = ops.Transpose()

    def init_weights(self):
        weight_shape = self.patch_embedding.weight.shape
        xavier_init = initializer("XavierUniform",
                                  [weight_shape[0], weight_shape[1] * weight_shape[2] * weight_shape[3]],
                                  ms.float32)

        self.patch_embedding.weight = self.reshape(xavier_init,
                                                   (weight_shape[0], weight_shape[1], weight_shape[2], weight_shape[3]))

    def construct(self, x):
        x = self.patch_embedding(x)
        x = self.reshape(x, (x.shape[0], x.shape[1], x.shape[2] * x.shape[3]))
        x = self.transpose(x, (0, 2, 1))
        return x


class Encoder(nn.Cell):
    r"""
     Encoder module with multi-layer stacked of `Block`, including multihead self attention and feedforward layer.

     Args:
          grid_size (tuple[int]): The grid_size size of input.
          in_channels (int): The input feature size of input. Default: ``3``.
          patch_size (int): The patch size of image. Default: ``16``.
          depths (int): The encoder depth of encoder layer.
          embed_dim (int): The encoder embedding dimension of encoder layer. Default: ``768``.
          num_heads (int): The encoder heads' number of encoder layer. Default: ``16``.
          mlp_ratio (int): The rate of mlp layer. Default: ``4``.
          dropout_rate (float): The rate of dropout layer. Default: ``1.0``.
          dtype (dtype): The data type for encoder, encoding_embedding, encoder and dense layer.
                                Default: ``ms.float16``.

     Inputs:
             - **input** (Tensor) - Tensor of shape :math:`(batch\_size, feature\_size, image\_height, image\_width)`.

     Outputs:
             - **output** (Tensor) - Tensor of shape :math:`(batch\_size, patchify\_size, embed\_dim)`.
             where patchify_size = (image_height * image_width) / (patch_size * patch_size).

     Supported Platforms:
         ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore as ms
        >>> from mindspore import Tensor, context, nn
        >>> from sciai.architecture.transformer.layer import Encoder
        >>> input_tensor = Tensor(np.ones((32, 3, 192, 384)), ms.float32)
        >>> print(input_tensor.shape)
        (32, 3, 192, 384)
        >>>encoder = Encoder(grid_size=(192 // 16, 384 // 16),
        >>>                  in_channels=3,
        >>>                  patch_size=16,
        >>>                  depths=6,
        >>>                  embed_dim=768,
        >>>                  num_heads=12,
        >>>                  mlp_ratio=4,
        >>>                  dropout_rate=1.0,
        >>>                  dtype=ms.float16)
        >>>output_tensor = encoder(input_tensor)
        >>> print(output_tensor.shape)
        (32, 288, 768)
    """

    def __init__(self,
                 grid_size,
                 in_channels,
                 patch_size,
                 depths,
                 embed_dim,
                 num_heads,
                 mlp_ratio=4,
                 dropout_rate=1.0,
                 dtype=ms.float16):
        super(Encoder, self).__init__()
        self.patch_embedding = Embedding(in_channels, embed_dim, patch_size, dtype=dtype)
        pos_embed = get_2d_sin_cos_pos_embed(embed_dim, grid_size)
        self.position_embedding = Parameter(Tensor(pos_embed, ms.float32),
                                            name="encoder_pos_embed",
                                            requires_grad=False)
        self.layer = nn.SequentialCell()
        for _ in range(depths):
            self.layer.append(Block(embed_dim, num_heads, mlp_ratio, dropout_rate, dtype=dtype))
        self.encoder_norm = nn.LayerNorm([embed_dim], epsilon=1e-6).to_float(ms.float32)

    def construct(self, x):
        """construct"""
        x = self.patch_embedding(x)
        x = x + self.position_embedding
        x = self.layer(x)
        x = self.encoder_norm(x)
        return x


class Decoder(nn.Cell):
    r"""
    Decoder module with multi-layer stacked of `Block`, including multihead self attention and feedforward layer.

    Args:
        grid_size (tuple[int]): The grid_size size of input.
        depths (int): The decoder depth of decoder layer.
        embed_dim (int): The decoder embedding dimension of decoder layer.
        num_heads (int): The decoder heads' number of decoder layer.
        mlp_ratio (int): The rate of mlp layer. Default: ``4``.
        dropout_rate (float): The rate of dropout layer. Default: ``1.0``.
        dtype (dtype): The data type for encoder, decoding_embedding, decoder and dense layer.
            Default: ``ms.float16``.

    Inputs:
        - **input** (Tensor) - Tensor of shape :math:`(batch\_size, patchify\_size, embed\_dim)`.

    Outputs:
        - **output** (Tensor) - Tensor of shape :math:`(batch\_size, patchify\_size, embed\_dim)`.
          where patchify_size = (image_height * image_width) / (patch_size * patch_size).

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore as ms
        >>> from mindspore import Tensor, context, nn
        >>> from sciai.architecture.transformer.layer import Decoder
        >>> input_tensor = Tensor(np.ones((32, 288, 512)), ms.float32)
        >>> print(input_tensor.shape)
        (32, 288, 768)
        >>> decoder = Decoder(grid_size=grid_size,
        >>>                   depths=6,
        >>>                   embed_dim=512,
        >>>                   num_heads=16,
        >>>                   mlp_ratio=4,
        >>>                   dropout_rate=1.0,
        >>>                   dtype=ms.float16)
        >>> output_tensor = decoder(input_tensor)
        >>> print(output_tensor.shape)
        (32, 288, 512)
    """

    def __init__(self,
                 grid_size,
                 depths,
                 embed_dim,
                 num_heads,
                 mlp_ratio=4,
                 dropout_rate=1.0,
                 dtype=ms.float16):
        super(Decoder, self).__init__()
        self.grid_size = grid_size
        self.layer = nn.SequentialCell()
        pos_embed = get_2d_sin_cos_pos_embed(embed_dim, grid_size)
        self.position_embedding = Parameter(Tensor(pos_embed, ms.float32),
                                            name="decoder_pos_embed",
                                            requires_grad=False)
        for _ in range(depths):
            self.layer.append(Block(embed_dim, num_heads, mlp_ratio, dropout_rate, dtype=dtype))
        self.decoder_norm = nn.LayerNorm([embed_dim], epsilon=1e-6).to_float(ms.float32)

    def construct(self, x):
        """construct"""
        x = x + self.position_embedding
        x = self.layer(x)
        x = self.decoder_norm(x)
        return x
