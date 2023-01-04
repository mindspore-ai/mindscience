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

import copy

import mindspore
import mindspore.nn as nn
import mindspore.ops.operations as P
from mindspore import dtype as mstype
from mindspore import Parameter, Tensor
from mindspore.common.initializer import initializer, XavierUniform

from ..utils import get_2d_sin_cos_pos_embed

__all__ = ['Decoder', 'Encoder']


class Attention(nn.Cell):
    """Attention modules"""

    def __init__(self,
                 embed_dim,
                 num_heads,
                 mlp_ratio,
                 dropout_rate=1.0,
                 compute_dtype=mstype.float16):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.embed_dim_per_head = embed_dim // num_heads
        self.embed_dim_per_head_fp32 = Tensor(self.embed_dim_per_head, mstype.float32)
        self.mlp_ratio = mlp_ratio
        self.compute_dtype = compute_dtype

        self.layer_norm = nn.LayerNorm([embed_dim], epsilon=1e-6).to_float(mstype.float32)

        self.query = nn.Dense(self.embed_dim, self.embed_dim, weight_init='XavierUniform').to_float(compute_dtype)
        self.key = nn.Dense(self.embed_dim, self.embed_dim, weight_init='XavierUniform').to_float(compute_dtype)
        self.value = nn.Dense(self.embed_dim, self.embed_dim, weight_init='XavierUniform').to_float(compute_dtype)

        self.proj = nn.Dense(self.embed_dim, self.embed_dim, weight_init='XavierUniform').to_float(compute_dtype)
        self.attention_dropout_rate = nn.Dropout(dropout_rate)
        self.proj_dropout = nn.Dropout(dropout_rate)

        self.softmax = nn.Softmax(axis=-1)

    def transpose_for_scores(self, x):
        """transpose_for_scores"""
        new_x_shape = P.Shape()(x)[:-1] + (self.num_heads, self.embed_dim_per_head)
        x = P.Reshape()(x, new_x_shape)
        return P.Transpose()(x, (0, 2, 1, 3))

    def construct(self, x):
        mixed_query_layer = self.query(x)
        mixed_key_layer = self.key(x)
        mixed_value_layer = self.value(x)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores_compute_dtype = mindspore.ops.matmul(query_layer, P.Transpose()(key_layer, (0, 1, 3, 2)))
        attention_scores_fp32 = P.Cast()(attention_scores_compute_dtype, mstype.float32)

        scaled_attention_scores_fp32 = attention_scores_fp32 / P.Sqrt()(self.embed_dim_per_head_fp32)
        attention_probs_fp32 = self.softmax(scaled_attention_scores_fp32)
        attention_probs_compute_dtype = P.Cast()(attention_probs_fp32, self.compute_dtype)
        attention_probs = self.attention_dropout_rate(attention_probs_compute_dtype)

        context_layer = mindspore.ops.matmul(attention_probs, value_layer)
        context_layer = P.Transpose()(context_layer, (0, 2, 1, 3))
        new_context_layer_shape = P.Shape()(context_layer)[:-2] + (self.embed_dim,)
        context_layer = P.Reshape()(context_layer, new_context_layer_shape)
        attention_output = self.proj(context_layer)
        attention_output = self.proj_dropout(attention_output)

        return attention_output


class Mlp(nn.Cell):
    """Mlp"""

    def __init__(self,
                 embed_dim,
                 mlp_ratio,
                 dropout_rate=1.0,
                 compute_dtype=mstype.float16):
        super(Mlp, self).__init__()
        self.fc1 = nn.Dense(embed_dim, embed_dim * mlp_ratio,
                            weight_init='XavierUniform').to_float(compute_dtype)
        self.fc2 = nn.Dense(embed_dim * mlp_ratio, embed_dim,
                            weight_init='XavierUniform').to_float(compute_dtype)
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
                 compute_dtype=mstype.float16):
        super(Block, self).__init__()
        self.embed_dim = embed_dim
        self.attention_norm = nn.LayerNorm([embed_dim], epsilon=1e-6).to_float(mstype.float32)
        self.ffn_norm = nn.LayerNorm([embed_dim], epsilon=1e-6).to_float(mstype.float32)
        self.ffn = Mlp(embed_dim, mlp_ratio, dropout_rate, compute_dtype=compute_dtype)
        self.attn = Attention(embed_dim, num_heads, mlp_ratio, dropout_rate, compute_dtype=compute_dtype)

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
                 compute_dtype=mstype.float16):
        super(Embedding, self).__init__()
        self.compute_dtype = compute_dtype
        self.patch_embedding = nn.Conv2d(in_channels=input_dims,
                                         out_channels=embed_dim,
                                         kernel_size=patch_size,
                                         stride=patch_size,
                                         has_bias=True).to_float(compute_dtype)
        self.init_weights()

    def init_weights(self):
        weight_shape = self.patch_embedding.weight.shape
        xavier_init = initializer(XavierUniform(),
                                  [weight_shape[0], weight_shape[1] * weight_shape[2] * weight_shape[3]],
                                  mstype.float32)

        self.patch_embedding.weight = P.Reshape()(xavier_init,
                                                  (weight_shape[0], weight_shape[1], weight_shape[2], weight_shape[3]))

    def construct(self, x):
        x = self.patch_embedding(x)
        x = P.Reshape()(x, (x.shape[0], x.shape[1], x.shape[2] * x.shape[3]))
        x = P.Transpose()(x, (0, 2, 1))
        return x


class Encoder(nn.Cell):
    r"""
     Encoder module with multi-layer stacked of `Block`, including multihead self attention and feedforward layer.

     Args:
          grid_size (tuple[int]): The grid_size size of input.
          in_channels (int): The input feature size of input. Default: 3.
          patch_size (int): The patch size of image. Default: 16.
          depths (int): The encoder depth of encoder layer.
          embed_dim (int): The encoder embedding dimension of encoder layer. Default: 768.
          num_heads (int): The encoder heads' number of encoder layer. Default: 16.
          mlp_ratio (int): The rate of mlp layer. Default: 4.
          dropout_rate (float): The rate of dropout layer. Default: 1.0.
          compute_dtype (dtype): The data type for encoder, encoding_embedding, encoder and dense layer.
                                Default: mstype.float16.

     Inputs:
             - **input** (Tensor) - Tensor of shape :math:`(batch\_size, feature\_size, image\_height, image\_width)`.

     Outputs:
             - **output** (Tensor) - Tensor of shape :math:`(batch\_size, patchify\_size, embed\_dim)`.
             where patchify_size = (image_height * image_width) / (patch_size * patch_size).

     Supported Platforms:
         ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore
        >>> import mindspore.nn as nn
        >>> from mindspore import Tensor
        >>> from mindspore import context
        >>> from mindspore import dtype as mstype
        >>> from mindflow.cell.transformer.layer import Encoder
        >>> input_tensor = Tensor(np.ones((32, 3, 192, 384)), mstype.float32)
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
        >>>                  compute_dtype=mstype.float16)
        >>>output_tensor = encoder(input_tensor)
        >>> print("output_tensor:",output_tensor.shape)
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
                 compute_dtype=mstype.float16):
        super(Encoder, self).__init__()
        self.patch_embedding = Embedding(in_channels, embed_dim, patch_size, compute_dtype=compute_dtype)
        pos_embed = get_2d_sin_cos_pos_embed(embed_dim, grid_size)
        self.position_embedding = Parameter(Tensor(pos_embed, mstype.float32),
                                            name="encoder_pos_embed",
                                            requires_grad=False)
        self.layer = nn.CellList([])
        self.encoder_norm = nn.LayerNorm([embed_dim], epsilon=1e-6).to_float(mstype.float32)
        for _ in range(depths):
            layer = Block(embed_dim, num_heads, mlp_ratio, dropout_rate, compute_dtype=compute_dtype)
            self.layer.append(copy.deepcopy(layer))

    def construct(self, x):
        """construct"""
        x = self.patch_embedding(x)
        x = x + self.position_embedding
        for layer_block in self.layer:
            x = layer_block(x)
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
        mlp_ratio (int): The rate of mlp layer. Default: 4.
        dropout_rate (float): The rate of dropout layer. Default: 1.0.
        compute_dtype (dtype): The data type for encoder, decoding_embedding, decoder and dense layer.
            Default: mstype.float16.

    Inputs:
        - **input** (Tensor) - Tensor of shape :math:`(batch\_size, patchify\_size, embed\_dim)`.

    Outputs:
        - **output** (Tensor) - Tensor of shape :math:`(batch\_size, patchify\_size, embed\_dim)`.
          where patchify_size = (image_height * image_width) / (patch_size * patch_size).

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore
        >>> import mindspore.nn as nn
        >>> from mindspore import Tensor
        >>> from mindspore import context
        >>> from mindspore import dtype as mstype
        >>> from mindflow.cell.transformer.layer import Decoder
        >>> input_tensor = Tensor(np.ones((32, 288, 512)), mstype.float32)
        >>> print(input_tensor.shape)
        (32, 288, 768)
        >>> decoder = Decoder(grid_size=grid_size,
        >>>                   depths=6,
        >>>                   embed_dim=512,
        >>>                   num_heads=16,
        >>>                   mlp_ratio=4,
        >>>                   dropout_rate=1.0,
        >>>                   compute_dtype=mstype.float16)
        >>> output_tensor = decoder(input_tensor)
        >>> print("output_tensor:",output_tensor.shape)
        (32, 288, 512)
    """

    def __init__(self,
                 grid_size,
                 depths,
                 embed_dim,
                 num_heads,
                 mlp_ratio=4,
                 dropout_rate=1.0,
                 compute_dtype=mstype.float16):
        super(Decoder, self).__init__()
        self.grid_size = grid_size
        self.layer = nn.CellList([])
        pos_embed = get_2d_sin_cos_pos_embed(embed_dim, grid_size)
        self.position_embedding = Parameter(Tensor(pos_embed, mstype.float32),
                                            name="decoder_pos_embed",
                                            requires_grad=False)
        self.decoder_norm = nn.LayerNorm([embed_dim], epsilon=1e-6).to_float(mstype.float32)
        for _ in range(depths):
            layer = Block(embed_dim, num_heads, mlp_ratio, dropout_rate, compute_dtype=compute_dtype)
            self.layer.append(copy.deepcopy(layer))

    def construct(self, x):
        """construct"""
        x = x + self.position_embedding
        for layer_block in self.layer:
            x = layer_block(x)
        x = self.decoder_norm(x)
        return x
