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
"""
The ViT model
"""


import mindspore
import mindspore.ops as ops
import mindspore.nn as nn
import mindspore.ops.operations as P

from ...common.math import to_2tuple
from .layer import Decoder, Encoder


class ViT(nn.Cell):
    r"""
    This module based on ViT backbone which including encoder, decoding_embedding, decoder and dense layer.

    Args:
         img_size(tuple[int]): The image size of input. Default: (192, 384).
         input_dims(int): The input feature size of input. Default: 7.
         output_dims(int): The output feature size of output. Default: 3.
         patch_size(int): The patch size of image. Default: 16.
         encoder_depth(int): The encoder depth of encoder layer. Default: 12.
         encoder_embed_dim(int): The encoder embedding dimension of encoder layer. Default: 768.
         encoder_num_heads(int): The encoder heads' number of encoder layer. Default: 12.
         decoder_depth(int): The decoder depth of decoder layer. Default: 8.
         decoder_embed_dim(int): The decoder embedding dimension of decoder layer. Default: 512.
         decoder_num_heads(int): The decoder heads' number of decoder layer. Default: 16.
         mlp_ratio(int): The rate of mlp layer. Default: 4.
         dropout_rate(float): The rate of dropout layer. Default: 1.0.
         compute_dtype(dtype): The data type for encoder, decoding_embedding, decoder and dense layer.
                               Default: mindspore.float16.

    Inputs:
            - **input** (Tensor) - Tensor of shape :math:`(batch\_size, feature\_size, image\_height, image\_width)`.

    Outputs:
            - **output** (Tensor) - Tensor of shape :math:`(batch\_size, patchify\_size, embed\_dim)`.
              where patchify_size = (image_height * image_width) / (patch_size * patch_size)

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> from mindspore import context
        >>> from mindspore import dtype as mstype
        >>> from mindflow.cell import ViT
        >>> input_tensor = Tensor(np.ones((32, 3, 192, 384)), mstype.float32)
        >>> print(input_tensor.shape)
        (32, 3, 192, 384)
        >>> model = ViT(input_dims=3,
        >>>             output_dims=3,
        >>>             encoder_depth=6,
        >>>             encoder_embed_dim=768,
        >>>             encoder_num_heads=12,
        >>>             decoder_depth=6,
        >>>             decoder_embed_dim=512,
        >>>             decoder_num_heads=16,
        >>>             )
        >>> output_tensor = model(input_tensor)
        >>> print(output_tensor.shape)
        (32, 288, 768)
    """
    def __init__(self,
                 img_size=(192, 384),
                 input_dims=7,
                 output_dims=3,
                 patch_size=16,
                 encoder_depth=12,
                 encoder_embed_dim=768,
                 encoder_num_heads=12,
                 decoder_depth=8,
                 decoder_embed_dim=512,
                 decoder_num_heads=16,
                 mlp_ratio=4,
                 dropout_rate=1.0,
                 compute_dtype=mindspore.float16):
        super(ViT, self).__init__()
        img_size = to_2tuple(img_size)
        grid_size = (img_size[0] // patch_size, img_size[1] // patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.output_dims = output_dims

        self.input_dims = input_dims

        self.encoder_depth = encoder_depth
        self.encoder_embed_dim = encoder_embed_dim
        self.encoder_num_heads = encoder_num_heads

        self.decoder_depth = decoder_depth
        self.decoder_embed_dim = decoder_embed_dim
        self.decoder_num_heads = decoder_num_heads

        self.transpose = ops.Transpose()

        self.encoder = Encoder(grid_size=grid_size,
                               input_dims=input_dims,
                               patch_size=patch_size,
                               depth=encoder_depth,
                               embed_dim=encoder_embed_dim,
                               num_heads=encoder_num_heads,
                               mlp_ratio=mlp_ratio,
                               dropout_rate=dropout_rate,
                               compute_dtype=compute_dtype)

        self.decoder_embedding = nn.Dense(encoder_embed_dim,
                                          decoder_embed_dim,
                                          has_bias=True,
                                          weight_init="XavierUniform").to_float(compute_dtype)

        self.decoder = Decoder(grid_size=grid_size,
                               depth=decoder_depth,
                               embed_dim=decoder_embed_dim,
                               num_heads=decoder_num_heads,
                               mlp_ratio=mlp_ratio,
                               dropout_rate=dropout_rate,
                               compute_dtype=compute_dtype)

        self.decoder_pred = nn.Dense(decoder_embed_dim,
                                     patch_size ** 2 * output_dims,
                                     has_bias=True,
                                     weight_init="XavierUniform").to_float(compute_dtype)

    def construct(self, x):
        x = self.encoder(x)
        x = self.decoder_embedding(x)
        x = self.decoder(x)
        imgs = self.decoder_pred(x)
        imgs = P.Cast()(imgs, mindspore.float32)
        return imgs
