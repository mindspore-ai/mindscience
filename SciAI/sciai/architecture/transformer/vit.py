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
"""The ViT model"""
import mindspore as ms
import mindspore.ops.operations as P
from mindspore import nn, ops

from sciai.architecture.transformer.layer import Decoder, Encoder
from sciai.utils.math_utils import _to_2tuple


class ViT(nn.Cell):
    r"""
    This module based on ViT backbone which including encoder, decoding_embedding, decoder and dense layer.

    Args:
        image_size (tuple[int]): The image size of input. Default: ``(192, 384)``.
        in_channels (int): The input feature size of input. Default: ``7``.
        out_channels (int): The output feature size of output. Default: ``3``.
        patch_size (int): The patch size of image. Default: ``16``.
        encoder_depths (int): The encoder depth of encoder layer. Default: ``12``.
        encoder_embed_dim (int): The encoder embedding dimension of encoder layer. Default: ``768``.
        encoder_num_heads (int): The encoder heads' number of encoder layer. Default: ``12``.
        decoder_depths (int): The decoder depth of decoder layer. Default: ``8``.
        decoder_embed_dim (int): The decoder embedding dimension of decoder layer. Default: ``512``.
        decoder_num_heads (int): The decoder heads' number of decoder layer. Default: ``16``.
        mlp_ratio (int): The rate of mlp layer. Default: ``4``.
        dropout_rate (float): The rate of dropout layer. Default: ``1.0``.
        dtype (dtype.Number): The data type for encoder, decoding_embedding, decoder and dense layer.
            Default: ``ms.float16``.

    Inputs:
        - **input** (Tensor) - Tensor of shape :math:`(batch\_size, feature\_size, image\_height, image\_width)`.

    Outputs:
        - **output** (Tensor) - Tensor of shape :math:`(batch\_size, patchify\_size, embed\_dim)`.
          where patchify_size = (image_height * image_width) / (patch_size * patch_size)

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore as ms
        >>> from mindspore import Tensor, context
        >>> from sciai.architecture.transformer import ViT
        >>> input_tensor = Tensor(np.ones((32, 3, 192, 384)), ms.float32)
        >>> print(input_tensor.shape)
        (32, 3, 192, 384)
        >>> model = ViT(in_channels=3,
        >>>             out_channels=3,
        >>>             encoder_depths=6,
        >>>             encoder_embed_dim=768,
        >>>             encoder_num_heads=12,
        >>>             decoder_depths=6,
        >>>             decoder_embed_dim=512,
        >>>             decoder_num_heads=16)
        >>> output_tensor = model(input_tensor)
        >>> print(output_tensor.shape)
        (32, 288, 768)
    """

    def __init__(self,
                 image_size=(192, 384),
                 in_channels=7,
                 out_channels=3,
                 patch_size=16,
                 encoder_depths=12,
                 encoder_embed_dim=768,
                 encoder_num_heads=12,
                 decoder_depths=8,
                 decoder_embed_dim=512,
                 decoder_num_heads=16,
                 mlp_ratio=4,
                 dropout_rate=1.0,
                 dtype=ms.float16):
        super(ViT, self).__init__()
        image_size = _to_2tuple(image_size)
        grid_size = (image_size[0] // patch_size, image_size[1] // patch_size)

        self.img_size = image_size
        self.patch_size = patch_size
        self.out_channels = out_channels

        self.in_channels = in_channels

        self.encoder_depths = encoder_depths
        self.encoder_embed_dim = encoder_embed_dim
        self.encoder_num_heads = encoder_num_heads

        self.decoder_depths = decoder_depths
        self.decoder_embed_dim = decoder_embed_dim
        self.decoder_num_heads = decoder_num_heads

        self.transpose = ops.Transpose()

        self.encoder = Encoder(grid_size=grid_size,
                               in_channels=in_channels,
                               patch_size=patch_size,
                               depths=encoder_depths,
                               embed_dim=encoder_embed_dim,
                               num_heads=encoder_num_heads,
                               mlp_ratio=mlp_ratio,
                               dropout_rate=dropout_rate,
                               dtype=dtype)

        self.decoder_embedding = nn.Dense(encoder_embed_dim,
                                          decoder_embed_dim,
                                          has_bias=True,
                                          weight_init="XavierUniform").to_float(dtype)

        self.decoder = Decoder(grid_size=grid_size,
                               depths=decoder_depths,
                               embed_dim=decoder_embed_dim,
                               num_heads=decoder_num_heads,
                               mlp_ratio=mlp_ratio,
                               dropout_rate=dropout_rate,
                               dtype=dtype)

        self.decoder_pred = nn.Dense(decoder_embed_dim,
                                     patch_size ** 2 * out_channels,
                                     has_bias=True,
                                     weight_init="XavierUniform").to_float(dtype)

    def construct(self, x):
        x = self.encoder(x)
        x = self.decoder_embedding(x)
        x = self.decoder(x)
        images = self.decoder_pred(x)
        images = P.Cast()(images, ms.float32)
        return images
