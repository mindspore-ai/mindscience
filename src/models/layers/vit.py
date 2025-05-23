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

from mindspore import ops, Parameter, Tensor, nn
import mindspore.ops.operations as P
from mindspore.common.initializer import initializer, XavierUniform
import mindspore.common.dtype as mstype

from .utils import to_2tuple, get_2d_sin_cos_pos_embed
from .attention import AttentionBlock


class PatchEmbedding(nn.Cell):
    """Construct patch embeddings with positional embeddings"""

    def __init__(self, in_channels, hidden_channels, patch_size=16, compute_dtype=mstype.float16
                 ):
        super().__init__()
        self.compute_dtype = compute_dtype
        self.patch_embedding = nn.Conv2d(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=patch_size,
            stride=patch_size,
            has_bias=True,
        ).to_float(compute_dtype)
        self.init_weights()

    def init_weights(self):
        weight_shape = self.patch_embedding.weight.shape
        xavier_init = initializer(
            XavierUniform(),
            [weight_shape[0], weight_shape[1] * weight_shape[2] * weight_shape[3]],
            mstype.float32,
        )

        self.patch_embedding.weight = P.Reshape()(
            xavier_init,
            (weight_shape[0], weight_shape[1],
             weight_shape[2], weight_shape[3]),
        )

    def construct(self, x):
        x = self.patch_embedding(x)
        x = P.Reshape()(x, (x.shape[0], x.shape[1], x.shape[2] * x.shape[3]))
        x = P.Transpose()(x, (0, 2, 1))
        return x


class VitEncoder(nn.Cell):
    r"""
     ViT Encoder module with multi-layer stacked of `MultiHeadAttention`,
     including multihead self attention and feedforward layer.

     Args:
          grid_size (tuple[int]): The grid_size size of input.
          in_channels (int): The input feature size of input. Default: ``3``.
          patch_size (int): The patch size of image. Default: ``16``.
          depths (int): The encoder depth of encoder layer.
          hidden_channels (int): The encoder embedding dimension of encoder layer. Default: ``768``.
          num_heads (int): The encoder heads' number of encoder layer. Default: ``16``.
          dropout_rate (float): The rate of dropout layer. Default: ``0.0``.
          compute_dtype (dtype): The data type for encoder, encoding_embedding, encoder and dense layer.
                                Default: ``mstype.float16``.

     Inputs:
             - **input** (Tensor) - Tensor of shape :math:`(batch\_size, feature\_size, image\_height, image\_width)`.

     Outputs:
             - **output** (Tensor) - Tensor of shape :math:`(batch\_size, patchify\_size, embed\_dim)`.
             where patchify_size = (image_height * image_width) / (patch_size * patch_size).

     Supported Platforms:
         ``Ascend`` ``GPU``

    Examples:
        >>> from mindspore import ops
        >>> from mindflow.cell.vit import VitEncoder
        >>> input_tensor = ops.rand(32, 3, 192, 384)
        >>> print(input_tensor.shape)
        (32, 3, 192, 384)
        >>>encoder = VitEncoder(grid_size=(192 // 16, 384 // 16),
        >>>                     in_channels=3,
        >>>                     patch_size=16,
        >>>                     depths=6,
        >>>                     hidden_channels=768,
        >>>                     num_heads=12,
        >>>                     dropout_rate=0.0,
        >>>                     compute_dtype=mstype.float16)
        >>>output_tensor = encoder(input_tensor)
        >>> print("output_tensor:",output_tensor.shape)
        (32, 288, 768)
    """

    def __init__(self,
                 in_channels,
                 hidden_channels,
                 grid_size,
                 patch_size,
                 depths,
                 num_heads,
                 dropout_rate=0.0,
                 compute_dtype=mstype.float16,
                 ):
        super().__init__()
        self.patch_embedding = PatchEmbedding(
            in_channels, hidden_channels, patch_size, compute_dtype=compute_dtype
        )
        pos_embed = get_2d_sin_cos_pos_embed(hidden_channels, grid_size)
        self.position_embedding = Parameter(
            Tensor(pos_embed, mstype.float32),
            name="encoder_pos_embed",
            requires_grad=False,
        )
        self.layer = nn.CellList([])
        self.encoder_norm = nn.LayerNorm([hidden_channels], epsilon=1e-6).to_float(
            mstype.float32
        )
        for _ in range(depths):
            layer = AttentionBlock(
                in_channels=hidden_channels,
                num_heads=num_heads,
                dropout_rate=dropout_rate,
                compute_dtype=compute_dtype,
            )
            self.layer.append(layer)

    def construct(self, x):
        """construct"""
        x = self.patch_embedding(x)
        x = x + self.position_embedding
        for layer_block in self.layer:
            x = layer_block(x)
        x = self.encoder_norm(x)
        return x


class VitDecoder(nn.Cell):
    r"""
    ViT Decoder module with multi-layer stacked of `MultiHeadAttention`,
    including multihead self attention and feedforward layer.

    Args:
        grid_size (tuple[int]): The grid_size size of input.
        depths (int): The decoder depth of decoder layer.
        hidden_channels (int): The decoder embedding dimension of decoder layer.
        num_heads (int): The decoder heads' number of decoder layer.
        dropout_rate (float): The rate of dropout layer. Default: ``0.0``.
        compute_dtype (dtype): The data type for encoder, decoding_embedding, decoder and dense layer.
            Default: ``mstype.float16``.

    Inputs:
        - **input** (Tensor) - Tensor of shape :math:`(batch\_size, patchify\_size, embed\_dim)`.

    Outputs:
        - **output** (Tensor) - Tensor of shape :math:`(batch\_size, patchify\_size, embed\_dim)`.
          where patchify_size = (image_height * image_width) / (patch_size * patch_size).

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> from mindspore import ops
        >>> from mindflow.cell.vit import VitDecoder
        >>> input_tensor = ops.rand(32, 288, 512)
        >>> print(input_tensor.shape)
        (32, 288, 768)
        >>> decoder = VitDecoder(grid_size=grid_size,
        >>>                      depths=6,
        >>>                      hidden_channels=512,
        >>>                      num_heads=16,
        >>>                      dropout_rate=0.0,
        >>>                      compute_dtype=mstype.float16)
        >>> output_tensor = VitDecoder(input_tensor)
        >>> print("output_tensor:",output_tensor.shape)
        (32, 288, 512)
    """

    def __init__(self,
                 grid_size,
                 depths,
                 hidden_channels,
                 num_heads,
                 dropout_rate=0.0,
                 compute_dtype=mstype.float16,
                 ):
        super().__init__()
        self.grid_size = grid_size
        self.layer = nn.CellList([])
        pos_embed = get_2d_sin_cos_pos_embed(hidden_channels, grid_size)
        self.position_embedding = Parameter(
            Tensor(pos_embed, mstype.float32),
            name="decoder_pos_embed",
            requires_grad=False,
        )
        self.decoder_norm = nn.LayerNorm([hidden_channels], epsilon=1e-6).to_float(
            mstype.float32
        )
        for _ in range(depths):
            layer = AttentionBlock(
                in_channels=hidden_channels,
                num_heads=num_heads,
                dropout_rate=dropout_rate,
                compute_dtype=compute_dtype,
            )
            self.layer.append(layer)

    def construct(self, x):
        """construct"""
        x = x + self.position_embedding
        for layer_block in self.layer:
            x = layer_block(x)
        x = self.decoder_norm(x)
        return x


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
        dropout_rate (float): The rate of dropout layer. Default: ``0.0``.
        compute_dtype (dtype): The data type for encoder, decoding_embedding, decoder and dense layer.
            Default: ``mstype.float16``.

    Inputs:
        - **input** (Tensor) - Tensor of shape :math:`(batch\_size, feature\_size, image\_height, image\_width)`.

    Outputs:
        - **output** (Tensor) - Tensor of shape :math:`(batch\_size, patchify\_size, embed\_dim)`.
          where patchify_size = (image_height * image_width) / (patch_size * patch_size)

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> from mindspore import ops
        >>> from mindflow.cell import ViT
        >>> input_tensor = ops.rand(32, 3, 192, 384)
        >>> print(input_tensor.shape)
        (32, 3, 192, 384)
        >>> model = ViT(in_channels=3,
        >>>             out_channels=3,
        >>>             encoder_depths=6,
        >>>             encoder_embed_dim=768,
        >>>             encoder_num_heads=12,
        >>>             decoder_depths=6,
        >>>             decoder_embed_dim=512,
        >>>             decoder_num_heads=16,
        >>>             )
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
                 dropout_rate=0.0,
                 compute_dtype=mstype.float16,
                 ):
        super().__init__()
        image_size = to_2tuple(image_size)
        grid_size = (image_size[0] // patch_size, image_size[1] // patch_size)

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

        self.encoder = VitEncoder(
            in_channels=in_channels,
            hidden_channels=encoder_embed_dim,
            patch_size=patch_size,
            grid_size=grid_size,
            depths=encoder_depths,
            num_heads=encoder_num_heads,
            dropout_rate=dropout_rate,
            compute_dtype=compute_dtype,
        )

        self.decoder_embedding = nn.Dense(
            encoder_embed_dim,
            decoder_embed_dim,
            has_bias=True,
            weight_init="XavierUniform",
        ).to_float(compute_dtype)

        self.decoder = VitDecoder(
            hidden_channels=decoder_embed_dim,
            grid_size=grid_size,
            depths=decoder_depths,
            num_heads=decoder_num_heads,
            dropout_rate=dropout_rate,
            compute_dtype=compute_dtype,
        )

        self.decoder_pred = nn.Dense(
            decoder_embed_dim,
            patch_size**2 * out_channels,
            has_bias=True,
            weight_init="XavierUniform",
        ).to_float(compute_dtype)

    def construct(self, x):
        x = self.encoder(x)
        x = self.decoder_embedding(x)
        x = self.decoder(x)
        images = self.decoder_pred(x)
        return images.astype(mstype.float32)
