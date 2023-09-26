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
'''Module providing AFNONet'''
import mindspore
from mindspore import nn, ops
from mindspore.common.initializer import initializer, Normal
from mindspore import dtype as mstype

from mindearth.cell.utils import to_2tuple
from .afno2d import ForwardFeatures


class AFNONet(nn.Cell):
    r"""
    The AFNO model is a deep learning model that based on the
    Fourier Neural Operator (AFNO) and the Vision Transformer structure.
    The details can be found in `Adaptive Fourier Neural Operators: Efficient
    Token Mixers For Transformers <https://arxiv.org/pdf/2111.13587.pdf>`_.

    Args:
        image_size (tuple[int]): The size of the input image. Default: (128, 256).
        in_channels (int): The number of channels in the input space. Default: 1.
        out_channels (int): The number of channels in the output space. Default: 1.
        patch_size (int): The patch size of image. Default: 8.
        encoder_depths (int): The encoder depth of encoder layer. Default: 12.
        encoder_embed_dim (int): The encoder embedding dimension of encoder layer. Default: 768.
        mlp_ratio (int): The rate of mlp layer. Default: 4.
        dropout_rate (float): The rate of dropout layer. Default: 1.0.
        compute_dtype (dtype): The data type for encoder, decoding_embedding, decoder and dense layer.
                Default: mindspore.float32.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(batch\_size, feature\_size, image\_height, image\_width)`.

    Outputs:
        - **output** (Tensor) -Tensor of shape :math:`(batch\_size, patch\_size, embed\_dim)`,
          where :math:`patch\_size = (image\_height * image\_width) / (patch\_size * patch\_size)`.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore.common.initializer import initializer, Normal
        >>> from mindearth.cell import AFNONet
        >>> B, C, H, W = 16, 20, 128, 256
        >>> input_ = initializer(Normal(), [B, C, H, W])
        >>> net = AFNONet(image_size=(H, W), in_channels=C, out_channels=C, compute_dtype=dtype.float32)
        >>> output = net(input_)
        >>> print(output.shape)
        (16, 128, 5120)

    """

    def __init__(self,
                 image_size=(128, 256),
                 in_channels=1,
                 out_channels=1,
                 patch_size=8,
                 encoder_depths=12,
                 encoder_embed_dim=768,
                 mlp_ratio=4,
                 dropout_rate=1.0,
                 compute_dtype=mindspore.float32):
        super(AFNONet, self).__init__()
        image_size = to_2tuple(image_size)
        try:
            grid_size = (image_size[0] // patch_size, image_size[1] // patch_size)
        except ZeroDivisionError:
            ops.Print()("Patch size can't be Zero")

        self.image_size = image_size
        self.patch_size = patch_size
        self.output_dims = out_channels
        self.input_dims = in_channels

        self.encoder_depths = encoder_depths
        self.encoder_embed_dim = encoder_embed_dim

        self.transpose = ops.Transpose()

        self.forward_features = ForwardFeatures(grid_size=grid_size,
                                                in_channels=in_channels,
                                                patch_size=patch_size,
                                                depth=encoder_depths,
                                                embed_dims=encoder_embed_dim,
                                                mlp_ratio=mlp_ratio,
                                                dropout_rate=dropout_rate,
                                                compute_dtype=compute_dtype)

        self.compute_type = compute_dtype

        self.head = nn.Dense(encoder_embed_dim, patch_size ** 2 * out_channels,
                             weight_init=initializer(Normal(sigma=0.02),
                                                     shape=(patch_size ** 2 * out_channels, encoder_embed_dim)),
                             has_bias=False).to_float(compute_dtype)

    def construct(self, x):
        x = self.forward_features(x)
        x = self.head(x)

        output = ops.Cast()(x, mstype.float32)
        return output
