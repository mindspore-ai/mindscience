# Copyright 2024 Huawei Technologies Co., Ltd
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
'''Module providing for SKNO'''
import mindspore
import mindspore.numpy as np
from mindspore import dtype as mstype
from mindspore import nn, Parameter, ops, Tensor
from mindspore.common.initializer import initializer, TruncatedNormal, One, Zero

from .skno_block import SKNOBlock, MLPNet, PatchEmbed


class SKNO(nn.Cell):
    r"""
    The SKNO is the new work of our team, which fuses the SHT algorithm into the ViT-KNO model.
    SKNO treats weather data as the spherical structure and performs spherical harmonic transformation on it.
    The details of SHT can be found in `Spherical Fourier Neural Operators:
    Learning Stable Dynamics on the Sphere <https://dl.acm.org/doi/10.5555/3618408.3618525>`_.

    Args:
        image_size (tuple[int], optional): The size of the input image. Default: (128, 256).
        patch_size (int, optional): The patch size of image. Default: 8.
        in_channels (int, optional): The number of channels in the input space. Default: 1.
        out_channels (int, optional): The number of channels in the output space. Default: 1.
        encoder_depths (int, optional): The encoder depth of encoder layer. Default: 12.
        encoder_embed_dims (int, optional): The encoder embedding dimension of encoder layer. Default: 768.
        mlp_ratio (int, optional): The rate of mlp layer. Default: 4.
        dropout_rate (float, optional): The rate of dropout layer. Default: 1.0.
        drop_path_rate (float, optional): The rate of droppath layer. Default: 0.0.
        num_blocks: (int, optional): The number of blocks. Default: 16.
        high_freq (bool, optional): if high-frequency information complement is applied. Default: True.
        encoder_network (bool, optional): if encoder_network is applied. Default: False
        compute_dtype (dtype, optional): The data type for encoder, decoding_embedding, decoder and dense layer.
                Default: mindspore.float32.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(batch\_size, feature\_size, h\_size, w\_size)`.

    Outputs:
        - **output** (Tensor) - Tensor of shape :math:`(batch\_size, feature\_size, h\_size, w\_size)`.
        - **recons** (Tensor) - Tensor of shape :math:`(batch\_size, feature\_size, h\_size, w\_size)`.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore
        >>> from mindearth.cell import ViTKNO
        >>> B, C, H, W = 16, 20, 128, 256
        >>> input_ = Tensor(np.random.randn(B, C, H, W), dtype=mindspore.float32)
        >>> net = SKNO(image_size=(H, W), in_channels=C, out_channels=C, compute_dtype=mstype.float32)
        >>> output, _ = net(input_)
        >>> print(output.shape)
        (16, 20, 128, 256)

    """

    def __init__(
            self,
            image_size=(128, 256),
            patch_size=8,
            in_channels=1,
            out_channels=1,
            encoder_embed_dims=768,
            encoder_depths=16,
            mlp_ratio=4,
            dropout_rate=1.,
            drop_path_rate=0.,
            num_blocks=16,
            settings="MLP",
            high_freq=True,
            encoder_network=False,
            compute_dtype=mstype.float32
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.num_feature = self.encoder_embed_dims = encoder_embed_dims
        self.num_blocks = num_blocks
        self.encoder_depths = encoder_depths
        self.settings = settings
        self.encoder_network = encoder_network

        self.patch_embed = PatchEmbed(in_channels=in_channels, latent_dims=self.encoder_embed_dims,
                                      patch_size=self.patch_size, compute_dtype=mstype.float32)
        num_patches = (self.image_size[1] // self.patch_size) * (self.image_size[0] // self.patch_size)

        self.pos_embed = Parameter(Tensor(np.zeros((1, num_patches, encoder_embed_dims)), dtype=mindspore.float32),
                                   name="pos_embed")
        self.pos_drop = nn.Dropout(keep_prob=dropout_rate)
        dpr_range = ops.linspace(Tensor(0, mindspore.float32), Tensor(drop_path_rate, mindspore.float32),
                                 self.encoder_depths)
        dpr = [x for x in dpr_range]

        self.h = image_size[0] // self.patch_size
        self.w = image_size[1] // self.patch_size

        self.blocks = nn.CellList([
            SKNOBlock(latent_dims=self.encoder_embed_dims, mlp_ratio=mlp_ratio, dropout_rate=dropout_rate,
                      drop_path=dpr[i], h_size=image_size[0], w_size=image_size[1], patch_size=patch_size,
                      num_blocks=self.num_blocks, high_freq=high_freq, compute_dtype=compute_dtype)
            for i in range(self.encoder_depths)])

        if self.encoder_network:
            self.head_encoder = MLPNet(self.encoder_embed_dims, 2)

        if self.settings == "MLP":
            self.head = nn.Dense(self.encoder_embed_dims, self.out_channels * self.patch_size * self.patch_size,
                                 has_bias=False).to_float(compute_dtype)
        elif self.settings == "Conv2d":
            self.head_conv2d = nn.Conv2d(encoder_embed_dims, self.out_channels * self.patch_size * self.patch_size, 1)

        self.pos_embed.set_data(initializer(TruncatedNormal(sigma=0.02), self.pos_embed.shape, self.pos_embed.dtype))

        self.init_weights()

    @staticmethod
    def no_weight_decay():
        return {'pos_embed', 'cls_token'}

    def init_weights(self):
        """init_weights"""
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Dense):
                cell.weight.set_data(initializer(TruncatedNormal(sigma=0.02), cell.weight.shape, cell.weight.dtype))
                if isinstance(cell, nn.Dense) and cell.bias is not None:
                    cell.bias.set_data(initializer(Zero(), cell.bias.shape, cell.bias.dtype))
            elif isinstance(cell, nn.LayerNorm):
                cell.gamma.set_data(initializer(One(), cell.gamma.shape, cell.gamma.dtype))
                cell.beta.set_data(initializer(Zero(), cell.beta.shape, cell.beta.dtype))

    def encoder(self, x: Tensor):
        '''encoder

        Args:
            x (Tensor): Input Tensor.
        '''
        b = x.shape[0]
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        if self.encoder_network:
            x = self.head_encoder(x)
        x = x.reshape(b, self.h, self.w, self.encoder_embed_dims)
        return x

    def decoder(self, x: Tensor):
        '''decoder

        Args:
            x (Tensor): Input Tensor.
        '''
        if self.settings == "MLP":
            x = self.head(x)
        elif self.settings == "Conv2d":
            x = x.transpose((0, 3, 1, 2))
            x = self.head_conv2d(x)
            x = x.transpose((0, 2, 3, 1))

        h = self.image_size[0] // self.patch_size
        w = self.image_size[1] // self.patch_size
        c_out = self.in_channels
        x = x.reshape(x.shape[0], h, w, self.patch_size, self.patch_size, c_out)
        x = x.transpose((0, 5, 1, 3, 2, 4)).reshape(x.shape[0], c_out, h * self.patch_size, w * self.patch_size)
        return x

    def construct(self, x: Tensor):
        '''SKNO construct function

        Args:
            x (Tensor): Input Tensor.
        '''
        x = self.encoder(x)

        # Reconstruction
        recons = self.decoder(x)

        # Prediction
        b, h, w, c = x.shape
        x = x.reshape(b, h * w, c)
        for blk in self.blocks:
            x = blk(x)
        b, _, c = x.shape
        x = x.reshape(b, h, w, c)
        output = self.decoder(x)
        return output, recons
