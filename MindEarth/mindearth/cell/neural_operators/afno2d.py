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
'''Module providing afno2d'''
import numpy as np

from mindspore import ops, nn, Tensor, Parameter
from mindspore import dtype as mstype
from mindspore.common.initializer import initializer, Normal, TruncatedNormal
from mindspore.nn.probability.distribution import Bernoulli

from .dft import dft2, idft2


class DropPath(nn.Cell):
    """

    Args:

    Inputs:

    Outputs:

    Supported Platforms:

    Examples:


    """
    def __init__(self, drop_prob=0.0, scale_by_keep=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.keep_prob = 1.0 - self.drop_prob
        self.scale_by_keep = scale_by_keep
        self.bernoulli = Bernoulli(probs=self.keep_prob)
        self.div = ops.Div()

    def construct(self, x):
        '''construct'''
        if self.drop_prob > 0.0:
            random_tensor = self.bernoulli.sample((x.shape[0],) + (1,) * (x.ndim - 1))
            if self.keep_prob > 0.0 and self.scale_by_keep:
                random_tensor = self.div(random_tensor, self.keep_prob)
            x = x * random_tensor
        return x


class Mlp(nn.Cell):
    """

    Args:

    Inputs:

    Outputs:

    Supported Platforms:

    Examples:


    """
    def __init__(self,
                 embed_dims,
                 mlp_ratio,
                 dropout_rate=1.0,
                 compute_dtype=mstype.float16):
        super(Mlp, self).__init__()
        self.fc1 = nn.Dense(embed_dims, embed_dims * mlp_ratio,
                            weight_init=initializer(Normal(sigma=0.02), shape=(embed_dims * mlp_ratio, embed_dims)),
                            ).to_float(compute_dtype)
        self.fc2 = nn.Dense(embed_dims * mlp_ratio, embed_dims,
                            weight_init=initializer(Normal(sigma=0.02), shape=(embed_dims, embed_dims * mlp_ratio)),
                            ).to_float(compute_dtype)

        self.act_fn = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)

    def construct(self, x):
        '''construct'''
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)

        return x


class AFNOBlock(nn.Cell):
    """

    Args:

    Inputs:

    Outputs:

    Supported Platforms:

    Examples:


    """
    def __init__(self,
                 embed_dims,
                 mlp_ratio,
                 dropout_rate=1.0,
                 drop_path=0.,
                 h_size=128,
                 w_size=256,
                 patch_size=8,
                 num_blocks=8,
                 high_freq=False,
                 compute_dtype=mstype.float16):
        super(AFNOBlock, self).__init__()
        self.embed_dims = embed_dims
        self.layer_norm = nn.LayerNorm([embed_dims], epsilon=1e-6).to_float(compute_dtype)

        self.ffn_norm = nn.LayerNorm([embed_dims], epsilon=1e-6).to_float(compute_dtype)
        self.mlp = Mlp(embed_dims, mlp_ratio, dropout_rate, compute_dtype=compute_dtype)
        self.filter = AFNO2D(h_size=h_size // patch_size,
                             w_size=w_size // patch_size,
                             embed_dims=embed_dims,
                             num_blocks=num_blocks,
                             high_freq=high_freq,
                             compute_dtype=compute_dtype)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def construct(self, x):
        '''construct'''
        h = x
        x = self.layer_norm(x)
        x = self.filter(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.mlp(x)
        x = self.drop_path(x)
        x = x + h
        return x


class PatchEmbed(nn.Cell):
    """

    Args:

    Inputs:

    Outputs:

    Supported Platforms:

    Examples:


    """
    def __init__(self,
                 in_channels,
                 embed_dims,
                 patch_size=16,
                 compute_dtype=mstype.float16):
        super(PatchEmbed, self).__init__()
        self.compute_dtype = compute_dtype
        self.proj = nn.Conv2d(in_channels=in_channels,
                              out_channels=embed_dims,
                              kernel_size=patch_size,
                              stride=patch_size,
                              has_bias=True,
                              bias_init='normal'
                              ).to_float(compute_dtype)

    def construct(self, x):
        '''construct'''
        x = self.proj(x)
        x = ops.Reshape()(x, (x.shape[0], x.shape[1], x.shape[2] * x.shape[3]))
        x = ops.Transpose()(x, (0, 2, 1))
        return x


class ForwardFeatures(nn.Cell):
    """

    Args:

    Inputs:

    Outputs:

    Supported Platforms:

    Examples:


    """
    def __init__(self,
                 grid_size,
                 in_channels,
                 patch_size,
                 depth,
                 embed_dims,
                 mlp_ratio=4,
                 dropout_rate=1.0,
                 compute_dtype=mstype.float16):
        super(ForwardFeatures, self).__init__()
        self.patch_embed = PatchEmbed(in_channels, embed_dims, patch_size, compute_dtype=compute_dtype)
        self.pos_embed = Parameter(
            initializer(TruncatedNormal(sigma=0.02), [1, grid_size[0] * grid_size[1], embed_dims], dtype=compute_dtype),
            requires_grad=True)
        self.layer = nn.CellList([])
        self.encoder_norm = nn.LayerNorm([embed_dims], epsilon=1e-6).to_float(compute_dtype)
        for _ in range(depth):
            self.layer.append(AFNOBlock(embed_dims, mlp_ratio, dropout_rate, patch_size=patch_size,
                                        compute_dtype=compute_dtype))

        self.pos_drop = nn.Dropout(keep_prob=dropout_rate)
        self.h = grid_size[0]
        self.w = grid_size[1]
        self.embed_dims = embed_dims

    def construct(self, x):
        '''construct'''
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for layer_block in self.layer:
            x = layer_block(x)

        return x


class AFNO2D(nn.Cell):
    """

    Args:

    Inputs:

    Outputs:

    Supported Platforms:

    Examples:


    """
    def __init__(self,
                 h_size=32,
                 w_size=64,
                 mean=None,
                 std=None,
                 embed_dims=768,
                 num_blocks=8,
                 high_freq=False,
                 compute_dtype=mstype.float16):
        super().__init__()

        self.compute_type = compute_dtype

        self.fno_seq = nn.SequentialCell()
        self.concat = ops.Concat(axis=-1)
        self.act = ops.GeLU()

        self.mean = Tensor(mean, dtype=compute_dtype) if mean is not None else None
        self.std = Tensor(std, dtype=compute_dtype) if std is not None else None

        self.h_size = h_size
        self.w_size = w_size

        self.dft2_cell = dft2(shape=(h_size, w_size), dim=(-3, -2),
                              modes=(h_size // 2, w_size // 2 + 1), compute_dtype=compute_dtype)
        self.idft2_cell = idft2(shape=(h_size, w_size), dim=(-3, -2),
                                modes=(h_size // 2, w_size // 2 + 1), compute_dtype=compute_dtype)

        self.scale = 0.02
        self.num_blocks = num_blocks
        self.block_size = embed_dims // self.num_blocks
        self.hidden_size_factor = 1
        w1 = self.scale * Tensor(np.random.randn(
            2, self.num_blocks, self.block_size, self.block_size * self.hidden_size_factor), dtype=compute_dtype)
        b1 = self.scale * Tensor(np.random.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor),
                                 dtype=compute_dtype)
        w2 = self.scale * Tensor(np.random.randn(
            2, self.num_blocks, self.block_size * self.hidden_size_factor, self.block_size), dtype=compute_dtype)
        b2 = self.scale * Tensor(np.random.randn(2, self.num_blocks, self.block_size), dtype=compute_dtype)

        self.w1 = Parameter(w1, requires_grad=True)
        self.b1 = Parameter(b1, requires_grad=True)
        self.w2 = Parameter(w2, requires_grad=True)
        self.b2 = Parameter(b2, requires_grad=True)

        self.relu = ops.ReLU()

        self.sparsity_threshold = 0.01
        self.hard_thresholding_fraction = 1.0

        self.high_freq = high_freq
        self.w = nn.Conv2d(embed_dims, embed_dims, 1)  # High Frequency

        self.cast = ops.Cast()

    @staticmethod
    def mul2d(inputs, weights):
        weight = weights.expand_dims(0)
        data = inputs.expand_dims(5)
        out = weight * data
        return out.sum(4)

    def construct(self, x: Tensor):
        '''construct'''
        if self.high_freq:
            b, n, c = x.shape
            h, w = self.h_size, self.w_size
            x = x.reshape(b, h, w, c)
            bias = self.w(x.transpose((0, 3, 1, 2))).transpose((0, 2, 3, 1))
            bias = bias.reshape(b, h * w, c)
        else:
            bias = x
            b, n, c = x.shape
            h, w = self.h_size, self.w_size
            x = x.reshape(b, h, w, c)

        x_re = x
        x_im = ops.zeros_like(x_re)

        x_ft_re, x_ft_im = self.dft2_cell((x_re, x_im))

        x_ft_re = x_ft_re.reshape(b, x_ft_re.shape[1], x_ft_re.shape[2], self.num_blocks, self.block_size)
        x_ft_im = x_ft_im.reshape(b, x_ft_im.shape[1], x_ft_im.shape[2], self.num_blocks, self.block_size)

        kept_modes = h // 2 + 1

        o1_real = self.relu(self.mul2d(x_ft_re, self.w1[0]) - self.mul2d(x_ft_im, self.w1[1]) + self.b1[0])
        o1_real[:, :, kept_modes:] = 0.0

        o1_imag = self.relu(self.mul2d(x_ft_im, self.w1[0]) + self.mul2d(x_ft_re, self.w1[1]) + self.b1[1])
        o1_imag[:, :, kept_modes:] = 0.0

        o2_real = (self.mul2d(o1_real, self.w2[0]) - self.mul2d(o1_imag, self.w2[1]) + self.b2[0])
        o2_real[:, :, kept_modes:] = 0.0

        o2_imag = (self.mul2d(o1_imag, self.w2[0]) + self.mul2d(o1_real, self.w2[1]) + self.b2[1])
        o2_imag[:, :, kept_modes:] = 0.0

        o2_real = self.cast(o2_real, self.compute_type)
        o2_imag = self.cast(o2_imag, self.compute_type)

        o2_real = ops.softshrink(o2_real, lambd=self.sparsity_threshold)
        o2_imag = ops.softshrink(o2_imag, lambd=self.sparsity_threshold)

        o2_real = o2_real.reshape(b, o2_real.shape[1], o2_real.shape[2], c)
        o2_imag = o2_imag.reshape(b, o2_imag.shape[1], o2_imag.shape[2], c)

        x, _ = self.idft2_cell((o2_real, o2_imag))

        x = x.reshape(b, n, c)
        return x + bias
