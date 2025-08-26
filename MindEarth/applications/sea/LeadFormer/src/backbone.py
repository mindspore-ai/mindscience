# right 2020-2022 Huawei Technologies Co., Ltd
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
"""model"""
import math

import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.numpy as msnp
from mindspore.ops import linspace
from mindspore.common.initializer import initializer, TruncatedNormal, Normal


def init_weights(m):
    """init_weights"""
    if isinstance(m, nn.Dense):
        m.weight.set_data(
            initializer(TruncatedNormal(0.02), m.weight.shape, m.weight.dtype)
        )
        if m.bias is not None:
            m.bias.set_data(initializer("zeros", m.bias.shape, m.bias.dtype))
    elif isinstance(m, nn.LayerNorm):
        m.gamma.set_data(initializer("ones", m.gamma.shape, m.gamma.dtype))
        m.beta.set_data(initializer("zeros", m.beta.shape, m.beta.dtype))
    elif isinstance(m, nn.Conv2d):
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        fan_out //= m.group
        m.weight.set_data(
            initializer(Normal(math.sqrt(2.0 / fan_out)), m.weight.shape, m.weight.dtype)
        )
        if m.bias is not None:
            m.bias.set_data(initializer("zeros", m.bias.shape, m.bias.dtype))


class OverlapPatchEmbed(nn.Cell):
    """
    Implements overlapping patch embedding with convolutional projection.

    This module splits input images into overlapping patches using a convolutional
    projection layer, then applies layer normalization. Designed for vision transformers
    with overlapping patch strategies.

    Args:
        patch_size (int): Size of the sliding window (kernel size). Default: 7
        stride (int): Stride for convolution operation. Controls patch overlap. Default: 4
        in_chans (int): Number of input channels. Default: 3 (RGB)
        embed_dim (int): Dimension of embedding output. Default: 768

    Shape:
        Input: (B, C, H, W) where B=batch_size, C=in_chans
        Output: (B, num_patches, embed_dim)
    """
    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=(
                patch_size[0] // 2,
                patch_size[1] // 2,
                patch_size[0] // 2,
                patch_size[1] // 2,
            ),
            pad_mode="pad",
            has_bias=True,
            bias_init=None,
        )
        self.norm = nn.LayerNorm((embed_dim,))
        self.apply(init_weights)

    def construct(self, x):
        x = self.proj(x)
        b, c, h, w = x.shape

        x = x.reshape(b, c, h * w)
        x = ops.swapaxes(x, 1, 2)
        x = self.norm(x)

        return x


class Attention(nn.Cell):
    """
    Implements multi-head self-attention with spatial reduction.

    This attention module optionally incorporates spatial reduction (SR) to reduce
    computational complexity for high-resolution inputs. When sr_ratio > 1, it applies
    convolution to reduce spatial dimensions before key/value computation.

    Args:
        dim (int): Input feature dimension
        num_heads (int): Number of attention heads. Default: 8
        qkv_bias (bool): Enable bias for query/key/value projections. Default: False
        qk_scale (float): Override default scale factor (1/sqrt(d_k)). Default: None
        attn_drop (float): Attention dropout probability. Default: 0.0
        proj_drop (float): Output projection dropout probability. Default: 0.0
        sr_ratio (int): Spatial reduction ratio. Default: 1 (no reduction)

    Shape:
        Input: (B, N, C) where:
            B = batch size
            N = sequence length (H * W)
            C = feature dimension (dim)
        Output: (B, N, C) (same shape as input)
    """
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_scale=None,
            attn_drop=0.0,
            proj_drop=0.0,
            sr_ratio=1,
    ):
        super().__init__()
        assert (
            dim % num_heads == 0
        ), f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.q = nn.Dense(dim, dim, has_bias=qkv_bias)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(
                dim, dim, kernel_size=sr_ratio, has_bias=True, stride=sr_ratio
            )
            self.norm = nn.LayerNorm(normalized_shape=(dim,))
        self.kv = nn.Dense(dim, dim * 2, has_bias=qkv_bias)
        self.attn_drop = nn.Dropout(p=attn_drop)
        self.proj = nn.Dense(dim, dim)
        self.proj_drop = nn.Dropout(p=proj_drop)
        self.apply(init_weights)

    def construct(self, x, h, w):
        """construct"""
        b, n, c = x.shape
        q = (
            self.q(x)
            .reshape(b, n, self.num_heads, c // self.num_heads)
            .permute(0, 2, 1, 3)
        )

        if self.sr_ratio > 1:
            x_ = ops.permute(x, (0, 2, 1))
            x_ = x_.reshape(b, c, h, w)
            x_ = self.sr(x_).reshape(b, c, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = (
                self.kv(x_)
                .reshape(b, -1, 2, self.num_heads, c // self.num_heads)
                .permute(2, 0, 3, 1, 4)
            )
        else:
            kv = (
                self.kv(x)
                .reshape(b, -1, 2, self.num_heads, c // self.num_heads)
                .permute(2, 0, 3, 1, 4)
            )
        k, v = kv[0], kv[1]

        attn = (q @ k.swapaxes(-2, -1)) * self.scale
        attn = nn.Softmax(axis=-1)(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).swapaxes(1, 2).reshape(b, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


def drop_path(
        x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True
):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    if keep_prob > 0.0 and scale_by_keep:
        mask = keep_prob + ops.rand(shape, dtype=x.dtype)
        mask = msnp.floor(mask)
        x = ops.div(x, keep_prob) * mask
    return x


class DropPath(nn.Cell):
    """
    Actual implementation of stochastic depth (drop path) regularization.

    Args:
        x (Tensor): Input tensor
        drop_prob (float): Probability of dropping a sample
        training (bool): Whether in training mode
        scale_by_keep (bool): Whether to scale outputs by keep probability

    Returns:
        Tensor: Output tensor after applying drop path
    """
    def __init__(self, drop_prob=None, scale_by_keep=True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def construct(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


class DWConv(nn.Cell):
    """
    Depthwise Convolution Layer

    This module implements a depthwise separable convolution operation.
    Depthwise convolution applies a single filter per input channel, making it
    computationally efficient while still capturing spatial features.

    Args:
        dim (int): Number of input/output channels (default: 768)
    """
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(
            dim,
            dim,
            kernel_size=3,
            stride=1,
            pad_mode="pad",
            padding=1,
            has_bias=True,
            group=dim,
        )

    def construct(self, x, h, w):
        """construct"""
        b, _, c = x.shape
        x = x.swapaxes(1, 2).view(b, c, h, w)
        x = self.dwconv(x)
        x = x.flatten(start_dim=2).swapaxes(1, 2)

        return x


class Mlp(nn.Cell):
    """
    Multi-Layer Perceptron with Depthwise Convolution

    This module implements a multi-layer perceptron with an intermediate depthwise convolution layer.
    It consists of two fully connected layers with an activation function and dropout in between.
    A depthwise convolution is applied after the first fully connected layer to incorporate spatial information.

    Args:
        in_features (int): Number of input features
        hidden_features (int): Number of hidden features (default: None, same as in_features)
        out_features (int): Number of output features (default: None, same as in_features)
        act_layer (nn.Cell): Activation function (default: nn.GELU)
        drop (float): Dropout probability (default: 0.0)
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Dense(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()

        self.fc2 = nn.Dense(hidden_features, out_features)

        self.drop = nn.Dropout(p=drop)
        self.apply(init_weights)

    def construct(self, x, h, w):
        """construct"""
        x = self.fc1(x)
        x = self.dwconv(x, h, w)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Cell):
    """
    Generic Transformer Block for all stages of the network architecture.

    Args:
        dim (int): Dimension of input features
        num_heads (int): Number of attention heads
        spatial_dims (tuple): Spatial dimensions (H, W) for current stage
        mlp_ratio (float): Ratio for hidden dimension in MLP (default: 4.0)
        qkv_bias (bool): Whether to include bias in QKV projection (default: False)
        qk_scale (float): Scaling factor for QK dot product (default: None)
        drop (float): Dropout rate for projection layers (default: 0.0)
        attn_drop (float): Dropout rate for attention weights (default: 0.0)
        drop_path (float): Drop path probability for this block (default: 0.0)
        act_layer (nn.Cell): Activation function (default: nn.GELU)
        sr_ratio (int): Spatial reduction ratio for attention (default: 1)
    """
    def __init__(
            self,
            dim,
            num_heads,
            spatial_dims,
            qkv_bias=False,
            qk_scale=None,
            drop=0.0,
            attn_drop=0.0,
            block_drop_path=0.0,
            act_layer=nn.GELU,
            sr_ratio=1,
            mlp_ratio=4.
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm((dim,))
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio,
        )
        self.norm2 = nn.LayerNorm((dim,))
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )
        self.drop_path = (
            DropPath(drop_prob=block_drop_path) if block_drop_path > 0.0 else nn.Identity()
        )
        self.spatial_dims = spatial_dims
        self.apply(init_weights)

    def construct(self, x):
        h, w = self.spatial_dims
        x = self.norm1(x)
        x = x + self.attn(x, h, w)
        x = x + self.mlp(self.norm2(x), h, w)
        return x


class MixVisionTransformer(nn.Cell):
    """Hierarchical Vision Transformer backbone with multi-scale feature extraction.

    Implements a multi-stage transformer architecture with spatial reduction for
    efficient high-resolution processing. Produces feature maps at multiple scales
    suitable for dense prediction tasks like segmentation.

    Args:
        in_chans (int): Input channels. Default: 6 (RGB + auxiliary)
        num_classes (int): Output classes for classification head. Default: 1000
        embed_dims (list): Feature dimensions for each stage. Default: [64, 128, 320, 512]
        num_heads (list): Attention heads per stage. Default: [1, 2, 4, 8]
        mlp_ratios (list): MLP expansion ratios. Default: [4, 4, 4, 4]
        qkv_bias (bool): Enable bias in QKV projections. Default: False
        qk_scale (float): Custom QK scaling factor. Default: None
        drop_rate (float): General dropout rate. Default: 0.0
        attn_drop_rate (float): Attention dropout rate. Default: 0.0
        drop_path_rate (float): Stochastic depth rate. Default: 0.0
        depths (list): Number of transformer blocks per stage. Default: [3, 4, 6, 3]
        sr_ratios (list): Spatial reduction ratios per stage. Default: [8, 4, 2, 1]

    Input Shape:
        (B, in_chans, H, W)  # Typically 512x512 for segmentation tasks

    Output Shape:
        List of 4 feature maps at different scales:
        [
            (B, embed_dims[0], H/4, W/4),   # Stage1: 128x128
            (B, embed_dims[1], H/8, W/8),   # Stage2: 64x64
            (B, embed_dims[2], H/16, W/16), # Stage3: 32x32
            (B, embed_dims[3], H/32, W/32)  # Stage4: 16x16
        ]
    """
    def __init__(
            self,
            in_chans=6,
            num_classes=1000,
            embed_dims=None,
            num_heads=None,
            mlp_ratios=None,
            qkv_bias=False,
            qk_scale=None,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            norm_layer=nn.LayerNorm,
            depths=None,
            sr_ratios=None,
            high=None,
            weight=None
    ):
        super().__init__()
        self.embed_dims = embed_dims if embed_dims is not None else [64, 128, 320, 512]
        self.num_heads = num_heads if num_heads is not None else [1, 2, 4, 8]
        self.mlp_ratios = mlp_ratios if mlp_ratios is not None else [4, 4, 4, 4]
        self.sr_ratios = sr_ratios if sr_ratios is not None else [8, 4, 2, 1]
        self.num_classes = num_classes
        self.depths = depths if depths is not None else [3, 4, 6, 3]
        self.h = high if high is not None else [500, 250, 125, 63]
        self.w = weight if weight is not None else [500, 250, 125, 63]
        dpr = [x.item() for x in linspace(0, drop_path_rate, sum(self.depths))]
        self.patch_embed1 = OverlapPatchEmbed(
            patch_size=7, stride=4, in_chans=in_chans, embed_dim=self.embed_dims[0]
        )
        current_depth = 0
        current_channels = in_chans

        for i in range(4):
            patch_embed, blocks, norm = self.create_stage(i, current_channels, current_depth,
                                                          qkv_bias, qk_scale, drop_rate,
                                                          attn_drop_rate, dpr, norm_layer)
            setattr(self, f'patch_embed{i+1}', patch_embed)
            setattr(self, f'block{i+1}', blocks)
            setattr(self, f'norm{i+1}', norm)
            current_channels = self.embed_dims[i]
            current_depth += self.depths[i]
        self.apply(init_weights)

    def create_stage(self, index, in_chans, cur_depth,
                     qkv_bias, qk_scale, drop_rate, attn_drop_rate, dpr, norm_layer):
        """create stage"""
        patch_sizes = [7, 3, 3, 3]
        strides = [4, 2, 2, 2]
        patch_embed = OverlapPatchEmbed(
            patch_size=patch_sizes[index],
            stride=strides[index],
            in_chans=in_chans,
            embed_dim=self.embed_dims[index]
        )
        blocks = nn.CellList([
            Block(
                dim=self.embed_dims[index],
                spatial_dims=(self.h[index], self.w[index]),
                num_heads=self.num_heads[index],
                mlp_ratio=self.mlp_ratios[index],
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                block_drop_path=dpr[cur_depth + i],
                sr_ratio=self.sr_ratios[index],
            )
            for i in range(self.depths[index])
        ])
        norm = norm_layer((self.embed_dims[index],))

        return patch_embed, blocks, norm

    def _forward_stage(self, x, patch_embed, blocks, norm, h, w, b):
        x = patch_embed(x)
        for blk in blocks:
            x = blk(x)
        x = norm(x)
        x = x.reshape(b, h, w, -1).permute(0, 3, 1, 2)
        return x

    def construct(self, x):
        """construct"""
        b = x.shape[0]
        outs = []

        stages = [
            (self.patch_embed1, self.block1, self.norm1, self.h[0], self.w[0]),
            (self.patch_embed2, self.block2, self.norm2, self.h[1], self.w[1]),
            (self.patch_embed3, self.block3, self.norm3, self.h[2], self.w[2]),
            (self.patch_embed4, self.block4, self.norm4, self.h[3], self.w[3]),
        ]

        for patch_embed, blocks, norm, h, w in stages:
            x = self._forward_stage(x, patch_embed, blocks, norm, h, w, b)
            outs.append(x)

        return outs
