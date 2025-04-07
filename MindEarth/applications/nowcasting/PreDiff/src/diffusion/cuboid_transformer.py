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
"""The Substructure of cuboid_transformer_unet"""
from functools import lru_cache
from collections import OrderedDict

import mindspore as ms
from mindspore import nn, ops, Parameter
from mindspore.common.initializer import initializer, TruncatedNormal

from src.utils import (
    get_activation,
    get_norm_layer,
    generalize_padding,
    generalize_unpadding,
    apply_initialization,
)


class PosEmbed(nn.Cell):
    """
    Spatiotemporal positional embedding layer combining temporal, height, and width embeddings.
    """
    def __init__(self, embed_dim, max_t, max_h, max_w):
        """
        Initialize positional embedding with separate temporal/spatial components.
        Args:
            embed_dim (int): Dimensionality of the embedding vectors.
            maxT (int): Maximum temporal length (number of time steps).
            maxH (int): Maximum height dimension size.
            maxW (int): Maximum width dimension size.
        """
        super().__init__()
        self.embed_dim = embed_dim
        # spatiotemporal learned positional embedding
        self.t_embed = nn.Embedding(vocab_size=max_t, embedding_size=embed_dim)
        self.h_embed = nn.Embedding(vocab_size=max_h, embedding_size=embed_dim)
        self.w_embed = nn.Embedding(vocab_size=max_w, embedding_size=embed_dim)
        self.reset_parameters()

    def reset_parameters(self):
        for cell in self.cells():
            apply_initialization(cell, embed_mode="0")

    def construct(self, x):
        """Forward pass of positional embedding.

        Args:
            x (Tensor): Input tensor of shape (B, T, H, W, C)

        Returns:
            Tensor: Output tensor with added positional embeddings
        """

        _, t, h, w, _ = x.shape

        t_idx = ops.arange(t)
        h_idx = ops.arange(h)
        w_idx = ops.arange(w)
        return (
            x
            + self.t_embed(t_idx).reshape(t, 1, 1, self.embed_dim)
            + self.h_embed(h_idx).reshape(1, h, 1, self.embed_dim)
            + self.w_embed(w_idx).reshape(1, 1, w, self.embed_dim)
        )


class PositionwiseFFN(nn.Cell):
    """The Position-wise Feed-Forward Network layer used in Transformer architectures.

    This implements a two-layer MLP with optional gating mechanism and normalization.
    The processing order depends on the pre_norm parameter:

    If pre_norm is True:
        norm(data) -> fc1 -> act -> act_dropout -> fc2 -> dropout -> residual_add(+data)
    Else:
        data -> fc1 -> act -> act_dropout -> fc2 -> dropout -> norm(residual_add(+data))

    When gated projection is enabled, uses:
        fc1_1 * act(fc1_2(data)) for the first projection
    """

    def __init__(
            self,
            units: int = 512,
            hidden_size: int = 2048,
            activation_dropout: float = 0.0,
            dropout: float = 0.1,
            gated_proj: bool = False,
            activation="relu",
            normalization: str = "layer_norm",
            layer_norm_eps: float = 1e-5,
            pre_norm: bool = False,
            linear_init_mode="0",
            ffn2_linear_init_mode="2",
            norm_init_mode="0",
    ):
        super().__init__()
        self.linear_init_mode = linear_init_mode
        self.ffn2_linear_init_mode = ffn2_linear_init_mode
        self.norm_init_mode = norm_init_mode

        self._pre_norm = pre_norm
        self._gated_proj = gated_proj
        self._kwargs = OrderedDict(
            [
                ("units", units),
                ("hidden_size", hidden_size),
                ("activation_dropout", activation_dropout),
                ("activation", activation),
                ("dropout", dropout),
                ("normalization", normalization),
                ("layer_norm_eps", layer_norm_eps),
                ("gated_proj", gated_proj),
                ("pre_norm", pre_norm),
            ]
        )
        self.dropout_layer = nn.Dropout(p=dropout)
        self.activation_dropout_layer = nn.Dropout(p=activation_dropout)
        self.ffn_1 = nn.Dense(
            in_channels=units, out_channels=hidden_size, has_bias=True
        )
        if self._gated_proj:
            self.ffn_1_gate = nn.Dense(
                in_channels=units, out_channels=hidden_size, has_bias=True
            )
        self.activation = get_activation(activation)
        self.ffn_2 = nn.Dense(
            in_channels=hidden_size, out_channels=units, has_bias=True
        )
        self.layer_norm = get_norm_layer(
            norm_type=normalization, in_channels=units, epsilon=layer_norm_eps
        )
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize all sublayers with specified initialization modes."""
        apply_initialization(self.ffn_1, linear_mode=self.linear_init_mode)
        if self._gated_proj:
            apply_initialization(self.ffn_1_gate, linear_mode=self.linear_init_mode)
        apply_initialization(self.ffn_2, linear_mode=self.ffn2_linear_init_mode)
        apply_initialization(self.layer_norm, norm_mode=self.norm_init_mode)

    def construct(self, data):
        """
        Forward pass of the Position-wise FFN.

        Args:
            data: Input tensor of shape (batch_size, sequence_length, units)

        Returns:
            Output tensor of same shape as input with transformed features
        """
        residual = data
        if self._pre_norm:
            data = self.layer_norm(data)
        if self._gated_proj:
            out = self.activation(self.ffn_1_gate(data)) * self.ffn_1(data)
        else:
            out = self.activation(self.ffn_1(data))
        out = self.activation_dropout_layer(out)
        out = self.ffn_2(out)
        out = self.dropout_layer(out)
        out = out + residual
        if not self._pre_norm:
            out = self.layer_norm(out)
        return out


class PatchMerging3D(nn.Cell):
    """3D Patch Merging Layer for spatial-temporal feature downsampling.
    This layer merges patches in 3D (temporal, height, width) and applies a linear transformation
    to reduce the feature dimension while increasing the channel dimension.
    """

    def __init__(
            self,
            dim,
            out_dim=None,
            downsample=(1, 2, 2),
            norm_layer="layer_norm",
            padding_type="nearest",
            linear_init_mode="0",
            norm_init_mode="0",
    ):
        super().__init__()
        self.linear_init_mode = linear_init_mode
        self.norm_init_mode = norm_init_mode
        self.dim = dim
        if out_dim is None:
            out_dim = max(downsample) * dim
        self.out_dim = out_dim
        self.downsample = downsample
        self.padding_type = padding_type
        self.reduction = nn.Dense(
            downsample[0] * downsample[1] * downsample[2] * dim, out_dim, has_bias=False
        )
        self.norm = get_norm_layer(
            norm_layer, in_channels=downsample[0] * downsample[1] * downsample[2] * dim
        )
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize all sublayers with specified initialization modes."""
        for cell in self.cells():
            apply_initialization(
                cell, linear_mode=self.linear_init_mode, norm_mode=self.norm_init_mode
            )

    def get_out_shape(self, data_shape):
        """
        Calculate the output shape given input dimensions.

        Args:
            data_shape: Input shape tuple (T, H, W, C_in)

        Returns:
            Tuple of output shape (T_out, H_out, W_out, C_out)
        """
        t, h, w, _ = data_shape
        pad_t = (self.downsample[0] - t % self.downsample[0]) % self.downsample[0]
        pad_h = (self.downsample[1] - h % self.downsample[1]) % self.downsample[1]
        pad_w = (self.downsample[2] - w % self.downsample[2]) % self.downsample[2]
        return (
            (t + pad_t) // self.downsample[0],
            (h + pad_h) // self.downsample[1],
            (w + pad_w) // self.downsample[2],
            self.out_dim,
        )

    def construct(self, x):
        """
        Forward pass of the 3D Patch Merging layer.

        Args:
            x: Input tensor of shape (B, T, H, W, C)

        Returns:
            Output tensor of shape:
            (B, T//downsample[0], H//downsample[1], W//downsample[2], out_dim)
        """
        b, t, h, w, c = x.shape

        # padding
        pad_t = (self.downsample[0] - t % self.downsample[0]) % self.downsample[0]
        pad_h = (self.downsample[1] - h % self.downsample[1]) % self.downsample[1]
        pad_w = (self.downsample[2] - w % self.downsample[2]) % self.downsample[2]
        if pad_h or pad_h or pad_w:
            t += pad_t
            h += pad_h
            w += pad_w
            x = generalize_padding(
                x, pad_t, pad_h, pad_w, padding_type=self.padding_type
            )

        x = (
            x.reshape(
                (
                    b,
                    t // self.downsample[0],
                    self.downsample[0],
                    h // self.downsample[1],
                    self.downsample[1],
                    w // self.downsample[2],
                    self.downsample[2],
                    c,
                )
            )
            .permute(0, 1, 3, 5, 2, 4, 6, 7)
            .reshape(
                b,
                t // self.downsample[0],
                h // self.downsample[1],
                w // self.downsample[2],
                self.downsample[0] * self.downsample[1] * self.downsample[2] * c,
            )
        )
        x = self.norm(x)
        x = self.reduction(x)

        return x


class Upsample3DLayer(nn.Cell):
    """3D Upsampling Layer combining interpolation and convolution.

    Performs spatial upsampling (with optional temporal upsampling) followed by convolution.
    The operation consists of:
    1. Spatial upsampling using nearest-neighbor interpolation
    2. 2D or 3D convolution to refine features and adjust channel dimensions

    Note: Currently only implements 2D upsampling (spatial only)
    """

    def __init__(
            self,
            dim,
            out_dim,
            target_size,
            kernel_size=3,
            conv_init_mode="0",
    ):
        super().__init__()
        self.conv_init_mode = conv_init_mode
        self.target_size = target_size
        self.out_dim = out_dim
        self.up = nn.Upsample(size=(target_size[1], target_size[2]), mode="nearest")
        self.conv = nn.Conv2d(
            in_channels=dim,
            out_channels=out_dim,
            kernel_size=(kernel_size, kernel_size),
            padding=kernel_size // 2,
            has_bias=True,
            pad_mode="pad",
        )
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize all sublayers with specified initialization modes."""
        for cell in self.cells():
            apply_initialization(cell, conv_mode=self.conv_init_mode)

    def construct(self, x):
        """Forward pass of the 3D Upsampling layer."""
        b, t, h, w, c = x.shape
        assert self.target_size[0] == t
        x = x.reshape(b * t, h, w, c).permute(0, 3, 1, 2)
        x = self.up(x)
        return (
            self.conv(x)
            .permute(0, 2, 3, 1)
            .reshape((b,) + self.target_size + (self.out_dim,))
        )


def cuboid_reorder(data, cuboid_size, strategy):
    """Reorder the tensor into (B, num_cuboids, bT * bH * bW, C)

    We assume that the tensor shapes are divisible to the cuboid sizes.

    Parameters
    ----------
    data
        The input data
    cuboid_size
        The size of the cuboid
    strategy
        The cuboid strategy

    Returns
    -------
    reordered_data
        Shape will be (B, num_cuboids, bT * bH * bW, C)
        num_cuboids = T / bT * H / bH * W / bW
    """
    b, t, h, w, c = data.shape
    num_cuboids = t // cuboid_size[0] * h // cuboid_size[1] * w // cuboid_size[2]
    cuboid_volume = cuboid_size[0] * cuboid_size[1] * cuboid_size[2]
    intermediate_shape = []

    nblock_axis = []
    block_axis = []
    for i, (block_size, total_size, ele_strategy) in enumerate(
            zip(cuboid_size, (t, h, w), strategy)
    ):
        if ele_strategy == "l":
            intermediate_shape.extend([total_size // block_size, block_size])
            nblock_axis.append(2 * i + 1)
            block_axis.append(2 * i + 2)
        elif ele_strategy == "d":
            intermediate_shape.extend([block_size, total_size // block_size])
            nblock_axis.append(2 * i + 2)
            block_axis.append(2 * i + 1)
        else:
            raise NotImplementedError

    a = (b,) + tuple(intermediate_shape) + (c,)
    data = data.reshape(a)
    reordered_data = data.permute((0,) + tuple(nblock_axis) + tuple(block_axis) + (7,))
    reordered_data = reordered_data.reshape((b, num_cuboids, cuboid_volume, c))
    return reordered_data


def cuboid_reorder_reverse(data, cuboid_size, strategy, orig_data_shape):
    """Reverse the reordered cuboid back to the original space

    Parameters
    ----------
    data
    cuboid_size
    strategy
    orig_data_shape

    Returns
    -------
    data
        The recovered data
    """
    b, _, _, c = data.shape
    t, h, w = orig_data_shape

    permutation_axis = [0]
    for i, (_, _, ele_strategy) in enumerate(
            zip(cuboid_size, (t, h, w), strategy)
    ):
        if ele_strategy == "l":
            permutation_axis.append(i + 1)
            permutation_axis.append(i + 4)
        elif ele_strategy == "d":
            permutation_axis.append(i + 4)
            permutation_axis.append(i + 1)
        else:
            raise NotImplementedError
    permutation_axis.append(7)
    data = data.reshape(
        b,
        t // cuboid_size[0],
        h // cuboid_size[1],
        w // cuboid_size[2],
        cuboid_size[0],
        cuboid_size[1],
        cuboid_size[2],
        c,
    )
    data = data.permute(permutation_axis)
    data = data.reshape((b, t, h, w, c))
    return data


@lru_cache()
def compute_cuboid_self_attention_mask(
        data_shape, cuboid_size, shift_size, strategy, padding_type
):
    """compute_cuboid_self_attention_mask"""
    t, h, w = data_shape
    pad_t = (cuboid_size[0] - t % cuboid_size[0]) % cuboid_size[0]
    pad_h = (cuboid_size[1] - h % cuboid_size[1]) % cuboid_size[1]
    pad_w = (cuboid_size[2] - w % cuboid_size[2]) % cuboid_size[2]

    data_mask = None
    if pad_t > 0 or pad_h > 0 or pad_w > 0:
        if padding_type == "ignore":
            data_mask = ops.ones((1, t, h, w, 1), dtype=ms.bool_)
            data_mask = ops.pad(
                data_mask, ((0, 0), (0, pad_t), (0, pad_h), (0, pad_w), (0, 0))
            )
    else:
        data_mask = ops.ones((1, t + pad_t, h + pad_h, w + pad_w, 1), dtype=ms.bool_)

    if any(i > 0 for i in shift_size):
        if padding_type == "ignore":
            data_mask = ops.roll(
                data_mask,
                shifts=(-shift_size[0], -shift_size[1], -shift_size[2]),
                dims=(1, 2, 3),
            )
    t_padded, h_padded, w_padded = t + pad_t, h + pad_h, w + pad_w
    if t_padded <= 0 or h_padded <= 0 or w_padded <= 0:
        raise ValueError(
            f"invalid padded dimensions: t={t_padded}, h={h_padded}, w={w_padded}"
        )

    shift_mask = ops.zeros((1, t_padded, h_padded, w_padded, 1))
    cnt = 0
    t_slices = (
        [
            slice(0, cuboid_size[0]),
            slice(cuboid_size[0] - shift_size[0], t_padded - shift_size[0]),
            slice(t_padded - cuboid_size[0], t_padded),
        ]
        if shift_size[0] > 0
        else [slice(0, t_padded)]
    )

    h_slices = (
        [
            slice(0, cuboid_size[1]),
            slice(cuboid_size[1] - shift_size[1], h_padded - shift_size[1]),
            slice(h_padded - cuboid_size[1], h_padded),
        ]
        if shift_size[1] > 0
        else [slice(0, h_padded)]
    )

    w_slices = (
        [
            slice(0, cuboid_size[2]),
            slice(cuboid_size[2] - shift_size[2], w_padded - shift_size[2]),
            slice(w_padded - cuboid_size[2], w_padded),
        ]
        if shift_size[2] > 0
        else [slice(0, w_padded)]
    )

    for t in t_slices:
        for h in h_slices:
            for w in w_slices:
                shift_mask[:, t, h, w, :] = cnt
                cnt += 1

    shift_mask = cuboid_reorder(shift_mask, cuboid_size, strategy=strategy)
    shift_mask = shift_mask.squeeze(-1).squeeze(0)  # num_cuboids, cuboid_volume
    attn_mask = (shift_mask.unsqueeze(1) - shift_mask.unsqueeze(2)) == 0

    if padding_type == "ignore":
        if padding_type == "ignore":
            data_mask = cuboid_reorder(data_mask, cuboid_size, strategy=strategy)
            data_mask = data_mask.squeeze(-1).squeeze(0)
            attn_mask = data_mask.unsqueeze(1) * data_mask.unsqueeze(2) * attn_mask

    return attn_mask


def masked_softmax(att_score, mask, axis: int = -1):
    """Computes softmax while ignoring masked elements with broadcastable masks.

    Parameters
    ----------
    att_score : Tensor
    mask : Tensor or None
        Binary mask tensor of shape (..., length, ...) where:
        - 1 indicates unmasked (valid) elements
        - 0 indicates masked elements
        Must be broadcastable with att_score
    axis : int, optional

    Returns
    -------
    Tensor
        Softmax output of same shape as input att_score, with:
        - Proper attention weights for unmasked elements
        - Zero weights for masked elements
    """
    if mask is not None:
        # Fill in the masked scores with a very small value
        if att_score.dtype == ms.float16:
            att_score = att_score.masked_fill(ops.logical_not(mask), -1e4)
        else:
            att_score = att_score.masked_fill(ops.logical_not(mask), -1e18)
        att_weights = ops.softmax(att_score, axis=axis) * mask
    else:
        att_weights = ops.softmax(att_score, axis=axis)
    return att_weights


def update_cuboid_size_shift_size(data_shape, cuboid_size, shift_size, strategy):
    """Update the

    Parameters
    ----------
    data_shape
        The shape of the data
    cuboid_size
        Size of the cuboid
    shift_size
        Size of the shift
    strategy
        The strategy of attention

    Returns
    -------
    new_cuboid_size
        Size of the cuboid
    new_shift_size
        Size of the shift
    """
    new_cuboid_size = list(cuboid_size)
    new_shift_size = list(shift_size)
    for i in range(len(data_shape)):
        if strategy[i] == "d":
            new_shift_size[i] = 0
        if data_shape[i] <= cuboid_size[i]:
            new_cuboid_size[i] = data_shape[i]
            new_shift_size[i] = 0
    return tuple(new_cuboid_size), tuple(new_shift_size)


class CuboidSelfAttentionLayer(nn.Cell):
    """
    A self-attention layer designed for 3D data (e.g., video or 3D images),
    implementing cuboid-based attention with optional global vectors and relative position encoding.
    """
    def __init__(
            self,
            dim,
            num_heads,
            cuboid_size=(2, 7, 7),
            shift_size=(0, 0, 0),
            strategy=("l", "l", "l"),
            padding_type="ignore",
            qkv_bias=False,
            qk_scale=None,
            attn_drop=0.0,
            proj_drop=0.0,
            use_final_proj=True,
            norm_layer="layer_norm",
            use_global_vector=False,
            use_global_self_attn=False,
            separate_global_qkv=False,
            global_dim_ratio=1,
            use_relative_pos=True,
            attn_linear_init_mode="0",
            ffn_linear_init_mode="2",
            norm_init_mode="0",
    ):
        """Initialize the CuboidSelfAttentionLayer.

        Args:
            dim (int): Input feature dimension.
            num_heads (int): Number of attention heads.
            cuboid_size (tuple): 3D dimensions (T, H, W) of the cuboid blocks.
            shift_size (tuple): Shift sizes for each dimension to avoid attention blindness.
            strategy (tuple): Strategy for each dimension ('l' for local, 'g' for global).
            padding_type (str): Padding method for attention computation ("ignore", "zeros", "nearest").
            qkv_bias (bool): Whether to include bias in QKV projections.
            qk_scale (float, optional): Scaling factor for QK dot product. Defaults to head_dim**-0.5.
            attn_drop (float): Dropout rate after attention softmax.
            proj_drop (float): Dropout rate after output projection.
            use_final_proj (bool): Whether to apply the final linear projection.
            norm_layer (str): Type of normalization layer ("layer_norm", etc.).
            use_global_vector (bool): Whether to include a global vector in attention.
            use_global_self_attn (bool): Whether to apply self-attention to global vectors.
            separate_global_qkv (bool): Whether to use separate QKV for global vectors.
            global_dim_ratio (int): Dimension ratio for global vector (requires separate_global_qkv=True if !=1).
            use_relative_pos (bool): Whether to use relative position embeddings.
            attn_linear_init_mode (str): Initialization mode for attention linear layers.
            ffn_linear_init_mode (str): Initialization mode for FFN linear layers.
            norm_init_mode (str): Initialization mode for normalization layers.
        """
        super().__init__()
        # initialization
        self.attn_linear_init_mode = attn_linear_init_mode
        self.ffn_linear_init_mode = ffn_linear_init_mode
        self.norm_init_mode = norm_init_mode

        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.dim = dim
        self.cuboid_size = cuboid_size
        self.shift_size = shift_size
        self.strategy = strategy
        self.padding_type = padding_type
        self.use_final_proj = use_final_proj
        self.use_relative_pos = use_relative_pos
        # global vectors
        self.use_global_vector = use_global_vector
        self.use_global_self_attn = use_global_self_attn
        self.separate_global_qkv = separate_global_qkv
        self.global_dim_ratio = global_dim_ratio
        assert self.padding_type in ["ignore", "zeros", "nearest"]
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        if self.use_relative_pos:
            self.relative_position_bias_table = Parameter(
                initializer(
                    TruncatedNormal(sigma=0.02),
                    [
                        (2 * cuboid_size[0] - 1)
                        * (2 * cuboid_size[1] - 1)
                        * (2 * cuboid_size[2] - 1),
                        num_heads,
                    ],
                    ms.float32,
                )
            )
            self.relative_position_bias_table.name = "relative_position_bias_table"
            coords_t = ops.arange(self.cuboid_size[0])
            coords_h = ops.arange(self.cuboid_size[1])
            coords_w = ops.arange(self.cuboid_size[2])
            coords = ops.stack(ops.meshgrid(coords_t, coords_h, coords_w))

            coords_flatten = ops.flatten(coords, start_dim=1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0)
            relative_coords[:, :, 0] += self.cuboid_size[0] - 1
            relative_coords[:, :, 1] += self.cuboid_size[1] - 1
            relative_coords[:, :, 2] += self.cuboid_size[2] - 1

            relative_coords[:, :, 0] *= (2 * self.cuboid_size[1] - 1) * (
                2 * self.cuboid_size[2] - 1
            )
            relative_coords[:, :, 1] *= 2 * self.cuboid_size[2] - 1
            relative_position_index = relative_coords.sum(-1)
            self.relative_position_index = Parameter(
                relative_position_index,
                name="relative_position_index",
                requires_grad=False,
            )
        self.qkv = nn.Dense(dim, dim * 3, has_bias=qkv_bias)
        self.attn_drop = nn.Dropout(p=attn_drop)

        if use_final_proj:
            self.proj = nn.Dense(dim, dim)
            self.proj_drop = nn.Dropout(p=proj_drop)

            if self.use_global_vector:
                self.global_proj = nn.Dense(
                    in_channels=global_dim_ratio * dim,
                    out_channels=global_dim_ratio * dim,
                )

        self.norm = get_norm_layer(norm_layer, in_channels=dim)
        if self.use_global_vector:
            self.global_vec_norm = get_norm_layer(
                norm_layer, in_channels=global_dim_ratio * dim
            )

        self.reset_parameters()

    def reset_parameters(self):
        '''set_parameters'''
        apply_initialization(self.qkv, linear_mode=self.attn_linear_init_mode)
        if self.use_final_proj:
            apply_initialization(self.proj, linear_mode=self.ffn_linear_init_mode)
        apply_initialization(self.norm, norm_mode=self.norm_init_mode)
        if self.use_global_vector:
            if self.separate_global_qkv:
                apply_initialization(
                    self.l2g_q_net, linear_mode=self.attn_linear_init_mode
                )
                apply_initialization(
                    self.l2g_global_kv_net, linear_mode=self.attn_linear_init_mode
                )
                apply_initialization(
                    self.g2l_global_q_net, linear_mode=self.attn_linear_init_mode
                )
                apply_initialization(
                    self.g2l_k_net, linear_mode=self.attn_linear_init_mode
                )
                apply_initialization(
                    self.g2l_v_net, linear_mode=self.attn_linear_init_mode
                )
                if self.use_global_self_attn:
                    apply_initialization(
                        self.g2g_global_qkv_net, linear_mode=self.attn_linear_init_mode
                    )
            else:
                apply_initialization(
                    self.global_qkv, linear_mode=self.attn_linear_init_mode
                )
            apply_initialization(self.global_vec_norm, norm_mode=self.norm_init_mode)

    def construct(self, x):
        """
        Constructs the output by applying normalization, padding, shifting, and attention mechanisms.

        Parameters:
        - x (Tensor): Input tensor with shape (batch, time, height, width, channels).
        - global_vectors (Tensor, optional): Global vectors used in global-local interactions. Defaults to None.

        Returns:
        - Tensor: Processed tensor after applying all transformations.
        - Tensor: Updated global vectors if global vectors are used; otherwise, returns only the processed tensor.
        """
        x = self.norm(x)
        batch, time, height, width, channels = x.shape
        assert channels == self.dim
        cuboid_size, shift_size = update_cuboid_size_shift_size(
            (time, height, width), self.cuboid_size, self.shift_size, self.strategy
        )
        pad_t = (cuboid_size[0] - time % cuboid_size[0]) % cuboid_size[0]
        pad_h = (cuboid_size[1] - height % cuboid_size[1]) % cuboid_size[1]
        pad_w = (cuboid_size[2] - width % cuboid_size[2]) % cuboid_size[2]
        x = generalize_padding(x, pad_t, pad_h, pad_w, self.padding_type)
        if any(i > 0 for i in shift_size):
            shifted_x = ops.roll(
                x,
                shifts=(-shift_size[0], -shift_size[1], -shift_size[2]),
                dims=(1, 2, 3),
            )
        else:
            shifted_x = x
        reordered_x = cuboid_reorder(
            shifted_x, cuboid_size=cuboid_size, strategy=self.strategy
        )
        _, num_cuboids, cuboid_volume, _ = reordered_x.shape
        attn_mask = compute_cuboid_self_attention_mask(
            (time, height, width),
            cuboid_size,
            shift_size=shift_size,
            strategy=self.strategy,
            padding_type=self.padding_type,
        )
        head_c = channels // self.num_heads
        qkv = (
            self.qkv(reordered_x)
            .reshape(batch, num_cuboids, cuboid_volume, 3, self.num_heads, head_c)
            .permute(3, 0, 4, 1, 2, 5)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )
        q = q * self.scale
        attn_score = q @ k.swapaxes(-2, -1)
        if self.use_relative_pos:
            relative_position_bias = self.relative_position_bias_table[
                self.relative_position_index[:cuboid_volume, :cuboid_volume].reshape(-1)
            ].reshape(cuboid_volume, cuboid_volume, -1)
            relative_position_bias = relative_position_bias.permute(2, 0, 1).unsqueeze(
                1
            )
            attn_score = attn_score + relative_position_bias
        attn_score = masked_softmax(attn_score, mask=attn_mask)
        attn_score = self.attn_drop(attn_score)
        reordered_x = (
            (attn_score @ v)
            .permute(0, 2, 3, 1, 4)
            .reshape(batch, num_cuboids, cuboid_volume, self.dim)
        )

        if self.use_final_proj:
            reordered_x = self.proj_drop(self.proj(reordered_x))
            if self.use_global_vector:
                new_global_vector = self.proj_drop(self.global_proj(new_global_vector))
        shifted_x = cuboid_reorder_reverse(
            reordered_x,
            cuboid_size=cuboid_size,
            strategy=self.strategy,
            orig_data_shape=(time + pad_t, height + pad_h, width + pad_w),
        )
        if any(i > 0 for i in shift_size):
            x = ops.roll(
                shifted_x,
                shifts=(shift_size[0], shift_size[1], shift_size[2]),
                dims=(1, 2, 3),
            )
        else:
            x = shifted_x
        x = generalize_unpadding(
            x, pad_t=pad_t, pad_h=pad_h, pad_w=pad_w, padding_type=self.padding_type
        )
        if self.use_global_vector:
            return x, new_global_vector
        return x


class StackCuboidSelfAttentionBlock(nn.Cell):
    """

    - "use_inter_ffn" is True
        x --> attn1 --> ffn1 --> attn2 --> ... --> ffn_k --> out
    - "use_inter_ffn" is False
        x --> attn1 --> attn2 --> ... attnk --> ffnk --> out
    If we have enabled global memory vectors, each attention will be a

    """

    def __init__(
            self,
            dim=None,
            num_heads=None,
            block_cuboid_size=None,
            block_shift_size=None,
            block_strategy=None,
            padding_type="ignore",
            qkv_bias=False,
            qk_scale=None,
            attn_drop=0.0,
            proj_drop=0.0,
            ffn_drop=0.0,
            activation="leaky",
            gated_ffn=False,
            norm_layer="layer_norm",
            use_inter_ffn=False,
            use_global_vector=False,
            use_global_vector_ffn=True,
            use_global_self_attn=False,
            separate_global_qkv=False,
            global_dim_ratio=1,
            use_relative_pos=True,
            use_final_proj=True,
            # initialization
            attn_linear_init_mode="0",
            ffn_linear_init_mode="0",
            ffn2_linear_init_mode="2",
            attn_proj_linear_init_mode="2",
            norm_init_mode="0",
    ):
        super().__init__()
        # initialization
        self.attn_linear_init_mode = attn_linear_init_mode
        self.ffn_linear_init_mode = ffn_linear_init_mode
        self.attn_proj_linear_init_mode = attn_proj_linear_init_mode
        self.norm_init_mode = norm_init_mode
        self.num_attn = len(block_cuboid_size)
        self.use_inter_ffn = use_inter_ffn
        # global vectors
        self.use_global_vector = use_global_vector
        self.use_global_vector_ffn = use_global_vector_ffn
        self.use_global_self_attn = use_global_self_attn
        self.global_dim_ratio = global_dim_ratio

        if self.use_inter_ffn:
            self.ffn_l = nn.CellList(
                [
                    PositionwiseFFN(
                        units=dim,
                        hidden_size=4 * dim,
                        activation_dropout=ffn_drop,
                        dropout=ffn_drop,
                        gated_proj=gated_ffn,
                        activation=activation,
                        normalization=norm_layer,
                        pre_norm=True,
                        linear_init_mode=ffn_linear_init_mode,
                        ffn2_linear_init_mode=ffn2_linear_init_mode,
                        norm_init_mode=norm_init_mode,
                    )
                    for _ in range(self.num_attn)
                ]
            )
            if self.use_global_vector_ffn and self.use_global_vector:
                self.global_ffn_l = nn.CellList(
                    [
                        PositionwiseFFN(
                            units=global_dim_ratio * dim,
                            hidden_size=global_dim_ratio * 4 * dim,
                            activation_dropout=ffn_drop,
                            dropout=ffn_drop,
                            gated_proj=gated_ffn,
                            activation=activation,
                            normalization=norm_layer,
                            pre_norm=True,
                            linear_init_mode=ffn_linear_init_mode,
                            ffn2_linear_init_mode=ffn2_linear_init_mode,
                            norm_init_mode=norm_init_mode,
                        )
                        for _ in range(self.num_attn)
                    ]
                )
        else:
            self.ffn_l = nn.CellList(
                [
                    PositionwiseFFN(
                        units=dim,
                        hidden_size=4 * dim,
                        activation_dropout=ffn_drop,
                        dropout=ffn_drop,
                        gated_proj=gated_ffn,
                        activation=activation,
                        normalization=norm_layer,
                        pre_norm=True,
                        linear_init_mode=ffn_linear_init_mode,
                        ffn2_linear_init_mode=ffn2_linear_init_mode,
                        norm_init_mode=norm_init_mode,
                    )
                ]
            )
            if self.use_global_vector_ffn and self.use_global_vector:
                self.global_ffn_l = nn.CellList(
                    [
                        PositionwiseFFN(
                            units=global_dim_ratio * dim,
                            hidden_size=global_dim_ratio * 4 * dim,
                            activation_dropout=ffn_drop,
                            dropout=ffn_drop,
                            gated_proj=gated_ffn,
                            activation=activation,
                            normalization=norm_layer,
                            pre_norm=True,
                            linear_init_mode=ffn_linear_init_mode,
                            ffn2_linear_init_mode=ffn2_linear_init_mode,
                            norm_init_mode=norm_init_mode,
                        )
                    ]
                )
        self.attn_l = nn.CellList(
            [
                CuboidSelfAttentionLayer(
                    dim=dim,
                    num_heads=num_heads,
                    cuboid_size=ele_cuboid_size,
                    shift_size=ele_shift_size,
                    strategy=ele_strategy,
                    padding_type=padding_type,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    attn_drop=attn_drop,
                    proj_drop=proj_drop,
                    norm_layer=norm_layer,
                    use_global_vector=use_global_vector,
                    use_global_self_attn=use_global_self_attn,
                    separate_global_qkv=separate_global_qkv,
                    global_dim_ratio=global_dim_ratio,
                    use_relative_pos=use_relative_pos,
                    use_final_proj=use_final_proj,
                    attn_linear_init_mode=attn_linear_init_mode,
                    ffn_linear_init_mode=attn_proj_linear_init_mode,
                    norm_init_mode=norm_init_mode,
                )
                for ele_cuboid_size, ele_shift_size, ele_strategy in zip(
                    block_cuboid_size, block_shift_size, block_strategy
                )
            ]
        )

    def reset_parameters(self):
        for m in self.ffn_l:
            m.reset_parameters()
        if self.use_global_vector_ffn and self.use_global_vector:
            for m in self.global_ffn_l:
                m.reset_parameters()
        for m in self.attn_l:
            m.reset_parameters()

    def construct(self, x, global_vectors=None):
        """
        Constructs the network output by processing input data with attention and feed-forward layers.

        Args:
            x (Tensor): Input data tensor.
            global_vectors (Tensor, optional): Global vectors for contextual processing. Defaults to None.

        Returns:
            Union[Tensor, Tuple[Tensor, Tensor]]:
                - If `global_vectors` is used, returns a tuple (processed_x, updated_global_vectors).
                - Otherwise, returns the processed input tensor x.
        """
        if self.use_inter_ffn:
            if self.use_global_vector:
                for idx, (attn, ffn) in enumerate(zip(self.attn_l, self.ffn_l)):
                    x_out, global_vectors_out = attn(x, global_vectors)
                    x = x + x_out
                    global_vectors = global_vectors + global_vectors_out
                    x = ffn(x)
                    if self.use_global_vector_ffn:
                        global_vectors = self.global_ffn_l[idx](global_vectors)
                return x, global_vectors
            for idx, (attn, ffn) in enumerate(zip(self.attn_l, self.ffn_l)):
                x_ = attn(x)
                x = x + x_
                x = ffn(x)
            return x
        if self.use_global_vector:
            for idx, attn in enumerate(self.attn_l):
                x_out, global_vectors_out = attn(x, global_vectors)
                x = x + x_out
                global_vectors = global_vectors + global_vectors_out
            x = self.ffn_l[0](x)
            if self.use_global_vector_ffn:
                global_vectors = self.global_ffn_l[0](global_vectors)
            return x, global_vectors
        for idx, attn in enumerate(self.attn_l):
            out = attn(x)
            x = x + out
        x = self.ffn_l[0](x)
        return x
