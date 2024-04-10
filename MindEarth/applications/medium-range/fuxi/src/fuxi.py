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
"""The Substructure of FuXiNet"""
import math

import mindspore
from mindspore import nn, ops, Parameter
import mindspore.numpy as mnp
import mindspore.ops.operations as P
import mindspore.common.dtype as mstype


class CubeEmbed(nn.Cell):
    """
    Cube Embedding for high dimension data.
    """
    def __init__(self, in_channels, h_size, w_size, level_feature_size, pressure_level_num, surface_feature_size):
        super().__init__()
        self.in_channels = in_channels
        self.h_size = h_size
        self.w_size = w_size
        self.level_feature_size = level_feature_size
        self.pressure_level_num = pressure_level_num
        self.surface_feature_size = surface_feature_size
        self.layer_norm = nn.LayerNorm([in_channels], epsilon=1e-5)
        self.conv3d_dtype = mstype.float16
        self.cube3d = nn.Conv3d(level_feature_size, in_channels, kernel_size=(2, 4, 4),
                                pad_mode="valid", stride=(2, 4, 4), has_bias=True, dtype=self.conv3d_dtype)
        self.conv2d = nn.Conv2d(surface_feature_size, in_channels, kernel_size=(4, 4),
                                pad_mode="valid", stride=(4, 4), has_bias=True)

    def construct(self, x):
        """CubeEmbed forward function."""
        x_input = x.reshape(1, self.h_size, self.w_size, -1)
        x = x_input[..., :-self.surface_feature_size]
        x_surface = x_input[..., -self.surface_feature_size:]
        x = x.reshape(1, self.h_size, self.w_size, self.level_feature_size, self.pressure_level_num)
        x = x.transpose(0, 3, 4, 1, 2)
        x_surface = x_surface.transpose(0, 3, 1, 2)
        pad_zeros = ops.zeros((1, self.level_feature_size, 1, self.h_size, self.w_size), dtype=x.dtype)
        x = ops.concat((pad_zeros, x), axis=2)
        x = ops.cast(x, self.conv3d_dtype)
        x = self.cube3d(x)
        x = ops.cast(x, x_surface.dtype)
        x_surface = self.conv2d(x_surface)
        x_surface = x_surface.reshape(1, self.in_channels, 1, self.h_size // 4, self.w_size // 4)
        x = ops.concat((x, x_surface), axis=2)
        output = self.layer_norm(x.transpose(0, 2, 3, 4, 1))
        output = output.transpose(0, 4, 1, 2, 3)
        return output


class ResidualBlock(nn.Cell):
    """
    Residual Block in down sample.
    """
    def __init__(self,
                 in_channels,
                 out_channels):
        super().__init__()
        self.conv2d_1 = nn.Conv2d(in_channels=in_channels,
                                  out_channels=out_channels,
                                  kernel_size=3,
                                  padding=1,
                                  pad_mode="pad",
                                  stride=1,
                                  has_bias=True)
        self.group_norm = nn.GroupNorm(8, 8)
        self.conv2d_2 = nn.Conv2d(in_channels=in_channels,
                                  out_channels=out_channels,
                                  kernel_size=3,
                                  padding=1,
                                  pad_mode="pad",
                                  stride=1,
                                  has_bias=True)
        self.silu = nn.SiLU()

    def construct(self, x):
        """Residual Block forward function."""
        x1 = x
        x = self.conv2d_1(x)
        x = self.silu(x)
        x = self.conv2d_2(x)
        output = x + x1
        return output


class DownSample(nn.Cell):
    """Down Sample module."""
    def __init__(self, in_channels=96, out_channels=192):
        super().__init__()
        self.conv2d = nn.Conv2d(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=(3, 3),
                                stride=2,
                                has_bias=True)
        self.residual_block = ResidualBlock(in_channels=out_channels,
                                            out_channels=out_channels)
        self.silu = nn.SiLU()

    def construct(self, x):
        """Down Sampe forward function."""
        _, channels, z_size, h_size, w_size = x.shape
        x = x.squeeze(0).transpose(1, 0, 2, 3)
        x = self.conv2d(x)
        x = self.residual_block(x)
        x = self.silu(x)
        x = x.transpose(0, 2, 3, 1)
        output = x.reshape(1, z_size, h_size // 2, w_size // 2, channels * 2)
        return output


class RelativeBias(nn.Cell):
    """RelativeBias for relative position encoding."""
    def __init__(self, type_windows, num_heads, window_size):
        super(RelativeBias, self).__init__()
        bias = mnp.zeros(((2 * window_size[2] - 1) * window_size[1] * window_size[1] * window_size[0] * window_size[0],
                          type_windows, num_heads), dtype=mstype.float32)
        self.relative_position_bias_table = Parameter(bias, requires_grad=True)
        self.window_size = window_size
        coords_zi = ops.arange(0, window_size[0])
        coords_zj = -ops.arange(0, window_size[0]) * window_size[0]
        coords_hi = ops.arange(0, window_size[1])
        coords_hj = -ops.arange(0, window_size[1]) * window_size[1]
        coords_w = ops.arange(0, window_size[2])
        coords_1 = ops.stack(ops.meshgrid(coords_zi, coords_hi, coords_w, indexing='ij'))
        coords_2 = ops.stack(ops.meshgrid(coords_zj, coords_hj, coords_w, indexing='ij'))
        coords_flatten_1 = ops.flatten(coords_1)
        coords_flatten_2 = ops.flatten(coords_2)
        relative_coords = mnp.expand_dims(coords_flatten_1, axis=-1) - mnp.expand_dims(coords_flatten_2,
                                                                                       axis=1)
        relative_coords = relative_coords.transpose(1, 2, 0).astype(mnp.float32)

        relative_coords[:, :, 2] += window_size[2] - 1
        relative_coords[:, :, 1] *= 2 * window_size[2] - 1
        relative_coords[:, :, 0] *= (2 * window_size[2] - 1) * window_size[1] * window_size[1]
        relative_position_index = relative_coords.sum(-1).astype(mnp.int32)
        self.relative_position_index = Parameter(relative_position_index, requires_grad=False)

        self.type_windows = type_windows
        self.num_heads = num_heads

    def construct(self):
        """Relative Bias forward function."""
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.reshape(-1)].reshape(
            self.window_size[0] * self.window_size[1] * self.window_size[2],
            self.window_size[0] * self.window_size[1] * self.window_size[2], self.type_windows, self.num_heads)
        relative_position_bias = relative_position_bias.transpose(2, 3, 0, 1)
        relative_position_bias = mnp.expand_dims(relative_position_bias, axis=0)
        return relative_position_bias


class WindowAttention11(nn.Cell):
    """Window Attention in Swin Block."""
    def __init__(self, in_channels, num_heads, window_size, input_shape):
        super().__init__()
        self.dim = in_channels
        self.num_heads = num_heads
        self.relative_bias = RelativeBias((input_shape[0] // window_size[0]) * (input_shape[1] // window_size[1]),
                                          num_heads, window_size)
        self.softmax = nn.Softmax(axis=-1)

        self.q = nn.Dense(in_channels=self.dim, out_channels=self.dim, has_bias=True)
        self.k = nn.Dense(in_channels=self.dim, out_channels=self.dim, has_bias=True)
        self.v = nn.Dense(in_channels=self.dim, out_channels=self.dim, has_bias=True)

        self.proj = nn.Dense(self.dim, self.dim)
        self.matmul = ops.BatchMatMul()
        if num_heads:
            self.scale = 1 / math.sqrt(self.dim // num_heads)
        else:
            raise ValueError("The dim is divided by zero!")

    def construct(self, x):
        """Window Attention forward function."""
        w_nums, z_h_nums, nums, channels = x.shape

        q = self.q(x).reshape(w_nums, z_h_nums, nums, self.num_heads,
                              self.dim // self.num_heads).transpose(0, 1, 3, 2, 4) * self.scale
        k = self.k(x).reshape(w_nums, z_h_nums, nums, self.num_heads,
                              self.dim // self.num_heads).transpose(0, 1, 3, 4, 2)
        v = self.v(x).reshape(w_nums, z_h_nums, nums, self.num_heads,
                              self.dim // self.num_heads).transpose(0, 1, 3, 2, 4)
        attn = (self.matmul(q, k))

        attn = attn + self.relative_bias()
        attn = self.softmax(attn)
        attn = P.Cast()(attn, v.dtype)
        x = self.matmul(attn, v).transpose(0, 1, 3, 2, 4).reshape(w_nums, z_h_nums, nums, channels)
        output = self.proj(x)
        return output


class WindowAttention12(nn.Cell):
    """Window Attention in Swin Block."""
    def __init__(self, dim, num_heads, window_size, input_shape):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.relative_bias = RelativeBias((input_shape[0] // window_size[0]) * (input_shape[1] // window_size[1]),
                                          num_heads, window_size)

        self.softmax = nn.Softmax(axis=-1)

        self.q = nn.Dense(in_channels=dim, out_channels=dim, has_bias=True)
        self.k = nn.Dense(in_channels=dim, out_channels=dim, has_bias=True)
        self.v = nn.Dense(in_channels=dim, out_channels=dim, has_bias=True)

        self.proj = nn.Dense(dim, dim)
        self.matmul = ops.BatchMatMul()
        if num_heads:
            self.scale = 1 / math.sqrt(dim // num_heads)
        else:
            raise ValueError("The dim is divided by zero!")

    def construct(self, x, mask=None):
        """Window Attention in Swin Block."""
        w_nums, z_h_nums, nums, channels = x.shape

        q = self.q(x).reshape(w_nums, z_h_nums, nums, self.num_heads,
                              self.dim // self.num_heads).transpose(0, 1, 3, 2, 4) * self.scale
        k = self.k(x).reshape(w_nums, z_h_nums, nums, self.num_heads,
                              self.dim // self.num_heads).transpose(0, 1, 3, 4, 2)
        v = self.v(x).reshape(w_nums, z_h_nums, nums, self.num_heads,
                              self.dim // self.num_heads).transpose(0, 1, 3, 2, 4)
        attn = (self.matmul(q, k))

        attn = attn + self.relative_bias()

        attn = attn.reshape(1, w_nums, z_h_nums, self.num_heads, nums, nums) + \
               mnp.expand_dims(mnp.expand_dims(mask, axis=2), axis=0)
        attn = attn.reshape(w_nums, z_h_nums, self.num_heads, nums, nums)
        attn = self.softmax(attn)
        attn = P.Cast()(attn, v.dtype)
        x = self.matmul(attn, v).transpose(0, 1, 3, 2, 4).reshape(w_nums, z_h_nums, nums, channels)
        output = self.proj(x)
        return output


def window_partition1(x,
                      z_win_size,
                      h_win_size,
                      w_win_size):
    """window partition"""
    _, z_size, h_size, w_size, channels = x.shape
    x = x.reshape(1, z_size // z_win_size, z_win_size,
                  h_size // h_win_size, h_win_size,
                  w_size // w_win_size, w_win_size, channels)
    windows = x.transpose(0, 5, 1, 3, 2, 4, 6, 7).reshape(-1, (z_size // z_win_size) * (h_size // h_win_size),
                                                          z_win_size, h_win_size, w_win_size, channels)
    return windows


class Mlp(nn.Cell):
    """MLP Layer."""
    def __init__(self, in_features):
        super(Mlp, self).__init__()
        self.fc2 = nn.Dense(in_features * 4, in_features)
        self.fc1 = nn.Dense(in_features, in_features * 4)
        self.act_layer = nn.GELU(approximate=False)

    def construct(self, x):
        """MLP Layer forward function"""
        x = self.fc1(x)
        x = self.act_layer(x)
        output = self.fc2(x)
        return output


def window_reverse1(windows, z_win_size, h_win_size, w_win_size, z_size, h_size):
    w_nums, _, _, _ = windows.shape # w_nums, z_h_nums, nums, channels
    x = windows.reshape(1, w_nums, z_size // z_win_size, h_size // h_win_size, z_win_size, h_win_size, w_win_size, -1)
    x = x.transpose(0, 2, 4, 3, 5, 1, 6, 7).reshape(1, z_size, h_size, w_nums * w_win_size, -1)
    return x


class TransformerBlock1(nn.Cell):
    """Swin Transformer Block."""
    def __init__(self,
                 in_channels,
                 z_win_size=2,
                 h_win_size=6,
                 w_win_size=12,
                 input_shape=None):
        super(TransformerBlock1, self).__init__()
        self.in_channels = in_channels

        self.z_win_size = z_win_size
        self.h_win_size = h_win_size
        self.w_win_size = w_win_size

        self.norm1 = nn.LayerNorm([in_channels], epsilon=1e-5)
        self.attn = WindowAttention11(in_channels,
                                      num_heads=6,
                                      window_size=(z_win_size, h_win_size, w_win_size),
                                      input_shape=input_shape)
        self.norm2 = nn.LayerNorm([in_channels], epsilon=1e-5)

        self.mlp = Mlp(in_channels)

    def construct(self, x):
        """Swin Transformer Block forward function."""
        _, z_size, h_size, w_size, channels = x.shape
        shortcut = x.reshape(1, -1, self.in_channels)
        shifted_x = x
        x_windows = window_partition1(shifted_x,
                                      self.z_win_size,
                                      self.h_win_size,
                                      self.w_win_size)
        x_windows = x_windows.reshape(w_size // self.w_win_size,
                                      (z_size // self.z_win_size) * (h_size // self.h_win_size),
                                      self.z_win_size * self.h_win_size * self.w_win_size,
                                      self.in_channels)
        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows)

        # merge windows
        shifted_x = window_reverse1(attn_windows, self.z_win_size, self.h_win_size, self.w_win_size, z_size,
                                    h_size)

        x = shifted_x
        x = x.reshape(1, -1, channels)

        x = shortcut + self.norm1(x)
        output = x + self.norm2(self.mlp(x))

        return output


class TransformerBlock2(nn.Cell):
    """Swin Transformer Block."""
    def __init__(self,
                 in_channels=384,
                 z_win_size=2,
                 h_win_size=6,
                 w_win_size=12,
                 input_shape=None):
        super().__init__()
        self.in_channels = in_channels

        self.z_win_size = z_win_size
        self.h_win_size = h_win_size
        self.w_win_size = w_win_size

        self.norm1 = nn.LayerNorm([in_channels], epsilon=1e-5)
        self.attn = WindowAttention12(in_channels,
                                      num_heads=6,
                                      window_size=(z_win_size, h_win_size, w_win_size),
                                      input_shape=input_shape,
                                      )
        self.norm2 = nn.LayerNorm([in_channels], epsilon=1e-5)
        self.mlp = Mlp(in_channels)

    def construct(self, x, mask_matrix, z_shift_size, h_win_size, w_shift_size):
        """Swin Transformer Block forward function"""
        _, z_size, h_size, w_size, channels = x.shape
        shortcut = x.reshape(1, -1, self.in_channels)
        shifted_x = mnp.roll(x, shift=(-z_shift_size, -h_win_size, -w_shift_size), axis=(1, 2, 3))
        attn_mask = mask_matrix
        x_windows = window_partition1(shifted_x,
                                      self.z_win_size,
                                      self.h_win_size,
                                      self.w_win_size)
        x_windows = x_windows.reshape(w_size // self.w_win_size,
                                      (z_size // self.z_win_size) * (h_size // self.h_win_size),
                                      self.z_win_size * self.h_win_size * self.w_win_size,
                                      self.in_channels)

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, attn_mask)

        # merge windows
        shifted_x = window_reverse1(attn_windows, self.z_win_size, self.h_win_size, self.w_win_size, z_size,
                                    h_size)
        x = mnp.roll(shifted_x, shift=(z_shift_size, h_win_size, w_shift_size), axis=(1, 2, 3))
        x = x.reshape(1, -1, channels)
        x = shortcut + self.norm1(x)
        output = x + self.norm2(self.mlp(x))

        return output


class BaseBlock(nn.Cell):
    """Base Block."""
    def __init__(self,
                 z_win_size=2,
                 h_win_size=6,
                 w_win_size=12,
                 z_shift_size=1,
                 h_shift_size=3,
                 w_shift_size=6,
                 in_channels=192,
                 input_shape=None):
        super().__init__()
        self.z_win_size = z_win_size
        self.h_win_size = h_win_size
        self.w_win_size = w_win_size
        self.z_shift_size = z_shift_size
        self.h_shift_size = h_shift_size
        self.w_shift_size = w_shift_size
        self.in_channels = in_channels
        self.blk1 = TransformerBlock1(in_channels=in_channels, z_win_size=z_win_size, h_win_size=h_win_size,
                                      w_win_size=w_win_size, input_shape=input_shape)
        self.blk2 = TransformerBlock2(in_channels=in_channels, z_win_size=z_win_size, h_win_size=h_win_size,
                                      w_win_size=w_win_size, input_shape=input_shape)

        self.blk1.recompute()
        self.blk2.recompute()

    def construct(self, x):
        """Base Block forward function."""
        _, z_size, h_size, w_size, channels = x.shape
        img_mask = ops.zeros((1, z_size, h_size, w_size, 1), mindspore.float32)
        z_slices = (slice(0, -self.z_win_size), slice(-self.z_win_size, -self.z_shift_size),
                    slice(-self.z_shift_size, None))
        h_slices = (slice(0, -self.h_win_size), slice(-self.h_win_size, -self.h_shift_size),
                    slice(-self.h_shift_size, None))

        cnt = 0
        for z in z_slices:
            for h in h_slices:
                img_mask[:, z, h, :, :] = cnt
                cnt += 1

        mask_windows = img_mask.reshape(1, z_size // self.z_win_size, self.z_win_size,
                                        h_size // self.h_win_size, self.h_win_size,
                                        w_size // self.w_win_size, self.w_win_size, 1)
        mask_windows = mask_windows.transpose(0, 5, 1, 3, 2, 4, 6, 7).reshape(-1, (z_size // self.z_win_size) * (
            h_size // self.h_win_size), self.z_win_size, self.h_win_size, self.w_win_size, 1)
        mask_windows = mask_windows.reshape(w_size // self.w_win_size,
                                            (z_size // self.z_win_size) * (h_size // self.h_win_size),
                                            self.z_win_size * self.h_win_size * self.w_win_size)
        attn_mask = mnp.expand_dims(mask_windows, axis=2) - mnp.expand_dims(mask_windows, axis=3)
        attn_mask = ops.masked_fill(ops.masked_fill(attn_mask, attn_mask != 0, float(-100.0)), attn_mask == 0,
                                    float(0.0))

        x = self.blk1(x)
        x = x.reshape(1, z_size, h_size, w_size, channels)
        x = self.blk2(x, attn_mask, self.z_shift_size, self.h_win_size, self.w_shift_size)
        output = x.reshape(1, z_size, h_size, w_size, channels)
        return output


class UpSample(nn.Cell):
    """Up Sample module."""
    def __init__(self, in_channels=768, out_channels=384):
        super().__init__()
        self.conv2dtrans = nn.Conv2dTranspose(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=2,
            has_bias=True)
        self.residual_block = ResidualBlock(in_channels=out_channels, out_channels=out_channels)
        self.silu = nn.SiLU()

    def construct(self, x):
        """Up Sample forward function."""
        _, z_size, h_size, w_size, channels = x.shape
        x = x.transpose(0, 1, 4, 2, 3).squeeze(0)
        x = self.conv2dtrans(x)
        x = self.residual_block(x)
        x = self.silu(x)
        x = x.transpose(0, 2, 3, 1)
        output = x.reshape(1, z_size, h_size * 2, w_size * 2, channels // 2)
        return output


class PatchRecover(nn.Cell):
    """Patch Recover module."""
    def __init__(self,
                 channels,
                 h_size,
                 w_size,
                 level_feature_size,
                 pressure_level_num,
                 surface_feature_size,
                 kernel_size):
        super().__init__()
        self.channels = channels
        self.h_size = h_size
        self.w_size = w_size
        self.level_feature_size = level_feature_size
        self.pressure_level_num = pressure_level_num
        self.surface_feature_size = surface_feature_size
        self.kernel_size = kernel_size
        self.proj = nn.Conv1d(channels,
                              level_feature_size * kernel_size[0] * kernel_size[1] * kernel_size[2],
                              kernel_size=1,
                              stride=1,
                              group=1,
                              has_bias=True)
        self.proj_surface = nn.Conv1d(channels,
                                      surface_feature_size * kernel_size[1] * kernel_size[2],
                                      kernel_size=1,
                                      stride=1,
                                      group=1,
                                      has_bias=True)

    def construct(self, x):
        """Patch Recover forward function."""
        x = x.transpose(0, 4, 1, 2, 3)

        x_3d = x[:, :, :-1, :, :].reshape(1, self.channels, -1)
        x_surface = x[:, :, -1, :, :].reshape(1, self.channels, -1)

        x_3d = self.proj(x_3d)
        x_surface = self.proj_surface(x_surface)
        x_3d = x_3d.reshape(1,
                            self.level_feature_size,
                            self.kernel_size[0],
                            self.kernel_size[1],
                            self.kernel_size[2],
                            self.pressure_level_num // 2 + 1,
                            self.h_size // 4,
                            self.w_size // 4).transpose(0, 1, 5, 2, 6, 3, 7, 4)
        x_surface = x_surface.reshape(1,
                                      self.surface_feature_size,
                                      self.kernel_size[1],
                                      self.kernel_size[2],
                                      self.h_size // 4,
                                      self.w_size // 4).transpose(0, 1, 4, 2, 5, 3)
        x_3d = x_3d.reshape(1, self.level_feature_size, self.pressure_level_num + 1, self.h_size, self.w_size)
        x_surface = x_surface.reshape(1, self.surface_feature_size, self.h_size, self.w_size)

        x_3d = x_3d[:, :, 1:, :self.h_size, :self.w_size]

        x_3d = x_3d.reshape([1, self.level_feature_size, 1, self.pressure_level_num, self.h_size, self.w_size])
        x_surface = x_surface.reshape([1, self.surface_feature_size, 1, self.h_size, self.w_size])

        output = x_3d.reshape([self.level_feature_size, self.pressure_level_num, self.h_size, self.w_size])
        output_surface = x_surface.reshape([self.surface_feature_size, self.h_size, self.w_size])

        return output, output_surface
