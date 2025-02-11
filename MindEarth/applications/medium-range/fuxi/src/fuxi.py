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
import numpy as np

import mindspore as ms
import mindspore.numpy as mnp
import mindspore.common.dtype as mstype
import mindspore.ops.operations as P
from mindspore import nn, ops, Parameter, Tensor
from mindspore.common.initializer import initializer, Uniform


def window_partition(x, window_size):
    batch_size, pressure_level_num, h_size, w_size, channel_size = x.shape
    x = x.reshape(batch_size, pressure_level_num // window_size[0], window_size[0], h_size // window_size[1],
                  window_size[1],
                  w_size // window_size[2],
                  window_size[2], channel_size)
    windows = x.transpose(0, 5, 1, 3, 2, 4, 6, 7).reshape(-1, (pressure_level_num // window_size[0]) *
                                                          (h_size // window_size[1]),
                                                          window_size[0], window_size[1], window_size[2], channel_size)
    return windows


def window_reverse(windows, window_size, pressure_level_num, h_size, w_size):
    batch_size, _, _, _ = windows.shape
    batch_size = int(batch_size / (w_size // window_size[2]))
    x = windows.reshape(batch_size, w_size // window_size[2], pressure_level_num // window_size[0],
                        h_size // window_size[1],
                        window_size[0],
                        window_size[1], window_size[2], -1)
    x = x.transpose(0, 2, 4, 3, 5, 1, 6, 7).reshape(batch_size, pressure_level_num, h_size, w_size, -1)
    return x

class CustomConv3d(nn.Cell):
    """
    Applies a 3D convolution over an input tensor which is typically of shape (N, C, D, H, W)
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, pad_mode='same',
                 has_bias=False, dtype=ms.float16):
        super(CustomConv3d, self).__init__()
        self.out_channels = out_channels
        self.conv2d_blocks = nn.CellList(
            [nn.Conv2d(in_channels,
                       out_channels,
                       kernel_size=(kernel_size[1], kernel_size[2]),
                       stride=(stride[1], stride[2]),
                       pad_mode=pad_mode,
                       dtype=dtype,
                       ) for _ in range(kernel_size[0])]
        )
        w = Tensor(np.identity(kernel_size[0]), dtype=dtype)
        self.conv2d_weight = ops.expand_dims(ops.expand_dims(w, axis=0), axis=0)
        self.k = kernel_size[0]
        self.stride = stride
        self.pad_mode = pad_mode
        self.conv2d_dtype = dtype
        self.has_bias = has_bias
        if self.has_bias:
            self.bias = Parameter(initializer(Uniform(), [1, out_channels, 1, 1, 1], dtype=dtype))

    def construct(self, x_):
        """
          Process the input tensor through a series of convolutional layers and perform shaping operations.

          Args:
              x: The input tensor with shape (B, C, D, H, W) representing batch size, channels, depth,
              height, and width.

          Returns:
              The output tensor after convolution, reshaping, and optional bias addition, with the final shape
              depending on the input and layer parameters.
          """
        b, c, d, h, w = x_.shape
        x_ = x_.transpose(0, 2, 1, 3, 4).reshape(b * d, c, h, w)
        out = []
        for i in range(self.k):
            out.append(self.conv2d_blocks[i](x_))
        out = ops.stack(out, axis=-1)
        _, cnew, hnew, wnew, _ = out.shape
        out = out.reshape(b, d, cnew, hnew, wnew, self.k).transpose(0, 2, 3, 4, 1, 5).reshape(-1, 1, d, self.k)
        out = ops.conv2d(out, self.conv2d_weight, stride=(self.stride[0], 1), pad_mode='valid')
        out = out.reshape(b, cnew, hnew, wnew, -1).transpose(0, 1, 4, 2, 3)
        if self.has_bias:
            out += self.bias
        return out
class CubeEmbed(nn.Cell):
    """
    Cube Embedding for high dimension data.
    """

    def __init__(self, in_channels, h_size, w_size, level_feature_size, pressure_level_num, surface_feature_size,
                 batch_size):
        super().__init__()
        self.in_channels = in_channels
        self.h_size = h_size
        self.w_size = w_size
        self.batch_size = batch_size
        self.level_feature_size = level_feature_size
        self.pressure_level_num = pressure_level_num
        self.surface_feature_size = surface_feature_size
        self.layer_norm = nn.LayerNorm([in_channels], epsilon=1e-5)
        self.conv3d_dtype = mstype.float16
        self.cube3d = CustomConv3d(level_feature_size, in_channels, kernel_size=(2, 4, 4),
                                   pad_mode="valid", stride=(2, 4, 4), has_bias=True, dtype=mstype.float32)
        self.conv2d = nn.Conv2d(surface_feature_size, in_channels, kernel_size=(4, 4),
                                pad_mode="valid", stride=(4, 4), has_bias=True)

    def construct(self, x):
        """CubeEmbed forward function."""
        x_input = x.reshape(self.batch_size, self.h_size, self.w_size, -1)
        x = x_input[..., :-self.surface_feature_size]
        x_surface = x_input[..., -self.surface_feature_size:]
        x = x.reshape(self.batch_size, self.h_size, self.w_size, self.level_feature_size, self.pressure_level_num)
        x = x.transpose(0, 3, 4, 1, 2)
        x_surface = x_surface.transpose(0, 3, 1, 2)
        pad_zeros = ops.zeros((self.batch_size, self.level_feature_size, 1, self.h_size, self.w_size), dtype=x.dtype)
        x = ops.concat((pad_zeros, x), axis=2)
        x = self.cube3d(x)
        x_surface = self.conv2d(x_surface)
        x_surface = x_surface.reshape(self.batch_size, self.in_channels, 1, self.h_size // 4, self.w_size // 4)
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
        batch_size, channels, patch_size, h_size, w_size = x.shape
        x = x.transpose(0, 2, 1, 3, 4).reshape(batch_size * patch_size, channels, h_size, w_size)
        x = self.conv2d(x)
        x = self.residual_block(x)
        x = self.silu(x)
        x = x.transpose(0, 2, 3, 1)
        output = x.reshape(batch_size, patch_size, h_size // 2, w_size // 2, channels * 2)
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
        relative_coords = mnp.expand_dims(coords_flatten_1, axis=-1) - mnp.expand_dims(coords_flatten_2, axis=1)
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


class WindowAttention(nn.Cell):
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
        if mask is not None:
            attn = attn.reshape(1, w_nums, z_h_nums, self.num_heads, nums, nums) + \
                   mnp.expand_dims(mnp.expand_dims(mask, axis=2), axis=0)
            attn = attn.reshape(w_nums, z_h_nums, self.num_heads, nums, nums)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        attn = P.Cast()(attn, v.dtype)
        x = self.matmul(attn, v).transpose(0, 1, 3, 2, 4).reshape(w_nums, z_h_nums, nums, channels)
        output = self.proj(x)
        return output


class Mlp(nn.Cell):
    """MLP Layer."""

    def __init__(self, in_features, hidden_features=None, out_features=None):
        super(Mlp, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Dense(in_features, hidden_features)
        self.act_layer = nn.GELU(approximate=False)
        self.fc2 = nn.Dense(hidden_features, out_features)

    def construct(self, x):
        """MLP Layer forward function"""
        x = self.fc1(x)
        x = self.act_layer(x)
        x = self.fc2(x)
        return x


class TransformerBlock(nn.Cell):
    """Swin Transformer Block."""

    def __init__(self,
                 shift_size,
                 window_size,
                 dim=192,
                 num_heads=6,
                 input_shape=None,
                 mlp_ratio=4.):
        super().__init__()
        self.dim = dim
        self.shift_size = shift_size
        self.window_size = window_size
        self.norm1 = nn.LayerNorm([dim], epsilon=1e-5)
        self.attn = WindowAttention(dim,
                                    num_heads=num_heads,
                                    window_size=window_size,
                                    input_shape=input_shape,
                                    )
        self.norm2 = nn.LayerNorm([dim], epsilon=1e-5)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim)

    def construct(self, x, mask_matrix, pressure_level_num, h_size, w_size):
        """Swin Transformer Block forward function"""
        batch_size = x.shape[0]
        channel_size = x.shape[2]
        shortcut = x
        x = x.reshape(batch_size, pressure_level_num, h_size, w_size, channel_size)

        # cyclic shift
        if self.shift_size[0] > 0 or self.shift_size[1] > 0 or self.shift_size[2] > 0:
            shifted_x = mnp.roll(x,
                                 shift=(-self.shift_size[0], -self.shift_size[1], -self.shift_size[2]),
                                 axis=(1, 2, 3))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None
        x_windows = window_partition(shifted_x, self.window_size)
        b_w, _, _, _, _, channel_size = x_windows.shape
        x_windows = x_windows.reshape(b_w,
                                      (pressure_level_num // self.window_size[0]) * (h_size // self.window_size[1]),
                                      self.window_size[0] * self.window_size[1] * self.window_size[2],
                                      self.dim)

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, attn_mask)

        # merge windows
        shifted_x = window_reverse(attn_windows, self.window_size, pressure_level_num, h_size, w_size)

        # reverse cyclic shift
        if mask_matrix is not None:
            x = mnp.roll(shifted_x,
                         shift=(self.shift_size[0], self.shift_size[1], self.shift_size[2]),
                         axis=(1, 2, 3))
        else:
            x = shifted_x
        x = x.reshape(batch_size, pressure_level_num * h_size * w_size, channel_size)

        # FFN
        x = shortcut + self.norm1(x)
        output = x + self.norm2(self.mlp(x))
        return output


class BaseBlock(nn.Cell):
    """Base Block."""
    def __init__(self,
                 window_size=(2, 6, 12),
                 shift_size=(1, 3, 6),
                 in_channels=192,
                 input_shape=None,
                 recompute=False):
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size
        self.in_channels = in_channels

        self.blk1 = TransformerBlock(dim=in_channels, shift_size=[0, 0, 0], window_size=self.window_size,
                                     input_shape=input_shape)
        self.blk2 = TransformerBlock(dim=in_channels, shift_size=[1, 6, 6], window_size=self.window_size,
                                     input_shape=input_shape)

        if recompute:
            self.blk1.recompute()
            self.blk2.recompute()

    def construct(self, x, batch_size, pressure_level_num, h_size, w_size):
        """Base Block forward function."""
        img_mask = ops.zeros((batch_size, pressure_level_num, h_size, w_size, 1), ms.float32)
        z_slices = (slice(0, -self.window_size[0]),
                    slice(-self.window_size[0], -self.shift_size[0]),
                    slice(-self.shift_size[0], None))
        h_slices = (slice(0, -self.window_size[1]),
                    slice(-self.window_size[1], -self.shift_size[1]),
                    slice(-self.shift_size[1], None))
        cnt = 0
        for z in z_slices:
            for h in h_slices:
                img_mask[:, z, h, :, :] = cnt
                cnt += 1
        mask_windows = img_mask.reshape(batch_size, pressure_level_num // self.window_size[0], self.window_size[0],
                                        h_size // self.window_size[1], self.window_size[1],
                                        w_size // self.window_size[2], self.window_size[2], 1)
        mask_windows = (
            mask_windows.transpose(0, 5, 1, 3, 2, 4, 6, 7).reshape(-1, (pressure_level_num // self.window_size[0]) *
                                                                   (h_size // self.window_size[1]), self.window_size[0],
                                                                   self.window_size[1], self.window_size[2], 1))
        b_w, _, _, _, _, _ = mask_windows.shape
        mask_windows = (
            mask_windows.reshape(b_w,
                                 (pressure_level_num // self.window_size[0]) *
                                 (h_size // self.window_size[1]),
                                 self.window_size[0] * self.window_size[1] * self.window_size[2]))
        attn_mask = mnp.expand_dims(mask_windows, axis=2) - mnp.expand_dims(mask_windows, axis=3)
        attn_mask = ops.masked_fill(ops.masked_fill(attn_mask, attn_mask != 0, float(-100.0)), attn_mask == 0,
                                    float(0.0))
        x = self.blk1(x, attn_mask, pressure_level_num, h_size, w_size)
        x = self.blk2(x, attn_mask, pressure_level_num, h_size, w_size)
        return x


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
        batch_size, patch_size, h_size, w_size, channels = x.shape
        x = x.transpose(0, 1, 4, 2, 3).reshape(batch_size * patch_size, channels, h_size, w_size)
        x = self.conv2dtrans(x)
        x = self.residual_block(x)
        x = self.silu(x)
        x = x.transpose(0, 2, 3, 1)
        output = x.reshape(batch_size, patch_size, h_size * 2, w_size * 2, channels // 2)
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
        batch_size, _, _, _, _ = x.shape
        x = x.transpose(0, 4, 1, 2, 3)
        x_3d = x[:, :, :-1, :, :].reshape(batch_size, self.channels, -1)
        x_surface = x[:, :, -1, :, :].reshape(batch_size, self.channels, -1)
        x_3d = self.proj(x_3d)


        x_surface = self.proj_surface(x_surface)
        x_3d = x_3d.reshape(batch_size,
                            self.level_feature_size,
                            self.kernel_size[0],
                            self.kernel_size[1],
                            self.kernel_size[2],
                            self.pressure_level_num // 2 + 1,
                            self.h_size // 4,
                            self.w_size // 4).transpose(0, 1, 5, 2, 6, 3, 7, 4)
        x_surface = x_surface.reshape(batch_size,
                                      self.surface_feature_size,
                                      self.kernel_size[1],
                                      self.kernel_size[2],
                                      self.h_size // 4,
                                      self.w_size // 4).transpose(0, 1, 4, 2, 5, 3)
        x_3d = x_3d.reshape(batch_size, self.level_feature_size, self.pressure_level_num + 1, self.h_size, self.w_size)
        x_surface = x_surface.reshape(batch_size, self.surface_feature_size, self.h_size, self.w_size)
        x_3d = x_3d[:, :, 1:, :self.h_size, :self.w_size]
        output = x_3d.reshape([batch_size, self.level_feature_size, self.pressure_level_num, self.h_size, self.w_size])
        output_surface = x_surface.reshape([batch_size, self.surface_feature_size, self.h_size, self.w_size])
        return output, output_surface
