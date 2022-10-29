'''
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
'''

import numpy as np
import mindspore.ops.operations as P

__all__ = ['to_2tuple', 'unpatchify', 'patchify', 'get_2d_sin_cos_pos_embed', 'get_grid_1d', 'get_grid_2d']


def to_2tuple(t):
    return t if isinstance(t, tuple) else (t, t)


def get_2d_sin_cos_pos_embed(embed_dim, grid_size):
    """
    embed_dim:
    grid_size: int of the grid height and width
    return:
    """
    grid_size = to_2tuple(grid_size)
    grid_h = np.arange(grid_size[0], dtype=np.float32)
    grid_w = np.arange(grid_size[1], dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size[0], grid_size[1]])
    pos_embed = get_2d_sin_cos_pos_embed_from_grid(embed_dim, grid)
    pos_embed = np.expand_dims(pos_embed, 0)
    return pos_embed


def get_2d_sin_cos_pos_embed_from_grid(embed_dim, grid):
    # use half of dimensions to encode grid_h
    emb_h = get_1d_sin_cos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sin_cos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sin_cos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    omega = np.arange(embed_dim // 2, dtype=np.float)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def patchify(label, patch_size=16):
    """
    :param label:
    :param patch_size:
    :return:
    """
    label_shape = label.shape
    label = np.reshape(label, (label_shape[0] // patch_size,
                               patch_size,
                               label_shape[1] // patch_size,
                               patch_size,
                               label_shape[2]))
    label = np.transpose(label, (0, 2, 1, 3, 4))
    label_new_shape = label.shape
    label = np.reshape(label, (label_new_shape[0] * label_new_shape[1],
                               label_new_shape[2] * label_new_shape[3] * label_new_shape[4]))
    return label


def unpatchify(labels, img_size=(192, 384), patch_size=16, nchw=False):
    """
    :param labels:
    :param img_size:
    :param patch_size:
    :return:
    """
    label_shape = labels.shape
    output_dim = label_shape[-1] // (patch_size * patch_size)
    labels = P.Reshape()(labels, (label_shape[0],
                                  img_size[0] // patch_size,
                                  img_size[1] // patch_size,
                                  patch_size,
                                  patch_size,
                                  output_dim))

    labels = P.Transpose()(labels, (0, 1, 3, 2, 4, 5))
    labels = P.Reshape()(labels, (label_shape[0],
                                  img_size[0],
                                  img_size[1],
                                  output_dim))
    if nchw:
        labels = P.Transpose()(labels, (0, 3, 1, 2))
    return labels


def get_grid_1d(resolution):
    grid_x = np.linspace(0, 1, resolution)
    return grid_x.reshape(1, resolution, 1)


def get_grid_2d(resolution):
    resolution = to_2tuple(resolution)
    res_x = resolution[0]
    res_y = resolution[1]
    grid_x = np.linspace(0, 1, res_x).reshape(1, res_x, 1, 1)
    grid_y = np.linspace(0, 1, res_y).reshape(1, 1, res_y, 1)
    grid_x = np.repeat(grid_x, res_y, axis=2)
    grid_y = np.repeat(grid_y, res_x, axis=1)
    return np.concatenate((grid_x, grid_y), axis=-1)
