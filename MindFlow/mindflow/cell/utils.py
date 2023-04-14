''''
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

__all__ = ['to_2tuple', 'to_3tuple', 'unpatchify',
           'patchify', 'get_2d_sin_cos_pos_embed']


def to_3tuple(t):
    """
    Args:
        t (Union[int, tuple(int)]): The grid height and width.

    Returns:
        Same as input or a tuple as (t,t,t).

    """
    return t if isinstance(t, tuple) else (t, t, t)


def to_2tuple(t):
    """
    Args:
        t (Union[int, tuple(int)]): The grid height and width.

    Returns:
        Same as input or a tuple as (t,t).

    """
    return t if isinstance(t, tuple) else (t, t)


def get_2d_sin_cos_pos_embed(embed_dim, grid_size):
    r"""
    Args:
        embed_dim (int): The output dimension for each position.
        grid_size (tuple(int)): The grid height and width.

    Returns:
        The numpy array with shape of :math:`(1, grid\_height*grid_width, embed\_dim)`

    """
    grid_size = to_2tuple(grid_size)
    grid_height = np.arange(grid_size[0], dtype=np.float32)
    grid_width = np.arange(grid_size[1], dtype=np.float32)
    grid = np.meshgrid(grid_width, grid_height)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size[0], grid_size[1]])
    pos_embed = get_2d_sin_cos_pos_embed_from_grid(embed_dim, grid)
    pos_embed = np.expand_dims(pos_embed, 0)
    return pos_embed


def get_2d_sin_cos_pos_embed_from_grid(embed_dim, grid):
    r"""
    use half of dimensions to encode grid_height

    Args:
        embed_dim (int): output dimension for each position.
        grid (int): a numpy array of positions to be encoded: size (M,).

    Returns:
        The numpy array with shape of :math:`(M/2, embed\_dim)`
    """
    emb_height = get_1d_sin_cos_pos_embed_from_grid(
        embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_width = get_1d_sin_cos_pos_embed_from_grid(
        embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_height, emb_width], axis=1)  # (H*W, D)
    return emb


def get_1d_sin_cos_pos_embed_from_grid(embed_dim, pos):
    r"""
    Args:
        embed_dim (int): output dimension for each position.
        pos (int): a numpy array of positions to be encoded: size (M,).

    Returns:
        The numpy array with shape of :math:`(M, embed\_dim)`
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
    Args:
        label (Union[int, float]): output dimension for each position.
        patch_size (int): The patch size of image. Default: ``16``.

    Returns:
        The numpy array with new shape of :math:`(H, W)`.
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
    Args:
        labels (Union[int, float]): output dimension for each position.
        img_size (tuple(int)): Input image size. Default (192, 384).
        patch_size (int): The patch size of image. Default: 16.
        nchw (bool): If True, the unpatchify shape contains N, C, H, W.

    Returns:
        The tensor with shape of :math:`(N, H, W, C)`.
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
