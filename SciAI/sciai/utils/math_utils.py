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
"""math operators"""
import mindspore.numpy as mnp
import numpy as np


def _to_3tuple(t):
    """
    Args:
        t (Union[int, tuple(int)]): The grid height and width.

    Returns:
        Same as input or a tuple as (t,t,t).

    """
    return t if isinstance(t, tuple) else (t, t, t)


def _to_2tuple(t):
    """
    Args:
        t (Union[int, tuple(int)]): The grid height and width.

    Returns:
        Same as input or a tuple as (t,t).

    """
    return t if isinstance(t, tuple) else (t, t)


def _get_grid_1d(resolution):
    grid_x = np.linspace(0, 1, resolution)
    return grid_x.reshape((1, resolution, 1))


def _get_grid_2d(resolution):
    resolution = _to_2tuple(resolution)
    res_x = resolution[0]
    res_y = resolution[1]
    grid_x = np.linspace(0, 1, res_x).reshape((1, res_x, 1, 1))
    grid_y = np.linspace(0, 1, res_y).reshape((1, 1, res_y, 1))
    grid_x = np.repeat(grid_x, res_y, axis=2)
    grid_y = np.repeat(grid_y, res_x, axis=1)
    return np.concatenate((grid_x, grid_y), axis=-1)


def _get_grid_3d(resolution):
    """get grid 3d"""
    resolution = _to_3tuple(resolution)
    res_x = resolution[0]
    res_y = resolution[1]
    res_z = resolution[2]
    grid_x = np.linspace(0, 1, res_x).reshape((1, res_x, 1, 1, 1))
    grid_y = np.linspace(0, 1, res_y).reshape((1, 1, res_y, 1, 1))
    grid_z = np.linspace(0, 1, res_z).reshape((1, 1, 1, res_z, 1))
    grid_x = np.repeat(grid_x, res_y, axis=2)
    grid_x = np.repeat(grid_x, res_z, axis=3)
    grid_y = np.repeat(grid_y, res_x, axis=1)
    grid_y = np.repeat(grid_y, res_z, axis=3)
    grid_z = np.repeat(grid_z, res_x, axis=1)
    grid_z = np.repeat(grid_z, res_y, axis=2)

    return np.concatenate((grid_x, grid_y, grid_z), axis=-1)


def _fftshift(x, axes=None):
    if axes is None:
        axes = tuple(range(x.ndim))
        shift = [dim // 2 for dim in x.shape]
    elif isinstance(axes, int):
        shift = x.shape[axes] // 2
    else:
        shift = [x.shape[ax] // 2 for ax in axes]
    return mnp.roll(x, shift, axes)


def _ifftshift(x, axes=None):
    if axes is None:
        axes = tuple(range(x.ndim))
        shift = [-(dim // 2) for dim in x.shape]
    elif isinstance(axes, int):
        shift = -(x.shape[axes] // 2)
    else:
        shift = [-(x.shape[ax] // 2) for ax in axes]
    return mnp.roll(x, shift, axes)


def get_2d_sin_cos_pos_embed(embed_dim, grid_size):
    r"""
    Args:
        embed_dim (int): The output dimension for each position.
        grid_size (tuple(int)): The grid height and width.

    Returns:
        The numpy array with shape of :math:`(1, grid\_height*grid_width, embed\_dim)`

    """
    grid_size = _to_2tuple(grid_size)
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
        pos (ndarray): a numpy array of positions to be encoded: size (M,).

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
