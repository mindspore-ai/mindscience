'''Util module that provide utilities for common calculations'''

import os
import yaml
import numpy as np
from mindflow.utils.check_func import check_param_odd

EPS = 1e-8
np.random.seed(0)


def make_paths_absolute(dir_, config):
    """
    Make all values for keys ending with `_path` absolute to dir_.

    Parameters
    ----------
    dir_ : str
    config : dict

    Returns
    -------
    config : dict
    """
    for key in config.keys():
        if key.endswith("_path"):
            config[key] = os.path.join(dir_, config[key])
            config[key] = os.path.abspath(config[key])
        if isinstance(config[key], dict):
            config[key] = make_paths_absolute(dir_, config[key])
    return config


def load_config(yaml_filepath):
    """
    Load a YAML configuration file.

    Parameters
    ----------
    yaml_filepath : str

    Returns
    -------
    config : dict
    """
    # Read YAML experiment definition file
    with open(yaml_filepath, 'r') as stream:
        config = yaml.safe_load(stream)
    config = make_paths_absolute(os.path.join(os.path.dirname(yaml_filepath), ".."), config)
    return config


def patchify(data, patch_size=16):
    """
    :param data:
    :param patch_size:
    :return:
    """
    data_shape = data.shape
    data = np.reshape(data, (data_shape[0],
                             data_shape[1] // patch_size,
                             patch_size,
                             data_shape[2] // patch_size,
                             patch_size,
                             data_shape[3]))
    data = np.transpose(data, (0, 1, 3, 2, 4, 5))
    data_new_shape = data.shape
    data = np.reshape(data, (data_new_shape[0],
                             data_new_shape[1] * data_new_shape[2],
                             data_new_shape[3] * data_new_shape[4] * data_new_shape[5]))
    return data


def unpatchify(data, img_size=(192, 384), patch_size=16, nchw=False):
    """
    :param data:
    :param img_size:
    :param patch_size:
    :return:
    """
    data_shape = data.shape
    output_dim = data_shape[-1] // (patch_size * patch_size)
    data = np.reshape(data, (data_shape[0],
                             img_size[0] // patch_size,
                             img_size[1] // patch_size,
                             patch_size,
                             patch_size,
                             output_dim))
    data = np.transpose(data, (0, 1, 3, 2, 4, 5))
    data = np.reshape(data, (data_shape[0],
                             img_size[0],
                             img_size[1],
                             output_dim))
    if nchw:
        data = np.transpose(data, (0, 3, 1, 2))

    return data


def to_2tuple(t):
    return t if isinstance(t, tuple) else (t, t)


def get_2d_sincos_pos_embed(embed_dim, grid_size):
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
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    pos_embed = np.expand_dims(pos_embed, 0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    check_param_odd(embed_dim, "embed_dim")

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    check_param_odd(embed_dim, "embed_dim")
    omega = np.arange(embed_dim // 2, dtype=np.float)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_grid_1d(resolution):
    grid_x = np.linspace(0, 1, resolution)
    return grid_x.reshape(1, resolution, 1)
