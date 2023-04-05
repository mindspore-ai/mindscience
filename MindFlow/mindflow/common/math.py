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
math operators
'''
import numpy as np
import mindspore.numpy as mnp

from ..cell.utils import to_2tuple, to_3tuple

__all__ = ['get_grid_1d', 'get_grid_2d', 'get_grid_3d', 'fftshift', 'ifftshift']


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


def get_grid_3d(resolution):
    """get grid 3d"""
    resolution = to_3tuple(resolution)
    res_x = resolution[0]
    res_y = resolution[1]
    res_z = resolution[2]
    grid_x = np.linspace(0, 1, res_x).reshape(1, res_x, 1, 1, 1)
    grid_y = np.linspace(0, 1, res_y).reshape(1, 1, res_y, 1, 1)
    grid_z = np.linspace(0, 1, res_z).reshape(1, 1, 1, res_z, 1)
    grid_x = np.repeat(grid_x, res_y, axis=2)
    grid_x = np.repeat(grid_x, res_z, axis=3)
    grid_y = np.repeat(grid_y, res_x, axis=1)
    grid_y = np.repeat(grid_y, res_z, axis=3)
    grid_z = np.repeat(grid_z, res_x, axis=1)
    grid_z = np.repeat(grid_z, res_y, axis=2)

    return np.concatenate((grid_x, grid_y, grid_z), axis=-1)


def fftshift(x, axes=None):
    if axes is None:
        axes = tuple(range(x.ndim))
        shift = [dim // 2 for dim in x.shape]
    elif isinstance(axes, int):
        shift = x.shape[axes] // 2
    else:
        shift = [x.shape[ax] // 2 for ax in axes]
    return mnp.roll(x, shift, axes)


def ifftshift(x, axes=None):
    if axes is None:
        axes = tuple(range(x.ndim))
        shift = [-(dim // 2) for dim in x.shape]
    elif isinstance(axes, int):
        shift = -(x.shape[axes] // 2)
    else:
        shift = [-(x.shape[ax] // 2) for ax in axes]
    return mnp.roll(x, shift, axes)
