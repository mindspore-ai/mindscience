# Copyright 2021 Huawei Technologies Co., Ltd
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
"""Visualization of the results 3D VTK form"""

import os
from pyevtk.hl import gridToVTK
import numpy as np


def vtk_structure(grid_tensor, eh_tensor, path_res):
    r"""
    Generates 3D vtk file for visualizaiton.

    Args:
        grid_tensor (numpy.ndarray): grid data (shape: (dim_t, dim_x, dim_y, dim_z, 4)).
        eh_tensor (numpy.ndarray): electric and magnetic data (np.array, shape: (dim_t, dim_x, dim_y, dim_z, 6)).
        path_res (str): save path for the output vtk file.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import numpy as np
        >>> from mindelec.vision import vtk_structure
        >>> grid_tensor = np.random.rand(20, 10, 10, 10, 4).astype(np.float32)
        >>> eh_tensor = np.random.rand(20, 10, 10, 10, 6).astype(np.float32)
        >>> path_res = './result_vtk'
        >>> vtk_structure(grid_tensor, eh_tensor, path_res)
    """
    if not isinstance(grid_tensor, np.ndarray):
        raise TypeError("The type of grid_tensor should be numpy array, but get {}".format(type(grid_tensor)))

    if not isinstance(eh_tensor, np.ndarray):
        raise TypeError("The type of eh_tensor should be numpy array, but get {}".format(type(eh_tensor)))

    if not isinstance(path_res, str):
        raise TypeError("The type of path_res should be str, but get {}".format(type(path_res)))
    if not os.path.exists(path_res):
        os.makedirs(path_res)

    input_grid = grid_tensor
    output_grid = eh_tensor

    shape_grid = input_grid.shape
    shape_eh = output_grid.shape

    if len(shape_grid) != 5 or shape_grid[-1] != 4:
        raise ValueError("grid_tensor shape should be (dim_t, dim_x, dim_y, dim_z, 4), but get {}"
                         .format(shape_grid))

    if len(shape_eh) != 5 or shape_eh[-1] != 6:
        raise ValueError("eh_tensor shape should be (dim_t, dim_x, dim_y, dim_z, 6), but get {}"
                         .format(shape_eh))

    if shape_grid[:4] != shape_eh[:4]:
        raise ValueError("grid_tensor and eh_tensor should have the same dimension except the last axis, "
                         "but get grid_tensor shape {} and eh_tensor shape{}".format(shape_grid, shape_eh))

    (dim_t, dim_x, dim_y, dim_z, d) = input_grid.shape
    input_grid = np.reshape(input_grid, (dim_t * dim_x * dim_y * dim_z, d))
    x_min, x_max = np.min(input_grid[:, 0]), np.max(input_grid[:, 0])
    y_min, y_max = np.min(input_grid[:, 1]), np.max(input_grid[:, 1])
    z_min, z_max = np.min(input_grid[:, 2]), np.max(input_grid[:, 2])

    x_all = np.linspace(x_min, x_max, dim_x, endpoint=True, dtype='float64')
    y_all = np.linspace(y_min, y_max, dim_y, endpoint=True, dtype='float64')
    z_all = np.linspace(z_min, z_max, dim_z, endpoint=True, dtype='float64')

    x = np.zeros((dim_x, dim_y, dim_z))
    y = np.zeros((dim_x, dim_y, dim_z))
    z = np.zeros((dim_x, dim_y, dim_z))

    for i in range(dim_x):
        for j in range(dim_y):
            for k in range(dim_z):
                x[i, j, k] = x_all[i]
                y[i, j, k] = y_all[j]
                z[i, j, k] = z_all[k]

    for t in range(dim_t):
        output_grid_show = output_grid[t]
        ex, ey, ez = output_grid_show[:, :, :, 0], output_grid_show[:, :, :, 1], output_grid_show[:, :, :, 2]
        hx, hy, hz = output_grid_show[:, :, :, 3], output_grid_show[:, :, :, 4], output_grid_show[:, :, :, 5]
        ex, ey, ez = ex.astype(np.float64), ey.astype(np.float64), ez.astype(np.float64)
        hx, hy, hz = hx.astype(np.float64), hy.astype(np.float64), hz.astype(np.float64)
        gridToVTK(os.path.join(path_res, 'eh_t' + str(t)),
                  x, y, z,
                  pointData={"Ex": ex, "Ey": ey, "Ez": ez, "Hx": hx, "Hy": hy, "Hz": hz})
