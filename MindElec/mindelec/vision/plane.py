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
# ============================================================================
"""Visualization of the results in 2D image form"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')


def plot_s11(s11_tensor, path_image_save, legend, dpi=300):
    r"""
    Draw s11-frequency curve and save it in path_image_save.

    Args:
        s11_tensor (numpy.ndarray): s11 data (shape: (dim_frequency, 2)).
        path_image_save (str): s11-frequency curve saved path.
        legend (str): the legend of s11, plotting parameters.
        dpi (int): plotting parameters. Default: 300.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import numpy as np
        >>> from mindelec.vision import plot_s11
        >>> s11 = np.random.rand(1001, 2).astype(np.float32)
        >>> s11[:, 0] = np.linspace(0, 4 * 10 ** 9, 1001)
        >>> s11 = s11.astype(np.float32)
        >>> s11_tensor = s11
        >>> path_image_save = './result_s11'
        >>> legend = 's11'
        >>> dpi = 300
        >>> plot_s11(s11_tensor, path_image_save, legend, dpi)
    """
    if not isinstance(s11_tensor, np.ndarray):
        raise TypeError("The type of s11_tensor should be numpy array, but get {}".format(type(s11_tensor)))

    if not isinstance(path_image_save, str):
        raise TypeError("The type of path_image_save should be str, but get {}".format(type(path_image_save)))
    if not os.path.exists(path_image_save):
        os.makedirs(path_image_save)

    if not isinstance(legend, str):
        raise TypeError("The type of legend should be str, but get {}".format(type(legend)))

    if not isinstance(dpi, int):
        raise TypeError("The type of dpi must be int, but get {}".format(type(dpi)))
    if isinstance(dpi, bool):
        raise TypeError("The type of dpi must be int, but get {}".format(type(dpi)))
    if dpi <= 0:
        raise ValueError("dpi must be > 0, but get {}".format(dpi))

    shape_s11_all = s11_tensor.shape
    if len(shape_s11_all) != 2 or shape_s11_all[-1] != 2:
        raise ValueError("s11_tensor shape should be (dim_frequency, 2), but get {}".format(shape_s11_all))

    s11_temp, frequency = s11_tensor[:, 0], s11_tensor[:, 1]
    plt.figure(dpi=dpi, figsize=(8, 4))
    plt.plot(frequency, s11_temp, '-', label=legend, linewidth=2)
    plt.title('s11(dB)')
    plt.xlabel('frequency(Hz)')
    plt.ylabel('dB')
    plt.legend()
    plt.savefig(os.path.join(path_image_save, 's11.jpg'))
    plt.close()


def plot_eh(simu_res_tensor, path_image_save, z_index, dpi=300):
    r"""
    Draw electric and magnetic field values of every timestep for 2D slices, and save them in path_image_save

    Args:
        simu_res_tensor (numpy..array): simulation result (shape (dim_t, dim_x, dim_y, dim_z, 6)).
        path_image_save (str): images saved path.
        z_index (int): show 2D image for z=z_index.
        dpi (int): plotting parameters. Default: 300.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import numpy as np
        >>> from mindelec.vision import plot_eh
        >>> simu_res_tensor = np.random.rand(20, 10, 10, 10, 6).astype(np.float32)
        >>> path_image_save = './result_eh'
        >>> z_index = 5
        >>> dpi = 300
        >>> plot_eh(simu_res_tensor, path_image_save, z_index, dpi)
    """
    if not isinstance(simu_res_tensor, np.ndarray):
        raise TypeError("The type of simu_res_tensor should be numpy array, but get {}".format(type(simu_res_tensor)))

    if not isinstance(path_image_save, str):
        raise TypeError("The type of path_image_save should be str, but get {}".format(type(path_image_save)))
    if not os.path.exists(path_image_save):
        os.makedirs(path_image_save)

    if not isinstance(z_index, int):
        raise TypeError("The type of z_index must be int, but get {}".format(type(z_index)))
    if isinstance(z_index, bool):
        raise TypeError("The type of z_index must be int, but get {}".format(type(z_index)))
    if z_index <= 0:
        raise ValueError("z_index must be > 0, but get {}".format(z_index))

    if not isinstance(dpi, int):
        raise TypeError("The type of dpi must be int, but get {}".format(type(dpi)))
    if isinstance(dpi, bool):
        raise TypeError("The type of dpi must be int, but get {}".format(type(dpi)))
    if dpi <= 0:
        raise ValueError("dpi must be > 0, but get {}".format(dpi))

    shape_simulation_res = simu_res_tensor.shape
    if len(shape_simulation_res) != 5 or shape_simulation_res[-1] != 6:
        raise ValueError("simu_res_tensor shape should be (dim_t, dim_x, dim_y, dim_z, 6), but get {}"
                         .format(shape_simulation_res))

    plot_order = ['Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz']

    for i in range(6):
        current = simu_res_tensor[:, :, :, z_index, i]
        min_val, max_val = np.min(current), np.max(current)
        timesteps = len(current)
        for t in range(timesteps):
            current_2d = current[t]
            plt.figure(dpi=dpi)
            plt.imshow(current_2d, vmin=min_val, vmax=max_val)
            plt.colorbar()
            plt.savefig(os.path.join(path_image_save, plot_order[i] + '_' + str(t) + '.jpg'))
            plt.close()
