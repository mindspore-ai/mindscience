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
"""visualization"""
import os
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, ticker

plt.rcParams["figure.autolayout"] = True


def generate_mesh():
    """Generate mesh coordinates corresponding to flow data."""
    x = np.linspace(-2, 4, 128, endpoint=True)
    y = np.linspace(-1.5, 1.5, 64, endpoint=True)
    z = np.linspace(-1.5, 1.5, 64, endpoint=True)

    yy, xx, zz = np.meshgrid(y, x, z)
    mesh = np.concatenate((xx.reshape(1, 128, 64, 64), yy.reshape(1, 128, 64, 64), zz.reshape(1, 128, 64, 64)), axis=0)
    return mesh


def plt_contourf(xgrid, ygrid, data, vmin, vmax, title, step, is_fmt=False):
    """
    Plot a single contourf of the flow field at a certain time and set the figure config.

    Args:
        xgrid(ndarray): The x-coordinate of the grid point. Array of shape: (Nx, Ny).
        ygrid(ndarray): The y-coordinate of the grid point. Array of shape: (Nx, Ny).
        data(ndarray): Flow data of a channel. Array of shape: (Nx, Ny).
        vmin(float): Minimum range of values for contours.
        vmax(float): Maximum range of values for contours.
        title(str): The title of the figure to be drawn.
        step(int): The infer step.
        is_fmt(bool): Whether the label uses scientific notation
    """
    plt.title(title, y=1, fontsize=16, fontweight=500)
    plt.text(-1.5, 1, f"T={step}", fontsize=14, verticalalignment="center", horizontalalignment="center",
             bbox=dict(facecolor='w', alpha=0.5))
    plt.ylabel('Y/D', fontsize=12, style='italic', weight='bold')
    plt.xlabel('X/D', fontsize=12, style='italic', weight='bold')
    plt.xlim((np.min(xgrid), np.max(xgrid)))
    plt.ylim((np.min(ygrid), np.max(ygrid)))
    plt.xticks((-2, 0, 2, 4))
    plt.yticks((-1, 0, 1))

    # plot flow field
    plt.contourf(xgrid, ygrid, data, 21, vmin=vmin, vmax=vmax, cmap='jet')
    # plot sphere
    plt.gca().add_patch(plt.Circle((0, 0), radius=0.5, color='w'))

    def fmt(x, _):
        a, b = '{:.2e}'.format(x).split('e')
        b = int(b)
        return r'${} \times 10^{{{}}}$'.format(a, b)

    cb = plt.colorbar(format=(ticker.FuncFormatter(fmt) if is_fmt else None))
    cb.ax.tick_params(labelsize=10)

    plt.axis('tight')


def plot_gif(true, pred, map_name):
    r"""
    GIF visualization of p/u/v/w on Z=0 cross-section over steps.

    Args:
        true(ndarray): The true flow of channel "map_name". shape: (T, H, W, D)
        pred(ndarray): The predicted flow of channel "map_name". shape: (T, H, W, D)
        map_name(str): channel mame including "P", "U", "V", "W".
    """
    l1_error = np.abs(true - pred)

    cross_section = 32
    mesh = generate_mesh()
    xgrid, ygrid = mesh[0, ..., cross_section], mesh[1, ..., cross_section]

    fig = plt.figure(1, figsize=(19, 3))

    def animate(t):
        plt.clf()

        plt.subplot(1, 3, 1)
        vmin, vmax = np.percentile(true[:, :, :, cross_section], [0.5, 99.5])
        plt_contourf(xgrid, ygrid, true[t, :, :, cross_section], vmin, vmax, 'CFD' + '-' + map_name, step=t)

        plt.subplot(1, 3, 2)
        plt_contourf(xgrid, ygrid, pred[t, :, :, cross_section], vmin, vmax, 'ReUnet3D' + '-' + map_name, step=t)

        plt.subplot(1, 3, 3)
        vmin1, vmax1 = np.percentile(l1_error[:, :, :, cross_section], [0.5, 99.5])
        plt_contourf(xgrid, ygrid, l1_error[t, :, :, cross_section], vmin1, vmax1, 'L1 error' + '-' + map_name, step=t,
                     is_fmt=True)
        plt.subplots_adjust(wspace=0.2, hspace=0, left=0.2, bottom=0.2, right=0.9, top=0.9)

    ani = animation.FuncAnimation(fig, animate, frames=len(true), interval=200)
    return ani


def plot_results(save_path):
    """
    plot final results obtained from model.

    Args:
        save_path(str): Path to save prediction results(pred.npz).
    """
    print("================================Start Plotting================================")
    time_beg = time.time()

    data = np.load(os.path.join(save_path, 'pred.npz'))
    true, pred = data['true'].transpose(0, 1, 3, 4, 2), data['pred'].transpose(0, 1, 3, 4, 2)

    for i, map_name in enumerate(['P', 'U', 'V', 'W']):
        ani = plot_gif(true[:, i], pred[:, i], map_name)
        ani.save(os.path.join(save_path, map_name + '.gif'), writer='pillow')  # 保存
        print(f'{map_name}.gif was saved!')

    print("================================End Plotting================================")
    print("Plot total time: {} s".format(time.time() - time_beg))
