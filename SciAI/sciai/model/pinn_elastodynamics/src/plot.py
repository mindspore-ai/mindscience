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
# ============================================================================
"""plot"""
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np


def plot_uvmag(*args):
    """plot uvmag"""
    x_star, y_star, u_star, v_star, a_star, x_pred, y_pred, u_pred, v_pred, figures_path, num, amp_pred = args
    cmap = plt.get_cmap('seismic')
    new_cmap = truncate_colormap(cmap, 0.5, 1.0)
    xmin, xmax, ymin, ymax, s, dpi = -15, 15, -15, 15, 4, 150
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(14, 9))
    fig.subplots_adjust(hspace=0.15, wspace=0.1)
    ax[0, 0].scatter(x_pred, y_pred, c=u_pred, alpha=0.9, edgecolors='none', cmap='seismic', marker='o', s=s,
                     vmin=-0.5, vmax=0.5)
    set_ax(ax[0, 0], xmin, xmax, ymin, ymax, r'$u$-PINN')
    ax[1, 0].scatter(x_star, y_star, c=u_star, alpha=0.9, edgecolors='none', cmap='seismic', marker='o', s=s,
                     vmin=-0.5, vmax=0.5)
    set_ax(ax[1, 0], xmin + 15, xmax + 15, ymin + 15, ymax + 15, r'$u$-FEM')
    cf = ax[0, 1].scatter(x_pred, y_pred, c=v_pred, alpha=0.9, edgecolors='none', cmap='seismic', marker='o', s=s,
                          vmin=-0.5, vmax=0.5)
    set_ax(ax[0, 1], xmin, xmax, ymin, ymax, r'$v$-PINN')
    cf.cmap.set_under('whitesmoke')
    cf.cmap.set_over('black')
    cf = ax[1, 1].scatter(x_star, y_star, c=v_star, alpha=0.9, edgecolors='none', cmap='seismic', marker='o', s=s,
                          vmin=-0.5, vmax=0.5)
    set_ax(ax[1, 1], xmin + 15, xmax + 15, ymin + 15, ymax + 15, r'$v$-FEM')
    cf.cmap.set_under('whitesmoke')
    cf.cmap.set_over('black')
    ax[0, 2].scatter(x_pred, y_pred, c=amp_pred, alpha=0.9, edgecolors='none', cmap=new_cmap, marker='o', s=s,
                     vmin=0, vmax=0.5)
    set_ax(ax[0, 2], xmin, xmax, ymin, ymax, 'Mag.-PINN')
    ax[1, 2].scatter(x_star, y_star, c=a_star, alpha=0.9, edgecolors='none', cmap=new_cmap, marker='o', s=s,
                     vmin=0, vmax=0.5)
    set_ax(ax[1, 2], xmin + 15, xmax + 15, ymin + 15, ymax + 15, 'Mag.-FEM')
    plt.savefig(f'{figures_path}/uv_comparison_' + str(num).zfill(3) + '.png', dpi=dpi)
    plt.close('all')


def set_ax(*inputs):
    """set axis"""
    sub_ax, xmin, xmax, ymin, ymax, title = inputs
    sub_ax.axis('square')
    sub_ax.set_xticks([])
    sub_ax.set_yticks([])
    sub_ax.set_xlim([xmin, xmax])
    sub_ax.set_ylim([ymin, ymax])
    sub_ax.set_title(title, fontsize=22)


def plot_sigma(*args):
    """plot sigma"""
    x_star, y_star, s11_star, s22_star, s12_star, x_pred, y_pred, s11_pred, s22_pred, s12_pred, figures_path, num = args

    xmin, xmax, ymin, ymax, s, dpi = -15, 15, -15, 15, 4, 150
    # Plot predicted stress
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(14, 9))
    fig.subplots_adjust(hspace=0.15, wspace=0.1)
    ax[0, 0].scatter(x_pred, y_pred, c=s11_pred, alpha=0.9, edgecolors='none', marker='o', cmap='seismic', s=s,
                     vmin=-1.5, vmax=1.5)
    set_ax(ax[0, 0], xmin, xmax, ymin, ymax, r'$\sigma_{11}$-PINN')
    ax[1, 0].scatter(x_star, y_star, c=s11_star, alpha=0.9, edgecolors='none', marker='s', cmap='seismic', s=s,
                     vmin=-1.5, vmax=1.5)
    set_ax(ax[1, 0], xmin + 15, xmax + 15, ymin + 15, ymax + 15, r'$\sigma_{11}$-FEM')
    ax[0, 1].scatter(x_pred, y_pred, c=s22_pred, alpha=0.7, edgecolors='none', marker='s', cmap='seismic', s=s,
                     vmin=-1.5, vmax=1.5)
    set_ax(ax[0, 1], xmin, xmax, ymin, ymax, r'$\sigma_{22}$-PINN')
    ax[1, 1].scatter(x_star, y_star, c=s22_star, alpha=0.7, edgecolors='none', marker='s', cmap='seismic', s=s,
                     vmin=-1.5, vmax=1.5)
    set_ax(ax[1, 1], xmin + 15, xmax + 15, ymin + 15, ymax + 15, r'$\sigma_{22}$-FEM')
    ax[0, 2].scatter(x_pred, y_pred, c=s12_pred, alpha=0.7, edgecolors='none', marker='s', cmap='seismic', s=s,
                     vmin=-1.5, vmax=1.5)
    set_ax(ax[0, 2], xmin, xmax, ymin, ymax, r'$\sigma_{12}$-PINN')
    ax[1, 2].scatter(x_star, y_star, c=s12_star, alpha=0.7, edgecolors='none', marker='s', cmap='seismic', s=s,
                     vmin=-1.5, vmax=1.5)
    set_ax(ax[1, 2], xmin + 15, xmax + 15, ymin + 15, ymax + 15, r'$\sigma_{12}$-FEM')
    plt.savefig(f'{figures_path}/stress_comparison_' + str(num).zfill(3) + '.png', dpi=dpi)
    plt.close('all')


def truncate_colormap(old_cmap, minval=0.0, maxval=1.0, n=100):
    """truncate color map"""
    cmap_new = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=old_cmap.name, a=minval, b=maxval),
        old_cmap(np.linspace(minval, maxval, n)))
    return cmap_new
