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
"""plotting functions"""
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sciai.utils.plot_utils import save_result_dir, newfig
from scipy.interpolate import griddata


def plot_inf_cont_results(*inputs, save_path=None, save_hp=None):
    """Plot 2d domain and three slices"""
    x_star, u_pred, x_train, u_train, u, x_mesh, t_mesh, x, t = inputs
    x_u_pred = griddata(x_star, u_pred, (x_mesh, t_mesh), method='cubic')

    fig, ax = newfig(1.0, 1.1)
    ax.axis('off')

    # Plot the 2D domain
    plot_domain(fig, x, t, x_u_pred, x_train, u_train)

    # Plot three time slices
    gs1 = gridspec.GridSpec(1, 3)
    gs1.update(top=1 - 1 / 3, bottom=0, left=0.1, right=0.9, wspace=0.5)
    plot_slice(gs1, u, x, x_u_pred)

    if save_path is None or save_hp is None:
        plt.show()
    else:
        save_result_dir(save_path, save_hp)


def plot_slice(gs1, u, x, x_u_pred):
    """Plot slices"""
    ax = plt.subplot(gs1[0, 0])
    ax.plot(x, u[25, :], 'b-', linewidth=2, label='Exact')
    ax.plot(x, x_u_pred[25, :], 'r--', linewidth=2, label='Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$T(t,x)$')
    ax.set_title('$t = 0.25$', fontsize=10)
    ax.axis('square')
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])

    ax = plt.subplot(gs1[0, 1])
    ax.plot(x, u[50, :], 'b-', linewidth=2, label='Exact')
    ax.plot(x, x_u_pred[50, :], 'r--', linewidth=2, label='Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$T(t,x)$')
    ax.axis('square')
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.set_title('$t = 0.50$', fontsize=10)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), ncol=5, frameon=False)

    ax = plt.subplot(gs1[0, 2])
    ax.plot(x, u[75, :], 'b-', linewidth=2, label='Exact')
    ax.plot(x, x_u_pred[75, :], 'r--', linewidth=2, label='Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$T(t,x)$')
    ax.axis('square')
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.set_title('$t = 0.75$', fontsize=10)


def plot_domain(*inputs):
    """Plot domain"""
    fig, x, t, x_u_pred, x_train, u_train = inputs
    gs0 = gridspec.GridSpec(1, 2)
    gs0.update(top=1 - 0.06, bottom=1 - 1 / 3, left=0.15, right=0.85, wspace=0)
    ax = plt.subplot(gs0[:, :])
    h = ax.imshow(x_u_pred.T, interpolation='nearest', cmap='rainbow',
                  extent=[t.min(), t.max(), x.min(), x.max()],
                  origin='lower', aspect='auto')
    ax.plot(x_train[:, 1], x_train[:, 0], 'kx', label='Data (%d points)' % (u_train.shape[0]), markersize=4,
            clip_on=False)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)
    line = np.linspace(x.min(), x.max(), 2)[:, None]
    ax.plot(t[25] * np.ones((2, 1)), line, 'w-', linewidth=1)
    ax.plot(t[50] * np.ones((2, 1)), line, 'w-', linewidth=1)
    ax.plot(t[75] * np.ones((2, 1)), line, 'w-', linewidth=1)
    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    ax.legend(frameon=False, loc='best')
    ax.set_title('$T(t,x)$', fontsize=10)
