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
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sciai.utils import newfig, savefig
from scipy.interpolate import griddata


def plot_train(*inputs):
    """
    plot train
    """
    exact_sol, t_sol, x_sol, x_sol_star_, lb_sol, u_pred, ub_sol, args = inputs
    u_pred_ = griddata(x_sol_star_, u_pred.flatten().asnumpy(), (t_sol, x_sol), method='cubic')
    fig, ax = newfig(1.0, 0.6)
    ax.axis('off')

    gs = gridspec.GridSpec(1, 2)
    gs.update(top=0.8, bottom=0.2, left=0.1, right=0.9, wspace=0.5)
    extent = [lb_sol[0], ub_sol[0], lb_sol[1], ub_sol[1]]

    plot_sub(gs[:, 0], exact_sol, extent, fig, 'Exact Dynamics')  # Exact p(t,x,y)
    plot_sub(gs[:, 1], u_pred_, extent, fig, 'Learned Dynamics')  # Predicted p(t,x,y)

    savefig(f'{args.figures_path}/{args.problem}')


def plot_sub(grid_index, subplot_data, extent, fig, title):
    """
    plot sub figure
    """
    ax = plt.subplot(grid_index)
    h = ax.imshow(subplot_data, interpolation='nearest', cmap='jet',
                  extent=extent,
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)
    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    ax.set_title(title, fontsize=10)
