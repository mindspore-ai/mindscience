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
"""plotting results"""
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib.patches import Polygon
from sciai.utils.plot_utils import newfig, savefig


class PlotElements:
    """Elements need for plotting"""
    def __init__(self, **kwargs):
        self.epochs = kwargs.get('epochs')
        self.mse_hists = kwargs.get('mse_hists')
        self.x_f_trains = kwargs.get('x_f_trains')
        self.x_fi_trains = kwargs.get('x_fi_trains')
        self.x_ub_train = kwargs.get('x_ub_train')
        self.l2_errs = kwargs.get('l2_errs')
        self.u = kwargs.get('u')
        self.x_stars = kwargs.get('x_stars')
        self.x = kwargs.get('x')
        self.y = kwargs.get('y')
        self.figures_path = kwargs.get('figures_path')


def plot(elements):
    """plot"""
    epochs = elements.epochs
    x_f1_train, x_f2_train, x_f3_train = elements.x_f_trains
    x_fi1_train, x_fi2_train = elements.x_fi_trains
    x_ub_train = elements.x_ub_train
    u_exact, u_pred = elements.u
    x_stars = elements.x_stars
    xb, xi1, xi2 = elements.x
    yb, yi1, yi2 = elements.y
    triang_total, x_fi1_train_plot, x_fi2_train_plot, xx = plot_preprocess(x_stars, xb, xi1, xi2, yb, yi1, yi2)
    if epochs and elements.mse_hists:
        plot_mse_history(epochs, elements.mse_hists, elements.figures_path)
    if epochs and elements.l2_errs:
        plot_err_history(epochs, elements.l2_errs, elements.figures_path)
    plot_exact_solution(xx, x_fi1_train_plot, x_fi2_train_plot, triang_total, u_exact, elements.figures_path)
    plot_prediction_solution(x_fi1_train_plot, x_fi2_train_plot, triang_total, u_pred, xx, elements.figures_path)
    plot_prediction_error(x_fi1_train_plot, x_fi2_train_plot, triang_total, u_exact, u_pred, xx, elements.figures_path)
    plot_data_points(x_f1_train, x_f2_train, x_f3_train, x_fi1_train, x_fi2_train, x_ub_train, elements.figures_path)


def plot_preprocess(*inputs):
    """plot preprocess"""
    x_stars, xb, xi1, xi2, yb, yi1, yi2 = inputs
    x_star1, x_star2, x_star3 = x_stars
    x1, y1 = x_star1[:, 0:1], x_star1[:, 1:2]
    x2, y2 = x_star2[:, 0:1], x_star2[:, 1:2]
    x3, y3 = x_star3[:, 0:1], x_star3[:, 1:2]
    x_tot = np.concatenate([x1, x2, x3])
    y_tot = np.concatenate([y1, y2, y3])

    aa1 = np.array([[np.squeeze(xb[-1]), np.squeeze(yb[-1])]])
    aa2 = np.array(
        [[1.8, np.squeeze(yb[-1])], [+1.8, -1.7], [-1.6, -1.7], [-1.6, 1.55], [1.8, 1.55], [1.8, np.squeeze(yb[-1])]])
    x_domain1 = np.squeeze(xb.flatten()[:, None])
    y_domain1 = np.squeeze(yb.flatten()[:, None])
    aa3 = np.array([x_domain1, y_domain1]).T
    xx = np.vstack((aa3, aa2, aa1))
    triang_total = tri.Triangulation(x_tot.flatten(), y_tot.flatten())
    x_fi1_train_plot = np.hstack((xi1.flatten()[:, None], yi1.flatten()[:, None]))
    x_fi2_train_plot = np.hstack((xi2.flatten()[:, None], yi2.flatten()[:, None]))
    return triang_total, x_fi1_train_plot, x_fi2_train_plot, xx


def plot_mse_history(epochs, mse_hists, figures_path):
    """plot mse history"""
    newfig(1.0, 1.1)
    mse_hist1, mse_hist2, mse_hist3 = mse_hists
    plt.plot(range(1, epochs + 1, 20), mse_hist1, 'r-', linewidth=1, label='Sub-Net1')
    plt.plot(range(1, epochs + 1, 20), mse_hist2, 'b-.', linewidth=1, label='Sub-Net2')
    plt.plot(range(1, epochs + 1, 20), mse_hist3, 'g--', linewidth=1, label='Sub-Net3')
    plt.xlabel(r'$\#$ iterations')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend(loc='upper right')
    savefig(f'{figures_path}/XPINN_PoissonMSEhistory')


def plot_err_history(epochs, l2_errs, figures_path):
    "plot error history"
    newfig(1.0, 1.1)
    l2_err2, l2_err3 = l2_errs
    plt.plot(range(1, epochs + 1, 20), l2_err2, 'r-', linewidth=1, label='Subdomain 2')
    plt.plot(range(1, epochs + 1, 20), l2_err3, 'b--', linewidth=1, label='Subdomain 3')
    plt.xlabel(r'$\#$ iterations')
    plt.ylabel('Rel. $L_2$ error')
    plt.yscale('log')
    plt.legend(loc='upper right')
    savefig(f'{figures_path}/XPINN_PoissonErrhistory')


def plot_exact_solution(*inputs):
    """plot exact solution"""
    xx, x_fi1_train_plot, x_fi2_train_plot, triang_total, u_exact, figures_path = inputs
    fig, ax = newfig(1.0, 1.1)
    gridspec.GridSpec(1, 1)
    ax = plt.subplot2grid((1, 1), (0, 0))
    tcf = ax.tricontourf(triang_total, np.squeeze(u_exact), 100, cmap='jet')
    ax.add_patch(Polygon(xx, closed=True, fill=True, color='w', edgecolor='w'))
    tcbar = fig.colorbar(tcf)
    tcbar.ax.tick_params(labelsize=28)
    ax.set_xlabel('$x$', fontsize=32)
    ax.set_ylabel('$y$', fontsize=32)
    ax.set_title('$u$ (Exact)', fontsize=34)
    ax.tick_params(axis="x", labelsize=28)
    ax.tick_params(axis="y", labelsize=28)
    plt.plot(x_fi1_train_plot[:, 0:1], x_fi1_train_plot[:, 1:2], 'w-', markersize=2, label='Interface Pts')
    plt.plot(x_fi2_train_plot[:, 0:1], x_fi2_train_plot[:, 1:2], 'w-', markersize=2, label='Interface Pts')
    fig.set_size_inches(w=12, h=9)
    savefig(f'{figures_path}/XPINN_PoissonEq_ExSol')


def plot_prediction_solution(*inputs):
    """plot prediction solution"""
    x_fi1_train_plot, x_fi2_train_plot, triang_total, u_pred, xx, figures_path = inputs
    fig, ax = newfig(1.0, 1.1)
    gridspec.GridSpec(1, 1)
    ax = plt.subplot2grid((1, 1), (0, 0))
    tcf = ax.tricontourf(triang_total, u_pred.flatten(), 100, cmap='jet')
    ax.add_patch(Polygon(xx, closed=True, fill=True, color='w', edgecolor='w'))
    tcbar = fig.colorbar(tcf)
    tcbar.ax.tick_params(labelsize=28)
    ax.set_xlabel('$x$', fontsize=32)
    ax.set_ylabel('$y$', fontsize=32)
    ax.set_title('$u$ (Predicted)', fontsize=34)
    ax.tick_params(axis="x", labelsize=28)
    ax.tick_params(axis="y", labelsize=28)
    plt.plot(x_fi1_train_plot[:, 0:1], x_fi1_train_plot[:, 1:2], 'w-', markersize=2, label='Interface Pts')
    plt.plot(x_fi2_train_plot[:, 0:1], x_fi2_train_plot[:, 1:2], 'w-', markersize=2, label='Interface Pts')
    fig.set_size_inches(w=12, h=9)
    savefig(f'{figures_path}/XPINN_PoissonEq_Sol')


def plot_prediction_error(*inputs):
    """plot prediction error"""
    x_fi1_train_plot, x_fi2_train_plot, triang_total, u_exact, u_pred, xx, figures_path = inputs
    fig, ax = newfig(1.0, 1.1)
    gridspec.GridSpec(1, 1)
    ax = plt.subplot2grid((1, 1), (0, 0))
    tcf = ax.tricontourf(triang_total, abs(np.squeeze(u_exact) - u_pred.flatten()), 100, cmap='jet')
    ax.add_patch(Polygon(xx, closed=True, fill=True, color='w', edgecolor='w'))
    tcbar = fig.colorbar(tcf)
    tcbar.ax.tick_params(labelsize=28)
    ax.set_xlabel('$x$', fontsize=32)
    ax.set_ylabel('$y$', fontsize=32)
    ax.set_title('Point-wise Error', fontsize=34)
    ax.tick_params(axis="x", labelsize=28)
    ax.tick_params(axis="y", labelsize=28)
    plt.plot(x_fi1_train_plot[:, 0:1], x_fi1_train_plot[:, 1:2], 'w-', markersize=2, label='Interface Pts')
    plt.plot(x_fi2_train_plot[:, 0:1], x_fi2_train_plot[:, 1:2], 'w-', markersize=2, label='Interface Pts')
    fig.set_size_inches(w=12, h=9)
    savefig(f'{figures_path}/XPINN_PoissonEq_Err')


def plot_data_points(*inputs):
    """plot data points"""
    x_f1_train, x_f2_train, x_f3_train, x_fi1_train, x_fi2_train, x_ub_train, figures_path = inputs
    fig, ax = newfig(1.0, 1.1)
    gridspec.GridSpec(1, 1)
    ax = plt.subplot2grid((1, 1), (0, 0))
    plt.plot(x_f1_train[:, 0:1], x_f1_train[:, 1:2], 'r*', markersize=4, label='Residual Pts  (sub-domain 1)')
    plt.plot(x_f2_train[:, 0:1], x_f2_train[:, 1:2], 'yo', markersize=4, label='Residual Pts (sub-domain 2)')
    plt.plot(x_f3_train[:, 0:1], x_f3_train[:, 1:2], 'gs', markersize=4, label='Residual Pts (sub-domain 3)')
    plt.plot(x_fi1_train[:, 0:1], x_fi1_train[:, 1:2], 'bs', markersize=7, label='Interface Pts 1')
    plt.plot(x_fi2_train[:, 0:1], x_fi2_train[:, 1:2], 'bs', markersize=7, label='Interface Pts 1')
    plt.plot(x_ub_train[:, 0:1], x_ub_train[:, 1:2], 'kx', markersize=9, label='Interface Pts 1')
    ax.set_xlabel('$x$', fontsize=30)
    ax.set_ylabel('$y$', fontsize=30)
    ax.tick_params(axis="x", labelsize=26)
    ax.tick_params(axis="y", labelsize=26)
    fig.set_size_inches(w=12, h=12)
    savefig(f'{figures_path}/XPINN_Poisson_dataPts')
