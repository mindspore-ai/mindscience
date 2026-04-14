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

"""cpinns plot"""
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import scipy.stats
from scipy.interpolate import griddata
from sciai.utils import print_log, newfig, savefig


def plot(*inputs):
    """plot"""
    history, t_mesh, x_mesh, x_star, args, model, u_star, x_interface, total_dict = inputs

    plt.figure()
    ax = plt.subplot2grid((1, 1), (0, 0))
    colors = ["b", "r", "g", "c"]
    for x_f_train, color in zip(total_dict["x_f_train_total"], colors):
        ax.scatter(x_f_train[:, 1], x_f_train[:, 0], color=color)
    for x_fi_train in total_dict["x_f_inter_total"]:
        ax.scatter(x_fi_train[:, 1], x_fi_train[:, 0], color='k')
    colors = ['c', 'g', 'b', 'y']
    for x_u_train, color in zip(total_dict["x_u_train_total"], colors):
        ax.scatter(x_u_train[:, 1], x_u_train[:, 0], color=color)
    total_dict["u_pred_total"] = model.predict(*total_dict["x_star_total"])
    u_star_ = griddata(x_star, u_star.flatten(), (x_mesh, t_mesh), method='cubic')
    total_dict["u_err_total"] = [abs(ui_pred - ui_star) for ui_pred, ui_star in
                                 zip(total_dict["u_pred_total"], total_dict["u_star_total"])]
    plot_exact(args, t_mesh, u_star_, x_mesh, total_dict["u_star_total"])
    plot_pred(args, x_interface, total_dict)
    plot_error(args, x_interface, total_dict)
    if history is not None:
        l2error_u = history["l2error_u"]
        plot_a_hist(args, history)
        plot_l2error_u(args, l2error_u)
        save_mat(args, l2error_u)
        plot_mse_hist(args, history)


def plot_error(args, x_interface, total_dict):
    """plot error"""
    u_errs_ = []
    for x, t, x_star, u_err in zip(total_dict["x_sd_total"], total_dict["t_sd_total"],
                                   total_dict["x_star_total"], total_dict["u_err_total"]):
        u_errs_.append(griddata(x_star, u_err.flatten(), (x, t), method='cubic'))
    fig, ax = newfig(1.0, 1.1)
    gridspec.GridSpec(1, 1)
    ax = plt.subplot2grid((1, 1), (0, 0))
    maxerr = max(max(u_err) for u_err in total_dict["u_err_total"])[0]
    levels = np.linspace(0, maxerr, 200)

    cs_errs = []
    for x, t, u_err_ in zip(total_dict["x_sd_total"], total_dict["t_sd_total"], u_errs_):
        cs_errs.append(ax.contourf(t, x, u_err_, levels=levels, cmap='jet', origin='lower'))
    cbar = fig.colorbar(cs_errs[0])
    cbar.ax.tick_params(labelsize=20)
    ax.set_xlim(-0.01, 1)
    ax.set_ylim(-1.01, 1.02)
    ax.set_aspect(0.25)
    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    ax.set_title('$ point-wise error $')
    for xc in x_interface:
        ax.axhline(y=xc, linewidth=1, color='w')
    fig.set_size_inches(w=15, h=8)
    savefig(f'{args.figures_path}/BurError_4sd')


def plot_pred(args, x_interface, total_dict):
    """plot prediction"""
    fig, ax = newfig(1.0, 1.1)
    gridspec.GridSpec(1, 1)
    ax = plt.subplot2grid((1, 1), (0, 0))

    max_level = max(max(u_star) for u_star in total_dict["u_star_total"])[0]
    min_level = min(min(u_star) for u_star in total_dict["u_star_total"])[0]
    levels = np.linspace(min_level - 0.01, max_level + 0.01, 200)

    u_preds_ = []
    for x, t, x_star, u_pred in zip(total_dict["x_sd_total"], total_dict["t_sd_total"],
                                    total_dict["x_star_total"], total_dict["u_pred_total"]):
        u_preds_.append(griddata(x_star, u_pred.flatten(), (x, t), method='cubic'))

    cs_preds = []
    for x, t, u_pred_ in zip(total_dict["x_sd_total"], total_dict["t_sd_total"], u_preds_):
        cs_preds.append(ax.contourf(t, x, u_pred_, levels=levels, cmap='jet', origin='lower'))

    cbar = fig.colorbar(cs_preds[0])
    cbar.ax.tick_params(labelsize=20)
    ax.set_xlim(-0.01, 1)
    ax.set_ylim(-1.01, 1.02)
    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    ax.set_title('$ u^{prediction} $')
    for xc in x_interface:
        ax.axhline(y=xc, linewidth=1, color='w')
    fig.set_size_inches(w=15, h=8)
    savefig(f'{args.figures_path}/BurPred_4sd')


def plot_exact(args, t_mesh, u_star_, x_mesh, u_star_total):
    """plot exact solution"""
    fig, ax = newfig(1.0, 1.1)
    gridspec.GridSpec(1, 1)
    ax = plt.subplot2grid((1, 1), (0, 0))

    max_level = max(max(u_star) for u_star in u_star_total)[0]
    min_level = min(min(u_star) for u_star in u_star_total)[0]
    levels = np.linspace(min_level - 0.01, max_level + 0.01, 200)
    cs_ext1 = ax.contourf(t_mesh, x_mesh, u_star_, levels=levels, cmap='jet', origin='lower')
    cbar = fig.colorbar(cs_ext1)
    cbar.ax.tick_params(labelsize=20)
    ax.set_xlim(-0.01, 1)
    ax.set_ylim(-1.01, 1.02)
    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    ax.set_title('$ u^{Exact} $')
    fig.set_size_inches(w=15, h=8)
    savefig(f'{args.figures_path}/BurExact_4sd')


def plot_l2error_u(args, l2error_u):
    """plot l2 error u"""
    newfig(1.0, 1.1)
    plt.plot(range(1, args.epochs - 1, 10), l2error_u[0:-1], 'b-', linewidth=1)
    plt.xlabel(r'$\#$ iterations')
    plt.ylabel('$L_2$-error')
    plt.yscale('log')
    savefig(f'{args.figures_path}/Bur_L2err_4sd')
    print_log(l2error_u[-1])


def save_mat(args, l2error_u):
    """save l2 error u mat"""
    with open(f'{args.save_data_path}/L2error_Bur4SD_200Wi.mat', mode='wb') as mat_file:
        scipy.io.savemat(mat_file, {'l2error_u': l2error_u})


def plot_mse_hist(args, history):
    """plot mse history"""
    newfig(1.0, 1.1)
    mse_hists = history["loss1"], history["loss2"], history["loss3"], history["loss4"]
    colors = ['r.-', 'b--', 'g-', 'k-']
    labels = [f'Sub-PINN-{i}' for i in range(1, 5)]
    for mse_hist, color, label in zip(mse_hists, colors, labels):
        plt.plot(range(1, args.epochs - 1, 10), mse_hist[0:-1], color, linewidth=1, label=label)
    plt.xlabel(r'$\#$ iterations')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend(loc='upper right')
    savefig(f'{args.figures_path}/Bur_MSEdomain_4sd')


def plot_a_hist(args, history):
    """plot `a` history"""
    newfig(1.0, 1.1)
    a_hists = [np.reshape(history["a" + str(i)], (-1, 1)) for i in range(1, 5)]
    colors = ['r.-', 'b--', 'g:', 'k-']
    labels = ['$a_1$', '$a_2$', '$a_3$', '$a_4$']
    for a_hist, color, label in zip(a_hists, colors, labels):
        plt.plot(range(1, args.epochs - 1, 10), 20 * a_hist[0:-1], color, linewidth=1, label=label)
    plt.legend(loc='lower right')
    plt.xlabel(r'$\#$ iterations')
    savefig(f'{args.figures_path}/Bur_Ahist_4sd')
