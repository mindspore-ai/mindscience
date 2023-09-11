
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
import numpy as np


def plot_fig(x_quad_train, x_u_train, args, total_record, plt_elems):
    """plot"""
    plot_points(args, x_quad_train, x_u_train)
    if total_record:
        plot_loss(args, total_record)
    plot_prediction(args, plt_elems)
    plot_error(args, plt_elems)


def plot_points(args, x_quad_train, x_u_train):
    """plot points"""
    fig = plt.figure(0)
    gridspec.GridSpec(3, 1)
    plot_sub(0, "Quadrature", x_quad_train, 'green')
    plot_sub(1, "Training", x_u_train, 'blue')
    fig.tight_layout()
    fig.set_size_inches(w=10, h=7)
    plt.savefig(f'{args.figures_path}/Train-Quad-pnts.pdf')


def plot_sub(ind, title, x, color):
    """plot sub figure"""
    y = np.ones(len(x))
    plt.subplot2grid((3, 1), (ind, 0))
    plt.tight_layout()
    plt.locator_params(axis='x', nbins=6)
    plt.yticks([])
    plt.title(rf'${title} \,\, Points$')
    plt.xlabel('$x$')
    plt.axhline(1, linewidth=1, linestyle='-', color='red')
    plt.axvline(-1, linewidth=1, linestyle='--', color='red')
    plt.axvline(1, linewidth=1, linestyle='--', color='red')
    plt.scatter(x, y, color=color)


def plot_loss(args, total_record):
    """plot loss"""
    fig, _ = plt.subplots()
    plt.tick_params(axis='y', which='both', labelleft='on', labelright='off')
    plt.xlabel('$iteration$', fontsize=args.font)
    plt.ylabel(r'$loss \,\, values$', fontsize=args.font)
    plt.yscale('log')
    plt.grid(True)
    iteration = [_[0] for _ in total_record]
    loss_his = [_[1] for _ in total_record]
    plt.plot(iteration, loss_his, 'gray')
    plt.tick_params(labelsize=20)
    fig.set_size_inches(w=11, h=5.5)
    plt.savefig(f'{args.figures_path}/loss.pdf')


def plot_prediction(args, plt_elems):
    """plot prediction"""
    grid, u_pred, u_test, x_test = plt_elems
    pnt_skip = 25
    fig, _ = plt.subplots()
    plt.locator_params(axis='x', nbins=6)
    plt.locator_params(axis='y', nbins=8)
    plt.xlabel('$x$', fontsize=args.font)
    plt.ylabel('$u$', fontsize=args.font)
    plt.axhline(0, linewidth=0.8, linestyle='-', color='gray')
    for xc in grid:
        plt.axvline(x=xc, linewidth=2, ls='--')
    plt.plot(x_test, u_test, linewidth=1, color='r', label='$exact$')
    plt.plot(x_test[::pnt_skip], u_pred[::pnt_skip], 'k*', label='$VPINN$')
    plt.tick_params(labelsize=20)
    plt.legend(shadow=True, loc='upper left', fontsize=18, ncol=1)
    fig.set_size_inches(w=11, h=5.5)
    plt.savefig(f'{args.figures_path}/prediction.pdf')


def plot_error(args, plt_elems):
    """plot error"""
    grid, u_pred, u_test, x_test = plt_elems
    fig, _ = plt.subplots()
    plt.locator_params(axis='x', nbins=6)
    plt.locator_params(axis='y', nbins=8)
    plt.xlabel('$x$', fontsize=args.font)
    plt.ylabel('point-wise error', fontsize=args.font)
    plt.yscale('log')
    plt.axhline(0, linewidth=0.8, linestyle='-', color='gray')
    for xc in grid:
        plt.axvline(x=xc, linewidth=2, ls='--')
    err = abs(u_test - u_pred.asnumpy())
    plt.plot(x_test.asnumpy(), err, 'k')
    plt.tick_params(labelsize=20)
    fig.set_size_inches(w=11, h=5.5)
    plt.savefig(f'{args.figures_path}/error.pdf')
