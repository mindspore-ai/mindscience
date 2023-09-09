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
"""pinns_swe plot"""
import numpy as np
from matplotlib import pyplot as plt


def plot_init_solution(problem, args):
    """plot initial solution"""
    plt.rcParams.update({'font.size': 14})

    x = np.linspace(problem.lmbd_left, problem.lmbd_right, 100)
    y = np.linspace(problem.tht_lower, problem.tht_upper, 100)
    x_mesh, y_mesh = np.meshgrid(x, y)
    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    plt.contourf(360 / 2 / np.pi * x_mesh, 360 / 2 / np.pi * y_mesh,
                 problem.h * problem.h0(x_mesh, y_mesh), 100, cmap=plt.cm.coolwarm)
    plt.colorbar()
    plt.subplot(132)
    plt.contourf(360 / 2 / np.pi * x_mesh, 360 / 2 / np.pi * y_mesh,
                 problem.u * problem.u0(x_mesh, y_mesh), 100, cmap=plt.cm.coolwarm)
    plt.colorbar()
    plt.subplot(133)
    plt.contourf(360 / 2 / np.pi * x_mesh, 360 / 2 / np.pi * y_mesh,
                 problem.v0(x_mesh), 100, cmap=plt.cm.coolwarm)
    plt.colorbar()
    plt.savefig(f"{args.figures_path}/initial_solution.png")


def plot_loss(problem, args, loss):
    plt.figure(figsize=(8, 5))
    plt.subplot(111)
    plt.semilogy(range(0, problem.epochs), loss)
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.grid()
    plt.savefig(f'{args.figures_path}/loss.png')


def plot_comparison_with_truth(problem, args, pdes):
    """plot comparison with truth"""
    t_p, x_p, y_p, h = grid_to_plot(problem, pdes)

    for i in range(0, problem.t_final + 1):
        plt.figure(figsize=(15, 5))
        plt.subplot(131)
        plt.contourf(x_p[i], y_p[i], h[i], 100, cmap=plt.cm.coolwarm)
        plt.title('Neural network solution')
        plt.xlabel(r'$\lambda$')
        plt.ylabel(r'$\theta$')
        plt.colorbar()

        plt.subplot(132)
        h_true = problem.true_solution(t_p[i], x_p[i], y_p[i])
        plt.contourf(x_p[i], y_p[i], h_true, 100, cmap=plt.cm.coolwarm)
        plt.title('True solution')
        plt.xlabel(r'$\lambda$')
        plt.ylabel(r'$\theta$')
        plt.colorbar()

        plt.subplot(133)
        plt.contourf(x_p[i], y_p[i], h[i] - h_true, 100, cmap=plt.cm.coolwarm)
        plt.xlabel(r'$\lambda$')
        plt.ylabel(r'$\theta$')
        plt.title("Difference NN solution and true solution")
        plt.colorbar()

        plt.savefig(f"{args.figures_path}/pict{i}.png")


def plot_comparison_with_initial(problem, args, pdes):
    """plot comparison with initial"""
    _, x_p, y_p, h = grid_to_plot(problem, pdes)
    plot_steps = [0, 1, 2, 3, 4]
    for i in plot_steps:
        plt.figure(figsize=(8, 5))
        plt.subplot(111)
        plt.contourf(x_p[i], y_p[i], problem.h00 * h[i * 3], np.linspace(-28, problem.h00 + 20, 200),
                     cmap=plt.cm.coolwarm)
        plt.colorbar(ticks=[0 + 200 * j for j in range(6)])
        plt.title('Solution for h on day {}'.format(i * 3))
        plt.xlabel(r'$\lambda$')
        plt.ylabel(r'$\theta$')
        plt.savefig(f"{args.figures_path}/Williamson_1_day_{i * 3}.png")

    plt.figure(figsize=(8, 5))
    plt.subplot(111)
    plt.contourf(x_p[0], y_p[0], problem.h00 * (h[0] - h[-1]), 200, cmap=plt.cm.coolwarm)
    plt.colorbar()
    plt.title('Difference initial and final solution')
    plt.xlabel(r'$\lambda$')
    plt.ylabel(r'$\theta$')
    plt.savefig(f"{args.figures_path}/Williamson_1_difference_initial_final.png")


def grid_to_plot(problem, pdes):
    """grid to plot"""
    l, m, n = problem.t_final + 1, 150, 75
    t = np.linspace(problem.t0, problem.t_final, l)
    x = np.linspace(problem.lmbd_left, problem.lmbd_right, m)
    y = np.linspace(problem.tht_lower, problem.tht_upper, n)

    t_p, x_p, y_p = np.meshgrid(t, x, y, indexing='ij')
    tt = np.expand_dims(t_p.flatten(), axis=1)
    xx = np.expand_dims(x_p.flatten(), axis=1)
    yy = np.expand_dims(y_p.flatten(), axis=1)

    h = problem.predict(pdes, tt, xx, yy)
    h = np.reshape(h, (l, m, n))

    return t_p, x_p, y_p, h
