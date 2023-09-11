
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
from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.interpolate import griddata
from sciai.utils import print_log

from .process import flatten


def plot_elements_namedtuple(args, model, data, test_data, logs=None):
    fields = 'model, epochs, layers, data, test_data, figures_path, logs, gradients_res_layers, gradients_bcs_layers'
    fields_default = ['M4', 40001, [2, 50, 50, 50, 1], None, None, './figures', None, None, None]
    PlotElements = namedtuple('PlotElements', fields, defaults=fields_default)
    return PlotElements(model=args.method,
                        epochs=args.epochs,
                        layers=args.layers,
                        data=data,
                        test_data=test_data,
                        figures_path=args.figures_path,
                        logs=logs,
                        gradients_res_layers=model.dict_gradients_res_layers,
                        gradients_bcs_layers=model.dict_gradients_bcs_layers)


def plot(elements):
    """plotting"""

    print_log("Start plotting results ...")

    fig_path = elements.figures_path
    model = elements.model
    x_star, u_star, u_pred = elements.data
    x1, x2 = elements.test_data

    data_gradients_res = elements.gradients_res_layers
    data_gradients_bcs = elements.gradients_bcs_layers

    Solution = namedtuple('Solution', ['u_pred', 'u_star', 'x1', 'x2', 'x_star'])
    plot_solutions(Solution(u_pred, u_star, x1, x2, x_star), fig_path, model)

    if elements.logs:
        adaptive_constant, loss_res, loss_bcs = elements.logs
        plot_loss(loss_bcs, loss_res, fig_path, model)
        plot_adaptive_constant(adaptive_constant, fig_path, model)
        plot_gradient_distributions(data_gradients_bcs, data_gradients_res, elements.layers, fig_path, model)

    print_log("Done plotting")


def plot_adaptive_constant(adaptive_constant, fig_path, model):
    """plot adaptive constant"""
    fig_3 = plt.figure(3)
    ax = fig_3.add_subplot(1, 1, 1)
    ax.plot(adaptive_constant, label=r'$\lambda_{u_b}$')
    ax.set_xlabel('iterations')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{fig_path}/{model}_adaptive_constant.png')


def plot_gradient_distributions(data_gradients_bcs, data_gradients_res, layers, fig_path, model):
    """plot gradients at the end of training"""
    num_hidden_layers = len(layers) - 1
    cnt = 1
    fig_4 = plt.figure(4, figsize=(13, 4))

    for j in range(num_hidden_layers):
        ax = plt.subplot(1, 4, cnt)
        ax.set_title('Layer {}'.format(j + 1))
        ax.set_yscale('symlog')
        gradients_res = data_gradients_res['layer_' + str(j + 1)][-1].asnumpy()
        gradients_bcs = data_gradients_bcs['layer_' + str(j + 1)][-1].asnumpy()
        sns.distplot(gradients_bcs, hist=False,
                     kde_kws={"shade": False},
                     norm_hist=True, label=r'$\nabla_\theta \lambda_{u_b} \mathcal{L}_{u_b}$')
        sns.distplot(gradients_res, hist=False,
                     kde_kws={"shade": False},
                     norm_hist=True, label=r'$\nabla_\theta \mathcal{L}_r$')

        ax.set_xlim([-3.0, 3.0])
        ax.set_ylim([0, 100])
        cnt += 1
    handles, labels = ax.get_legend_handles_labels()
    fig_4.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.35, -0.01),
                 borderaxespad=0, bbox_transform=fig_4.transFigure, ncol=2)
    plt.tight_layout()
    plt.savefig(f'{fig_path}/{model}_gradient_distributions.png')


def plot_loss(loss_bcs, loss_res, fig_path, model):
    """plot residual loss & boundary loss"""
    fig_2 = plt.figure(2)
    ax = fig_2.add_subplot(1, 1, 1)
    ax.plot(loss_res, label=r'$\mathcal{L}_{r}$')
    ax.plot(loss_bcs, label=r'$\mathcal{L}_{u_b}$')
    ax.set_yscale('log')
    ax.set_xlabel('iterations')
    ax.set_ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{fig_path}/{model}_loss.png')


def plot_solutions(solution, fig_path, model):
    """plot solution to Helmholtz Eqn"""
    # Exact soluton
    u_star = griddata(solution.x_star, flatten(solution.u_star), (solution.x1, solution.x2), method='cubic')
    # Predicted solution
    u_pred = griddata(solution.x_star, flatten(solution.u_pred), (solution.x1, solution.x2), method='cubic')

    _ = plt.figure(1, figsize=(18, 5))
    plt.subplot(1, 3, 1)
    plt.pcolor(solution.x1, solution.x2, u_star, cmap='jet')
    plt.colorbar()
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.title('Exact $u(x)$')
    plt.subplot(1, 3, 2)
    plt.pcolor(solution.x1, solution.x2, u_pred, cmap='jet')
    plt.colorbar()
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.title('Predicted $u(x)$')
    plt.subplot(1, 3, 3)
    plt.pcolor(solution.x1, solution.x2, np.abs(u_star - u_pred), cmap='jet')
    plt.colorbar()
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.title('Absolute error')
    plt.tight_layout()
    plt.savefig(f'{fig_path}/{model}_solutions.png')
