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

"""dgm plot"""
import matplotlib.pyplot as plt
import numpy as np
from sciai.utils import print_log


def visualize(advection, figures_path, x_range, y):
    """visualize"""
    path = figures_path + "/exact_solution.png"
    fig = plt.figure()
    plt.plot(x_range, y)
    plt.plot(x_range, advection.exact_solution(x_range), '--')
    plt.savefig(path)
    plt.close(fig)
    # Error - diff exact and obtained solution
    path = figures_path + "/error.png"
    fig = plt.figure()
    plt.plot(x_range, y - advection.exact_solution(x_range))
    plt.savefig(path)
    plt.close(fig)


def plot_activation_mean(train_process):
    """plot activation mean"""
    if not train_process.debug:
        print_log("plot activation mean: debug is off , turn it on and train again")
        return
    history = np.array(train_process.history_mean_hooks)
    jet = plt.get_cmap('jet')
    colors = iter(jet(np.linspace(0, 1, 10)))
    fig, ax = plt.subplots()
    for i in range(history.shape[1]):
        ax.plot(history[:, i], '--', label=i, color=next(colors))

    fig.suptitle('Layers activation mean value', fontsize=10)
    path = train_process.figures_path + "/activation.png"
    plt.savefig(path)
    plt.close(fig)


def plot_report(train_process):
    """plot report"""
    if not train_process.debug:
        print_log("plot report: debug is off , turn it on and train again")
        return
    fig, ax = plt.subplots(3, 1, constrained_layout=True)
    ax[0].plot(np.log(train_process.history_tl), '-b', label='total')

    ax[0].set_title('total')
    fig.suptitle('Training Loss', fontsize=10)

    ax[1].plot(np.log(train_process.history_dl))
    ax[1].set_title('diff operator')

    ax[2].plot(np.log(train_process.history_il))
    ax[2].set_title('initial condition')

    path = train_process.figures_path + "/report.png"
    plt.savefig(path)
    plt.close(fig)
