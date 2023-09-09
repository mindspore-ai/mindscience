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
"""plot functions"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.interpolate import griddata


def plot_train_val(*args):
    """plot train val"""
    figures_path, log_dict, u_pred, x_star, _, u_star, t, x = args

    u_star_list = griddata(x_star, u_star.flatten(), (t, x), method='cubic')
    u_pred_list = griddata(x_star, u_pred.flatten(), (t, x), method='cubic')
    u_err_list = np.abs(u_star_list - u_pred_list)

    plt.figure(1, figsize=(18, 5))
    plot_element(t, x, u_star_list, 1, 'Exact')
    plot_element(t, x, u_pred_list, 2, 'Predicted')
    plot_element(t, x, u_err_list, 3, 'Absolute error')
    plt.savefig(f"{figures_path}/absolute_error.png")
    if log_dict:
        plt.figure(2, figsize=(6, 5))
        iters = 100 * np.arange(len(log_dict["loss_res_log"]))
        with sns.axes_style("darkgrid"):
            plt.plot(iters, log_dict["loss_res_log"], label=r'$\mathcal{L}_{r}$', linewidth=2)
            plt.plot(iters, log_dict["loss_bcs_log"], label=r'$\mathcal{L}_{bc}$', linewidth=2)
            plt.plot(iters, log_dict["loss_ics_log"], label=r'$\mathcal{L}_{ic}$', linewidth=2)
            plt.plot(iters, log_dict["l2_error_log"], label=r'$L^2$ error', linewidth=2)
            plt.yscale('log')
            plt.xlabel('iterations')
            plt.legend(ncol=2, fontsize=17)
            plt.tight_layout()
            plt.savefig(f"{figures_path}/loss.png")


def plot_element(t, x, u_star_list, plot_index, title):
    """plot element"""
    plt.subplot(1, 3, plot_index)
    plt.pcolor(t, x, u_star_list, cmap='jet')
    plt.colorbar()
    plt.xlabel('$t$')
    plt.ylabel('$x$')
    plt.title(title)
    plt.tight_layout()
