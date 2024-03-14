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
# ==============================================================================
"""Plot"""
import matplotlib.pyplot as plt
import numpy as np


def change_alpha(x):
    alpha = np.zeros(x.shape)
    alpha[x >= 2] = 1
    return alpha


def plt_img(field, label, idx, plot_evo=False, evo=None, interval=10, fig_name="", vmin=1, vmax=40, cmap="viridis"):
    """plot figures"""
    if plot_evo:
        _, axs = plt.subplots(3, 3)
    else:
        _, axs = plt.subplots(2, 3)
    axs[0][0].set_axis_off()
    axs[0][1].set_axis_off()
    axs[0][2].set_axis_off()
    axs[1][0].set_axis_off()
    axs[1][1].set_axis_off()
    axs[1][2].set_axis_off()
    alpha = change_alpha(label[idx[0]])
    _ = axs[0][0].imshow(label[idx[0]], alpha=alpha, vmin=vmin, vmax=vmax, cmap=cmap)
    axs[0][0].set_title(f"label {idx[0] * interval + interval} min")
    alpha = change_alpha(label[idx[1]])
    _ = axs[0][1].imshow(label[idx[1]], alpha=alpha, vmin=vmin, vmax=vmax, cmap=cmap)
    axs[0][1].set_title(f"label {idx[1] * interval + interval} min")
    alpha = change_alpha(label[idx[2]])
    _ = axs[0][2].imshow(label[idx[2]], alpha=alpha, vmin=vmin, vmax=vmax, cmap=cmap)
    axs[0][2].set_title(f"label {idx[2] * interval + interval} min")
    alpha = change_alpha(field[idx[0]])
    _ = axs[1][0].imshow(field[idx[0]], alpha=alpha, vmin=vmin, vmax=vmax, cmap=cmap)
    axs[1][0].set_title(f"pred {idx[0] * interval + interval} min")
    alpha = change_alpha(field[idx[1]])
    _ = axs[1][1].imshow(field[idx[1]], alpha=alpha, vmin=vmin, vmax=vmax, cmap=cmap)
    axs[1][1].set_title(f"pred {idx[1] * interval + interval} min")
    alpha = change_alpha(field[idx[2]])
    _ = axs[1][2].imshow(field[idx[2]], alpha=alpha, vmin=vmin, vmax=vmax, cmap=cmap)
    axs[1][2].set_title(f"pred {idx[2] * interval + interval} min")
    if plot_evo:
        axs[2][0].set_axis_off()
        axs[2][1].set_axis_off()
        axs[2][2].set_axis_off()
        alpha = change_alpha(evo[idx[0]])
        _ = axs[2][0].imshow(evo[idx[0]], alpha=alpha, vmin=vmin, vmax=vmax, cmap=cmap)
        axs[2][0].set_title(f"evo results {idx[0] * interval + interval} min")
        alpha = change_alpha(evo[idx[1]])
        _ = axs[2][1].imshow(evo[idx[1]], alpha=alpha, vmin=vmin, vmax=vmax, cmap=cmap)
        axs[2][1].set_title(f"evo results {idx[1] * interval + interval} min")
        alpha = change_alpha(evo[idx[2]])
        _ = axs[2][2].imshow(evo[idx[2]], alpha=alpha, vmin=vmin, vmax=vmax, cmap=cmap)
        axs[2][2].set_title(f"evo results {idx[2] * interval + interval} min")
    plt.savefig(fig_name, dpi=180)
    plt.close()
