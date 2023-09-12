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
import matplotlib.pyplot as plt
import mindspore.numpy as mnp
import numpy as np


def plot_loss(loss_ics, loss_bcs, loss_res, figures_path):
    """plot the loss"""
    plt.rcParams.update(plt.rcParamsDefault)
    plt.rc('font', family='serif')
    plt.rcParams.update({
        "font.family": "serif",
        'font.size': 20,
        'lines.linewidth': 3,
        'axes.labelsize': 20,  # fontsize for x and y labels (was 10)
        'axes.titlesize': 22,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'legend.fontsize': 20,
        'axes.linewidth': 2})
    # Losses
    plt.figure(figsize=(6, 5))
    iters = 1000 * mnp.arange(len(loss_ics))
    plt.plot(iters, loss_bcs, lw=2, label='$L_{bc}$')
    plt.plot(iters, loss_ics, lw=2, label='$L_{ic}$')
    plt.plot(iters, loss_res, lw=2, label='$L_{r}$')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend()
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    plt.savefig(f"{figures_path}/loss_history.png")


def plot_train(*inputs):
    """plot the result"""
    d_t, n_test, nt_test, nx_test, s_pred_, s_test, figures_path = inputs
    x = np.linspace(0, 1, nx_test)
    t = np.linspace(0, n_test * d_t, nt_test)
    xx, tt = np.meshgrid(x, t)
    # Prediction
    plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1)
    plt.pcolor(xx, tt, s_test.T.asnumpy(), cmap='seismic')
    plt.colorbar()
    plt.subplot(1, 3, 2)
    plt.pcolor(xx, tt, s_pred_.asnumpy(), cmap='seismic')
    plt.colorbar()
    plt.subplot(1, 3, 3)
    plt.pcolor(xx, tt, np.abs(s_pred_.asnumpy() - s_test.T.asnumpy()), cmap='seismic')
    plt.colorbar()
    plt.savefig(f"{figures_path}/result.png")
