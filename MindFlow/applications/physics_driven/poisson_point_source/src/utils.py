# ============================================================================
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
"""utility functions"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size


def plot_2d(u_label, u_predict, file_name=None):
    """Draw the image containing the label and the prediction."""
    u_error = np.abs(u_label - u_predict)

    vmin_u = u_label.min()
    vmax_u = u_label.max()
    vmin_error = u_error.min()
    vmax_error = u_error.max()
    vmin = [vmin_u, vmin_u, vmin_error]
    vmax = [vmax_u, vmax_u, vmax_error]

    sub_titles = ["Reference", "Predict", "Error"]

    plt.rcParams['figure.figsize'] = [9.6, 3.2]
    fig = plt.figure()
    gs_ = gridspec.GridSpec(2, 6)
    slice_ = [gs_[0:2, 0:2], gs_[0:2, 2:4], gs_[0:2, 4:6]]
    for i, data in enumerate([u_label, u_predict, u_error]):
        ax_ = fig.add_subplot(slice_[i])

        img = ax_.imshow(
            data.T, vmin=vmin[i],
            vmax=vmax[i],
            cmap=plt.get_cmap("jet"),
            origin='lower')

        ax_.set_title(sub_titles[i], fontsize=10)
        plt.xticks(())
        plt.yticks(())

        aspect = 20
        pad_fraction = 0.5
        divider = make_axes_locatable(ax_)
        width = axes_size.AxesY(ax_, aspect=1 / aspect)
        pad = axes_size.Fraction(pad_fraction, width)
        cax = divider.append_axes("right", size=width, pad=pad)
        cb_ = plt.colorbar(img, cax=cax)
        cb_.ax.tick_params(labelsize=6)

    gs_.tight_layout(fig, pad=1.0, w_pad=3.0, h_pad=1.0)
    if file_name is None:
        plt.show()
    else:
        os.makedirs("./images", exist_ok=True)
        fig.savefig(os.path.join("./images", file_name))
    plt.close()


def visual(model, ds_test, n_samps_per_axis=100, file_name=None):
    """Visual comparison of label and prediction"""
    mesh, label = ds_test[0], ds_test[1]
    pred = model(mesh).asnumpy()
    label = label.asnumpy()
    plot_2d(label.reshape(n_samps_per_axis, n_samps_per_axis),
            pred.reshape(n_samps_per_axis, n_samps_per_axis),
            file_name=file_name)


def calculate_l2_error(model, ds_test):
    """Calculate the relative L2 error."""
    mesh, label = ds_test[0], ds_test[1]
    pred = model(mesh).asnumpy().flatten()
    label = label.asnumpy().flatten()

    error_norm = np.linalg.norm(pred - label, ord=2)
    label_norm = np.linalg.norm(label, ord=2)
    relative_l2_error = error_norm / label_norm

    print(f"Relative L2 error: {relative_l2_error:>8.4f}")
