# Copyright 2024 Huawei Technologies Co., Ltd
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
"""utils"""
import os
import time

import numpy as np
from scipy.interpolate import interp2d
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mindspore import Tensor
from mindspore import dtype as mstype
from mindflow.utils import print_log
from mindflow.cell import poly_data


def _calculate_error(label, prediction):
    """calculate l2-error to evaluate accuracy"""
    rel_error = np.sqrt(np.sum(np.square(label.flatten() - prediction.flatten()))) / \
                np.sqrt(np.sum(np.square(label.flatten())))
    return rel_error


def calculate_l2_error(model, inputs, label, labels_unif, data_config):
    """
    Evaluate the model sequentially on Gauss grid and on regular (original) grid.

    Args:
        model (Cell): Prediction network cell.
        inputs (numpy.Array): Input data, interpolated on Gauss grid.
        label (numpy.Array): Label data, interpolated on Gauss grid.
        labels_unif (numpy.Array): original label data.
        data_config (dict): dict with data configurations.
    """
    print_log("================================Start Evaluation================================")
    time_beg = time.time()

    resolution = data_config['resolution']
    batch_size = data_config['batch_size']
    test_size = data_config['test_size']
    poly_type = data_config['poly_type']
    n_batches = test_size // batch_size

    rel_rmse_error = 0.0
    rel_rmse_error_unif = 0.0
    t = 10  # number of time steps to predict sequentially

    x = np.linspace(-1., 1.0, resolution)
    x_p = poly_data[poly_type](resolution, None)

    prediction = None
    sample_shape = (1, 1, resolution, resolution)
    for i in range(n_batches):
        for j in range(t - 1, t + 9):
            cur_label = label[i:i + 1, j]
            unif_label = labels_unif[i:i + 1, j]
            if j == t - 1:
                test_batch = Tensor(inputs[i:i + 1, j].reshape(sample_shape), dtype=mstype.float32)
            else:
                test_batch = prediction
            prediction = model(test_batch)
            prediction_np = prediction.asnumpy()

            rel_rmse_error += _calculate_error(cur_label, prediction_np)

            f = interp2d(x_p, x_p, prediction_np.flatten(), kind='cubic')
            pred_unif = f(x, x).T
            rel_rmse_error_unif += _calculate_error(unif_label, pred_unif)

    rel_rmse_error = rel_rmse_error / (n_batches * 10)
    rel_rmse_error_unif = rel_rmse_error_unif / (n_batches * 10)
    print_log(f"on Gauss grid: {rel_rmse_error}, on regular grid: {rel_rmse_error_unif}")
    print_log("=================================End Evaluation=================================")
    print_log("predict total time: {} s".format(time.time() - time_beg))


def visual_animate(yy, yp, ye):
    """ Plot animate figures.

    Args:
        yy (numpy.array): Label data with shape e.g. :math:`[T, C, H, W]`.
        yp (numpy.array): Label data with shape e.g. :math:`[T, C, H, W]`.
        ye (numpy.array): Error data with shape e.g. :math:`[T, C, H, W]`.
    """
    cmap = matplotlib.colormaps['jet']
    fig, ax = plt.subplots(1, 3, figsize=[7, 3])

    ax[0].set_title(f'Label')
    im0 = ax[0].imshow(yy[0], cmap=cmap)
    ax[1].set_title(f'Prediction')
    im1 = ax[1].imshow(yp[0], cmap=cmap)
    ax[2].set_title(f'Error')
    im2 = ax[2].imshow(ye[0], cmap=cmap)
    title = fig.suptitle(f't=0')
    fig.tight_layout()
    fig.colorbar(im1, ax=ax)

    def animate(i):
        y, p, e = yy[i], yp[i], ye[i]

        im0.set_data(y)
        im1.set_data(p)
        im2.set_data(e)
        vmin, vmax = np.min(y), np.max(y)
        im0.set_clim(vmin, vmax)
        im1.set_clim(vmin, vmax)
        im2.set_clim(vmin, vmax)
        title.set_text(f't={i}')

    ani = animation.FuncAnimation(fig, animate, interval=200, blit=False, frames=10,
                                  repeat_delay=1000)
    ani.save('images/result.gif', writer='imagemagick')


def visual_static(x, yy, yp):
    """ Plot static figures.

    Args:
        x (Array): Input data with shape e.g. :math:`[C, H, W]`.
        yy (Array): Label data with shape e.g. :math:`[T, C, H, W]`.
        yp (Array): Predicted data with shape e.g. :math:`[T, C, H, W]`.
    """
    cmap = matplotlib.colormaps['jet']

    plt.figure(figsize=(20, 12))
    plt.subplot(4, 9, 1)
    plt.title("Input")
    plt.imshow(x, cmap=cmap)
    plt.axis('off')

    for i in range(9):
        label = yy[i]
        vmin, vmax = np.min(label), np.max(label)
        plt.subplot(4, 9, i + 10)
        plt.title(f"Label {i}")
        plt.imshow(yy[i], cmap=cmap, vmin=vmin, vmax=vmax)
        plt.axis('off')
        plt.subplot(4, 9, i + 19)
        plt.title(f"Predict {i}")
        plt.imshow(yp[i], cmap=cmap, vmin=vmin, vmax=vmax)
        plt.axis('off')
        plt.subplot(4, 9, i + 28)
        plt.title(f"Error {i}")
        plt.imshow(np.abs(yy[i] - yp[i]), cmap=cmap, vmin=vmin, vmax=vmax)
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(f'images/result.jpg')
    plt.close()


def visual(model, inputs, labels_unif, data_config):
    """ Infer the model sequentially and visualize the results.

    Args:
        Prediction network cell.
        inputs (numpy.array): Input data, interpolated on Gauss grid.
        labels_unif (numpy.Array): original label data.
        data_config (dict): dict with data configurations.
    """
    if not os.path.exists('images'):
        os.makedirs('images')

    res = data_config['resolution']
    poly_type = data_config['poly_type']

    x = np.linspace(-1., 1.0, res)
    x_p = poly_data[poly_type](res, None)

    prediction = None

    t = 10  # number of time steps to predict sequentially
    pred_unif = []
    for j in range(t - 1, t + 9):
        unif_label = labels_unif[0, j]
        if j == t - 1:
            test_batch = Tensor(inputs[0, j].reshape((1, 1, res, res)), dtype=mstype.float32)
        else:
            test_batch = prediction
        prediction = model(test_batch)
        prediction_np = prediction.asnumpy()

        f = interp2d(x_p, x_p, prediction_np.flatten(), kind='cubic')
        pred_unif.append(f(x, x).T)

    pred_unif = np.array(pred_unif)
    unif_label = labels_unif[0, 9:19].reshape(10, res, res)
    x = inputs[0, 9].reshape((res, res))
    visual_static(x, unif_label, pred_unif)
    err = np.abs(pred_unif - unif_label)
    visual_animate(unif_label, pred_unif, err)
