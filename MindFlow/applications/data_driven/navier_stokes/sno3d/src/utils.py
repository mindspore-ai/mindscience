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
from mindspore import jit_class, ops
from mindflow.core import RelativeRMSELoss
from mindflow.utils import print_log
from mindflow.cell import poly_data


@jit_class
class UnitGaussianNormalizer():
    '''normalizer'''
    def __init__(self, x, eps=0.00001):
        super(UnitGaussianNormalizer, self).__init__()

        self.mean = ops.mean(x)
        self.std = ops.std(x)
        self.eps = eps

    def encode(self, x):
        '''encoder'''
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x):
        '''decoder'''
        std = self.std + self.eps
        mean = self.mean
        x = (x * std) + mean
        return x


def calculate_l2_error(model, inputs, labels, test_u_unif, data_config, a_normalizer, u_normalizer):
    """
    Evaluate the model on Gauss grid and on regular (original) grid.

    Args:
        model (nn.Cell): Prediction network cell.
        inputs (Tensor): Input data, interpolated on Gauss grid.
        labels (Tensor): Label data, interpolated on Gauss grid.
        test_u_unif (np.Array): Label data on original grid.
        data_config (dict): dict with data configurations.
        a_normalizer (UnitGaussianNormalizer): input normalizer (encoder)
        u_normalizer (UnitGaussianNormalizer): output normalizer (decoder)
    """
    print_log("================================Start Evaluation================================")
    time_beg = time.time()
    grid_size = data_config['resolution']
    poly_type = data_config['poly_type']

    input_timestep = data_config["input_timestep"]
    output_timestep = data_config["output_timestep"]

    rms_error, rms_error_unif = 0.0, 0.0
    x = np.linspace(-1., 1., grid_size)
    x_p = poly_data[poly_type](grid_size, None)

    loss_fn = RelativeRMSELoss()
    for i in range(labels.shape[0]):
        label = labels[i:i + 1]
        test_batch = inputs[i:i + 1]
        test_batch = a_normalizer.encode(test_batch)

        test_batch = test_batch.reshape(1, input_timestep, grid_size,
                                        grid_size, 1).repeat(output_timestep, axis=-1)
        prediction = model(test_batch).reshape(1, output_timestep, grid_size, grid_size)
        prediction = u_normalizer.decode(prediction)

        rms_error_step = loss_fn(prediction.reshape(1, -1), label.reshape(1, -1))
        rms_error += rms_error_step

        test_batch_interp = np.zeros((output_timestep, grid_size, grid_size))
        for j in range(output_timestep):
            prediction_np = prediction[0, j].asnumpy().flatten()
            f = interp2d(x_p, x_p, prediction_np, kind='cubic')
            test_batch_interp[j] = f(x, x).T

        test_unif = test_u_unif[i].flatten()
        test_batch_interp = test_batch_interp.flatten()
        err_i = np.linalg.norm(test_unif - test_batch_interp) / np.linalg.norm(test_unif)
        rms_error_unif += err_i

    rms_error = rms_error / labels.shape[0]
    rms_error_unif = rms_error_unif / labels.shape[0]

    print_log(f"Error on Gauss grid: {rms_error}, on regular grid: {rms_error_unif}")
    print_log("predict total time: {} s".format(time.time() - time_beg))
    print_log("=================================End Evaluation=================================")


def visual_animate(yy, yp, ye, start):
    """ Plot animate figures.

    Args:
        yy (numpy.array): Label data with shape e.g. :math:`[C, H, W]`.
        yp (numpy.array): Label data with shape e.g. :math:`[C, H, W]`.
        ye (numpy.array): Error data with shape e.g. :math:`[C, H, W]`.
    """
    cmap = matplotlib.colormaps['jet']
    fig, ax = plt.subplots(1, 3, figsize=[9, 3])

    ax[0].set_title(f'Label')
    im0 = ax[0].imshow(yy[0], cmap=cmap)
    ax[1].set_title(f'Prediction')
    im1 = ax[1].imshow(yp[0], cmap=cmap)
    ax[2].set_title(f'Error')
    im2 = ax[2].imshow(ye[0], cmap=cmap)
    title = fig.suptitle(f't={start}')
    fig.tight_layout()
    fig.colorbar(im1, ax=ax)
    for i in range(3):
        ax[i].set_xlabel('x')
        ax[i].set_ylabel('y')

    vmin, vmax = np.min(yy), np.max(yy)

    def animate(i):
        y, p, e = yy[i], yp[i], ye[i]

        im0.set_data(y)
        im1.set_data(p)
        im2.set_data(e)
        im0.set_clim(vmin, vmax)
        im1.set_clim(vmin, vmax)
        im2.set_clim(vmin, vmax)
        title.set_text(f't={start+i}')

    ani = animation.FuncAnimation(fig, animate, interval=200, blit=False, frames=yy.shape[0],
                                  repeat_delay=1000)
    ani.save('images/result.gif', writer='imagemagick')


def visual(model, inputs, test_a_unif, test_u_unif, data_config, a_normalizer, u_normalizer):
    """ Infer the model and visualize the input and prediction for the first sample.

    Args:
        model (nn.Cell): Prediction network cell.
        inputs (numpy.array): Input data with shape e.g. :math:`[N, C, H, W]`.
        test_a_unif (np.Array): Input data on original grid.
        test_u_unif (np.Array): Label data on original grid.
        data_config (dict): dict with data configurations.
        a_normalizer (UnitGaussianNormalizer): normalizer for input data.
        u_normalizer (UnitGaussianNormalizer): normalizer for output data.
    """
    if not os.path.exists('images'):
        os.makedirs('images')

    grid_size = data_config['resolution']

    poly_type = data_config['poly_type']
    input_timestep = data_config['input_timestep']
    output_timestep = data_config['output_timestep']

    x = np.linspace(-1., 1., grid_size)
    x_p = poly_data[poly_type](grid_size, None)

    test_sample = inputs[0]
    test_sample = a_normalizer.encode(test_sample)
    test_sample = test_sample.reshape(1, input_timestep, grid_size,
                                      grid_size, 1).repeat(output_timestep, axis=-1)
    prediction = model(test_sample).reshape(1, output_timestep, grid_size, grid_size)
    prediction = u_normalizer.decode(prediction)

    prediction_interp = np.zeros((output_timestep, grid_size, grid_size))
    for i in range(output_timestep):
        prediction_np = prediction[0, i].asnumpy().flatten()
        f = interp2d(x_p, x_p, prediction_np, kind='cubic')
        prediction_interp[i] = f(x, x).T

    label = test_u_unif[0].reshape((output_timestep, grid_size, grid_size))

    input_sample = test_a_unif[0].reshape(input_timestep, grid_size, grid_size)
    cmap = matplotlib.colormaps['jet']
    fig, ax = plt.subplots(1, 1, figsize=[3, 3])
    ax.set_title(f'Input sample')
    im = ax.imshow(input_sample[0], cmap=cmap)
    title = fig.suptitle(f't=0')
    fig.tight_layout()
    fig.colorbar(im, ax=ax)
    ax.set_xlabel('x')
    ax.set_xlabel('y')

    vmin, vmax = np.min(input_sample), np.max(input_sample)
    def animate(i):
        y = input_sample[i]
        im.set_data(y)
        im.set_clim(vmin, vmax)
        title.set_text(f't={i}')

    ani = animation.FuncAnimation(fig, animate, interval=200, blit=False, frames=input_timestep,
                                  repeat_delay=1000)
    ani.save('images/input.gif', writer='imagemagick')

    err = np.abs(label - prediction_interp)
    visual_animate(label, prediction_interp, err, start=input_timestep)
