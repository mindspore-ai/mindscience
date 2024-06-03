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
from scipy.interpolate import interp1d
import matplotlib
import matplotlib.pyplot as plt
from mindflow import RelativeRMSELoss
from mindflow.utils import print_log

from .sno_utils import poly_data


def test_error(model, input_tensor, label_tensor, label_unif, data_config):
    """
    Evaluate the model on Gauss grid and on regular (original) grid.

    Args:
        model (Cell): Prediction network cell.
        input_tensor (mindspore.Tensor): Input data, interpolated on Gauss grid.
        label_tensor (mindspore.Tensor): Label data, interpolated on Gauss grid.
        label_unif (numpy.array): original label data.
        data_config (dict): dict with data configurations.
    """
    print_log("================================Start Evaluation================================")
    time_beg = time.time()

    resolution = data_config['resolution']
    test_size = data_config['test']['num_samples']
    poly_type = data_config['poly_type']

    rel_rmseloss = RelativeRMSELoss(reduction='mean')

    x = np.linspace(-1., 1.0, resolution)
    x_p = poly_data[poly_type](resolution, None)

    x_p[0] = -1.
    x_p[-1] = 1.

    output = model(input_tensor)
    rms_error = rel_rmseloss(output, label_tensor)

    pred_unif = np.zeros((test_size, resolution))
    for i in range(test_size):
        output_i = output[i].asnumpy().flatten()
        f = interp1d(x_p, output_i, kind='cubic')
        pred_unif[i] = f(x)
    pred_unif = pred_unif.flatten()
    label_unif_f = label_unif.flatten()

    err_unif = 0.0
    norm = np.linalg.norm(label_unif_f, 2)
    if norm != 0:
        err_unif = np.linalg.norm(pred_unif - label_unif_f, 2) / norm
    print('poly err', rms_error, 'unif err', err_unif)

    print_log("mean rel_rmse_error:")
    print_log(f"on Gauss grid: {rms_error}, on regular grid: {err_unif}")
    print_log("=================================End Evaluation=================================")
    print_log(f"predict total time: {time.time() - time_beg} s")


def visual(model, input_tensor, data_config, t_out=10):
    """ Infer the model sequentially and visualize the results.

    Args:
        model (Cell): Prediction network cell.
        input_tensor (mindspore.Tensor): Input data, interpolated on Gauss grid.
        data_config (dict): Dict with data configurations.
        t_out (int): Number of time steps to forward the model sequentially.
    """
    if not os.path.exists('images'):
        os.makedirs('images')

    resolution = data_config['resolution']
    poly_type = data_config['poly_type']

    x = np.linspace(-1., 1.0, resolution)
    x_p = poly_data[poly_type](resolution, None)

    x_p[0] = -1.
    x_p[-1] = 1.

    input_tensor_slice = input_tensor[:6]
    model.set_train(False)
    outputs = []
    for i in range(6):
        inp = input_tensor_slice[i].expand_dims(0)
        output = np.zeros((t_out, resolution))
        for j in range(t_out):
            out = model(inp)
            out_np = out.asnumpy().flatten()
            f = interp1d(x_p, out_np, kind='cubic')
            output[j] = f(x)
            inp = out
        outputs.append(output)

    cmap = matplotlib.colormaps['jet']
    fig, axes = plt.subplots(6, 1)
    im = None
    for i in range(6):
        im = axes[i].imshow(outputs[i], cmap=cmap, interpolation='nearest', aspect='auto')
        axes[i].set_ylabel('t')
    for i in range(5):
        axes[i].set_xticks([])
    axes[-1].set_xlabel('x')
    fig.tight_layout()
    cbar = fig.colorbar(im, ax=axes)
    cbar.set_label('u(t,x)')
    fig.savefig('images/result.jpg')
    plt.close()
