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
import time
import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
import numpy as np

from mindspore import Tensor
import mindspore.common.dtype as mstype

from .dataset import create_test_dataset

plt.rcParams["figure.dpi"] = 300


def visual(model, config):
    """visual result of model prediction and ground-truth"""
    name = "result"
    test_input, label = create_test_dataset(config)
    visual_input = test_input.reshape(-1, config["model"]["in_channels"])
    visual_label = label.reshape(-1, config["model"]["out_channels"])
    prediction = np.zeros(label.shape)
    index = 0
    while index < len(visual_input):
        index_end = min(index + config["data"]["train"]["batch_size"], len(visual_input))
        for i in range(config["model"]["in_channels"]):
            test_batch = Tensor(visual_input[index:index_end, :], dtype=mstype.float32)
            predict = model(test_batch)
            predict = predict.asnumpy()
            prediction[index:index_end, i] = predict[:, i]
        index = index_end

    visual_fn(visual_label, prediction.reshape(visual_label.shape), "./images", name)


def visual_fn(label, predict, path, name):
    """visulization of ux/uy/p"""
    ux_min, ux_max = np.percentile(label[:, 0], [0.5, 99.5])
    uy_min, uy_max = np.percentile(label[:, 1], [0.5, 99.5])
    p_min, p_max = np.percentile(label[:, 2], [0.5, 99.5])

    min_list = [ux_min, uy_min, p_min]
    max_list = [ux_max, uy_max, p_max]

    mean_abs_ux_label = 1.0
    mean_abs_uy_label = 1.0
    mean_abs_p_label = 1.0

    output_names = ["ux", "uy", "p"]

    if not os.path.isdir(path):
        os.makedirs(path)

    ux_label = label[:, 0]
    uy_label = label[:, 1]
    p_label = label[:, 2]

    ux_label_2d = np.array(ux_label).ravel()
    uy_label_2d = np.array(uy_label)
    p_label_2d = np.array(p_label)

    ux_pred = predict[:, 0]
    uy_pred = predict[:, 1]
    p_pred = predict[:, 2]

    ux_pred_2d = np.array(ux_pred)
    uy_pred_2d = np.array(uy_pred)
    p_pred_2d = np.array(p_pred)

    ux_error_2d = np.abs(ux_pred_2d - ux_label_2d) / mean_abs_ux_label
    uy_error_2d = np.abs(uy_pred_2d - uy_label_2d) / mean_abs_uy_label
    p_error_2d = np.abs(p_pred_2d - p_label_2d) / mean_abs_p_label

    label_2d = [ux_label_2d, uy_label_2d, p_label_2d]
    pred_2d = [ux_pred_2d, uy_pred_2d, p_pred_2d]
    error_2d = [ux_error_2d, uy_error_2d, p_error_2d]

    lpe_2d = [label_2d, pred_2d, error_2d]
    lpe_names = ["label", "predict", "error"]

    fig = plt.figure()

    grid_spec = gridspec.GridSpec(3, 3)

    grid_spec_idx = int(0)

    for i, data_2d in enumerate(lpe_2d):
        for j, data in enumerate(data_2d):
            subfig = fig.add_subplot(grid_spec[grid_spec_idx])
            grid_spec_idx += 1

            if lpe_names[i] == "error":
                img = subfig.imshow(
                    data.T.reshape(101, 101),
                    vmin=0,
                    vmax=1,
                    cmap=plt.get_cmap("jet"),
                    origin="lower",
                )
            else:
                img = subfig.imshow(
                    data.T.reshape(101, 101),
                    vmin=min_list[j],
                    vmax=max_list[j],
                    cmap=plt.get_cmap("jet"),
                    origin="lower",
                )

            subfig.set_title(output_names[j] + " " + lpe_names[i], fontsize=4)
            plt.xticks(size=4)
            plt.yticks(size=4)

            aspect = 20
            pad_fraction = 0.5
            divider = make_axes_locatable(subfig)
            width = axes_size.AxesY(subfig, aspect=1 / aspect)
            pad = axes_size.Fraction(pad_fraction, width)
            cax = divider.append_axes("right", size=width, pad=pad)
            colorbar = plt.colorbar(img, cax=cax)
            colorbar.ax.tick_params(labelsize=4)

    grid_spec.tight_layout(fig, pad=0.4, w_pad=0.4, h_pad=0.4)
    fig.savefig(path + "/" + str(name) + ".png", format="png")


def _calculate_error(label, prediction):
    """calculate l2-error to evaluate accuracy"""
    error = label - prediction
    l2_error = np.sqrt(np.sum(np.square(error[..., 0]))) / np.sqrt(
        np.sum(np.square(label[..., 0]))
    )

    return l2_error


def _get_prediction(model, inputs, label_shape, batch_size):
    """calculate the prediction respect to the given inputs"""
    prediction = np.zeros(label_shape)
    prediction = prediction.reshape((-1, label_shape[1]))
    inputs = inputs.reshape((-1, inputs.shape[1]))

    time_beg = time.time()

    index = 0
    while index < inputs.shape[0]:
        index_end = min(index + batch_size, inputs.shape[0])
        test_batch = Tensor(inputs[index:index_end, :], mstype.float32)
        prediction[index:index_end, :] = model(test_batch).asnumpy()
        index = index_end

    print("    predict total time: {} ms".format((time.time() - time_beg) * 1000))
    prediction = prediction.reshape(label_shape)
    prediction = prediction.reshape((-1, label_shape[1]))
    return prediction


def calculate_l2_error(model, inputs, label, batch_size):
    """
    Evaluate the model respect to input data and label.

    Args:
         model (Cell): list of expressions node can by identified by mindspore.
         inputs (Tensor): the input data of network.
         label (Tensor): the true output value of given inputs.
         batch_size (int): data size in one step, which is the same as that in training.

    """
    label_shape = label.shape
    prediction = _get_prediction(model, inputs, label_shape, batch_size)
    label = label.reshape((-1, label_shape[1]))
    l2_error = _calculate_error(label, prediction)
    print("    l2_error: ", l2_error)
    print(
        "=================================================================================================="
    )
