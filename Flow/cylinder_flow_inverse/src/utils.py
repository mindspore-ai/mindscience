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
'''utils for  inverse navier stokes equation'''
import collections
import os
import time

import io
import cv2
import PIL
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size

from mindspore import Tensor
from mindspore import dtype as mstype


plt.rcParams['figure.dpi'] = 300


def visual(model, epochs, input_data, label, path="./videos"):
    """visulization of u/v/p"""
    predict = model(Tensor(input_data, mstype.float32)).asnumpy()
    [sample_t, sample_x, sample_y, _] = np.shape(input_data)

    u_vmin, u_vmax = np.percentile(label[:, :, :, 0], [0.5, 99.5])
    v_vmin, v_vmax = np.percentile(label[:, :, :, 1], [0.5, 99.5])
    p_vmin, p_vmax = np.percentile(label[:, :, :, 2], [0.5, 99.5])

    vmin_list = [u_vmin, v_vmin, p_vmin]
    vmax_list = [u_vmax, v_vmax, p_vmax]

    output_names = ["U", "V", "P"]

    if not os.path.isdir(os.path.abspath(path)):
        os.makedirs(path)

    fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
    fps = 10
    size = (1920, 1440)
    video = cv2.VideoWriter(os.path.join(path, "FlowField_" + str(epochs) + ".avi"), fourcc, fps, size)

    t_set = []
    if sample_t < 100:
        t_set = np.arange(sample_t, dtype=np.int32)
    else:
        for t in range(sample_t):
            if t % int(sample_t / 50) == 0 or t == sample_t - 1:
                t_set.append(t)

    for t in t_set:
        u_label = label[t, :, :, 0]
        v_label = label[t, :, :, 1]
        p_label = label[t, :, :, 2]

        u_predict = predict[t, :, :, 0]
        v_predict = predict[t, :, :, 1]
        p_predict = predict[t, :, :, 2]

        u_label_2d = np.reshape(np.array(u_label), (sample_x, sample_y))
        v_label_2d = np.reshape(np.array(v_label), (sample_x, sample_y))
        p_label_2d = np.reshape(np.array(p_label), (sample_x, sample_y))

        u_predict_2d = np.reshape(np.array(u_predict), (sample_x, sample_y))
        v_predict_2d = np.reshape(np.array(v_predict), (sample_x, sample_y))
        p_predict_2d = np.reshape(np.array(p_predict), (sample_x, sample_y))

        u_error_2d = np.abs(u_predict_2d - u_label_2d)
        v_error_2d = np.abs(v_predict_2d - v_label_2d)
        p_error_2d = np.abs(p_predict_2d - p_label_2d)

        label_2d = [u_label_2d, v_label_2d, p_label_2d]
        predict_2d = [u_predict_2d, v_predict_2d, p_predict_2d]
        error_2d = [u_error_2d, v_error_2d, p_error_2d]

        lpe_2d = [label_2d, predict_2d, error_2d]
        lpe_names = ["label", "predict", "error"]

        fig = plt.figure()

        gs = gridspec.GridSpec(3, 3)

        title = "t={:d}".format(t)
        plt.suptitle(title, fontsize=14)

        gs_idx = int(0)

        for i, data_2d in enumerate(lpe_2d):
            for j, data in enumerate(data_2d):
                ax = fig.add_subplot(gs[gs_idx])
                gs_idx += 1

                if lpe_names[i] == "error":
                    img = ax.imshow(data.T, vmin=0, vmax=1,
                                    cmap=plt.get_cmap("jet"), origin='lower')
                else:
                    img = ax.imshow(data.T, vmin=vmin_list[j], vmax=vmax_list[j],
                                    cmap=plt.get_cmap("jet"), origin='lower')

                ax.set_title(output_names[j] + " " + lpe_names[i], fontsize=4)
                plt.xticks(size=4)
                plt.yticks(size=4)

                aspect = 20
                pad_fraction = 0.5
                divider = make_axes_locatable(ax)
                width = axes_size.AxesY(ax, aspect=1 / aspect)
                pad = axes_size.Fraction(pad_fraction, width)
                cax = divider.append_axes("right", size=width, pad=pad)
                cb = plt.colorbar(img, cax=cax)
                cb.ax.tick_params(labelsize=4)

        gs.tight_layout(fig, pad=0.4, w_pad=0.4, h_pad=0.4)

        buffer_ = io.BytesIO()
        fig.savefig(buffer_, format="jpg")
        buffer_.seek(0)
        image = PIL.Image.open(buffer_)

        video.write(np.asarray(image))

        buffer_.close()

        plt.close()

    video.release()

    numt, _, _, output_size = label.shape
    label = label.reshape((numt, -1, output_size))
    predict = predict.reshape((numt, -1, output_size))
    error = label - predict
    l2_error_u = np.sqrt(np.sum(np.square(error[:, :, 0]), axis=1)) / np.sqrt(np.sum(np.square(label[:, :, 0]), axis=1))
    l2_error_v = np.sqrt(np.sum(np.square(error[:, :, 1]), axis=1)) / np.sqrt(np.sum(np.square(label[:, :, 1]), axis=1))
    l2_error_total = np.sqrt(np.sum(np.square(error[:, :, :2]), axis=(1, 2))) / \
                     np.sqrt(np.sum(np.square(label[:, :, :2]), axis=(1, 2)))

    plt.figure()
    plt.plot(input_data[:, 0, 0, 2], l2_error_u, 'b--', label="l2_error of U")
    plt.plot(input_data[:, 0, 0, 2], l2_error_v, 'g-.', label="l2_error of V")
    plt.plot(input_data[:, 0, 0, 2], l2_error_total, 'r-', label="l2_error of All")
    plt.legend()
    plt.xlabel('time')
    plt.ylabel('l2_error')
    plt.xticks(np.arange(0, 20.0, 4))

    plt.savefig(os.path.join(path, "TimeError_" + str(epochs) + ".png"))


InvNSError = collections.namedtuple("Inv_NS_error", ["l2_error", "l2_error_u", "l2_error_v", "l2_error_p"])


def _calculate_error(label, prediction):
    '''calculate l2-error to evaluate accuracy'''
    error = label - prediction

    l2_error_u = np.sqrt(np.sum(np.square(error[..., 0]))) / np.sqrt(np.sum(np.square(label[..., 0])))
    l2_error_v = np.sqrt(np.sum(np.square(error[..., 1]))) / np.sqrt(np.sum(np.square(label[..., 1])))
    l2_error_p = np.sqrt(np.sum(np.square(error[..., 2]))) / np.sqrt(np.sum(np.square(label[..., 2])))
    l2_error = np.sqrt(np.sum(np.square(error))) / np.sqrt(np.sum(np.square(label)))
    errors = InvNSError(l2_error, l2_error_u, l2_error_v, l2_error_p)
    return errors


def _get_prediction(model, inputs, label_shape, config):
    '''calculate the prediction respect to the given inputs'''
    output_size = config.get("output_size", 3)
    input_size = config.get("input_size", 3)
    prediction = np.zeros(label_shape)
    prediction = prediction.reshape((-1, output_size))
    inputs = inputs.reshape((-1, input_size))

    time_beg = time.time()

    index = 0
    while index < inputs.shape[0]:
        index_end = min(index + config["data"]["test"]["batch_size"], inputs.shape[0])
        test_batch = Tensor(inputs[index: index_end, :], mstype.float32)
        prediction[index: index_end, :] = model(test_batch).asnumpy()
        index = index_end

    print("    predict total time: {} ms".format((time.time() - time_beg)*1000))
    prediction = prediction.reshape(label_shape)
    prediction = prediction.reshape((-1, output_size))
    return prediction


def calculate_l2_error(model, inputs, label, config):
    """
    Evaluate the model respect to input data and label.

    Args:
         model (mindspore.nn.Cell): list of expressions node can by identified by mindspore.
         inputs (Tensor): the input data of network.
         label (Tensor): the true output value of given inputs.
         config (dict): the configuration of dataset.

    """
    label_shape = label.shape
    prediction = _get_prediction(model, inputs, label_shape, config)
    output_size = config.get("output_size", 3)
    label = label.reshape((-1, output_size))
    l2_errors = _calculate_error(label, prediction)
    print("    l2_error, U: ", l2_errors.l2_error_u, ", V: ", l2_errors.l2_error_v, ", P: ", l2_errors.l2_error_p,
          ", Total: ", l2_errors.l2_error)
    print("==================================================================================================")


def plot_params(param1, param2):
    """
    Evaluate the model respect to input data and label.

    Args:
         the history value of the training parameters

    """
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(param1, 'b--', label="value of parameter 1")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel('value')

    plt.subplot(2, 1, 2)
    plt.plot(param2, 'b--', label="value of parameter 2")
    plt.legend()
    plt.xlabel("epoch")
    plt.show()
    plt.savefig("Parameter.png")
