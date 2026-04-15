# Copyright 2022 Huawei Technologies Co., Ltd
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
"""
visualization functions
"""
import time
import numpy as np
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt

from mindspore import Tensor
from mindspore import dtype as mstype

from mindflow.utils import print_log


def visual(model, epochs=1, resolution=100):
    """visulization of ex/ey/hz"""
    t_flat = np.linspace(0, 1, resolution)
    x_flat = np.linspace(-1, 1, resolution)
    t_grid, x_grid = np.meshgrid(t_flat, x_flat)
    x = x_grid.reshape((-1, 1))
    t = t_grid.reshape((-1, 1))
    xt = Tensor(np.concatenate((x, t), axis=1), dtype=mstype.float32)
    u_predict = model(xt)
    u_predict = u_predict.asnumpy()
    gs = GridSpec(2, 3)
    plt.subplot(gs[0, :])
    plt.scatter(t, x, c=u_predict, cmap=plt.cm.rainbow)
    plt.xlabel('t')
    plt.ylabel('x')
    cbar = plt.colorbar(pad=0.05, aspect=10)
    cbar.set_label('u(t,x)')
    cbar.mappable.set_clim(-1, 1)
    t_cross_sections = [0.25, 0.5, 0.75]
    for i, t_cs in enumerate(t_cross_sections):
        plt.subplot(gs[1, i])
        xt = Tensor(np.stack([x_flat, np.full(x_flat.shape, t_cs)], axis=-1), dtype=mstype.float32)
        u = model(xt).asnumpy()
        plt.plot(x_flat, u)
        plt.title('t={}'.format(t_cs))
        plt.xlabel('x')
        plt.ylabel('u(t,x)')
    plt.tight_layout()
    plt.savefig(f'images/{epochs + 1}-result.jpg')


def _calculate_error(label, prediction):
    '''calculate l2-error to evaluate accuracy'''
    error = label - prediction
    l2_error = np.sqrt(np.sum(np.square(error[..., 0]))) / np.sqrt(np.sum(np.square(label[..., 0])))

    return l2_error


def _get_prediction(model, inputs, label_shape, batch_size):
    '''calculate the prediction respect to the given inputs'''
    prediction = np.zeros(label_shape)
    prediction = prediction.reshape((-1, label_shape[1]))
    inputs = inputs.reshape((-1, inputs.shape[1]))

    time_beg = time.time()

    index = 0
    while index < inputs.shape[0]:
        index_end = min(index + batch_size, inputs.shape[0])
        test_batch = Tensor(inputs[index: index_end, :], mstype.float32)
        prediction[index: index_end, :] = model(test_batch).asnumpy()
        index = index_end

    print_log("    predict total time: {} ms".format((time.time() - time_beg)*1000))
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
    print_log("    l2_error: ", l2_error)
    print_log("==================================================================================================")
