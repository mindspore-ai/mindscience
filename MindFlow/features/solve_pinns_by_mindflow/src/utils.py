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
import matplotlib.pyplot as plt

from mindspore import Tensor
from mindspore import dtype as mstype


def visual(model, inputs, label, epochs=1):
    '''visual result for poisson 2D'''
    fig, ax = plt.subplots(2, 1)
    ax = ax.flatten()
    plt.subplots_adjust(hspace=0.5)
    ax0 = ax[0].scatter(inputs[:, 0], inputs[:, 1], c=label[:, 0], cmap=plt.cm.rainbow, s=0.5)
    ax[0].set_title("true")
    ax[0].set_xlabel('x')
    ax[0].set_ylabel('y')
    ax[0].axis('equal')
    ax[1].scatter(inputs[:, 0], inputs[:, 1], c=model(Tensor(inputs, mstype.float32)), cmap=plt.cm.rainbow, s=0.5)
    ax[1].set_title("prediction")
    ax[1].set_xlabel('x')
    ax[1].set_ylabel('y')
    ax[1].axis('equal')
    cbar = fig.colorbar(ax0, ax=[ax[0], ax[1]])
    cbar.set_label('u(x, y)')

    plt.savefig(f"images/{epochs}-result.jpg", dpi=600)


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

    print("    predict total time: {} ms".format((time.time() - time_beg) * 1000))
    prediction = prediction.reshape(label_shape)
    prediction = prediction.reshape((-1, label_shape[1]))
    return prediction


def calculate_l2_error(model, inputs, label, batch_size):
    """
    Evaluate the model respect to input data and label.

    Args:
         model (mindspore.nn.Cell): List of expressions node can by identified by mindspore.
         inputs (Tensor): The input data of network.
         label (Tensor): The true output value of given inputs.
         batch_size (int): Data size in one step, which is the same as that in training.

    Supported Platforms:
        ``Ascend`` ``GPU``
    """
    label_shape = label.shape
    prediction = _get_prediction(model, inputs, label_shape, batch_size)
    label = label.reshape((-1, label_shape[1]))
    l2_error = _calculate_error(label, prediction)
    print("    l2_error: ", l2_error)
    print("==================================================================================================")
