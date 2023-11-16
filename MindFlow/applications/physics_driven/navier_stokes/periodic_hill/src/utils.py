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
'''utils for 2D periodic hill'''
import collections
import time

import numpy as np
import matplotlib.pyplot as plt

from mindspore import Tensor
from mindspore import dtype as mstype


plt.rcParams['figure.dpi'] = 300


def _calculate_error(label, prediction):
    '''calculate l2-error to evaluate accuracy'''
    errors = collections.namedtuple("PeriodicHillError", ["l2_error", "l2_error_u", "l2_error_v", "l2_error_p",
                                                          "l2_error_uu", "l2_error_uv", "l2_error_vv"])
    error = label - prediction
    # x, y, u, v, p, uu, uv, vv, rho, nu
    l2_error_u = np.sqrt(np.sum(np.square(error[..., 0]))) / np.sqrt(np.sum(np.square(label[..., 0])))
    l2_error_v = np.sqrt(np.sum(np.square(error[..., 1]))) / np.sqrt(np.sum(np.square(label[..., 1])))
    l2_error_p = np.sqrt(np.sum(np.square(error[..., 2]))) / np.sqrt(np.sum(np.square(label[..., 2])))
    l2_error_uu = np.sqrt(np.sum(np.square(error[..., 3]))) / np.sqrt(np.sum(np.square(label[..., 3])))
    l2_error_uv = np.sqrt(np.sum(np.square(error[..., 4]))) / np.sqrt(np.sum(np.square(label[..., 4])))
    l2_error_vv = np.sqrt(np.sum(np.square(error[..., 5]))) / np.sqrt(np.sum(np.square(label[..., 5])))


    l2_error = np.sqrt(np.sum(np.square(error))) / np.sqrt(np.sum(np.square(label)))
    errors = errors(l2_error, l2_error_u, l2_error_v, l2_error_p,
                    l2_error_uu, l2_error_uv, l2_error_vv)
    return errors


def _get_prediction(model, inputs, label_shape, config):
    '''calculate the prediction respect to the given inputs'''
    output_size = config['model']['out_channels']
    input_size = config['model']['in_channels']

    prediction = np.zeros(label_shape)
    prediction = prediction.reshape((-1, output_size))
    inputs = inputs.reshape((-1, input_size))

    time_beg = time.time()

    index = 0
    while index < inputs.shape[0]:
        index_end = min(index + config["data"]['batch_size'], inputs.shape[0])
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
    output_size = config['model']['out_channels']
    label = label.reshape((-1, output_size))
    l2_errors = _calculate_error(label, prediction)
    print("    l2_error, U: ", l2_errors.l2_error_u, ", V: ", l2_errors.l2_error_v, ", P: ", l2_errors.l2_error_p)
    print("    l2_error, uu: ", l2_errors.l2_error_uu, ", uv: ", l2_errors.l2_error_uv, ", vv: ", l2_errors.l2_error_vv,
          ", Total: ", l2_errors.l2_error)
    print("==================================================================================================")
