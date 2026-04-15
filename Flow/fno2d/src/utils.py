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
utils
"""
import time
import numpy as np

from mindspore import Tensor
from mindspore import dtype as mstype

from mindflow.utils import print_log


def _calculate_error(label, prediction, batch_size):
    """calculate l2-error to evaluate accuracy"""
    rel_error = np.sqrt(np.sum(np.square(label.reshape(batch_size, -1) - prediction.reshape(batch_size, -1)))) / \
                np.sqrt(np.sum(np.square(label.reshape(batch_size, -1))))
    return rel_error


def calculate_l2_error(model, inputs, label, batch_size):
    """
    Evaluate the model respect to input data and label.

    Args:
        model (Cell): Prediction network cell.
        inputs (Array): Input data of prediction.
        label (Array): Label data of prediction.
        batch_size (int): size of prediction batch.
    """
    print_log("================================Start Evaluation================================")
    time_beg = time.time()
    rel_rmse_error = 0.0
    prediction = 0.0
    length = label.shape[0]
    t = 10
    for i in range(length):
        for j in range(t - 1, t + 9):
            cur_label = label[i:i + 1, j]
            if j == t - 1:
                test_batch = Tensor(inputs[i:i + 1, j], dtype=mstype.float32)
            else:
                test_batch = Tensor(np.expand_dims(prediction, axis=-2), dtype=mstype.float32)
            prediction = model(test_batch[..., -1, :])
            prediction = prediction.asnumpy()
            rel_rmse_error_step = _calculate_error(cur_label[..., -1, :], prediction, batch_size)
            rel_rmse_error += rel_rmse_error_step

    rel_rmse_error = rel_rmse_error / (length * 10)
    print_log("mean rel_rmse_error:", rel_rmse_error)
    print_log("=================================End Evaluation=================================")
    print_log("predict total time: {} s".format(time.time() - time_beg))
