# Copyright 2021 Huawei Technologies Co., Ltd
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
"""metrics"""

import numpy as np
import mindspore.nn as nn
from mindspore.ops import functional as F


class MyMSELoss(nn.LossBase):
    """mse loss function"""
    def construct(self, base, target):
        x = F.square(base - target)
        return self.get_loss(x)

class EvalMetric(nn.Metric):
    """evaluation metrics"""
    def __init__(self, scale_s11, length):
        super(EvalMetric, self).__init__()
        self.clear()
        self.scale_s11 = scale_s11
        self.length = length

    def clear(self):
        """clear"""
        self.error_sum_l2_error = 0
        self.error_sum_mse_error = 0

    def update(self, *inputs):
        """update"""
        y_pred = self._convert_data(inputs[0])
        y = self._convert_data(inputs[1])

        test_predict, test_label = y_pred, y
        test_predict[:, :] = test_predict[:, :] * self.scale_s11
        test_label[:, :] = test_label[:, :] * self.scale_s11
        test_predict[:, :] = 1.0 - np.power(10, test_predict[:, :])
        test_label[:, :] = 1.0 - np.power(10, test_label[:, :])

        for i in range(len(test_label)):
            predict_real_temp = test_predict[i]
            label_real_temp = test_label[i]
            self.error_sum_l2_error += np.sqrt(np.sum(np.square(label_real_temp - predict_real_temp))) / \
                                       np.sqrt(np.sum(np.square(label_real_temp)))
            self.error_sum_mse_error += np.mean((label_real_temp - predict_real_temp) ** 2)

    def eval(self):
        return {'l2_error': self.error_sum_l2_error / self.length,
                'mse_error': self.error_sum_mse_error / self.length}


def l2_error_np(label_real_temp, predict_real_temp):
    res = np.sqrt(np.sum(np.square(label_real_temp - predict_real_temp))) / np.sqrt(np.sum(np.square(label_real_temp)))
    return res
