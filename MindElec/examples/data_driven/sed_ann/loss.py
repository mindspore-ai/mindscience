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
# ==============================================================================
"""
loss
"""

import os
import shutil

import mindspore.nn as nn
import numpy as np


class EvalMetric(nn.Metric):
    """
    eval metric
    """

    def __init__(self, scale_s11, length, show_pic_number, file_path):
        super(EvalMetric, self).__init__()
        self.clear()
        self.scale_s11 = scale_s11
        self.length = length
        self.show_pic_number = show_pic_number
        self.file_path = file_path
        self.show_pic_id = np.random.choice(length, self.show_pic_number, replace=False)
        self.pic_res = None

    def clear(self):
        """
        clear error
        """
        self.error_sum_l2_error = 0
        self.error_sum_loss_error = 0
        self.pic_res = None

    def update(self, *inputs):
        """
        update error
        """
        if not os.path.exists(self.file_path):
            os.mkdir(self.file_path)
        else:
            shutil.rmtree(self.file_path)
            os.mkdir(self.file_path)

        y_pred = self._convert_data(inputs[0])
        y_label = self._convert_data(inputs[1])

        test_predict, test_label = y_pred, y_label
        test_predict[:, :] = test_predict[:, :] * self.scale_s11
        test_label[:, :] = test_label[:, :] * self.scale_s11
        self.pic_res = []

        for predict_real_temp, label_real_temp in zip(test_predict, test_label):
            l2_error_temp = np.sqrt(np.sum(np.square(label_real_temp - predict_real_temp))) / \
                            np.sqrt(np.sum(np.square(label_real_temp)))
            self.error_sum_l2_error += l2_error_temp
            self.error_sum_loss_error += np.mean((label_real_temp - predict_real_temp) ** 2)
        self.pic_res = np.array(self.pic_res).astype(np.float32)

    def eval(self):
        """
        compute final error
        """
        return {'l2_error': self.error_sum_l2_error / self.length,
                'loss_error': self.error_sum_loss_error / self.length,
                'pic_res': self.pic_res}
