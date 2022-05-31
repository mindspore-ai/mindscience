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
metric
"""

import os
import shutil
from mindspore import nn
import matplotlib.pyplot as plt
import numpy as np


class EvalMetric(nn.Metric):
    """
    eval metric
    """

    def __init__(self, length, file_path):
        super(EvalMetric, self).__init__()
        self.clear()
        self.length = length
        self.file_path = file_path

    def clear(self):
        """
        clear error
        """
        self.error_sum_l2_error = 0
        self.error_sum_loss_error = 0


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
        test_predict[:, :] = test_predict[:, :]
        test_label[:, :] = test_label[:, :]

        for i in range(len(test_label)):
            predict_real_temp_0 = test_predict[i, 0]
            label_real_temp_0 = test_label[i, 0]
            predict_real_temp_1 = test_predict[i, 1]
            label_real_temp_1 = test_label[i, 1]
            l2_error_temp = np.sqrt(np.sum(np.square(label_real_temp_0 - predict_real_temp_0) + np.square(
                label_real_temp_1 - predict_real_temp_1))) / \
                            np.sqrt(np.sum(np.square(label_real_temp_0) + np.square(label_real_temp_1)))
            self.error_sum_l2_error += l2_error_temp
            self.error_sum_loss_error += np.mean(
                (label_real_temp_0 - predict_real_temp_0) ** 2 + (label_real_temp_1 - predict_real_temp_1) ** 2)
        plt.plot(dpi=250)
        samples = list(range(self.length))
        plt.subplots_adjust(wspace=0, hspace=0.2)
        plt.subplot(2, 1, 1)
        plt.plot(samples, test_predict[:, 0], '*', label='AI model', linewidth=2)
        plt.plot(samples, test_label[:, 0], '.r', label='gprMax', linewidth=1)
        plt.ylabel('X_pos')
        plt.legend(loc='upper right')
        plt.subplot(2, 1, 2)
        plt.plot(samples, test_predict[:, 1], '*', label='AI model', linewidth=2)
        plt.plot(samples, test_label[:, 1], '.r', label='gprMax', linewidth=1)
        plt.ylabel('Y_pos')
        plt.legend(loc='upper right')
        plt.savefig(self.file_path + '/' + 'result.jpg')
        plt.close()

    def eval(self):
        """
        compute final error
        """
        return {'l2_error': self.error_sum_l2_error / self.length,
                'loss_error': self.error_sum_loss_error / self.length}
