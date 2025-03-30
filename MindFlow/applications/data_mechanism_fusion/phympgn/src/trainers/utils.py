# Copyright 2025 Huawei Technologies Co., Ltd
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
"""trainer utils"""
import numpy as np
from mindspore import ops


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def compute_armse(pred, truth):
    """compute armse"""
    armses = []
    for i in range(pred.shape[0]):
        nume = ops.norm(pred[:i+1] - truth[:i+1])
        deno = ops.norm(truth[:i+1])
        res = nume / deno
        armses.append(res)

    return armses


def correlation(u, truth):
    """compute correlation"""
    u = u.reshape(1, -1)
    truth = truth.reshape(1, -1)
    u_truth = np.concatenate((u, truth), axis=0)
    coef = np.corrcoef(u_truth)[0][1]
    return coef


def cal_cur_time_corre(u, truth):
    """
    compute correlation per time
    """
    coef_list = []
    for i in range(u.shape[0]):
        cur_truth = truth[i]
        cur_u = u[i]
        cur_coef = correlation(cur_u, cur_truth)
        coef_list.append(cur_coef)
    return coef_list


def compute_average_correlation(pred_list, truth_list):
    """compute average correlation
    """
    corr_data = []
    for i in range(len(pred_list)):
        pred = pred_list[i]
        truth = truth_list[i]
        coef_list = cal_cur_time_corre(pred, truth)
        corr_data.append(np.array(coef_list))

    corr = np.mean(corr_data, axis=0)  # [b, t] -> [t,]

    return corr
