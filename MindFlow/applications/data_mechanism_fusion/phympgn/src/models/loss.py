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
"""loss"""
from mindspore import nn, ops


class TwoStepLoss(nn.Cell):
    """TwoStepLoss"""
    def __init__(self):
        super().__init__()
        self.loss_func = nn.MSELoss()

    def construct(self, u_pred, truth, mask=None):
        """construct"""
        pred1 = u_pred[1]  # [bxn, 2]
        predn = u_pred[-1]
        new_pred = ops.stack((pred1, predn), axis=0)

        truth1 = truth[1]
        truthn = truth[-1]
        new_truth = ops.stack((truth1, truthn), axis=0)

        if mask is None:
            loss = self.loss_func(new_pred, new_truth)
        else:
            loss = self.loss_func(new_pred[:, mask], new_truth[:, mask])
        return loss
