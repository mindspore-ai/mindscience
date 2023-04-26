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
"""
loss
"""
from mindspore import nn


class LpLoss(nn.LossBase):
    '''loss function'''
    def __init__(self, d=2, p=2, reduction="mean"):
        super(LpLoss, self).__init__(reduction=reduction)

        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.norm = nn.Norm(axis=1)

    def construct(self, x, y):
        num_examples = x.shape[0]

        diff_norms = self.norm(
            x.reshape(num_examples, -1) - y.reshape(num_examples, -1))
        y_norms = self.norm(y.reshape(num_examples, -1))
        return self.get_loss(diff_norms/y_norms)
