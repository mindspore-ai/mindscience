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
"""loss"""
import mindspore.nn as nn
import mindspore.ops as ops

class RMSELoss(nn.LossBase):
    def __init__(self):
        super(RMSELoss, self).__init__()

    def construct(self, output, target):
        _pointwise_loss = lambda a, b: ((a - b) / b)**2
        d = _pointwise_loss(output, target)
        return self.get_loss(d)

class WMSELoss(nn.LossBase):
    def __init__(self):
        super(WMSELoss, self).__init__()
    
    def construct(self, output, target):
        _pointwise_loss = lambda a, b: ((a - b) ** 2) * ops.exp(b)
        d = _pointwise_loss(output, target)
        return self.get_loss(d)

class MSELoss_zsym(nn.LossBase):
    def __init__(self, coef=1.):
        super(MSELoss_zsym, self).__init__()
        self.coef = coef

    def construct(self, output, target):
        # output contains prediction from two channels.
        # true output = (xm + xp) / 2
        # LOSS = MSELoss(true output, target) + c * (xm - xp)**2
        _pointwise_loss = lambda a, b: ((a[0] + a[1]) / 2. - b)**2 + self.coef * ((a[0] - a[1])**2)
        d = _pointwise_loss(output, target)
        return self.get_loss(d)

