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
# ==============================================================================
"""
loss
"""
import mindspore
import mindspore.nn as nn
from mindspore import ops
from mindspore.ops import functional as F
from mindspore.ops import operations as P


class RelativeRMSELoss(nn.LossBase):
    '''Relative Root Mean Squared Error Loss'''
    def __init__(self, reduction="sum"):
        super(RelativeRMSELoss, self).__init__(reduction=reduction)

    def construct(self, prediction, label):
        prediction = P.Cast()(prediction, mindspore.float32)
        batch_size = prediction.shape[0]
        diff_norms = F.square(prediction.reshape(batch_size, -1) - label.reshape(batch_size, -1)).sum(axis=1)
        label_norms = F.square(label.reshape(batch_size, -1)).sum(axis=1)
        rel_error = ops.div(F.sqrt(diff_norms), F.sqrt(label_norms))
        return self.get_loss(rel_error)
