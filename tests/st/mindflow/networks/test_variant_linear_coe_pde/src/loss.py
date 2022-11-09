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
construct loss function
"""
import mindspore.common.dtype as mstype
from mindspore import nn, ops


class LpLoss(nn.Cell):
    """
    Define loss function.
    """

    def __init__(self, p=2, batch_size=28, size_average=True, reduction=True):
        super(LpLoss, self).__init__()
        self.p = p
        self.reduction = reduction
        self.size_average = size_average
        self.batch_size = batch_size
        self.norm = ops.LpNorm(axis=1, p=self.p)

    def construct(self, x, y):
        """construct function of LpLoss"""
        num_examples = x.shape[0]
        cast = ops.Cast()
        x = cast(x, mstype.float32)
        y = cast(y, mstype.float32)
        diff_norms = self.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1))
        y_norms = self.norm(y.reshape(num_examples, -1))

        res = 0
        if self.reduction:
            if self.size_average:
                res = (diff_norms / y_norms).mean()
            else:
                res = (diff_norms / y_norms).sum()
        else:
            res = diff_norms / y_norms

        return res
