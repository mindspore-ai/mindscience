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
"""test loss"""
from mindspore import ops


class TestLoss:
    """test loss"""
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super().__init__()

        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        """abs"""
        num_examples = x.size()[0]

        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h ** (self.d / self.p)) * ops.norm(x.view(num_examples, -1) - y.view(num_examples, -1), self.p,
                                                          1)

        if self.reduction:
            if self.size_average:
                all_norms = ops.mean(all_norms)
            else:
                all_norms = ops.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        """rel"""
        num_examples = x.shape[0]

        diff_norms = ops.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1)
        y_norms = ops.norm(y.reshape(num_examples, -1), self.p, 1)
        if self.reduction:
            if self.size_average:
                out = ops.mean(diff_norms / y_norms)
            else:
                out = ops.sum(diff_norms / y_norms)
        else:
            out = diff_norms / y_norms
        return out

    def __call__(self, x, y):
        """call"""
        return self.rel(x, y)
