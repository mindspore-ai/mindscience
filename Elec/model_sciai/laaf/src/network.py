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

"""LAAF network"""
from mindspore import ops, nn

from sciai.architecture.basic_block import MSE


class RecoverySlopeLoss(nn.Cell):
    """Recovery slope loss"""
    def __init__(self, mlp):
        super(RecoverySlopeLoss, self).__init__()
        self.mlp = mlp
        self.mse = MSE()

    def construct(self, x_train, y_train):
        """Network forward pass"""
        y_pred = self.mlp(x_train)
        loss = self.mse(y_pred - y_train) + \
               1.0 / (ops.reduce_mean(sum(ops.exp(ops.reduce_mean(a)) for a in self.mlp.a_value())))
        return loss, y_pred

    def a_values_np(self):
        return ops.stack(self.mlp.a_value())
