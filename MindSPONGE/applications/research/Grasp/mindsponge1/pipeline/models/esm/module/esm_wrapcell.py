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
"""TrainOneStepCell"""
import mindspore as ms
import mindspore.nn as nn


class TrainOneStepCell(nn.Cell):
    """training"""
    def __init__(self, network, optimizer):
        super(TrainOneStepCell, self).__init__(auto_prefix=False)
        self.network = network
        self.network.set_grad()
        self.optimizer = optimizer
        self.weights = self.optimizer.parameters
        self.grad = ms.ops.GradOperation(get_by_list=True)

    def construct(self, inputs):
        """Train net construction"""
        loss = self.network((inputs['coords'], inputs['padding_mask'], inputs['confidence'],
                             inputs['prev_output_tokens']), label=inputs['target'])
        grads = \
            self.grad(self.network, self.weights)((inputs['coords'],
                                                   inputs['padding_mask'],
                                                   inputs['confidence'],
                                                   inputs['prev_output_tokens']),
                                                  inputs['target'])
        self.optimizer(grads)
        return loss
