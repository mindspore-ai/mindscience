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
"""class for calculating gradients"""
from mindspore import nn, ops


class UNet(nn.Cell):
    """return first dimension u"""

    def __init__(self, model):
        super(UNet, self).__init__()
        self.model = model

    def construct(self, database):
        return self.model(database)[:, 0]


class VNet(nn.Cell):
    """return second dimension v"""

    def __init__(self, model):
        super(VNet, self).__init__()
        self.model = model

    def construct(self, database):
        return self.model(database)[:, 1]


class PNet(nn.Cell):
    """return third dimension p"""

    def __init__(self, model):
        super(PNet, self).__init__()
        self.model = model

    def construct(self, database):
        return self.model(database)[:, 2]


class Grad(nn.Cell):
    def __init__(self, net):
        super(Grad, self).__init__()
        self.net = net
        self.grad_op = ops.GradOperation()

    def construct(self, database):
        gradient_function = self.grad_op(self.net)
        return gradient_function(database)
