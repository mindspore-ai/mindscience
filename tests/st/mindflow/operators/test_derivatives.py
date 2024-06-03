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
# ==============================================================================
# pylint: disable=W0235
"""
test derivation
"""
import numpy as np
import pytest

from mindspore import dtype as mstype
from mindspore import nn, ops, Tensor

from mindflow.operators import batched_jacobian, batched_hessian

np.random.seed(123456)


# pylint: disable=C0111
class Net(nn.Cell):
    def __init__(self, cin=2, cout=1, hidden=10):
        """Test Net for Derivation"""
        super().__init__()
        self.fc1 = nn.Dense(cin, hidden)
        self.fc2 = nn.Dense(hidden, hidden)
        self.fcout = nn.Dense(hidden, cout)
        self.act = ops.Tanh()

    def construct(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.fcout(x)
        return x


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_batched_jacobian_ascend():
    """
    Feature: Test batched jacobian in platform ascend.
    Description: The input type of batched_jacobian is Tensor.
    Expectation: Success or throw AssertionError.
    """
    model = Net()
    jacobian = batched_jacobian(model)
    inputs = np.random.random(size=(3, 2))
    res = jacobian(Tensor(inputs, mstype.float32))
    assert res.shape == (1, 3, 2)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_batched_hessian_ascend():
    """
    Feature: Test batched hessian in platform ascend.
    Description: The input type of batched_hessian is Tensor.
    Expectation: Success or throw AssertionError.
    """
    model = Net()
    hessian = batched_hessian(model)
    inputs = np.random.random(size=(3, 2))
    res = hessian(Tensor(inputs, mstype.float32))
    assert res.shape == (1, 2, 3, 2)
